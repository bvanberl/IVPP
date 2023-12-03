import argparse
import datetime
import os
import time
import json

import yaml
import torch.distributed as dist
import torchsummary
import torchvision

from src.models.joint_embedding import JointEmbeddingModel
from src.losses.losses import *
from src.data.utils import load_data_for_pretrain
from src.experiments.utils import *
from src.custom.optimizers import LARS
from src.custom.schedulers import WarmupCosineDecayLR

torchvision.disable_beta_transforms_warning()
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
if os.path.exists("./wandb.yml"):
    wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
else:
    wandb_cfg = {'MODE': 'disabled', 'RESUME_ID': None}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', required=False, type=str, help='SSL Pre-training method')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, default='', type=str, help='Root directory containing splits information')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default="localhost", type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Backend for distributed package')
    parser.add_argument('--log_interval', default=1, type=int, help='Number of steps after which to log')
    parser.add_argument('--max_time_delta', required=False, default=None, type=float, help='Maximum number of seconds between positive pairs')
    parser.add_argument('--max_x_delta', required=False, default=None, type=int, help='Maximum number of seconds between positive pairs')
    parser.add_argument('--sample_weights', type=int, required=False, default=None, help='If 0, no sample weights. If 1, sample weights.')
    parser.add_argument('--lambda_', required=False, default=None, type=float, help='Invariance term weight')
    parser.add_argument('--augment_pipeline', required=False, type=str, default=None, help='Augmentation pipeline')
    parser.add_argument('--num_workers', required=False, type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--checkpoint_name', required=False, type=str, help='Augmentation pipeline')
    parser.add_argument('--us_mode', required=False, type=str, help='US mode. Either "bmode" or "mmode".')
    parser.add_argument('--min_crop_area', required=False, type=float, help='Min crop fraction for NCUS augmentations')
    parser.add_argument('--max_crop_area', required=False, type=float, help='Max crop fraction for NCUS augmentations')
    parser.add_argument('--min_crop_ratio', required=False, type=float, help='Min crop aspect ratio for NCUS augmentations')
    parser.add_argument('--max_crop_ratio', required=False, type=float, help='Max crop aspect ratio for NCUS augmentations')
    parser.add_argument('--blur_prob', required=False, type=float, help='Chance of applying random blur for NCUS augmentations')
    parser.add_argument('--height', required=False, type=int, help='Image height')
    parser.add_argument('--width', required=False, type=int, help='Image width')
    args = vars(parser.parse_args())
    print(f"Args: {args}")

    world_size = args["world_size"]
    if world_size > 1:
        current_device, rank = init_distributed_mode(args["dist_backend"], world_size, args["dist_url"])
    else:
        print("Not using distributed mode.")
        current_device = 0
        rank = 0
        torch.cuda.set_device(current_device)
    seed = args['seed'] if args['seed'] else cfg['pretrain']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Update config with values from command-line args
    for k in cfg['data']:
        if k in args and args[k] is not None:
            cfg['data'][k] = args[k]
    for k in cfg['pretrain']:
        if k in args and args[k] is not None:
            cfg['pretrain'][k] = args[k]
    print(f"Config after parsing args: {cfg}")
    
    method = cfg['pretrain']['method'].lower()
    assert method in ['simclr', 'barlow_twins', 'vicreg', 'uscl', 'ncus_vicreg', 'ncus_barlow_twins', 'ncus_simclr'], \
        f"Unsupported pretraining method: {method}"
    hparams = {k.lower(): v for k, v in cfg['pretrain']['hparams'].pop(method.lower()).items()}
    for k in args:
        if k in hparams and args[k] is not None:
            hparams[k] = args[k]
    hparams["sample_weights"] = bool(hparams["sample_weights"])

    image_dir = args['image_dir'] if args['image_dir'] else cfg["paths"]["images"]
    splits_dir = args['splits_dir'] if args['splits_dir'] else cfg["paths"]["splits"]
    height = cfg['data']['height']
    width = cfg['data']['width']
    batch_size = cfg['pretrain']['batch_size']
    use_unlabelled = cfg['pretrain']['use_unlabelled']
    use_imagenet = cfg['pretrain']['imagenet_weights']

    # Determine data augmentation pipeline
    if args["augment_pipeline"] is not None:
        augment_pipeline = args["augment_pipeline"]
    else:
        augment_pipeline = cfg['pretrain']['augment_pipeline']
    if augment_pipeline in ['ncus', 'uscl']:
        aug_params = cfg['augment'][augment_pipeline]
        for k in aug_params:
            if k in args and args[k] is not None:
                aug_params[k] = args[k]
        hparams['augmentation'] = aug_params
    else:
        hparams['augmentation'] = {}
    print(f"Method hyperparameters: {hparams}")

    channels = 3
    us_mode = cfg['data']['us_mode']   # TODO: Add M-mode
    n_workers = args["num_workers"]

    train_ds, train_df, val_ds, val_set = load_data_for_pretrain(
        image_dir,
        splits_dir,
        method,
        batch_size,
        augment_pipeline=augment_pipeline,
        use_unlabelled=use_unlabelled,
        channels=channels,
        width=width,
        height=height,
        us_mode=us_mode,
        world_size=world_size,
        n_workers=n_workers,
        **hparams
    )

    n_examples = train_df.shape[0]
    batches_per_epoch = len(train_ds)
    img_dim = (batch_size, channels, height, width)
    extractor_name = cfg['pretrain']['extractor']
    use_bias = cfg['pretrain']['use_bias']
    proj_nodes = cfg['pretrain']['proj_nodes']
    n_cutoff_layers = cfg['pretrain']['n_cutoff_layers']

    use_wandb = wandb_cfg["mode"] == "online"
    resume_id = wandb_cfg["resume_id"]

    run_cfg = {
        'seed': seed
    }
    run_cfg.update(cfg['data'])
    run_cfg.update(cfg['pretrain'])
    run_cfg = {k.lower(): v for k, v in run_cfg.items()}
    run_cfg.update(hparams)
    run_cfg.update({"batches_per_epoch": batches_per_epoch})
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_cfg['project'],
            job_type=f"pretrain",
            entity=wandb_cfg['entity'],
            config=run_cfg,
            sync_tensorboard=False,
            tags=["pretrain", method],
            id=resume_id
        )
        print(f"Run config: {wandb_run}")
        model_artifact = wandb.Artifact(f"pretrained_{method}", type="model")
    else:
        wandb_run = None
        model_artifact = None

    # Define the base loss and initialize any regularizers
    if method.lower() in ['simclr', 'ncus_simclr']:
        loss_fn = SimCLRLoss(tau=hparams["tau"], distributed=(world_size > 1)).cuda()
    elif method.lower() in ['barlow_twins', 'ncus_barlow_twins']:
        loss_fn = BarlowTwinsLoss(batch_size, lambda_=hparams["lambda_"], distributed=(world_size > 1)).cuda()
    elif method.lower() in ['vicreg', 'ncus_vicreg']:
        loss_fn = VICRegLoss(batch_size, lambda_=hparams["lambda_"], mu=hparams["mu"], nu=hparams["nu"], distributed=(world_size > 1)).cuda()
    else:
        raise NotImplementedError(f'{method} is not currently supported.')

    model = JointEmbeddingModel(
        img_dim,
        extractor_name,
        use_imagenet,
        proj_nodes,
        extractor_cutoff_layers=n_cutoff_layers,
        projector_bias=use_bias
    ).cuda()
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
    print(model.extractor)
    print(model.projector)
    torchsummary.summary(model.extractor, input_size=(channels, width, height))
    torchsummary.summary(model.projector, input_size=(model.h_dim,))

    epochs = cfg['pretrain']['epochs']
    base_lr = cfg['pretrain']['init_lr']
    warmup_epochs = cfg['pretrain']['warmup_epochs']
    weight_decay = cfg['pretrain']['weight_decay']
    optimizer_name = cfg['pretrain']['optimizer']
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=weight_decay
    )
    scheduler = WarmupCosineDecayLR(
        optimizer,
        warmup_epochs,
        epochs,
        batches_per_epoch,
        base_lr,
        batch_size
    )

    # Set checkpoint/log dir
    if args['checkpoint_name']:
        checkpoint_name = args['checkpoint_name']      
    else:               
        checkpoint_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(cfg['paths']['model_weights'], 'pretrained', method,
                                  us_mode + checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    run_cfg_path = os.path.join(checkpoint_dir, "run_cfg.json")
    with open(run_cfg_path, 'w') as f:
        json.dump(run_cfg, f)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(checkpoint_dir, "logs")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop
    scaler = torch.cuda.amp.GradScaler()
    log_interval = args["log_interval"]
    pretrain_state = {}

    for epoch in range(epochs):
        model.train(True)
        print(f"Epoch {epoch + 1}/{epochs} " + "=" * 30 + "\n")
        if world_size > 1:
            train_ds.sampler.set_epoch(epoch)

        start_step = epoch * len(train_ds)
        epoch_step = 0
        for global_step, (x1, x2, sw) in enumerate(train_ds, start=start_step):
            start_time = time.time()
            epoch_step += 1

            # Pass inputs to device
            x1 = x1.cuda()
            x2 = x2.cuda()
            sw = sw.cuda()

            # Forward & backward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z1, z2 = model.forward(x1, x2)
                loss = loss_fn(z1, z2, sw=sw)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # Update LR

            # Log the loss and pertinent metrics.
            batch_dur = time.time() - start_time
            scalars = {"epoch": int(epoch), "loss": loss, "lr": scheduler.get_lr()[0]}
            scalars.update(loss_fn.get_instance_vars())
            if global_step % log_interval == 0:
                log_scalars("train", scalars, global_step, writer, use_wandb)
                print(f"Step {epoch_step}/{batches_per_epoch}: " +
                      ", ".join([f"{m}: {scalars[m]:.4f}" for m in scalars]) +
                      f", time: {batch_dur:.3f}, rank: {rank}")

        # Log validation set metrics
        if val_ds:
            if world_size > 1:
                dist.barrier()
            model.eval()
            with torch.no_grad():
                val_scalars = {"loss": 0.}
                val_scalars.update({m: 0. for m in loss_fn.get_instance_vars()})
                for val_step, (x1, x2, sw) in enumerate(val_ds, start=0):
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    sw = sw.cuda()

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        z1, z2 = model.forward(x1, x2)
                        loss = loss_fn(z1, z2, sw=sw)

                    # Accumulate validation loss components
                    val_scalars["loss"] += loss
                    step_metrics = loss_fn.get_instance_vars()
                    for m in step_metrics:
                        val_scalars[m] += step_metrics[m]

                # Log validation loss components
                for m in val_scalars:
                    val_scalars[m] /= len(val_ds)
                log_scalars("val", val_scalars, start_step + epoch_step, writer, use_wandb)
                print(f"Epoch {epoch + 1}: " +
                      ", ".join([f"val/{m}: {val_scalars[m]:.4f}" for m in val_scalars]))

        # Save checkpoint
        if rank == 0:
            pretrain_state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                extractor=model.extractor.state_dict(),
                optimizer=optimizer.state_dict(),
                loss_fn=loss_fn.state_dict(),
                pretrain_method=method
            )
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
            torch.save(pretrain_state, checkpoint_path)
            # if use_wandb:
            #     model_artifact.add_file(checkpoint_path)
        model.to(current_device)

    # Save the final pretrained model
    if rank == 0:
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(pretrain_state, model_path)
        if use_wandb:
            model_artifact.add_file(model_path)

    if use_wandb:
        wandb.finish()
