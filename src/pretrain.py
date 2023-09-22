import os
import argparse
import datetime
import json

import tqdm
import yaml
import wandb
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchsummary

from src.models.joint_embedding import JointEmbeddingModel
from src.losses.losses import *
from src.data.utils import load_data_for_pretrain
from src.experiments.utils import *
from src.custom.optimizers import LARS
from src.custom.schedulers import WarmupCosineDecayLR

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
if os.path.exists("./wandb.yml"):
    wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
else:
    wandb_cfg = {'MODE': 'disabled'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', required=False, default='', type=str, help='SSL Pre-training method')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, default='', type=str, help='Root directory containing splits information')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default="localhost", type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Backend for distributed package')
    parser.add_argument('--dist_port', default='12355', type=str, help='Backend for distributed package')
    parser.add_argument('--log_interval', default=20, type=int, help='Number of steps after which to log')
    args = vars(parser.parse_args())

    os.environ['MASTER_ADDR'] = args['dist_url']
    os.environ['MASTER_PORT'] = args['dist_port']
    init_distributed_mode(args["dist_backend"], args["world_size"], args["dist_url"])
    torch.manual_seed(cfg['PRETRAIN']['SEED'])
    np.random.seed(cfg['PRETRAIN']['SEED'])
    gpu = torch.device("cuda")

    if args['method']:
        method = args['method'].lower()
    else:
        method = cfg['PRETRAIN']['METHOD'].lower()
    assert method in ['simclr', 'barlow_twins', 'vicreg', 'uscl', 'ncus_vicreg', 'ncus_barlow_twins', 'ncus_simclr'], \
        f"Unsupported pretraining method: {method}"
    hparams = {k.lower(): v for k, v in cfg['PRETRAIN']['HPARAMS'].pop(method.upper()).items()}

    image_dir = cfg["PATHS"]["IMAGES"]
    splits_dir = cfg["PATHS"]["SPLITS"]
    height = cfg['DATA']['HEIGHT']
    width = cfg['DATA']['WIDTH']
    batch_size = cfg['PRETRAIN']['BATCH_SIZE']
    use_unlabelled = cfg['PRETRAIN']['USE_UNLABELLED']
    use_imagenet = cfg['PRETRAIN']['IMAGENET_WEIGHTS']
    augment_pipeline = cfg['PRETRAIN']['AUGMENT_PIPELINE']
    channels = 3 if use_imagenet else 1
    us_mode = "bmode"   # TODO: Add M-mode

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
        **hparams
    )
    n_examples = train_df.shape[0]
    batches_per_epoch = len(train_ds)
    img_dim = (batch_size, channels, height, width)
    backbone_name = cfg['PRETRAIN']['BACKBONE']
    use_bias = cfg['PRETRAIN']['USE_BIAS']
    proj_nodes = cfg['PRETRAIN']['PROJ_NODES']
    n_cutoff_layers = cfg['PRETRAIN']['N_CUTOFF_LAYERS']

    use_wandb = wandb_cfg["MODE"] == "online"
    resume_id = wandb_cfg["RESUME_ID"]
    if use_wandb:
        run_cfg = {
            'seed': cfg['PRETRAIN']['SEED']
        }
        run_cfg.update(cfg['DATA'])
        run_cfg.update(cfg['PRETRAIN'])
        run_cfg = {k.lower(): v for k, v in run_cfg.items()}
        run_cfg.update(hparams)
        run_cfg.update({"batches_per_epoch": batches_per_epoch})

        wandb_run = wandb.init(
            project=wandb_cfg['PROJECT'],
            job_type=f"pretrain",
            entity=wandb_cfg['ENTITY'],
            config=run_cfg,
            sync_tensorboard=False,
            tags=["pretrain", method],
            id=resume_id
        )
        print(f"Run config: {wandb_run}")
    else:
        wandb_run = None


    # Define the base loss and initialize any regularizers
    if method.lower() in ['simclr']:
        loss_fn = SimCLRLoss(tau=hparams["tau"])
    elif method.lower() in ['barlow_twins', 'ncus_barlow_twins']:
        loss_fn = BarlowTwinsLoss(batch_size, lambda_=hparams["lambda_"])
    elif method.lower() in ['vicreg', 'ncus_vicreg']:
        loss_fn = VICRegLoss(batch_size, lambda_=hparams["lambda_"], mu=hparams["mu"], nu=hparams["nu"])
    else:
        raise NotImplementedError(f'{method} is not currently supported.')

    model = JointEmbeddingModel(
        img_dim,
        backbone_name,
        use_imagenet,
        proj_nodes,
        loss_fn,
        backbone_cutoff_layers=n_cutoff_layers,
        projector_bias=use_bias
    ).cuda(gpu)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #torchsummary.summary(model, input_size=(channels, width, height))

    epochs = cfg['PRETRAIN']['EPOCHS']
    base_lr = cfg['PRETRAIN']['INIT_LR']
    warmup_epochs = cfg['PRETRAIN']['WARMUP_EPOCHS']
    weight_decay = cfg['PRETRAIN']['WEIGHT_DECAY']
    optimizer_name = cfg['PRETRAIN']['OPTIMIZER']
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
    cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(cfg['PATHS']['MODEL_WEIGHTS'], 'pretrained', cfg['PRETRAIN']['METHOD'],
                                  us_mode + cur_datetime)
    os.makedirs(checkpoint_dir)
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # Training loop
    scaler = torch.cuda.amp.GradScaler()
    log_interval = args["log_interval"]
    pretrain_state = {}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} " + "=" * 30 + "\n")

        start_step = epoch * len(train_ds)
        epoch_step = 0
        for global_step, (x1, x2, sw) in enumerate(train_ds, start=start_step):
            epoch_step += 1

            # Pass inputs to device
            x1 = x1.cuda(gpu, non_blocking=True)
            x2 = x2.cuda(gpu, non_blocking=True)
            sw = sw.cuda(gpu, non_blocking=True)

            # Forward & backward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x1, x2, sw=sw)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # Update LR

            # Log the loss and pertinent metrics.
            scalars = {"loss": loss, "lr": scheduler.get_lr()[0], "epoch": epoch}
            scalars.update(model.loss.get_instance_vars())
            if global_step % log_interval == 0:
                log_scalars("train", scalars, global_step, writer, use_wandb)
                print(f"Step {epoch_step + 1}/{batches_per_epoch}: " +
                      ", ".join([f"{m}: {scalars[m]:.4f}" for m in scalars]))

        # Log validation set metrics
        with torch.no_grad():
            val_scalars = {"loss": 0.}
            val_scalars.update({m: 0. for m in model.loss.get_instance_vars()})
            for val_step, (x1, x2, sw) in enumerate(val_ds, start=0):
                x1 = x1.cuda(gpu, non_blocking=True)
                x2 = x2.cuda(gpu, non_blocking=True)
                sw = sw.cuda(gpu, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = model.forward(x1, x2, sw=sw)

                # Accumulate validation loss components
                val_scalars["loss"] += loss
                step_metrics = model.loss.get_instance_vars()
                for m in step_metrics:
                    val_scalars[m] += step_metrics[m]

            # Log validation loss components
            for m in val_scalars:
                val_scalars[m] /= len(val_ds)
            log_scalars("val", val_scalars, start_step + epoch_step, writer, use_wandb)
            print(f"Epoch {epoch + 1}: " +
                  ", ".join([f"val/{m}: {val_scalars[m]:.4f}" for m in val_scalars]))

        # Save checkpoint
        pretrain_state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            backbone=model.backbone.state_dict(),
            optimizer=optimizer.state_dict(),
            pretrain_method=method
        )
        torch.save(pretrain_state, os.path.join(checkpoint_dir, "checkpoint.pth"))

    # Save the final pretrained model
    if use_wandb:
        model_path = os.path.join(wandb.run.dir, f'{method}_pretrained')
    else:
        cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(pretrain_state, model_path)

    if use_wandb:
        wandb.finish()