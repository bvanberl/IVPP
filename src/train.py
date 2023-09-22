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
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torchsummary

from src.data.utils import load_data_supervised
from src.experiments.utils import *
from src.models.backbones import get_backbone

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
if os.path.exists("./wandb.yml"):
    wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
else:
    wandb_cfg = {'MODE': 'disabled'}


def evaluate_on_dataset(
        ds: DataLoader,
        split: str,
        classifier: Module,
        n_classes: int,
        loss_fn: Module,
        gpu: torch.device,
        log_step: Optional[int] = None,
        writer: Optional[SummaryWriter] = None,
        use_wandb: bool = False,
):
    """Computes and logs classification metrics

    Given a dataset, obtains a classifier's predictions
    and logs the loss, along with classification metrics.
    :param ds: Evaluation dataset
    :param split: Dataset split. 'train', 'val', or 'test'
    :param classifier: Neural network classifier
    :param n_classes: Number of classes
    :param loss_fn: Loss function
    :param gpu: GPU device
    :param log_step: Global training step. If not set, logs as
                     summary text (atemporal).
    :param writer: TensorBoard writer
    :param use_wandb: If True, log to wandb.
    :return:
    """
    all_y_pred = np.ones((0, 1))
    all_y_prob = np.ones((0, 1 if n_classes == 2 else n_classes))
    all_y_true = np.ones((0, 1))
    with torch.no_grad():
        scalars = {"loss": 0.}
        for step, (x, y_true) in enumerate(ds, start=0):
            y_true = y_true.cuda(gpu, non_blocking=True)
            x = x.cuda(gpu, non_blocking=True)

            with torch.cuda.amp.autocast():
                y_prob = classifier.forward(x)
                loss = loss_fn(y_prob, y_true)
            if n_classes == 2:
                y_pred = torch.greater_equal(y_prob, 0.5).to(torch.int64)
            else:
                y_pred = torch.argmax(y_prob, 1, keepdim=True).numpy()
            all_y_prob = np.concatenate([all_y_prob, y_prob.cpu().detach().numpy()])
            all_y_pred = np.concatenate([all_y_pred, y_pred.cpu().detach().numpy()])
            all_y_true = np.concatenate([all_y_true, y_true.cpu().detach().numpy()])
            scalars["loss"] += loss
        scalars["loss"] /= len(ds)
        scalars.update(
            get_classification_metrics(n_classes, all_y_prob, all_y_pred, all_y_true)
        )
        log_scalars(split, scalars, log_step, writer, use_wandb)
    return scalars


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--experiment', required=False, default='', type=str, help='Type of training experiment')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, default='', type=str, help='Root directory containing splits information')
    parser.add_argument('--redownload_data', required=False, type=str, default="N", help='Redownload image data from wandb')
    parser.add_argument('--task', required=False, type=str,
                        help='Downstream task ID. One of {"lung_sliding", "view" "ab_lines", "pe"')
    parser.add_argument('--backbone_weights', required=False, type=str,
                        help='Wandb artefact ID, path to .pth checkpoint, or "scratch"')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default="localhost", type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Backend for distributed package')
    parser.add_argument('--dist_port', default='12355', type=str, help='Backend for distributed package')
    parser.add_argument('--log_interval', default=20, type=int, help='Number of steps after which to log')
    parser.add_argument('--test_eval', required=False, type=str, default='N', help='Evaluate on test set')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    args = vars(parser.parse_args())

    os.environ['MASTER_ADDR'] = args['dist_url']
    os.environ['MASTER_PORT'] = args['dist_port']
    init_distributed_mode(args["dist_backend"], args["world_size"], args["dist_url"])
    torch.manual_seed(cfg['TRAIN']['SEED'])
    np.random.seed(cfg['TRAIN']['SEED'])
    gpu = torch.device("cuda")

    # Set up configuration for this run
    run_cfg = {k.lower(): v for k, v in cfg['TRAIN'].items()}
    run_cfg.update({k.lower(): v for k, v in cfg['DATA'].items()})
    if args['experiment']:
        run_cfg['experiment'] = args['experiment'].lower()
    for arg in args:
        if args[arg]:
            run_cfg[arg] = args[arg]

    # Verify experiment type
    experiment_type = run_cfg['experiment']
    assert experiment_type in ['linear', 'fine-tune', 'mlp'], f"Unsupported evaluation type: {experiment_type}"

    # Associate artifact with corresponding pre-training method
    use_wandb = wandb_cfg["MODE"] == "online"
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_cfg['PROJECT'],
            job_type=f"train_{experiment_type}",
            entity=wandb_cfg['ENTITY'],
            config=run_cfg,
            sync_tensorboard=True,
            mode=wandb_cfg['MODE']
        )
        print(f"Run config: {wandb_run}")
    else:
        wandb_run = None
        print("Running experiment offline.")

    # Load backbone weights, if necessary
    backbone_weights = run_cfg['backbone_weights']
    backbone_type = run_cfg['backbone_type']
    n_cutoff_layers = run_cfg['n_cutoff_layers']
    backbone = get_backbone(
        backbone_type,
        backbone_weights == 'imagenet',
        n_cutoff_layers
    )
    if backbone_weights in ['scratch', 'imagenet']:
        pretrain_method = "fully_supervised"
    else:
        backbone, pretrain_method = restore_backbone(
            backbone,
            backbone_weights,
            use_wandb,
            wandb_run
        )

    # Get identifying info for data artefacts
    data_artifact = wandb_cfg['DATA_ARTIFACT']
    data_version =  wandb_cfg['DATA_VERSION']
    splits_artifact = wandb_cfg['SPLITS_ARTIFACT']
    splits_version = wandb_cfg['SPLITS_VERSION']

    # Obtain remaining experiment attributes
    label_col = run_cfg['label']
    redownload_data = args['redownload_data'] in ['Yes', 'yes', 'y', 'Y']
    percent_train = run_cfg['prop_train']
    seed = run_cfg['seed']
    channels = run_cfg['num_channels']
    n_layers_frozen = run_cfg['n_cutoff_layers']
    batch_size = run_cfg['batch_size']
    height = run_cfg['height']
    width = run_cfg['width']
    img_dim = (channels, height, width)
    run_cfg['img_dim'] = img_dim
    print(f"Run config:\n {run_cfg}")

    # Prepare training, validation, and test sets.
    train_ds, val_ds, test_ds, train_df, val_df, test_df = load_data_supervised(
        cfg,
        batch_size,
        label_col,
        data_artifact,
        splits_artifact,
        run=wandb_run,
        data_version=data_version,
        splits_version=splits_version,
        redownload_data=redownload_data,
        percent_train=percent_train,
        channels=channels,
        seed=seed
    )

    run_test = args['test_eval'] == 'Y'

    # Define training callbacks
    cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(
        cfg['PATHS']['MODEL_WEIGHTS'],
        'supervised',
        pretrain_method,
        experiment_type,
        cur_datetime
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # Save run config
    with open(os.path.join(checkpoint_dir, 'run_cfg.json'), 'w') as f:
        json.dump(run_cfg, f)

    # Freeze backbone if using linear eval
    backbone.training = (experiment_type in ['linear', 'mlp'])
    if experiment_type == 'mlp':
        fc_nodes = run_cfg['mlp_hidden_layers']
    else:
        fc_nodes = []

    # Initialize classifier
    n_classes = train_df[label_col].nunique()
    h_dim = backbone(torch.randn((1,) + img_dim)).shape[-1]
    head = get_classifier_head(h_dim, fc_nodes, n_classes)
    classifier = Sequential(backbone, head).cuda(gpu)

    # Define an optimizer that assigns different learning rates to the
    # backbone and head.
    epochs = run_cfg['epochs']
    batches_per_epoch = len(train_ds)
    lr_backbone = run_cfg['lr_backbone']
    lr_head = run_cfg['lr_head']
    weight_decay = run_cfg['weight_decay']
    param_groups = [dict(params=head.parameters(), lr=lr_head)]
    if experiment_type == "fine-tune":
        param_groups.append(dict(params=backbone.parameters(), lr=lr_backbone))
    optimizer = SGD(param_groups, 0, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    loss_fn = BCEWithLogitsLoss() if n_classes == 2 else CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    log_interval = args["log_interval"]
    best_val_loss = np.inf

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} " + "=" * 50 + "\n")

        start_step = epoch * batches_per_epoch
        epoch_step = 0
        for global_step, (x, y_true) in enumerate(train_ds, start=start_step):
            epoch_step += 1

            # Pass inputs to device
            x = x.cuda(gpu, non_blocking=True)
            y_true = y_true.cuda(gpu, non_blocking=True)

            # Forward & backward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                y_prob = classifier.forward(x)
                loss = loss_fn(y_prob, y_true)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # Update LR

            # Log the loss and pertinent metrics.
            if global_step % log_interval == 0:
                scalars = {"loss": loss, "lr": scheduler.get_last_lr()[-1], "epoch": epoch}
                if n_classes == 2:
                    y_pred = torch.greater_equal(y_prob, 0.5).to(torch.int64)
                else:
                    y_pred = torch.argmax(y_prob, 1, keepdim=True)
                train_metrics = get_classification_metrics(
                    n_classes,
                    y_prob.cpu().detach().numpy(),
                    y_pred.cpu().detach().numpy(),
                    y_true.cpu().detach().numpy()
                )
                scalars.update(train_metrics)
                log_scalars("train", scalars, global_step, writer, use_wandb)
                print(f"Step {epoch_step + 1}/{batches_per_epoch}: " +
                      ", ".join([f"{m}: {scalars[m]:.4f}" for m in scalars]))

        # Log validation set metrics
        val_metrics = evaluate_on_dataset(
            val_ds,
            "val",
            classifier,
            n_classes,
            loss_fn,
            gpu,
            start_step + epoch_step,
            writer,
            use_wandb
        )
        print(f"Epoch {epoch + 1}: " +
              ", ".join([f"val/{m}: {val_metrics[m]:.4f}" for m in val_metrics]))

        # Save checkpoint if validation loss has improved
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            print(f"Val loss improved from {best_val_loss} to {val_loss}. "
                  f"Updating checkpoint.")
            best_val_loss  = val_loss
            torch.save(classifier.state_dict(), checkpoint_path)

    # Evaluate checkpoint with lowest validation loss on the test set
    if args["test_eval"]:
        classifier.load_state_dict(torch.load(checkpoint_path))
        test_metrics = evaluate_on_dataset(
            test_ds,
            "test",
            classifier,
            n_classes,
            loss_fn,
            gpu,
            None,
            writer,
            use_wandb
        )
        print(f"Test metrics:")
        print(json.dumps(test_metrics, indent=4))

    if use_wandb:
        wandb.finish()
