import argparse
import datetime
from typing import Union

import json

import numpy as np
import torch.optim
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torchsummary
import torchvision
torchvision.disable_beta_transforms_warning()

from src.data.utils import load_data_supervised
from src.experiments.utils import *
from src.models.extractors import get_extractor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
if os.path.exists("./wandb.yml"):
    wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
else:
    wandb_cfg = {'mode': 'disabled'}
use_wandb = wandb_cfg["mode"] != "disabled"


def evaluate_on_dataset(
        ds: DataLoader,
        split: str,
        classifier: Module,
        n_classes: int,
        loss_fn: Module,
        current_device: Union[int, torch.device],
        class_thresh: float = 0.5,
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
    :param current_device: Device for training
    :param class_thresh: Classification threshold (for binary classification)
    :param log_step: Global training step. If not set, logs as
                     summary text (atemporal).
    :param writer: TensorBoard writer
    :param use_wandb: If True, log to wandb.
    :return:
    """
    classifier.eval()

    all_y_pred = np.ones((0,))
    all_y_prob = np.ones((0, 1 if n_classes == 2 else n_classes))
    all_y_true = np.ones((0, 1 if n_classes == 2 else n_classes))
    with torch.no_grad():
        scalars = {"loss": 0.}
        for step, (x, y_true) in enumerate(ds, start=0):
            y_true = y_true.cuda().float()
            x = x.cuda()

            with torch.cuda.amp.autocast():
                y_prob = classifier.forward(x)
                loss = loss_fn(y_prob, y_true)
            if n_classes == 2:
                y_pred = torch.greater_equal(torch.sigmoid(y_prob), class_thresh).to(torch.int64).squeeze()
            else:
                y_pred = torch.argmax(y_prob, 1, keepdim=False)
            all_y_prob = np.concatenate([all_y_prob, y_prob.cpu().numpy()])
            all_y_pred = np.concatenate([all_y_pred, y_pred.cpu().numpy()])
            all_y_true = np.concatenate([all_y_true, y_true.cpu().numpy()])
            scalars["loss"] += loss.item()
        scalars["loss"] /= len(ds)
        scalars.update(
            get_classification_metrics(n_classes, all_y_prob, all_y_pred, all_y_true)
        )
        log_scalars(split, scalars, log_step, writer, use_wandb)
    classifier.to(current_device)
    return scalars


def train_classifier(
        classifier: Module,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        epochs: int,
        n_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_path: str,
        writer: SummaryWriter,
        test_eval: bool = False,
        class_thresh: float = 0.5,
        metric_of_interest: str = "loss",
        use_class_weights: bool = True
):
    # Define loss function
    if n_classes == 2:
        if use_class_weights:
            pos_weight = torch.tensor(train_ds.dataset.label_freqs[0] / train_ds.dataset.label_freqs[1]).cuda()
            print("POS WEIGHT", pos_weight)
            loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = BCEWithLogitsLoss()
    else:
        if use_class_weights:
            class_weights = torch.tensor(np.reciprocal(train_ds.dataset.label_freqs) / 2.).cuda()
            loss_fn = CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()
    log_interval = args["log_interval"]
    batches_per_epoch = len(train_ds)
    best_val_metric = np.inf if "loss" in metric_of_interest else -np.inf

    for epoch in range(epochs):
        classifier.train(True)
        print(f"Epoch {epoch + 1}/{epochs} " + "=" * 50 + "\n")

        start_step = epoch * batches_per_epoch
        epoch_step = 0
        for global_step, (x, y_true) in enumerate(train_ds, start=start_step):
            epoch_step += 1

            # Pass inputs to device
            x = x.cuda()
            y_true = y_true.cuda().float()

            # Forward & backward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                y_prob = classifier.forward(x)
                loss = loss_fn(y_prob, y_true)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log the loss and pertinent metrics.
            if global_step % log_interval == 0:
                scalars = {"loss": loss, "lr_head": optimizer.param_groups[0]['lr'], "epoch": epoch}
                if n_classes == 2:
                    y_pred = torch.greater_equal(torch.sigmoid(y_prob), class_thresh).to(torch.int64)
                else:
                    y_pred = torch.argmax(y_prob, 1, keepdim=False)
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
        if val_ds is not None:
            val_metrics = evaluate_on_dataset(
                val_ds,
                "val",
                classifier,
                n_classes,
                loss_fn,
                current_device,
                class_thresh,
                start_step + epoch_step,
                writer,
                use_wandb
            )
            print(f"Epoch {epoch + 1}: " +
                  ", ".join([f"val/{m}: {val_metrics[m]:.4f}" for m in val_metrics]))

            val_loss = val_metrics["loss"]
            scheduler.step()  # Update LR if necessary

            # Save checkpoint if validation loss has improved
            old_best_val_metric = best_val_metric
            best, best_val_metric = check_model_improvement(
                metric_of_interest,
                val_metrics[metric_of_interest],
                best_val_metric
            )
            if best and rank == 0:
                print(f"Val loss improved from {old_best_val_metric} to {best_val_metric}. "
                      f"Updating checkpoint.")
                torch.save(classifier.state_dict(), checkpoint_path)

    # Evaluate checkpoint with lowest validation loss on the test set
    if test_eval:
        classifier.load_state_dict(torch.load(checkpoint_path))
        test_metrics = evaluate_on_dataset(
            test_ds,
            "test",
            classifier,
            n_classes,
            loss_fn,
            current_device,
            class_thresh,
            None,
            writer,
            use_wandb
        )
        print(f"Test metrics:\n {test_metrics}")
        print(json.dumps(
            {m: test_metrics[m] for m in test_metrics},
            indent=4)
        )
    else:
        test_metrics = None
    return test_metrics


def single_train(run_cfg):

    # Associate artifact with corresponding pre-training method
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_cfg['project'],
            job_type=f"train_{experiment_type}",
            entity=wandb_cfg['entity'],
            config=run_cfg,
            sync_tensorboard=True,
            mode=wandb_cfg['mode']
        )
        print(f"Run config: {wandb_run}")
    else:
        wandb_run = None
        print("Running experiment offline.")

    # Load extractor weights, if necessary
    extractor_weights = run_cfg['extractor_weights']
    extractor_type = run_cfg['extractor_type']
    n_cutoff_layers = run_cfg['n_cutoff_layers']
    freeze_prefix = run_cfg['freeze_prefix']
    extractor = get_extractor(
        extractor_type,
        extractor_weights == 'imagenet',
        n_cutoff_layers,
        freeze_prefix
    )
    if extractor_weights in ['scratch', 'imagenet']:
        pretrain_method = "fully_supervised"
    else:
        extractor, pretrain_method = restore_extractor(
            extractor,
            extractor_weights,
            use_wandb,
            wandb_run,
            freeze_prefix
        )

    # Get identifying info for data artefacts
    data_artifact = wandb_cfg['data_artifact']
    data_version = wandb_cfg['data_version']
    splits_artifact = wandb_cfg['splits_artifact']
    splits_version = wandb_cfg['splits_version']

    # Obtain remaining experiment attributes
    us_mode = run_cfg['us_mode']
    label_col = run_cfg['label']
    redownload_data = args['redownload_data'] in ['Yes', 'yes', 'y', 'Y']
    percent_train = run_cfg['prop_train']
    seed = run_cfg['seed']
    channels = run_cfg['num_channels']
    batch_size = run_cfg['batch_size']
    height = run_cfg['height']
    width = run_cfg['width']
    augment_pipeline = run_cfg['augment_pipeline']
    if augment_pipeline == 'ncus':
        augment_kwargs = {k.lower(): v for k, v in cfg['augment']['ncus'].items()}
    else:
        augment_kwargs = {}
    img_dim = (channels, height, width)
    run_cfg['img_dim'] = img_dim
    n_workers = run_cfg["num_workers"]
    priority_metric = run_cfg["priority_metric"]
    print(f"Run config:\n {run_cfg}")

    # Prepare training, validation, and test sets.
    train_ds, val_ds, test_ds, train_df, val_df, test_df = load_data_supervised(
        cfg,
        batch_size,
        us_mode,
        label_col,
        data_artifact,
        splits_artifact,
        image_dir=run_cfg['image_dir'],
        splits_dir=run_cfg['splits_dir'],
        run=wandb_run,
        data_version=data_version,
        splits_version=splits_version,
        redownload_data=redownload_data,
        percent_train=percent_train,
        height=height,
        width=width,
        channels=channels,
        augment_pipeline=augment_pipeline,
        n_workers=n_workers,
        seed=seed,
        **augment_kwargs
    )
    run_test = args['test_eval'] == 'Y'

    # Define training callbacks
    if run_cfg['checkpoint_name'] is None:
        checkpoint_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        checkpoint_name = run_cfg['checkpoint_name']
    checkpoint_dir = os.path.join(
        cfg['paths']['model_weights'],
        'supervised',
        pretrain_method,
        experiment_type,
        checkpoint_name
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Save run config
    with open(os.path.join(checkpoint_dir, 'run_cfg.json'), 'w') as f:
        json.dump(run_cfg, f)

    # Freeze extractor if using linear eval
    if experiment_type in ['linear', 'mlp']:
        for param in extractor.parameters():
            param.requires_grad = False
    if experiment_type == 'mlp':
        fc_nodes = run_cfg['mlp_hidden_layers']
    else:
        fc_nodes = []

    # Initialize classifier
    n_classes = train_df[label_col].nunique()
    h_dim = extractor(torch.randn((1,) + img_dim)).shape[-1]
    head = get_classifier_head(h_dim, fc_nodes, n_classes)
    classifier = Sequential(extractor, head).cuda()
    torchsummary.summary(extractor, input_size=(channels, width, height))
    torchsummary.summary(head, input_size=(h_dim,))
    print(classifier)

    # Define an optimizer that assigns different learning rates to the
    # extractor and head.
    epochs = run_cfg['epochs']
    lr_extractor = run_cfg['lr_extractor']
    lr_head = run_cfg['lr_head']
    weight_decay = run_cfg['weight_decay']
    momentum = run_cfg['momentum']
    param_groups = [dict(params=head.parameters(), lr=lr_head)]
    if experiment_type == "fine-tune":
        param_groups.append(dict(params=extractor.parameters(), lr=lr_extractor))
    if run_cfg['optimizer'] == "adam":
        optimizer = Adam(param_groups, 0, weight_decay=weight_decay)
    elif run_cfg['optimizer'] == "sgd":
        optimizer = SGD(param_groups, 0, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"{run_cfg['optimizer']} is an unsupported optimizer.")
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1, verbose=True) 

    train_classifier(
        classifier,
        train_ds,
        val_ds,
        test_ds,
        epochs,
        n_classes,
        optimizer,
        scheduler,
        checkpoint_path,
        writer,
        run_test,
        metric_of_interest=priority_metric
    )


def kfold_cross_validation(run_cfg):

    test_metrics = {}
    if run_cfg['checkpoint_name'] is None:
        checkpoint_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        checkpoint_name = run_cfg['checkpoint_name']
    k = run_cfg["folds"]

    for i in range(1, k + 1):
        if use_wandb:
            wandb_run = wandb.init(
                project=wandb_cfg['project'],
                job_type=f"train_{experiment_type}",
                entity=wandb_cfg['entity'],
                config=run_cfg,
                sync_tensorboard=True,
                mode=wandb_cfg['mode']
            )
            print(f"Run config: {wandb_run}")
        else:
            wandb_run = None
            print("Running experiment offline.")

        # Load extractor weights, if necessary
        extractor_weights = run_cfg['extractor_weights']
        extractor_type = run_cfg['extractor_type']
        n_cutoff_layers = run_cfg['n_cutoff_layers']
        freeze_prefix = run_cfg['freeze_prefix']
        extractor = get_extractor(
            extractor_type,
            extractor_weights == 'imagenet',
            n_cutoff_layers
        )
        if extractor_weights in ['scratch', 'imagenet']:
            pretrain_method = "fully_supervised"
        else:
            extractor, pretrain_method = restore_extractor(
                extractor,
                extractor_weights,
                use_wandb,
                wandb_run,
                freeze_prefix
            )

        # Get identifying info for data artefacts
        data_artifact = wandb_cfg['data_artifact']
        data_version = wandb_cfg['data_version']
        splits_artifact = wandb_cfg['splits_artifact']
        splits_version = wandb_cfg['splits_version']

        # Obtain remaining experiment attributes
        us_mode = run_cfg['us_mode']
        label_col = run_cfg['label']
        redownload_data = args['redownload_data'] in ['Yes', 'yes', 'y', 'Y']
        percent_train = run_cfg['prop_train']
        seed = run_cfg['seed']
        channels = run_cfg['num_channels']
        batch_size = run_cfg['batch_size']
        height = run_cfg['height']
        width = run_cfg['width']
        img_dim = (channels, height, width)
        augment_pipeline = run_cfg['augment_pipeline']
        run_cfg['img_dim'] = img_dim
        n_workers = run_cfg["num_workers"]
        priority_metric = run_cfg["priority_metric"]
        print(f"Run config:\n {run_cfg}")

        # Prepare training, validation, and test sets.
        train_ds, val_ds, test_ds, train_df, val_df, test_df = load_data_supervised(
            cfg,
            batch_size,
            us_mode,
            label_col,
            data_artifact,
            splits_artifact,
            image_dir=run_cfg['image_dir'],
            splits_dir=run_cfg['splits_dir'],
            run=wandb_run,
            data_version=data_version,
            splits_version=splits_version,
            redownload_data=redownload_data,
            percent_train=percent_train,
            height=height,
            width=width,
            channels=channels,
            augment_pipeline=augment_pipeline,
            n_workers=n_workers,
            seed=seed,
            fold=i,
            k=k
        )

        # Define training callbacks
        checkpoint_dir = os.path.join(
            cfg['paths']['model_weights'],
            'supervised',
            pretrain_method,
            experiment_type,
            checkpoint_name,
            f"fold{i}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        log_dir = os.path.join(checkpoint_dir, "logs", f"fold{i}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        # Save run config
        with open(os.path.join(checkpoint_dir, 'run_cfg.json'), 'w') as f:
            json.dump(run_cfg, f)

        # Freeze extractor if using linear eval
        if experiment_type in ['linear', 'mlp']:
            for param in extractor.parameters():
                param.requires_grad = False
        if experiment_type == 'mlp':
            fc_nodes = run_cfg['mlp_hidden_layers']
        else:
            fc_nodes = []

        # Initialize classifier
        n_classes = train_df[label_col].nunique()
        h_dim = extractor(torch.randn((1,) + img_dim)).shape[-1]
        head = get_classifier_head(h_dim, fc_nodes, n_classes)
        classifier = Sequential(extractor, head).cuda()
        if i == 1:
            torchsummary.summary(extractor, input_size=(channels, width, height))
            torchsummary.summary(head, input_size=(h_dim,))

        # Define an optimizer that assigns different learning rates to the
        # extractor and head.
        epochs = run_cfg['epochs']
        lr_extractor = run_cfg['lr_extractor']
        lr_head = run_cfg['lr_head']
        weight_decay = run_cfg['weight_decay']
        momentum = run_cfg['momentum']
        param_groups = [dict(params=head.parameters(), lr=lr_head)]
        if experiment_type == "fine-tune":
            param_groups.append(dict(params=extractor.parameters(), lr=lr_extractor))
        if run_cfg['optimizer'] == "adam":
            optimizer = Adam(param_groups, 0, weight_decay=weight_decay)
        elif run_cfg['optimizer'] == "sgd":
            optimizer = SGD(param_groups, 0, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"{run_cfg['optimizer']} is an unsupported optimizer.")
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0., last_epoch=-1, verbose=True) 

        fold_test_metrics = train_classifier(
            classifier,
            train_ds,
            val_ds,
            test_ds,
            epochs,
            n_classes,
            optimizer,
            scheduler,
            checkpoint_path,
            writer,
            True,
            metric_of_interest=priority_metric
        )
        if i == 1:
            test_metrics = {m: [fold_test_metrics[m]] for m in fold_test_metrics}
        else:
            for m in test_metrics:
                test_metrics[m].append(fold_test_metrics[m])

    metric_names = list(test_metrics.keys())
    for m in metric_names:
        test_metrics[f"{m}_mean"] = np.mean(test_metrics[m])
        test_metrics[f"{m}_std"] = np.std(test_metrics[m])
    print("Cross-validation results:")
    for key, val in test_metrics.items():
        print(f'{key}: {val}')
    with open(os.path.join(os.path.dirname(checkpoint_dir), 'kfold_results.json'), 'w') as f:
        json.dump(test_metrics, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--experiment', required=False, type=str, help='Type of training experiment')
    parser.add_argument('--train', required=False, type=str, default="single", help='"single" or "cross_validation"')
    parser.add_argument('--image_dir', required=False, type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, type=str, help='Root directory containing splits information')
    parser.add_argument('--redownload_data', required=False, type=str, default="N", help='Redownload image data from wandb')
    parser.add_argument('--task', required=False, type=str,
                        help='Downstream task ID. One of {"lung_sliding", "view" "ab_lines", "pe"')
    parser.add_argument('--extractor_weights', required=False, type=str,
                        help='Wandb artefact ID, path to .pth checkpoint, or "scratch"')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default="localhost", type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Backend for distributed package')
    parser.add_argument('--log_interval', default=20, type=int, help='Number of steps after which to log')
    parser.add_argument('--test_eval', required=False, type=str, default='N', help='Evaluate on test set')
    parser.add_argument('--augment_pipeline', required=False, type=str, default="ncus", help='Augmentation pipeline')
    parser.add_argument('--label', required=False, type=str, default="label", help='Label column name')
    parser.add_argument('--num_workers', required=False, type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--checkpoint_name', required=False, type=str, default=None, help='Checkpoint folder name')
    parser.add_argument('--priority_metric', required=False, type=str, help='Metric to prioritize in model evaluation')
    parser.add_argument('--us_mode', required=False, type=str, help='US mode. Either "bmode" or "mmode".')
    parser.add_argument('--min_crop_area', required=False, type=float, help='Min crop fraction for NCUS augmentations')
    parser.add_argument('--max_crop_area', required=False, type=float, help='Max crop fraction for NCUS augmentations')
    parser.add_argument('--min_crop_ratio', required=False, type=float, help='Min crop aspect ratiofor NCUS augmentations')
    parser.add_argument('--max_crop_ratio', required=False, type=float, help='Max crop aspect ratio for NCUS augmentations')
    parser.add_argument('--height', required=False, type=int, help='Image height')
    parser.add_argument('--width', required=False, type=int, help='Image width')
    parser.add_argument('--epochs', required=False, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, help='Batch size')
    parser.add_argument('--prop_train', required=False, type=float, help='Fraction of training set to use, in [0, 1]')
    parser.add_argument('--lr_extractor', required=False, type=float, help='Learning rate for feature extractor')
    parser.add_argument('--lr_head', required=False, type=float, help='Learning rate for model head')
    parser.add_argument('--extractor_type', required=False, type=str, help='Architecture of feature extractor')
    parser.add_argument('--freeze_prefix', nargs='*', default=[], help='Prefixes for layers we wish to freeze')
    args = vars(parser.parse_args())

    torch.manual_seed(cfg['train']['seed'])
    np.random.seed(cfg['train']['seed'])
    world_size = args["world_size"]
    if world_size > 1:
        current_device, rank = init_distributed_mode(args["dist_backend"], world_size, args["dist_url"])
    else:
        print("Not using distributed mode.")
        current_device = 0
        rank = 0
        torch.cuda.set_device(current_device)

    # Set up configuration for this run
    print("ARGUMENTS:", args)
    run_cfg = {k.lower(): v for k, v in cfg['train'].items()}
    run_cfg.update({k.lower(): v for k, v in cfg['data'].items()})
    run_cfg.update({k: args[k] for k in args if args[k] is not None})
    print(f"RUN CONFIG:\n {run_cfg}")

    # Update config with values from command-line args
    for k in cfg['data']:
        if k in args and args[k] is not None:
            cfg['data'][k] = args[k]
    for k in cfg['train']:
        if k in args and args[k] is not None:
            cfg['train'][k] = args[k]
    if cfg['train']['augment_pipeline'] == 'ncus':
        aug_params = cfg['augment']['ncus']
        for k in aug_params:
            if k in args and args[k] is not None:
                aug_params[k] = args[k]
        cfg['augment']['ncus'] = aug_params

    # Verify experiment type
    experiment_type = run_cfg['experiment']
    assert experiment_type in ['linear', 'fine-tune', 'mlp'], f"Unsupported evaluation type: {experiment_type}"

    if args['train'] == 'single':
        single_train(run_cfg)
    elif args['train'] == 'cross_validation':
        kfold_cross_validation(run_cfg)
    else:
        raise NotImplementedError(f"No training run type named {args['train']}")

    if use_wandb:
        wandb.finish()
