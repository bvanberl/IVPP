from typing import Dict
import os

import torch
import wandb

from torch.utils.tensorboard import SummaryWriter

def log_scalars(
        step: int,
        split: str,
        metrics_dict: Dict[str, float],
        writer: SummaryWriter,
        use_wandb: bool
):

    for m in metrics_dict:
        writer.add_scalar(f"{m}/{split}", metrics_dict[m], step)
        if use_wandb:
            if m not in ["epoch", "lr"]:
                wandb.log({f"{split}/{m}": metrics_dict[m]}, step=step)
            else:
                wandb.log({f"{m}": metrics_dict[m]}, step=step)

def init_distributed_mode(
        backend: str = "gloo",
        world_size: int = 1,
        dist_url: str = "env://"
):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        init_method = dist_url
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['WORLD_SIZE'] = str(world_size)
        init_method = dist_url
    else:
        rank = 0
        gpu = 0
        init_method = None
        print('Not using distributed mode.')

    torch.cuda.set_device(gpu)
    print('Distributed init (rank {}): {}, gpu {}'.format(
        rank, dist_url, gpu), flush=True)
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    torch.distributed.barrier()