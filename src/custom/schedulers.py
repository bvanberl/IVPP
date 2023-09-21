import math
from typing import List

from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineDecayLR(LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_epochs: int,
            total_epochs: int,
            steps_per_epoch: int,
            base_lr: float,
            batch_size: int,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        self.warmup_steps = steps_per_epoch * warmup_epochs
        self.total_steps = steps_per_epoch * total_epochs
        self.decay_steps = self.total_steps - self.warmup_steps
        self.max_lr = base_lr * batch_size / 256.
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        step = self.last_epoch      # This is actually number of steps
        if step < self.warmup_steps:
            lr = self.max_lr * step / self.warmup_steps
        else:
            step -= self.warmup_steps
            q = 0.5 * (1. + math.cos(math.pi * step / self.decay_steps))
            final_lr = self.max_lr * 0.001
            lr = self.max_lr * q + final_lr * (1. - q)
        return [lr for _ in self.optimizer.param_groups]
