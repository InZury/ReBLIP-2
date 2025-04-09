# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import math

from registry import registry


@registry.register_lr_scheduler("linear_warmup_step_lr")
class LinearWarmupStepLRScheduler:
    def __init__(
        self, optimizer, max_epoch, min_lr, init_lr,
        decay_rate=1, warmup_start_lr=-1, warmup_steps=0, **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, current_epoch, current_step):
        if current_epoch == 0:
            warmup_lr_schedule(
                step=current_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr
            )
        else:
            step_lr_schedule(
                epoch=current_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate
            )


@registry.register_lr_scheduler("linear_warmup_cosine_lr")
class LinearWarmupCosineLRScheduler:
    def __init__(
        self, optimizer, max_epoch, min_lr, init_lr,
        warmup_steps=0, warmup_start_lr=-1, **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, current_epoch, current_step):
        if current_epoch == 0:
            warmup_lr_schedule(
                step=current_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr
            )
        else:
            cosine_lr_schedule(
                epoch=current_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr
            )


@registry.register_lr_scheduler("constant_lr")
class ConstantLRScheduler:
    def __init__(
        self, optimizer, init_lr,
        warmup_start_lr=-1, warmup_steps=0, **kwargs
    ):
        self.optimizer = optimizer
        self.lr = init_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.warmup_steps = warmup_steps

    def step(self, current_epoch, current_step):
        if current_epoch == 0:
            warmup_lr_schedule(
                step=current_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.lr
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch)) + min_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    lr = max(min_lr, init_lr * (decay_rate**epoch))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
