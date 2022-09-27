# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Learning rate schedule generator."""

from collections import Counter
import math
import numpy as np


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)

def warmup_cosine_annealing_lr_v1(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)

def warmup_cosine_annealing_lr_v2(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate V2."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    last_lr = 0
    last_epoch_v1 = 0

    t_max_v2 = int(max_epoch / 3)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            if i < total_steps * (2 / 3):
                lr = eta_min + (base_lr - eta_min) *\
                     (1. + math.cos(math.pi * last_epoch / t_max)) / 2
                last_lr = lr
                last_epoch_v1 = last_epoch
            else:
                base_lr = last_lr
                last_epoch = last_epoch - last_epoch_v1
                lr = eta_min + (base_lr - eta_min) *\
                     (1. + math.cos(math.pi * last_epoch / t_max_v2)) / 2

        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)

def cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def dynamic_lr(base_lr, steps_per_epoch, warmup_steps, warmup_ratio, epoch_size):
    """dynamic learning rate generator"""
    total_steps = steps_per_epoch * (epoch_size + 1)
    warmup_steps = int(warmup_steps)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * warmup_ratio))
        else:
            lr.append(cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr
