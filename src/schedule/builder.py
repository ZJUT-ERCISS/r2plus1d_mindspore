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
# ==============================================================================
"""Builder of learning rate schedule."""

from src.utils.check_param import Validator
from src.schedule import warmup_cosine_annealing_lr_v1
from src.schedule import warmup_cosine_annealing_lr_v2
from src.schedule import warmup_step_lr


def get_lr(args):
    """generate learning rate."""
    Validator.check_string(args.lr_scheduler, ['exponential',
                                               'cosine_annealing',
                                               'cosine_annealing_V2',
                                               'cosine_annealing_sample',
                                               'dynamic_lr',
                                               'multi_warmup_epochs_lr',
                                               'multistep',
                                               'lr_ssd'])
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr_v1(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_v2(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)

    return lr
