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
""" Builder of optimizers. """

import inspect

import mindspore as ms

from msvideo.utils.class_factory import ClassFactory, ModuleType


def build_optimizer(cfg, default_args=None):
    opt = ClassFactory.get_instance_from_cfg(
        cfg, ModuleType.OPTIMIZER, default_args)
    return opt


def register_optimizers():
    """Register optimizers from nn.Optimizer in engine. """
    for module_name in dir(ms.nn):
        if not module_name.startswith('__'):
            opt = getattr(ms.nn, module_name)
            if inspect.isclass(opt) and issubclass(opt, ms.nn.Optimizer):
                ClassFactory.register_cls(opt, ModuleType.OPTIMIZER)
