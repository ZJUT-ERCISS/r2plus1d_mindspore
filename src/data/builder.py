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
"""This module is used to load the corresponding cfg parameters to the corresponding registration class."""

import inspect

import mindspore as ms

from src.utils.class_factory import ClassFactory, ModuleType


def build_dataset_sampler(cfg, default_args=None):
    """ build sampler. """
    dataset_sampler = ClassFactory.get_instance_from_cfg(
        cfg, ModuleType.DATASET_SAMPLER, default_args)
    return dataset_sampler


def build_dataset(cfg, default_args=None):
    """ build dataset. """
    dataset = ClassFactory.get_instance_from_cfg(
        cfg, ModuleType.DATASET, default_args)
    return dataset


def build_transforms(cfg):
    """ build data transform pipeline. """
    cfg_pipeline = cfg
    if not isinstance(cfg_pipeline, list):
        return ClassFactory.get_instance_from_cfg(cfg_pipeline,
                                                  ModuleType.PIPELINE)

    transforms = []
    for transform in cfg_pipeline:
        transform_op = build_transforms(transform)
        transforms.append(transform_op)

    return transforms


def register_builtin_dataset():
    """ register MindSpore builtin dataset class. """
    for module_name in dir(ms.dataset):
        if not module_name.startswith('__'):
            dataset = getattr(ms.dataset, module_name)
            if inspect.isclass(dataset):
                ClassFactory.register_cls(dataset, ModuleType.DATASET)
