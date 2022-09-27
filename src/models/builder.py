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
"""builder of model and layer"""

from src.utils.class_factory import ClassFactory, ModuleType


def build_model(cfg):
    """build model"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.MODEL)


def build_layer(cfg):
    """build layer"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.LAYER)
