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
"""Utility function for weight initialization. TODO: Why should add these? """

import math

import mindspore as msp
from mindspore import ops
from mindspore import Tensor
from mindspore.common import initializer as init
from mindspore.common.initializer import _assignment
from mindspore.common.initializer import _calculate_fan_in_and_fan_out


class UniformBias(init.Initializer):
    """bias uniform initializer"""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def _initialize(self, arr):
        fan_in, _ = _calculate_fan_in_and_fan_out(self.shape)
        bound = 1 / math.sqrt(fan_in)
        bound = Tensor(bound, msp.float32)
        data = ops.uniform(arr.shape, -bound, bound,
                           dtype=msp.float32).asnumpy()
        _assignment(arr, data)
