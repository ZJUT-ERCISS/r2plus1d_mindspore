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
""" Dense head."""

from typing import Optional, Union

from mindspore import nn

from src.utils.class_factory import ClassFactory, ModuleType

__all__ = ['DropoutDense']


@ClassFactory.register(ModuleType.LAYER)
class DropoutDense(nn.Cell):
    """
    Dropout + Dense architecture.

    Args:
        input_channel (int): The number of input channel.
        out_channel (int): The number of output channel.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output.
                Eg. `ReLU`. Default: None.
        keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9,
                means dropping out 10% of input. Default: 1.0.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 input_channel: int,
                 out_channel: int,
                 has_bias: bool = True,
                 activation: Optional[Union[str, nn.Cell]] = None,
                 keep_prob: float = 1.0
                 ) -> None:
        super(DropoutDense, self).__init__()

        self.dropout = nn.Dropout(keep_prob)
        self.dense = nn.Dense(input_channel, out_channel, has_bias=has_bias, activation=activation)

    def construct(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.dense(x)
        return x
