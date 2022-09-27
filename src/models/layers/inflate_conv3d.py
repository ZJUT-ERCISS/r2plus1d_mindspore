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
"""Inflate conv3d block."""

from typing import List, Optional, Union

from mindspore import nn

from src.models.layers.unit3d import Unit3D
from src.utils.class_factory import ClassFactory, ModuleType

__all__ = ['Inflate3D']


@ClassFactory.register(ModuleType.LAYER)
class Inflate3D(nn.Cell):
    """
    Inflate3D block definition.

    Args:
        in_channel (int):  The number of channels of input frame images.
        out_channel (int):  The number of channels of output frame images.
        mid_channel (int): The number of channels of inner frame images.
        kernel_size (tuple): The size of the spatial-temporal convolutional layer kernels.
        stride (Union[int, Tuple[int]]): Stride size for the second convolutional layer. Default: 1.
        conv2_group (int): Splits filter into groups for the second conv layer,
            in_channels and out_channels
            must be divisible by the number of groups. Default: 1.
        norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.BatchNorm3d.
        activation (List[Optional[Union[nn.Cell, str]]]): Activation function which will be stacked
            on top of the normalization layer (if not None), otherwise on top of the conv layer.
            Default: nn.ReLU, None.
        inflate (int): Whether to inflate two conv3d layers and with different kernel size.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.msvideo.models.blocks import Inflate3D
        >>> Inflate3D(3, 64, 64)
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 mid_channel: int = 0,
                 stride: tuple = (1, 1, 1),
                 kernel_size: tuple = (3, 3, 3),
                 conv2_group: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm3d,
                 activation: List[Optional[Union[nn.Cell, str]]] = (nn.ReLU, None),
                 inflate: int = 1,
                 ):
        super(Inflate3D, self).__init__()
        if not norm:
            norm = nn.BatchNorm3d
        self.in_channel = in_channel
        if mid_channel == 0:
            self.mid_channel = (in_channel * out_channel * kernel_size[1] * kernel_size[2] * 3) // \
                               (in_channel * kernel_size[1] * kernel_size[2] + 3 * out_channel)
        else:
            self.mid_channel = mid_channel
        self.inflate = inflate
        if self.inflate == 0:
            conv1_kernel_size = (1, 1, 1)
            conv2_kernel_size = (1, kernel_size[1], kernel_size[2])
        elif self.inflate == 1:
            conv1_kernel_size = (kernel_size[0], 1, 1)
            conv2_kernel_size = (1, kernel_size[1], kernel_size[2])
        elif self.inflate == 2:
            conv1_kernel_size = (1, 1, 1)
            conv2_kernel_size = (kernel_size[0], kernel_size[1], kernel_size[2])
        self.conv1 = Unit3D(
            self.in_channel,
            self.mid_channel,
            stride=(1, 1, 1),
            kernel_size=conv1_kernel_size,
            norm=norm,
            activation=activation[0])
        self.conv2 = Unit3D(
            self.mid_channel,
            self.mid_channel,
            stride=stride,
            kernel_size=conv2_kernel_size,
            group=conv2_group,
            norm=norm,
            activation=activation[1])

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
