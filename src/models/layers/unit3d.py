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
"""Unit3d Module."""

from typing import Optional, Union, Tuple

from mindspore import nn

from src.utils.six_padding import six_padding
from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.LAYER)
class Unit3D(nn.Cell):
    """
    Conv3d fused with normalization and activation blocks definition.

    Args:
        in_channels (int):  The number of channels of input frame images.
        out_channels (int):  The number of channels of output frame images.
        kernel_size (tuple): The size of the conv3d kernel.
        stride (Union[int, Tuple[int]]): Stride size for the first convolutional layer. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad".
            Default: "pad".
        padding (Union[int, Tuple[int]]): Implicit paddings on both sides of the input x.
            If `pad_mode` is "pad" and `padding` is not specified by user, then the padding
            size will be `(kernel_size - 1) // 2` for C, H, W channel.
        dilation (Union[int, Tuple[int]]): Specifies the dilation rate to use for dilated
            convolution. Default: 1
        group (int): Splits filter into groups, in_channels and out_channels must be divisible
            by the number of groups. Default: 1.
        activation (Optional[nn.Cell]): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.
        norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.BatchNorm3d.
        pooling (Optional[nn.Cell]): Pooling layer (if not None) will be stacked on top of all the
            former layers. Default: None.
        has_bias (bool): Whether to use Bias.

    Returns:
        Tensor, output tensor.

    Examples:
        Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(1, 1, 1))
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 pad_mode: str = 'pad',
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 group: int = 1,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 norm: Optional[nn.Cell] = nn.BatchNorm3d,
                 pooling: Optional[nn.Cell] = None,
                 has_bias: bool = False
                 ) -> None:
        super().__init__()
        if pad_mode == 'pad' and padding == 0:
            padding = tuple((k - 1) // 2 for k in six_padding(kernel_size))
        else:
            padding = 0
        layers = [nn.Conv3d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            pad_mode=pad_mode,
                            padding=padding,
                            dilation=dilation,
                            group=group,
                            has_bias=has_bias)
                  ]

        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation())

        self.pooling = None
        if pooling:
            self.pooling = pooling

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        """ construct unit3d"""
        output = self.features(x)
        if self.pooling:
            output = self.pooling(output)
        return output
