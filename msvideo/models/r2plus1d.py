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
""" nonlocal network."""

import math
from typing import Optional, Tuple

from mindspore import nn
from mindspore.common import initializer as init
from msvideo.models.layers import ResNet3D, ResidualBlockBase3D, ResidualBlock3D
from msvideo.models.layers import Unit3D, DropoutDense, AdaptiveAvgPool3D
from msvideo.utils.class_factory import ClassFactory, ModuleType


__all__ = ['R2Plus1dNet',
           'R2Plus1d18',
           'R2Plus1d50']


class Conv2Plus1d(nn.Cell):
    """R(2+1)d conv12 block. It implements spatial-temporal feature extraction in
        a sperated way.

    Args:
        in_channels (int):  The number of channels of input frame images.
        out_channels (int):  The number of channels of output frame images.
        kernel_size (tuple): The size of the spatial-temporal convolutional layer kernels.
        stride (Union[int, Tuple[int]]): Stride size for the convolutional layer. Default: 1.
        group (int): Splits filter into groups, in_channels and out_channels must be divisible by the number
            of groups. Default: 1.
        norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.BatchNorm3d.
        activation (Optional[nn.Cell]): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, its channel size is calculated from in_channel, out_channel and kernel_size.
    """

    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 norm=nn.BatchNorm3d,
                 activation=nn.ReLU,
                 ):
        super(Conv2Plus1d, self).__init__()
        self.mid_channel = mid_channel
        if not self.mid_channel:
            self.mid_channel = (in_channel * out_channel * kernel_size[1] * kernel_size[2] * 3) // \
                (in_channel * kernel_size[1] * kernel_size[2] + 3 * out_channel)
        self.conv1_0 = Unit3D(in_channel,
                              self.mid_channel,
                              kernel_size=(1, kernel_size[1], kernel_size[2]),
                              stride=(1, stride[1], stride[2]),
                              norm=norm,
                              activation=activation)
        self.conv1_1 = Unit3D(self.mid_channel,
                              out_channel,
                              kernel_size=(kernel_size[0], 1, 1),
                              stride=(stride[0], 1, 1),
                              norm=norm,
                              activation=activation)
        self.conv2_0 = Unit3D(out_channel,
                              self.mid_channel,
                              kernel_size=(1, kernel_size[1], kernel_size[2]),
                              stride=(1, 1, 1),
                              norm=norm,
                              activation=activation)
        self.conv2_1 = Unit3D(self.mid_channel,
                              out_channel,
                              kernel_size=(kernel_size[0], 1, 1),
                              stride=(1, 1, 1),
                              norm=norm,
                              activation=None)

    def construct(self, x):
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        return x


@ClassFactory.register(ModuleType.MODEL)
class R2Plus1dNet(ResNet3D):
    """Generic R(2+1)d generator.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (Tuple[int]): The numbers of block in different layers.
        stage_channels (Tuple[int]): Output channel for every res stage. Default: (64, 128, 256, 512).
        stage_strides (Tuple[Tuple[int]]): Strides for every res stage.
            Default:((1, 1, 1),
                     (2, 2, 2),
                     (2, 2, 2),
                     (2, 2, 2).
        conv12 (nn.Cell, optional): Conv1 and conv2 config in resblock. Default: Conv2Plus1D.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.
        num_classes(int): Number of categories in the action recognition dataset.
        keep_prob(float): Dropout probability in classification stage.
        kwargs (dict, optional): Key arguments for "make_res_layer" and resblocks.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.msvideo.models.backbones.r2plus1d import *
        >>> from mindvision.msvideo.models.backbones.resnet3d import ResidualBlockBase3D
        >>> data = Tensor(np.random.randn(2, 3, 16, 112, 112), dtype=mindspore.float32)
        >>>
        >>> net = R2Plus1dNet(block=ResidualBlockBase3D, layer_nums=[2, 2, 2, 2])
        >>>
        >>> predict = net(data)
        >>> print(predict.shape)
    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                     (2, 2, 2),
                                                     (2, 2, 2),
                                                     (2, 2, 2)),
                 num_classes: int = 400,
                 **kwargs) -> None:
        super().__init__(block=block,
                         layer_nums=layer_nums,
                         stage_channels=stage_channels,
                         stage_strides=stage_strides,
                         conv12=Conv2Plus1d,
                         **kwargs)
        self.conv1 = nn.SequentialCell([nn.Conv3d(3, 45,
                                                  kernel_size=(1, 7, 7),
                                                  stride=(1, 2, 2),
                                                  pad_mode='pad',
                                                  padding=(0, 0, 3, 3, 3, 3),
                                                  has_bias=False),
                                        nn.BatchNorm3d(45),
                                        nn.ReLU(),
                                        nn.Conv3d(45, 64,
                                                  kernel_size=(3, 1, 1),
                                                  stride=(1, 1, 1),
                                                  pad_mode='pad',
                                                  padding=(1, 1, 0, 0, 0, 0),
                                                  has_bias=False),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU()])
        self.avgpool = AdaptiveAvgPool3D((1, 1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(stage_channels[-1] * block.expansion,
                                   num_classes)
        # init weights
        self._initialize_weights()

    def construct(self, x):
        """R2Plus1dNet construct."""
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Init the weight of Conv3d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(
                    init.HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias:
                    cell.bias.set_data(init.initializer(
                        init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(
                    init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(
                    init.Zero(), cell.beta.shape, cell.beta.dtype))


@ClassFactory.register(ModuleType.MODEL)
class R2Plus1d18(R2Plus1dNet):
    """
    The class of R2Plus1d-18 uses the registration mechanism to register,
    need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(R2Plus1d18, self).__init__(block=ResidualBlockBase3D,
                                         layer_nums=(2, 2, 2, 2),
                                         **kwargs)


@ClassFactory.register(ModuleType.MODEL)
class R2Plus1d50(R2Plus1dNet):
    """
    The class of R2Plus1d-50 uses the registration mechanism to register,
    need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(R2Plus1d50, self).__init__(block=ResidualBlock3D,
                                         layer_nums=(3, 4, 6, 3),
                                         **kwargs)
