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
"""Resnet3d backbone."""

from typing import List, Optional, Tuple, Union

from mindspore import nn
from mindspore import ops

from src.models.layers.unit3d import Unit3D
from src.models.layers.inflate_conv3d import Inflate3D
from src.utils.check_param import Validator

__all__ = [
    "ResidualBlockBase3D",
    "ResidualBlock3D",
    'ResNet3D',
    'ResNet3D18',  # registration mechanism to use yaml configuration
    'ResNet3D34',  # registration mechanism to use yaml configuration
    'ResNet3D50',  # registration mechanism to use yaml configuration
    'ResNet3D101',  # registration mechanism to use yaml configuration
    'ResNet3D152'  # registration mechanism to use yaml configuration
]


class ResidualBlockBase3D(nn.Cell):
    """
    ResNet3D residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers.
            It can be `Inflate3D`, Conv2Plus1D` or other custom blocks, this block should
            construct a layer where the name of output feature channel size is `mid_channel`
            for the third conv layers. Default: Inflate3D.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.
        **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride",
            "inflate", etc.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase3D(3, 256, conv12=Inflate3D)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 mid_channel: int = 0,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        if not norm:
            norm = nn.BatchNorm3d
        assert group != 1 or base_width == 64, \
            "ResidualBlockBase3D only supports groups=1 and base_width=64"

        self.conv12 = conv12(in_channel=in_channel,
                             mid_channel=mid_channel,
                             out_channel=out_channel,
                             **kwargs)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase3D construct."""
        identity = x

        out = self.conv12(x)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out


class ResidualBlock3D(nn.Cell):
    """
    ResNet3D residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        mid_channel (int): Inner channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers.
            It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this block should
            construct a layer where the name of output feature channel size is `mid_channel`
            for the third conv layers. Default: Inflate3D.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        activation (List[Optional[Union[nn.Cell, str]]]): Activation function which will be stacked
            on top of the normalization layer (if not None), otherwise on top of the conv layer.
            Default: nn.ReLU, None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.
        **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride",
            "inflate", etc.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.msvideo.models.backbones import ResidualBlock3D
        >>> ResidualBlock3D(3, 256, conv12=Inflate3D)
    """

    expansion: int = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 mid_channel: int = 0,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 activation: List[Optional[Union[nn.Cell, str]]] = (nn.ReLU, None),
                 down_sample: Optional[nn.Cell] = None,
                 **kwargs
                 ) -> None:
        super(ResidualBlock3D, self).__init__()
        if not norm:
            norm = nn.BatchNorm3d
        # conv3d doesn't support group>1 now at 1.6.1 version

        out_channel = int(out_channel * (base_width / 64.0)) * group

        self.conv12 = conv12(in_channel=in_channel,
                             out_channel=out_channel,
                             mid_channel=mid_channel,
                             activation=activation,
                             **kwargs)
        self.conv3 = Unit3D(
            self.conv12.mid_channel,
            out_channel * self.expansion,
            kernel_size=1,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlock3D construct."""
        identity = x

        out = self.conv12(x)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Cell):
    """
    ResNet3D architecture.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (Tuple[int]): The numbers of block in different layers.
        stage_channels (Tuple[int]): Output channel for every res stage.
            Default: [64, 128, 256, 512].
        stage_strides (Tuple[Tuple[int]]): Strides for every res stage.
            Default:[[1, 1, 1],
                     [1, 2, 2],
                     [1, 2, 2],
                     [1, 2, 2]].
        group (int): The number of Group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use.
            Default: None.
        down_sample(nn.Cell, optional): Residual block in every resblock, it can transfer the input
            feature into the same channel of output. Default: Unit3D.
        kwargs (dict, optional): Key arguments for "make_res_layer" and resblocks.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, T_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, 2048, 7, 7, 7)`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.msvideo.models.backbones import ResNet3D, ResidualBlock3D
        >>> net = ResNet(ResidualBlock3D, [3, 4, 23, 3])
        >>> x = ms.Tensor(np.ones([1, 3, 16, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 2048, 7, 7)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than
        those used previously.
    The model explicitly reformulate the layers as learning residual functions with
        reference to the layer inputs, instead of learning unreferenced functions.

    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                     (1, 2, 2),
                                                     (1, 2, 2),
                                                     (1, 2, 2)),
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = Unit3D,
                 **kwargs
                 ) -> None:
        super().__init__()
        if not norm:
            norm = nn.BatchNorm3d
        self.norm = norm
        self.in_channels = stage_channels[0]
        self.group = group
        self.base_with = base_width
        self.down_sample = down_sample
        self.conv1 = Unit3D(3, self.in_channels, kernel_size=7, stride=2, norm=norm)
        self.max_pool = ops.MaxPool3D(kernel_size=3, strides=2, pad_mode='same')
        self.layer1 = self._make_layer(
            block,
            stage_channels[0],
            layer_nums[0],
            stride=stage_strides[0],
            norm=self.norm,
            **kwargs)
        self.layer2 = self._make_layer(
            block,
            stage_channels[1],
            layer_nums[1],
            stride=stage_strides[1],
            norm=self.norm,
            **kwargs)
        self.layer3 = self._make_layer(
            block,
            stage_channels[2],
            layer_nums[2],
            stride=stage_strides[2],
            norm=self.norm,
            **kwargs)
        self.layer4 = self._make_layer(
            block,
            stage_channels[3],
            layer_nums[3],
            stride=stage_strides[3],
            norm=self.norm,
            **kwargs)

    def _make_layer(self,
                    block: Optional[nn.Cell],
                    channel: int,
                    block_nums: int,
                    stride: Tuple[int] = (1, 2, 2),
                    norm: Optional[nn.Cell] = nn.BatchNorm3d,
                    **kwargs):
        """Block layers."""
        down_sample = None
        if stride[1] != 1 or self.in_channels != channel * block.expansion:
            down_sample = self.down_sample(
                self.in_channels,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                norm=norm,
                activation=None)
        self.stride = stride
        bkwargs = [{} for _ in range(block_nums)]  # block specified key word args
        temp_args = kwargs.copy()
        for pname, pvalue in temp_args.items():
            if isinstance(pvalue, (list, tuple)):
                Validator.check_equal_int(len(pvalue), block_nums, f'len({pname})')
                for idx, v in enumerate(pvalue):
                    bkwargs[idx][pname] = v
                kwargs.pop(pname)
        layers = []
        layers.append(
            block(
                self.in_channels,
                channel,
                stride=self.stride,
                down_sample=down_sample,
                group=self.group,
                base_width=self.base_with,
                norm=norm,
                **(bkwargs[0]),
                **kwargs
            )
        )
        self.in_channels = channel * block.expansion
        for i in range(1, block_nums):
            layers.append(
                block(self.in_channels,
                      channel,
                      stride=(1, 1, 1),
                      group=self.group,
                      base_width=self.base_with,
                      norm=norm,
                      **(bkwargs[i]),
                      **kwargs
                      )
            )
        return nn.SequentialCell(layers)

    def construct(self, x):
        """Resnet3D construct."""
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet3D18(ResNet3D):
    """
    The class of ResNet3D18 uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(ResNet3D18, self).__init__(ResidualBlockBase3D, (2, 2, 2, 2), **kwargs)


class ResNet3D34(ResNet3D):
    """
    The class of ResNet3D18 uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(ResNet3D34, self).__init__(ResidualBlockBase3D, (3, 4, 6, 3), **kwargs)


class ResNet3D50(ResNet3D):
    """
    The class of ResNet3D18 uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(ResNet3D50, self).__init__(ResidualBlock3D, (3, 4, 6, 3), **kwargs)


class ResNet3D101(ResNet3D):
    """
    The class of ResNet3D101 uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(ResNet3D101, self).__init__(ResidualBlock3D, (3, 4, 23, 3), **kwargs)


class ResNet3D152(ResNet3D):
    """
    The class of ResNet3D152 uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(ResNet3D152, self).__init__(ResidualBlock3D, (3, 8, 36, 3), **kwargs)
