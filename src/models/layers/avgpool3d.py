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
"""Average pooling 3D."""

from mindspore import nn, ops

from src.utils.class_factory import ClassFactory, ModuleType

__all__ = ['AvgPool3D', 'GlobalAvgPooling3D']


@ClassFactory.register(ModuleType.LAYER)
class AvgPool3D(nn.Cell):
    """Average pooling for 3d feature.

    Args:
        kernel_size(Union[int, tuple[int]]):
                The size of kernel window used to take the average value, Default: (1, 1, 1).
        strides(Union[int, tuple[int]]): The distance of kernel moving. Default: (1, 1, 1).

    Inputs:
        x(Tensor): The input Tensor.

    Returns:
        Tensor, the pooled Tensor.
    """

    def __init__(self, kernel_size=(1, 1, 1), strides=(1, 1, 1)):
        super(AvgPool3D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        self.kernel_size = kernel_size
        self.strides = strides
        self.mean_op = ops.ReduceMean(keep_dims=True)
        self.concat_op = ops.Concat(axis=2)
        self.stride_slice_op = ops.StridedSlice()
        self.reshape = ops.Reshape()
        self.k0 = self.kernel_size[0]
        self.k1 = self.kernel_size[1]
        self.k2 = self.kernel_size[2]
        self.s0 = self.strides[0]
        self.s1 = self.strides[1]
        self.s2 = self.strides[2]

    def construct(self, x):
        """Average pooling 3D construct."""
        n, c, in_d, in_height, in_width = x.shape

        out_d = (in_d - self.kernel_size[0]) // self.strides[0] + 1
        out_height = (in_height - self.kernel_size[1]) // self.strides[1] + 1
        out_width = (in_width - self.kernel_size[2]) // self.strides[2] + 1

        out = []
        for i in range(out_d):
            for j in range(out_height):
                for k in range(out_width):
                    start_i = i * self.strides[0]
                    start_j = j * self.strides[1]
                    start_k = k * self.strides[2]
                    end_i = start_i + self.kernel_size[0]
                    end_j = start_j + self.kernel_size[1]
                    end_k = start_k + self.kernel_size[2]
                    out.append(self.mean_op(self.stride_slice_op(x,
                                                                 (0, 0, start_i, start_j, start_k),
                                                                 (n, c, end_i, end_j, end_k),
                                                                 (1, 1, 1, 1, 1)), (2, 3, 4)))
        out = self.reshape(self.concat_op(
            out), (n, c, out_d, out_height, out_width))
        return out


@ClassFactory.register(ModuleType.LAYER)
class GlobalAvgPooling3D(nn.Cell):
    """
    A module of Global average pooling for 3D video features.

    Args:
        keep_dims (bool): Specifies whether to keep dimension shape the same as input feature.
            E.g. `True`. Default: False

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 keep_dims: bool = True
                 ) -> None:
        super(GlobalAvgPooling3D, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3, 4))
        return x
