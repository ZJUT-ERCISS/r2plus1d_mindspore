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
"""Adaptive Average pooling 3D."""

from mindspore import nn, ops

__all__ = ['AdaptiveAvgPool3D']


class AdaptiveAvgPool3D(nn.Cell):
    """Applies a 3D adaptive average pooling over an input tensor which is typically of shape
    :math:`(N, C, D_{in}, H_{in}, W_{in})` and output shape
    :math:`(N, C, D_{out}, H_{out}, W_{out})`. where :math:`N` is batch size.
                :math:`C` is channel number.

    Args:
        output_size(Union[int, tuple[int]]): The target output size of the form D x H x W.
            Can be a tuple (D, H, W) or a single number D for a cube D x D x D.

    Inputs:
        x(Tensor): The input Tensor in the form of :math:`(N, C, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, the pooled Tensor in the form of :math:`(N, C, D_{out}, H_{out}, W_{out})`.
    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool3D, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.out_d = output_size[0]
        self.out_h = output_size[1]
        self.out_w = output_size[2]
        self.mean_op = ops.ReduceMean(keep_dims=True)
        self.concat_op = ops.Concat(axis=2)
        self.stride_slice_op = ops.StridedSlice()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Average pooling 3D construct."""
        n, c, in_d, in_h, in_w = x.shape

        stride_d = in_d // self.out_d
        stride_h = in_h // self.out_h
        stride_w = in_w // self.out_w
        kernel_d = in_d - stride_d * (self.out_d - 1)
        kernel_h = in_h - stride_h * (self.out_h - 1)
        kernel_w = in_w - stride_w * (self.out_w - 1)

        out = []
        for i in range(self.out_d):
            for j in range(self.out_h):
                for k in range(self.out_w):
                    start_i = i * stride_d
                    start_j = j * stride_h
                    start_k = k * stride_w
                    end_i = start_i + kernel_d
                    end_j = start_j + kernel_h
                    end_k = start_k + kernel_w
                    out.append(self.mean_op(self.stride_slice_op(x,
                                                                 (0, 0, start_i, start_j, start_k),
                                                                 (n, c, end_i, end_j, end_k),
                                                                 (1, 1, 1, 1, 1)), (2, 3, 4)))
        out = self.reshape(self.concat_op(
            out), (n, c, self.out_d, self.out_h, self.out_w))
        return out
