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
"""Video transforms functions."""

import numpy as np
import mindspore.dataset.transforms.py_transforms as trans

from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoToTensor(trans.PyTensorOperation):
    """
    Convert the input video frames in type numpy.ndarray of shape (T, H, W, C)
    in the range [0, 255] to numpy.ndarray of shape (C, T, H, W)  in the range
    [-1.0, 1.0] with the desired dtype.

    Args:
       new_order(tuple), new_order of output.

    Examples:
      >>> #  Convert the input video frames in type numpy
      >>> transforms_list1 = [transform.VideoToTensor((3,0,1,2))]
      >>> dataset = video_folder_dataset_1.map(operations=transforms_list1, input_columns=["video"])
   """

    def __init__(self, order=(3, 0, 1, 2)):
        self.order = tuple(order)

    def __call__(self, x):
        """
        Args:
           Video(list): Video to be tensor.

        Returns:
           seq video: Tensor of seq video.
        """
        if isinstance(x, np.ndarray):
            x = x / 255.0
            x = x * 2 - 1.0
            return np.transpose(x, self.order).astype(np.float32)
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
