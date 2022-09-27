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
class VideoRescale(trans.PyTensorOperation):
    """
    Rescale the input video frames with the given rescale and shift. This operator will rescale the input video
    with: output = image * rescale + shift.

    Args:
        rescale (float): Rescale factor.
        shift (float, str): Shift factor, if `shift` is a string, it should be the path to a `.npy` file with
            shift data in it.

    Examples:
        >>> #  Rescale the input video frames with the given rescale and shift.
        >>> transforms_list1 = [transform.VideoRescale(0.5,0.5)]
        >>> dataset = video_folder_dataset_1.map(operations=transforms_list1, input_columns=["video"])
    """

    def __init__(self, rescale=1 / 255.0, shift=0.0):
        self.rescale = rescale
        self.shift = shift
        if isinstance(shift, str):
            self.shift = -1.0 * np.load(shift)

    def __call__(self, x):
        """
        Args:
          x(numpy.ndarray): Video to be rescaled.

        Returns:
          seq video: Rescaled of seq video.
        """
        x = x * self.rescale + self.shift
        return x.astype(np.float32)
