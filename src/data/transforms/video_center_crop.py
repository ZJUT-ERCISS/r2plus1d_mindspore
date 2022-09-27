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
class VideoCenterCrop(trans.PyTensorOperation):
    """
    Crop each frame of the input video at the center to the given size.
    If input frame of video size is smaller than output size,
    input video will be padded with 0 before cropping.

    Args:
       size (Union[int, sequence]): The output size of the cropped image.
           If size is an integer, a square crop of size (size, size) is returned.
           If size is a sequence of length 2, it should be (height, width).Default:(224,224)

    Examples:
       >>> # crop video frame to a square
       >>> transforms_list1 = [transform.CenterCrop(50)]
       >>> dataset = image_folder_dataset.map(operations=transforms_list1, input_columns=["video"])
       >>>
       >>> # crop image to portrait style
       >>> transforms_list2 = [transform.CenterCrop((60, 40))]
       >>> dataset = image_folder_dataset.map(operations=transforms_list2, input_columns=["video"])
    """

    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, video):
        """
        Args:
            Video(list): Video to be cropped.

        Returns:
            seq video: Cropped seq video.
        """
        _, h, w, _ = video.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))
        return video[:, i:i + th, j:j + tw, :]
