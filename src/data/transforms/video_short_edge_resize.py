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

import cv2
import numpy as np

import mindspore.dataset.transforms.py_transforms as trans

from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoShortEdgeResize(trans.PyTensorOperation):
    """
    Resize the given video sequences (t, h, w, x, c) at the given size.
    And make sure the smallest dimension in (h, w) is 256 pixels.
    Args:
       size(int): Desired output size after resize.
        interpolation (str): TO DO. Default: "bilinear".
    Examples:
       >>> # Resize the given video sequences
       >>> transforms_list1 = [transform.VideoShortEdgeResize((80))]
       >>> video_folder_dataset = video_folder_dataset_1.map(operations=transforms_list1,
       ...                                                 input_columns=["video"])
    """

    def __init__(self, size, interpolation="bilinear"):
        self.size = size
        self.inter = interpolation
        self.interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
        }

    def __call__(self, x):
        """
        Args:
            Video(list): Video to be resized.

        Returns:
            seq video: ResizeD seq video.
        """
        _, h, w, _ = x.shape
        if h < w:
            scale = self.size * 1.0 / h
        else:
            scale = self.size * 1.0 / w

        new_size = (int(scale * w + 0.5), int(scale * h + 0.5))

        resized_img_array_list = [
            cv2.resize(
                img_array,
                new_size,  # The input order for OpenCV is w, h.
                interpolation=self.interpolation_map[self.inter],
            )
            for img_array in x
        ]
        img_array = np.concatenate(
            [np.expand_dims(arr, axis=0) for arr in resized_img_array_list],
            axis=0,
        )
        return img_array
