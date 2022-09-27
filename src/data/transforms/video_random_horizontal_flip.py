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

import random
import numpy as np

import mindspore.dataset.transforms.py_transforms as trans

from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoRandomHorizontalFlip(trans.PyTensorOperation):
    """
    Flip every frame of the video with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default: 0.5.

    Examples:
       >>> transforms_list = [transform.VideoRandomHorizontalFlip(0.3)]
       >>> dataset = video_dataset.map(operations=transforms_list, input_columns=["video"])
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video):
        """
        Args:
            video (seq Images): seq video to be flipped.

        Returns:
            seq video: Randomly flipped seq video.
        """
        if random.random() < self.prob:
            # t x h x w
            return np.flip(video, axis=2).copy()
        return video
