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
import numbers

import mindspore.dataset.transforms.py_transforms as trans

from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoRandomCrop(trans.PyTensorOperation):
    """
    Crop the given video sequences (t x h x w x c) at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.

    Examples:
       >>> # Randomly crop the given video at a random location.
       >>> transforms_list1 = [transform.VideoRandomCrop((120,120))]
       >>> dataset = video_folder_dataset_1.map(operations=transforms_list1, input_columns=["video"])
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = tuple(size)

    @staticmethod
    def get_params(img, output_size):
        """
        Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, video):
        """
        Args:
            Video(list): Video to be cropped.

        Returns:
            seq video: Randomly cropped seq video.
        """

        i, j, h, w = self.get_params(video, self.size)

        video = video[:, i:i + h, j:j + w, :]
        return video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
