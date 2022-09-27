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

from src.utils.check_param import Validator
from src.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoNormalize(trans.PyTensorOperation):
    r"""
    VideoNormalize the input numpy.ndarray video of shape (C, T, H, W) with the specified
        mean and standard deviation.

    .. math::

        output_{c} = \frac{input_{c} - mean_{c}}{std_{c}}

    Note:
        The values of the input image need to be in the range [0.0, 1.0]. If not so,
            call `VideoReOrder` and `VideoRescale` first.

    Args:
        mean (Union[float, sequence]): list or tuple of mean values for each channel,
            arranged in channel order. The values must be in the range [0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.
        std (Union[float, sequence]): list or tuple of standard deviation values for each channel,
            arranged in channel order. The values must be in the range (0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.

    Raises:
        TypeError: If the input is not numpy.ndarray.
        TypeError: If the dimension of input is not 4.
        NotImplementedError: If the dtype of input is a subdtype of np.integer.
        ValueError: If the lengths of the mean and std are not equal.
        ValueError: If the length of the mean or std is neither equal to the channel length nor 1.

    Examples:
        >>> from mindvision.msvideo.dataset import transforms
        >>> transforms_list = Compose([transforms.VideoReOrder(((3, 0, 1, 2))),
        ...                            transforms.VideoRescale((rescale=1 / 255.0, shift=0))
        ...                            transforms.Normalize((0.43216, 0.394666, 0.37645), (0.247, 0.243, 0.262))])
        >>> # apply the transform to dataset through map function
        >>> video_dataset = video_dataset.map(operations=transforms_list,
        ...                                     input_columns="video")
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        Validator.check_equal_int(len(self.mean), len(self.std), "length of mean and std")

    def __call__(self, video):
        r"""
        Call method.

        Args:
            video (numpy.ndarray): numpy.ndarray to be normalized.

        Returns:
            video (numpy.ndarray), Normalized video.
        """

        Validator.check_value_type("input_video", video, np.ndarray)
        Validator.check_equal_int(video.ndim, 4, "video ndim")

        if np.issubdtype(video.dtype, np.integer):
            raise NotImplementedError(
                f'''Unsupported video datatype: `{video.dtype}`, pls execute `VideoReOrder`
                    and `VideoRescale` before `VideoNormalize`.''')

        num_channels = video.shape[0]  # shape is (C, T ,H, W)
        mean = self.mean
        std = self.std
        # if length equal to 1, adjust the mean and std arrays to have the correct
        # number of channels (replicate the values)
        if len(mean) == 1:
            mean = [mean[0]] * num_channels
            std = [std[0]] * num_channels
        elif len(self.mean) != num_channels:
            raise ValueError(
                f"Length of mean and std must both be 1 or equal to the number of channels({num_channels}).")

        mean = np.array(mean, dtype=video.dtype)
        std = np.array(std, dtype=video.dtype)

        video = (video - mean[:, None, None, None]) / std[:, None, None, None]
        return video
