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
"""Calculate resized video mean."""

import os
import numpy as np
import cv2


def reisze_mean(data_dir, save_dir=None, height=240, width=320, interpolation='bilinear', norm=True):
    """Calculate mean of resized video frames.

    Args:
        data_dir (str): The directory of videos, the file structure should be like this:
            |-- data_dir
                |-- class1
                    |-- video1-1
                    |-- video1-2
                    ...
                |-- class2
                    |-- video2-1
                    |-- video2-2
        save_dir (Union[str, None]): The directory where saves the resized mean. If None,
            this function will not save it to disk.
        height (int): Height of resized video frames.
        width (int): Width of reiszed video frames.
        interpolation (str): Method of resize the frames, it can be 'bilinear', 'nearest', 'linear',
            'bicubic'. Default: 'bilinear'.
        norm (bool): Whether to normalize resized frames, if True, the resize mean will divided by 255.
    Returns:
        resized mean (numpy.ndarray): Resized mean of video frames in shape of (height, width, 3).

    Example:
        >>> vmean = reisze_mean(data_dir="/home/publicfile/UCF101/train",
        >>>                     save_dir="./",
        >>>                     height=128,
        >>>                     width=128)
        >>> print(vmean.shape)
    """

    inter = interpolation
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_AREA,
        "bicubic": cv2.INTER_CUBIC,
    }
    sum_video = np.zeros((height, width, 3), dtype=np.float64)
    cnt = 0

    classes = os.listdir(data_dir)
    for cls in classes:
        filenames = os.listdir(os.path.join(data_dir, cls))
        for filename in filenames:
            filepath = os.path.join(data_dir, cls, filename)
            cap = cv2.VideoCapture(filepath)
            while True:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (width, height),
                                   interpolation=interpolation_map[inter]).astype(np.float64)
                sum_video = sum_video + frame
                cnt += 1
                print(filepath)
                if ret:
                    break
    cap.release()

    sum_video = sum_video / cnt
    if norm:
        sum_video = sum_video / 255.0
    sum_video = sum_video.astype(np.float32)

    if save_dir:
        np.save(os.path.join(save_dir, "resized_mean.npy"), sum_video)
        print("\nResized mean calculated, the resized mean file is saved at:",
              os.path.join(save_dir, "resized_mean.npy"))
    return sum_video
