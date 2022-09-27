# Copyright 2021 Huawei Technologies Co., Ltd
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
""" Image io for read and write image. """

import os
import os.path
import pathlib
import cv2
from PIL import Image
import numpy as np

from src.utils.check_param import Validator
from src.data import path

image_format = (
    '.JPEG',
    '.jpeg',
    '.PNG',
    '.png',
    '.JPG',
    '.jpg',
    '.PPM',
    '.ppm',
    '.BMP',
    '.bmp',
    '.PGM',
    '.pgm',
    '.WEBP',
    '.webp',
    '.TIF',
    '.tif',
    '.TIFF',
    '.tiff')

image_mode = (
    '1',
    'L',
    'RGB',
    'RGBA',
    'CMYK',
    'YCbCr',
    'LAB',
    'HSV',
    'I',
    'F'
)


def imread(image, mode=None):
    """
    Read an image.

    Args:
        image (ndarray or str or Path): Ndarry, str or pathlib.Path.
        mode (str): Image mode.

    Returns:
        ndarray: Loaded image array.
    """
    Validator.check_string(mode, image_mode)

    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        path.check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """
    Write image to file.

    Args:
        image (ndarray): Image array to be written.
        image_path (str): Image file path to be written.
        auto_mkdir (bool): `image_path` does not exist create it automatically.

    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def imshow(img, win_name='', wait_time=0):
    """
    Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)
