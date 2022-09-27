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
""" Convert padding list into tuple in length 6."""

from src.utils.check_param import Validator


def six_padding(padding):
    r"""
    Convert padding list into a tuple of 6 integer.
    If padding is an int, returns `(padding, padding, padding, padding, padding, padding)`,
    If padding's length is 3, returns
        `(padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])`,
    If padding's length is 6, returns
        `(padding[0], padding[1], padding[2], padding[3], padding[4], padding[5])`,

    Args:
        padding(Union[int, tuple, list]): Padding list that has the length of 1, 3 or 6.

    Returns:
        Tuple of shape (6,).
    """

    Validator.check_value_type('padding', padding, (int, tuple, list), 'six_padding')
    if isinstance(padding, int):
        return (padding, padding, padding, padding, padding, padding)
    p_length = len(padding)
    if p_length == 3:
        return (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    if p_length == 6:
        return tuple(padding)
    raise ValueError(
        f'For `six_padding` the length of `padding` must be 1, 3 or 6, but got `{p_length}`.'
    )
