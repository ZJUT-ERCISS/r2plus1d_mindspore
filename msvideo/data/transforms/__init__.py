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
""" Init video transforms """

from .video_center_crop import *
from .video_random_crop import *
from .video_random_horizontal_flip import *
from .video_reorder import *
from .video_rescale import *
from .video_reshape import *
from .video_resize import *
from .video_short_edge_resize import *
from .video_to_tensor import *
from .video_normalize import *
