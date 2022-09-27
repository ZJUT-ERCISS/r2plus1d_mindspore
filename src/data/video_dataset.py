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
""" The public API for video dataset. """

import os
from typing import List, Optional, Callable, Union, Tuple
import mindspore.dataset as ds

from src.data.meta import Dataset
from src.data import transforms
from src.data.generator import DatasetGenerator


class VideoDataset(Dataset):
    """
        VideoDataset is the base class for making video dataset which are compatible
                with MindSpore Vision.
     Args:
        path (str): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: "infer".
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label.
                Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(bool): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.
                Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into.
                Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.
        frame_interval(int):The number of frame interval when reading video. Default: 1.
        num_clips(int):The number of video clips read per video.
    """

    def __init__(self,
                 path: str,
                 split: str,
                 load_data: Union[Callable, Tuple],
                 transform: Optional[Callable],
                 target_transform: Optional[Callable],
                 seq: int,
                 seq_mode: str,
                 align: bool,
                 batch_size: int,
                 repeat_num: int,
                 shuffle: bool,
                 num_parallel_workers: Optional[int],
                 num_shards: int,
                 shard_id: int,
                 download: bool,
                 columns_list: List = ['video', 'label'],
                 suffix: str = "video",
                 frame_interval: int = 1,
                 num_clips: int = 1):
        if columns_list[0] == 'image':
            super(VideoDataset, self).__init__(path=path,
                                               split=split,
                                               load_data=load_data,
                                               transform=transform,
                                               target_transform=target_transform,
                                               batch_size=batch_size,
                                               repeat_num=repeat_num,
                                               resize=None,
                                               shuffle=shuffle,
                                               num_parallel_workers=num_parallel_workers,
                                               num_shards=num_shards,
                                               shard_id=shard_id)
        ds.config.set_enable_shared_mem(False)
        self.path = os.path.expanduser(path)
        self.split = split
        self.download = download

        if self.download:
            self.download_dataset()
        self.transform = transform
        self.video_path, self.label = load_data()
        self.target_transform = target_transform
        self.seq = seq
        self.seq_mode = seq_mode
        self.align = align
        self.batch_size = batch_size
        self.repeat_num = repeat_num
        self.shuffle = shuffle
        self.num_parallel_workers = num_parallel_workers
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.columns_list = columns_list
        self.suffix = suffix
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.dataset = ds.GeneratorDataset(DatasetGenerator(path=self.video_path,
                                                            label=self.label,
                                                            seq=self.seq,
                                                            mode=self.seq_mode,
                                                            suffix=self.suffix,
                                                            align=self.align,
                                                            frame_interval=self.frame_interval,
                                                            num_clips=self.num_clips),
                                           column_names=self.columns_list,
                                           num_parallel_workers=num_parallel_workers,
                                           shuffle=self.shuffle,
                                           num_shards=self.num_shards,
                                           shard_id=self.shard_id)

    def download_dataset(self):
        """Download the dataset."""
        raise NotImplementedError

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise NotImplementedError

    def default_transform(self):
        """Set the default transform for video dataset."""
        size = (224, 224)
        order = (3, 0, 1, 2)
        trans = [
            transforms.VideoResize(size),
            transforms.VideoReOrder(order),
        ]

        return trans
