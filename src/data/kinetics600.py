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
""" KINETICS600 dataset. TODO: finish these dataset API. """
import json
import os
import csv
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.dataset.video_dataset import VideoDataset

__all__ = ["Kinetic600", "ParseKinetic600"]


@ClassFactory.register(ModuleType.DATASET)
class Kinetic600(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" , "val" or "infer". Default: train.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(boolean): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.msvideo.dataset.kinetics600 import Kinetic600
        >>> dataset = Kinetic600("./data","train")
        >>> dataset = dataset.run()

    The directory structure of Kinetic-600 dataset looks like:

        .
        |-kinetic-600
            |-- train
            |   |-- ___qijXy2f0_000011_000021.mp4      // video file
            |   |-- ___dTOdxzXY_000022_000032.mp4      // video file
            |    ...
            |-- test
            |   |-- __Zh0xijkrw_000042_000052.mp4       // video file
            |   |-- __zVSUyXzd8_000070_000080.mp4   // video file
            |-- val
            |   |-- __wsytoYy3Q_000055_000065.mp4       // video file
            |   |-- __vzEs2wzdQ_000026_000036.mp4   // video file
            |    ...
            |-- kinetics-600_train.csv             //training dataset label file.
            |-- kinetics-600_test.csv              //testing dataset label file.
            |-- kinetics-600_val.csv               //validation dataset label file.
            |-- kinetics-600_infer.csv               //infer dataset label file defined by user.
            ...
    """

    def __init__(self,
                 path,
                 split=None,
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False
                 ):
        load_data = ParseKinetic600(os.path.join(path, split)).parse_dataset
        super().__init__(path=path,
                         split=split,
                         load_data=load_data,
                         transform=transform,
                         target_transform=target_transform,
                         seq=seq,
                         seq_mode=seq_mode,
                         align=align,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         download=download)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        parse_kinetic600 = ParseKinetic600(os.path.join(self.path, self.split))
        mapping, _ = parse_kinetic600.load_cls_file()
        return mapping

    def download_dataset(self):
        """Download the Kinetic600 data if it doesn't exist already."""
        raise ValueError("Kinetic600 dataset download is not supported.")


class ParseKinetic600(ParseDataset):
    """
    Parse kinetic-600 dataset.
    """
    urlpath = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz"

    def load_cls_file(self):
        """Parse the category file."""
        base_path = os.path.dirname(self.path)
        csv_file = os.path.join(base_path, "kinetics-600_train.csv")
        cls2id = {}
        id2cls = []
        cls_file = os.path.join(base_path, "cls2index.txt")
        if os.path.isfile(cls_file):
            with open(cls_file, "r")as f:
                cls2id = json.load(f)
            id2cls = [*cls2id]
            return id2cls, cls2id
        with open(csv_file, "r")as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                if row['label'] not in cls2id:
                    cls2id.setdefault(row['label'], len(cls2id))
                    id2cls.append(row['label'])
        f.close()
        os.mknod(cls_file)
        with open(cls_file, "w")as f:
            f.write(json.dumps(cls2id))
        return id2cls, cls2id

    def parse_dataset(self, *args):
        """Traverse the Kinetic600 dataset file to get the path and label."""
        parse_kinetic600 = ParseKinetic600(self.path)
        split = os.path.split(parse_kinetic600.path)[-1]
        video_label, video_path = [], []
        _, cls2id = self.load_cls_file()
        csv_file = os.path.join(os.path.dirname(parse_kinetic600.path), f"kinetics-600_{split}.csv")
        with open(csv_file, "rb")as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                start = row['time_start'].zfill(6)
                end = row['time_end'].zfill(6)
                file_name = f"{row['youtube_id']}_{start}_{end}.mp4"
                video_path.append(os.path.join(self.path, file_name))
                video_label.append(cls2id[row['label']])
        return video_path, video_label
