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
""" The public API for dataset. """

from abc import ABCMeta, abstractmethod
from typing import Optional, Callable
import os
import cv2
import numpy as np

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from src.data.download import DownLoad
from src.data.images import imread


class DatasetGenerator:
    """ Dataset generator for getting image path and its corresponding label. """

    def __init__(self, load_data: Callable):
        self.load_data = load_data

    def __getitem__(self, item):
        """Get the image and label for each item."""

        return self.load_data(item)

    def __len__(self):
        """Get the the size of dataset."""
        return len(self.load_data()[0])


class DatasetToMR:
    """Transform dataset to MindRecord."""

    def __init__(self, load_data, destination, split, partition_number, schema_json, shard_id):
        self.load_data = load_data
        self.partition_number = partition_number
        if shard_id is None:
            self.file_name = "{}/{}.mindrecord".format(destination, split)
        else:
            self.file_name = "{}/{}{}.mindrecord".format(destination, split, shard_id)
        self.writer = FileWriter(file_name=self.file_name,
                                 shard_num=partition_number,
                                 overwrite=True)
        self.schema_json = schema_json

    def trans_to_mr(self):
        """Execute transformation from dataset to MindRecord."""
        # Set the header size.
        self.writer.set_header_size(1 << 24)
        # Set the page size.
        self.writer.set_page_size(1 << 26)

        # Create the schema.
        self.writer.add_schema(self.schema_json)

        if list(self.schema_json.keys()) == ["image", "label"]:
            images, labels = self.load_data()
            if isinstance(images, np.ndarray) and isinstance(labels, np.ndarray):
                for data, label in zip(images, labels):
                    data = data[..., [2, 1, 0]] if data.shape[-1] == 3 else data
                    _, img = cv2.imencode('.jpeg', data)
                    data_list = [{"image": img.tobytes(), "label": int(label)}]
                    self.writer.write_raw_data(data_list)
            elif isinstance(images, list) and isinstance(labels, list):
                for data, label in zip(images, labels):
                    with open(data, 'rb') as f:
                        image_data = f.read()
                    data_list = [{"image": image_data, "label": int(label)}]
                    self.writer.write_raw_data(data_list)

        if list(self.schema_json.keys()) == ["img_id", "image", "annotation"]:
            img_ids, image_path_dict, image_anno_dict = self.load_data()
            for img_id in img_ids:
                image_path = image_path_dict[img_id]
                with open(image_path, 'rb') as f:
                    img = f.read()
                annos = np.array(image_anno_dict[img_id], dtype=np.int32)
                img_id = np.array([img_id], dtype=np.int32)
                row = {"img_id": img_id, "image": img, "annotation": annos}
                self.writer.write_raw_data([row])

        self.writer.commit()

        return self.file_name


class Dataset:
    """
    Dataset is the base class for making dataset which are compatible with MindSpore Vision.
    """

    def __init__(self,
                 path: str,
                 split: str,
                 load_data: Callable,
                 batch_size: int,
                 repeat_num: int,
                 shuffle: bool,
                 num_parallel_workers: Optional[int],
                 num_shards: int,
                 shard_id: int,
                 resize: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 mode: Optional[str] = None,
                 columns_list: Optional[list] = None,
                 schema_json: Optional[dict] = None,
                 trans_record: Optional[bool] = None,
                 ) -> None:
        ds.config.set_enable_shared_mem(False)
        self.path = os.path.expanduser(path)
        self.split = split

        if self.split == "infer":
            img_list, id_list = load_data(self.path)
            if mode:
                load_data = lambda *args: [imread(img_list[args[0]], mode), id_list[args[0]]]\
                    if args else [img_list, id_list]
            else:
                load_data = lambda *args: [np.fromfile(img_list[args[0]], dtype="int8"),
                                           id_list[args[0]]] if args else [img_list, id_list]

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat_num = repeat_num
        self.resize = resize
        self.shuffle = shuffle
        self.num_parallel_workers = num_parallel_workers
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.columns_list = columns_list
        self.schema_json = schema_json
        self.trans_record = trans_record

        if self.trans_record:
            file_name = DatasetToMR(load_data=load_data,
                                    destination=self.path,
                                    split=self.split,
                                    partition_number=1,
                                    shard_id=self.shard_id,
                                    schema_json=self.schema_json).trans_to_mr()
            self.dataset = ds.MindDataset(dataset_files=file_name,
                                          num_parallel_workers=num_parallel_workers,
                                          shuffle=self.shuffle,
                                          num_shards=self.num_shards,
                                          shard_id=self.shard_id)
        else:
            self.dataset = ds.GeneratorDataset(DatasetGenerator(load_data),
                                               column_names=self.columns_list,
                                               num_parallel_workers=self.num_parallel_workers,
                                               shuffle=self.shuffle,
                                               num_shards=self.num_shards,
                                               shard_id=self.shard_id)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise NotImplementedError

    def default_transform(self):
        """Default data augmentation."""
        raise NotImplementedError

    def pipelines(self):
        """Data augmentation."""
        if not self.dataset:
            raise ValueError("dataset is None")

        trans = self.transform if self.transform else self.default_transform()

        self.dataset = self.dataset.map(operations=trans,
                                        input_columns=self.columns_list[0],
                                        num_parallel_workers=self.num_parallel_workers)
        if self.target_transform:
            self.dataset = self.dataset.map(operations=self.target_transform,
                                            input_columns=self.columns_list[1],
                                            num_parallel_workers=self.num_parallel_workers)

    def run(self):
        """Dataset pipeline."""
        self.pipelines()
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.repeat(self.repeat_num)

        return self.dataset


class ParseDataset(metaclass=ABCMeta):
    """
    Parse dataset.
    """

    def __init__(self, path: str, shard_id: Optional[int] = None):
        self.download = DownLoad()
        self.path = os.path.expanduser(path)
        self.shard_id = shard_id

    @abstractmethod
    def parse_dataset(self, *args):
        """parse dataset from internet or compression file."""
