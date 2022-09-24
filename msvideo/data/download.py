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
"""Download dataset from the internet."""

import hashlib
import pathlib
import os
import bz2
import gzip
import tarfile
import zipfile
import ssl
import urllib
import urllib.error
import urllib.request
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

FILE_TYPE_ALIASES = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz")
}

ARCHIVE_TYPE_SUFFIX = [".tar", ".zip"]

COMPRESS_TYPE_SUFFIX = [".bz2", ".gz"]

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


def detect_file_type(filename: str):  # pylint: disable=inconsistent-return-statements
    """Detect file type by suffixes and return tuple(suffix, archive_type, compression)."""
    suffixes = pathlib.Path(filename).suffixes
    if not suffixes:
        raise RuntimeError(f"File `{filename}` has no suffixes that could be used to detect.")
    suffix = suffixes[-1]

    # Check if the suffix is a known alias.
    if suffix in FILE_TYPE_ALIASES:
        return suffix, FILE_TYPE_ALIASES[suffix][0], FILE_TYPE_ALIASES[suffix][1]

    # Check if the suffix is an archive type.
    if suffix in ARCHIVE_TYPE_SUFFIX:
        return suffix, suffix, None

    # Check if the suffix is a compression.
    if suffix in COMPRESS_TYPE_SUFFIX:
        # Check for suffix hierarchy.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]
            # Check if the suffix2 is an archive type.
            if suffix2 in ARCHIVE_TYPE_SUFFIX:
                return suffix2 + suffix, suffix2, suffix
        return suffix, None, suffix


def read_dataset(path: str) -> Tuple[List[str], List[int]]:
    """
    Get the path list and index list of images.
    """
    img_list = list()
    id_list = list()

    idx = 0
    if os.path.isdir(path):
        for img_name in os.listdir(path):
            if pathlib.Path(img_name).suffix in image_format:
                img_path = os.path.join(path, img_name)
                img_list.append(img_path)
                id_list.append(idx)
                idx += 1
    else:
        img_list.append(path)
        id_list.append(idx)
    return img_list, id_list


def label2index(path: str) -> Dict[str, int]:
    """
    Read images directory for getting label and its corresponding index.
    """
    label = sorted(i.name for i in os.scandir(path) if i.is_dir())

    if not label:
        raise ValueError(f"Cannot find any folder in {path}.")

    return dict((j, i) for i, j in enumerate(label))


class DownLoad:
    """Download dataset."""
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                      "AppleWebKit/537.36 (KHTML, like Gecko) " \
                      "Chrome/92.0.4515.131 Safari/537.36"

    @staticmethod
    def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024) -> str:
        """Calculate md5 value."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, file_path: str, md5: Optional[str] = None) -> bool:
        """Check md5 value."""
        return md5 == self.calculate_md5(file_path)

    @staticmethod
    def extract_tar(from_path: str,
                    to_path: Optional[str] = None,
                    compression: Optional[str] = None) -> None:
        """Extract tar format file."""

        with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
            tar.extractall(to_path)

    @staticmethod
    def extract_zip(from_path: str,
                    to_path: Optional[str] = None,
                    compression: Optional[str] = None) -> None:
        """Extract zip format file."""

        compression_mode = zipfile.ZIP_BZIP2 if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(from_path, "r", compression=compression_mode) as zip_file:
            zip_file.extractall(to_path)

    def extract_archive(self, from_path: str, to_path: str = None) -> str:
        """ Extract and  archive from path to path. """
        archive_extractors = {
            ".tar": self.extract_tar,
            ".zip": self.extract_zip,
        }
        compress_file_open = {
            ".bz2": bz2.open,
            ".gz": gzip.open
        }

        if not to_path:
            to_path = os.path.dirname(from_path)

        suffix, archive_type, compression = detect_file_type(
            from_path)  # pylint: disable=unused-variable

        if not archive_type:
            to_path = from_path.replace(suffix, "")
            compress = compress_file_open[compression]
            with compress(from_path, "rb") as rf, open(to_path, "wb") as wf:
                wf.write(rf.read())
            return to_path

        extractor = archive_extractors[archive_type]
        extractor(from_path, to_path, compression)

        return to_path

    def download_file(self, url: str, file_path: str, chunk_size: int = 1024):
        """Download a file."""
        # Define request headers.
        headers = {"User-Agent": self.USER_AGENT}

        with open(file_path, 'wb') as f:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:
                with tqdm(total=response.length, unit='B') as pbar:
                    for chunk in iter(
                            lambda: response.read(chunk_size), b''):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def download_url(self,
                     url: str,
                     path: str = './',
                     filename: Optional[str] = None,
                     md5: Optional[str] = None) -> None:
        """Download a file from a url and place it in root."""
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        if not filename:
            filename = os.path.basename(url)

        file_path = os.path.join(path, filename)

        # Check if the file is exists.
        if os.path.isfile(file_path):
            if not md5 or self.check_md5(file_path, md5):
                return

        # Download the file.
        try:
            self.download_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                url = url.replace("https", "http")
                try:
                    self.download_file(url, file_path)
                except (urllib.error.URLError, IOError) as e:
                    # pylint: disable=protected-access
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.download_file(url, file_path)
                    ssl._create_default_https_context = ssl.create_default_context
            else:
                raise e

    def download_and_extract_archive(self,
                                     url: str,
                                     download_path: str,
                                     extract_path: Optional[str] = None,
                                     filename: Optional[str] = None,
                                     md5: Optional[str] = None) -> None:
        """ Download and extract archive. """
        download_path = os.path.expanduser(download_path)

        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_path, filename, md5)

        archive = os.path.join(download_path, filename)
        self.extract_archive(archive, extract_path)
