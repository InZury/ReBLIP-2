# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import io
import json
import os
import pickle
import re
import shutil
import tarfile
import yaml
import urllib
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

from typing import Optional
from urllib.parse import urlparse
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive
from lavis.utils.logger import logger
from lavis.utils.dist_utils import download_cached_file
from lavis.utils.registry import registry


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_file_name):
    return re.match(r"^https?://", url_or_file_name, re.IGNORECASE) is not None


def get_cache_path(relative_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), relative_path))


def get_abs_path(relative_path):
    return os.path.join(registry.get_path("library_root"), relative_path)


def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def makedir(dir_path):
    is_success = False

    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)

        is_success = True
    except OSError as error:
        logger.error(f"Error creating directory: {dir_path} : {error}")

    return is_success


def get_redirected_url(url: str):
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url


def to_google_drive_download_url(view_url: str) -> str:
    splits = view_url.split('/')

    assert splits[-1] == "view"

    return f"https://drive.google.com/uc?export=download&id={splits[-2]}"  # file_id


def download_google_drive_url(url: str, output_path: str, output_file_name: str):
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    url = f"{url}&confirm{value}"

        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get("Context-length", 0))

            with open(path, "wb") as file:
                from tqdm import tqdm

                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                        file.write(block)
                        progress_bar.update(len(block))


def get_google_drive_file_id(url: str) -> Optional[str]:
    parse = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parse.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parse.path)

    if match is None:
        return None

    return match.group("id")


def url_retrieve(url: str, file_name: str, chunk_size: int = 1024) -> None:
    with open(file_name, "wb") as file:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as progress_bar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break

                    progress_bar.update(chunk_size)
                    file.write(chunk)


def download_url(url: str, root: str, file_name: Optional[str] = None, md5: Optional[str] = None) -> None:
    root = os.path.expanduser(root)

    if not file_name:
        file_name = os.path.basename(url)

    file_path = os.path.join(root, file_name)

    makedir(root)

    if check_integrity(file_path, md5):
        logger.info(f"Using downloaded and verified file: {file_path}")
        return

    url = get_redirected_url(url)
    file_id = get_google_drive_file_id(url)

    if file_id is not None:
        return download_file_from_google_drive(file_id, root, file_name, md5)

    try:
        logger.info(f"Downloading {url} to {file_path}")
        url_retrieve(url, file_path)
    except (urllib.error.URLError, IOError) as error:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            logger.warning(f"Failed download. Trying https -> http instead.")
            logger.warning(f"Downloading {url} to {file_path}")

            url_retrieve(url, file_path)
        else:
            raise error

    if not check_integrity(file_path, md5):
        raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
        url: str, download_root: str,
        extract_root: Optional[str] = None, file_name: Optional[str] = None,
        md5: Optional[str] = None, remove_finished: bool = False
) -> None:
    download_root = os.path.expanduser(download_root)

    if extract_root is None:
        extract_root = download_root
    if not file_name:
        file_name = os.path.basename(url)

    download_url(url, download_root, file_name, md5)

    archive = os.path.join(download_root, file_name)
    logger.info(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)


def cache_url(url: str, cache_dir: str) -> str:
    parse = urlparse(url)
    dir_name = os.path.join(cache_dir, os.path.dirname(parse.path.lstrip("/")))

    makedir(dir_name)

    file_name = url.split('/')[-1]
    cached = os.path.join(dir_name, file_name)

    with file_lock(cached):
        if not os.path.isfile(cached):
            logger.info(f"Downloading {url} to {cached} ...")
            cached = download(url, dir_name, filename=file_name)

    logger.info(f"URL {url} cached in {cached}")

    return cached


def create_file_symlink(source_file, link_file):
    try:
        if g_pathmgr.exists(link_file):
            g_pathmgr.rm(link_file)

        g_pathmgr.symlink(source_file, link_file)
    except OSError as error:
        logger.error(f"Could Not create symlink. : {error}")


def save_file(data, file_name, append_to_json=True, verbose=True):
    if verbose:
        logger.info(f"Saving data to file: {file_name}")

    file_extension = os.path.splitext(file_name)[1]

    if file_extension in [".pkl", ".pickle"]:
        with g_pathmgr.open(file_name, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    elif file_extension == ".npy":
        with g_pathmgr.open(file_name, "wb") as file:
            np.save(file, data)
    elif file_extension == ".json":
        if append_to_json:
            with g_pathmgr.open(file_name, 'a') as file:
                file.write(f"{json.dumps(data, sort_keys=True)}\n")
                file.flush()
        else:
            with g_pathmgr.open(file_name, 'w') as file:
                file.write(f"{json.dumps(data, sort_keys=True)}\n")
                file.flush()
    elif file_extension == ".yaml":
        with g_pathmgr.open(file_name, 'w') as file:
            dump = yaml.dump(data)
            file.write(dump)
            file.flush()
    else:
        raise Exception(f"Saving {file_extension} is not supported yet.")
    if verbose:
        logger.info(f"Saved data to file: {file_name}")


def load_file(file_name, mmap_mode=None, verbose=True, allow_pickle=False):
    if verbose:
        logger.info(f"Loading data from file: {file_name}")

    file_extension = os.path.splitext(file_name)[1]

    if file_extension == ".txt":
        with g_pathmgr.open(file_name, 'r') as file:
            data = file.readlines()
    elif file_extension in [".pkl", "pickle"]:
        with g_pathmgr.open(file_name, "rb") as file:
            data = pickle.load(file, encoding="latin1")  # Compatibility to python2 else encoding="utf-8"
    elif file_extension == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(file_name, "rb") as file:
                    data = np.load(file, allow_pickle=allow_pickle, encoding="latin1", mmap_mode=mmap_mode)
            except ValueError as error:
                logger.error(f"Could not mmap {file_name}: {error}. Trying without g_pathmgr")
                data = np.load(file_name, allow_pickle=allow_pickle, encoding="latin1", mmap_mode=mmap_mode)
                logger.info("Successfully loaded without g_pathmgr")
            except OSError:
                logger.error("Could not mmap without g_pathmgr. Trying without mmap")
                with g_pathmgr.open(file_name, "rb") as file:
                    data = np.load(file, allow_pickle=allow_pickle, encoding="latin1")
        else:
            with g_pathmgr.open(file_name, "rb") as file:
                data = np.load(file, allow_pickle=allow_pickle, encoding="latin1")
    elif file_extension == ".json":
        with g_pathmgr.open(file_name, 'r') as file:
            data = json.load(file)
    elif file_extension == ".yaml":
        with g_pathmgr.open(file_name, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
    elif file_extension == ".csv":
        with g_pathmgr.open(file_name, 'r') as file:
            data = pd.read_csv(file)
    else:
        raise Exception(f"Reading from {file_extension} is not supported yet.")

    return data


def abs_path(resource_path: str):
    regex = re.compile(r"^\w+://")

    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def download_and_untar(url):
    cached_file = download_cached_file(url, check_hash=False, progress=True)
    untarred_dir = os.path.basename(url).split('.')[0]
    parent_dir = os.path.dirname(cached_file)
    full_dir = os.path.join(parent_dir, untarred_dir)

    if not os.path.exists(full_dir):
        with tarfile.open(cached_file) as tar:
            tar.extractall(parent_dir)

    return full_dir


def cleanup_dir(directory):
    if os.path.exists(directory):
        logger.info(f"Deleting directory: {directory}")
        shutil.rmtree(directory)

    logger.info(f"Deleted contents of directory: {directory}")


def get_file_size(file_name):
    return os.path.getsize(file_name) / (1024.0 * 1024.0)


def is_serializable(value):
    try:
        json.dumps(value)

        return True
    except (TypeError, OverflowError):
        return False


def is_convertible_to_int(value):
    return bool(re.match(r"^-?\d+$", str(value)))
