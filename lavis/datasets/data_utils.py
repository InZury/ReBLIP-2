# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import gzip
import os
import random
import tarfile
import zipfile
import cv2
import decord
import torch
import webdataset
import numpy as np

from decord import VideoReader
from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset, ChainDataset
from lavis.utils.logger import logger
from lavis.utils.registry import registry
from lavis.datasets.datasets.base_dataset import ConcatDataset

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


def load_video(video_path, frames=MAX_INT, height=-1, width=-1, sampling="uniform"):
    reader = VideoReader(uri=video_path, height=height, width=width)
    reader_length = len(reader)
    start, end = 0, reader_length
    num_frames = min(frames, reader_length)

    if sampling == "uniform":
        indices = np.arange(start, end, reader_length / num_frames).astype(int)
    elif sampling == "headtail":
        indices_head = sorted(random.sample(range(reader_length // 2), num_frames // 2))
        indices_tail = sorted(random.sample(range(reader_length // 2, reader_length), num_frames // 2))
        indices = indices_head + indices_tail
    else:
        raise NotImplementedError

    # get_batch: C, T, H, W --> T, H, W, C
    return reader.get_batch(indices).permute(3, 0, 1, 2).float()


def apply_to_sample(types, sample):
    if sample is None or len(sample) == 0:
        return {}

    def apply(x):
        if torch.is_tensor(x):
            return types(x)
        elif isinstance(types, dict):
            return {key: apply(value) for key, value in types.items()}
        elif isinstance(types, list):
            return [apply(x) for x in x]
        else:
            return x

    return apply(sample)


def move_to_cuda(sample):
    def move(tensor):
        return tensor.cuda()

    return apply_to_sample(move, sample)


def prepare_sample(samples, enabled=True):
    if enabled:
        samples = move_to_cuda(samples)

    return samples


def reorganize_datasets_by_split(datasets):
    reorganized_datasets = dict()

    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorganized_datasets:
                reorganized_datasets[split_name] = [dataset_split]
            else:
                reorganized_datasets[split_name].append(dataset_split)

    return reorganized_datasets


def concat_datasets(datasets):
    for split_name in datasets:
        if split_name != "train":
            assert (len(datasets[split_name]) == 1
                    ), f"Do not support multiple {split_name} datasets."
        else:
            iterable_datasets, map_datasets = [], []

            for dataset in datasets[split_name]:
                if isinstance(dataset, webdataset.DataPipeline):
                    logger.info(f"Dataset {dataset} is IterableDataset, can not be concatenated.")
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError("Do not support concatenation of generic IterableDataset")
                else:
                    map_datasets.append(dataset)

            chained_datasets = (ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None)
            concatenated_datasets = (ConcatDataset(map_datasets) if len(map_datasets) > 0 else None)

            train_datasets = concatenated_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (train_datasets[0] if len(train_datasets) == 1 else train_datasets)
            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)
    if from_path.endswith((".tar.gz", ".tgz")):
        logger.info(f"Opening tar file {from_path} to {to_path}")
        with tarfile.open(from_path, 'r') as tar:
            files = []

            for file in tqdm(tar):
                file_path = os.path.join(to_path, file.name)

                if file.isfile():
                    files.append(file_path)

                    if os.path.exists(file_path):
                        logger.info(f"{file_path} already extracted.")

                        if not overwrite:
                            continue

                tar.extract(file, to_path)

            logger.info(f"Finished extracting tar file {from_path}.")

            return files
    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logger.info(f"Opening zip file {from_path} to {to_path}.")
        with zipfile.ZipFile(from_path, 'r') as zip_file:
            files = []

            for file in tqdm(zip_file.namelist()):
                file_path = os.path.join(to_path, file)
                files.append(file_path)

                if os.path.exists(file_path):
                    logger.info(f"{file_path} already extracted.")

                    if not overwrite:
                        continue

                zip_file.extract(file, to_path)

        files = [file for file in files if os.path.isfile(file)]
        logger.info(f"Finished extracting zip file {from_path}.")

        return files
    elif from_path.endswith(".gz"):
        logger.info(f"Opening gz file file {from_path} to {to_path}.")
        default_block_size = 65536
        file_name = from_path[:-3]
        files = [file_name]
        with gzip.open(from_path, "rb") as gz_file, open(file_name, "wb") as target_file:
            while True:
                block = gz_file.read(default_block_size)

                if not block:
                    break
                else:
                    target_file.write(block)

            target_file.write(block)

        logger.info(f"Finished extracting gz file {from_path}.")

        return files
    else:
        raise NotImplementedError("We currently only support tar.gz, .tgz, .gz and zip archives.")


def save_frames_grid(image_array, out_path):
    from PIL import Image
    from torchvision.utils import make_grid

    if len(image_array.shape) == 3:
        image_array = image_array.unsqueeze(0)
    elif len(image_array.shape) == 4:
        pass
    elif len(image_array.shape) == 5:
        _, _, channel, height, width = image_array.shape
        image_array = image_array.view(-1, channel, height, width)
    else:
        raise NotImplementedError(
            "Support only (batch, time, channel, height, width)-shaped inputs. "
            "first two dimensions can be ignored."
        )

    assert image_array.shape[1] == 3, "Expecting input shape of (height, width, 3), i.e. RGB-only"

    grid = make_grid(image_array)
    ndarray = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    image = Image.fromarray(ndarray)

    image.save(out_path)


def uniform_frame_sampling(video_path, num_frames, target_height, target_width, start_time=None, end_time=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    start_time = start_time if start_time is not None else 0
    end_time = end_time if end_time is not None else total_frames / frame_rate
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    frame_indices = list(range(start_frame, end_frame + 1, (end_frame - start_frame + 1) // num_frames))

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        result, frame = cap.read()

        if not result:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    cap.release()

    return frames


def head_tail_frame_sampling(video_path, num_frames, target_height, target_width, start_time=None, end_time=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    start_time = start_time if start_time is not None else 0
    end_time = end_time if end_time is not None else total_frames / frame_rate
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    frame_indices = [start_frame] + [start_frame + (end_frame - start_frame) // (num_frames - 1) * i
                                     for i in range(1, num_frames - 1)] + [end_frame]

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        result, frame = cap.read()

        if not result:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return torch.stack([torch.tensor(frame).permute(2, 0, 1).float() for frame in frames], dim=1)


def load_clip(video_path, num_frames, target_height, target_width, start_time=None, end_time=None, sampling="headtail"):
    if sampling == "headtail":
        return head_tail_frame_sampling(video_path, num_frames, target_height, target_width, start_time, end_time)
    elif sampling == "uniform":
        return uniform_frame_sampling(video_path, num_frames, target_height, target_width, start_time, end_time)
    else:
        raise NotImplementedError
