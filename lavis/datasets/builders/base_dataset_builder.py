# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import shutil
import warnings
import torch.distributed as dist

from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url
from lavis.utils.utils import get_abs_path, get_cache_path
from lavis.utils.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.utils.logger import logger
from lavis.utils.registry import registry
from lavis.processors.base_processor import BaseProcessor


class BaseDatasetBuilder:
    train_dataset_class, eval_dataset_class = None, None
    DATASET_CONFIG_DICT = None

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(config, str):
            self.config = load_dataset_config(config)
        else:
            self.config = config

        self.data_type = self.config.data_type
        self.vision_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.kwargs_processors = {}

    def build_datasets(self):
        if is_main_process():
            self.download_data()
        if is_dist_avail_and_initialized():
            dist.barrier()

        logger.info("Building datasets...")
        datasets = self.build()

        return datasets

    def build_processors(self):
        vision_processor_config = self.config.get("vision_processor")
        text_processor_config = self.config.get("text_processor")

        if vision_processor_config is not None:
            vision_train_config = vision_processor_config.get("train")
            vision_eval_config = vision_processor_config.get("eval")

            self.vision_processors["train"] = self.build_processor_from_config(vision_train_config)
            self.vision_processors["eval"] = self.build_processor_from_config(vision_eval_config)

        if text_processor_config is not None:
            text_train_config = text_processor_config.get("train")
            text_eval_config = text_processor_config.get("eval")

            self.text_processors["train"] = self.build_processor_from_config(text_train_config)
            self.text_processors["eval"] = self.build_processor_from_config(text_eval_config)

        kwargs_processor_config = self.config.get("kwargs_processor")

        if kwargs_processor_config is not None:
            for name, config in kwargs_processor_config.items():
                self.kwargs_processors[name] = self.build_processor_from_config(config)

    @staticmethod
    def build_processor_from_config(config):
        return registry.get_processor_class(config.name).from_config(config) if config is not None else None

    @classmethod
    def default_config_path(cls, types="default"):
        if not hasattr(cls, "DATASET_CONFIG_DICT"):
            raise NotImplementedError("DATASET_CONFIG_DICT is not declared in class.")
        return get_abs_path(cls.DATASET_CONFIG_DICT[types])

    def download_data(self):
        self.download_annotation()
        self.download_vision()

    def download_annotation(self):
        annotations = self.config.build_info.annotations
        splits = annotations.keys()
        cache_root = registry.get_path("cache_root")

        for split in splits:
            info = annotations[split]
            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(urls, str):
                urls = [urls]
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            assert len(urls) == len(storage_paths)

            for url_or_file_name, storage_path in zip(urls, storage_paths):
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                directory_name = os.path.dirname(storage_path)

                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                if os.path.isfile(url_or_file_name):
                    src, dst = url_or_file_name, storage_path

                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logger.info(f"Using existing file {dst}.")
                else:
                    if os.path.isdir(storage_path):
                        raise ValueError(f"Expecting storage_path to be a file path, got directory {storage_path}")
                    else:
                        file_name = os.path.basename(storage_path)

                    download_url(url=url_or_file_name, root=directory_name, filename=file_name)

    def download_vision(self):
        storage_path = self.config.build_info.get(self.data_type).storage
        storage_path = get_cache_path(storage_path)

        if not os.path.exists(storage_path):
            warnings.warn(
                f"""
                The specified path {storage_path} for visual inputs does not exist.
                 Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
            )

    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        annotation_info = build_info.annotations
        vision_info = build_info.get(self.data_type)

        datasets = dict()

        for split in annotation_info.keys():
            if split not in ["train", "eval", "test"]:
                continue

            is_train = split == "train"

            vision_processors = self.vision_processors["train"] if is_train else self.vision_processors["eval"]
            text_processors = self.text_processors["train"] if is_train else self.text_processors["eval"]

            annotation_paths = annotation_info.get(split).storage

            if isinstance(annotation_paths, str):
                annotation_paths = [annotation_paths]

            abs_annotation_paths = []

            for annotation_path in annotation_paths:
                if not os.path.isabs(annotation_path):
                    annotation_path = get_cache_path(annotation_path)

                abs_annotation_paths.append(annotation_path)

            annotation_paths = abs_annotation_paths
            vision_path = vision_info.storage

            if not os.path.isabs(vision_path):
                vision_path = get_cache_path(vision_path)

            if not os.path.exists(vision_path):
                warnings.warn(f"storage path {vision_path} does not exist.")

            dataset_class = self.train_dataset_class if is_train else self.eval_dataset_class
            datasets[split] = dataset_class(
                vision_processors=vision_processors,
                text_processors=text_processors,
                annotation_paths=annotation_paths,
                vision_root=vision_path
            )

        return datasets


class MultiModalDatasetBuilder(BaseDatasetBuilder):
    train_dataset_class, eval_dataset_class = None, None

    def __init__(self, config=None):
        super().__init__(config)

        if isinstance(self.data_type, str):
            self.data_type = [self.data_type]

        self.processors = None

    def build_processor(self, config_name):
        config = self.config.get(config_name)

        return {
            split: self.build_processor_from_config(config.get(split))
            if config is not None else None
            for split in ["train", "eval"]
        }

    def build_processors(self):
        self.text_processors = self.build_processor("text_processor")
        self.processors = {
            split: {
                modality: self.build_processor_from_config(
                    self.config.get(f"{'vision' if 'image' in modality else modality}")
                )
                for modality in self.data_type
            }
            for split in ["train", "eval"]
        }

    def download_multimodal(self, modality):
        storage_path = get_cache_path(self.config.build_info.get(modality).storage)

        if not os.path.exists(storage_path):
            warnings.warn(f"The specified path {storage_path} for {modality} inputs does not exist.")

    def download_data(self):
        self.download_annotation()

        for modality in self.data_type:
            self.download_multimodal(modality)

    @staticmethod
    def get_absolute_path(path):
        if not os.path.isabs(path):
            return get_cache_path(path)

        return path

    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        datasets = {}

        for split, info in build_info.annotations.items():
            if split not in ["train", "eval", "test"]:
                continue

            is_train = split == "train"
            dataset_args = self.get_dataset_args(info, is_train)
            dataset_class = self.train_dataset_class if is_train else self.eval_dataset_class
            datasets[split] = dataset_class(**dataset_args)

        return datasets

    def get_dataset_args(self, info, is_train):
        dataset_args = dict(self.config.build_info.get("kwargs", {}))

        for modality in self.data_type:
            processor_name = f"{'vision' if 'image' in modality else modality}_processor"

            if self.processors is not None:
                dataset_args[processor_name] = self.processors["train" if is_train else "eval"][modality]
            else:
                raise AttributeError()

            mm_path = self.get_absolute_path(self.config.build_info.get(modality).storage)
            dataset_args[f"{'vision' if 'image' in modality else modality}_root"] = mm_path

        dataset_args["text_processor"] = self.text_processors["train" if is_train else "eval"]
        dataset_args["annotation_paths"] = [self.get_absolute_path(path) for path in info.storage]
        dataset_args["modalities"] = self.data_type

        for key in ["vision_processor", "vision_root", "test_processor"]:
            dataset_args.setdefault(key, None)

        return dataset_args


def load_dataset_config(config_path):
    config = OmegaConf.load(config_path).datasets

    return next(iter(config.values))
