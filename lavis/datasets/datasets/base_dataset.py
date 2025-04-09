# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import json
import torch
import pandas as pd

from typing import Iterable
from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self, vision_processor=None, text_processor=None, vision_root=None, annotation_paths: list = None):
        self.vision_root = vision_root
        self.annotation = []

        for annotation_path in annotation_paths:
            if any(extend in annotation_path for extend in ["csv", "tsv"]):
                data_frame = pd.read_csv(annotation_path)
                self.annotation.extend(data_frame.to_dict(orient="records"))
            elif "jsonl" in annotation_path:
                with open(annotation_path, 'r') as f:
                    self.annotation.extend([json.loads(line) for line in f])
            else:
                with open(annotation_path, 'r') as f:
                    loaded = json.load(f)

                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                        self.annotation.extend([{"sample_id": key, **value} if isinstance(value, dict)
                                                else {"sample_id": key, "data": value}
                                                for key, value in loaded.items()])

        self.vision_processor = vision_processor
        self.text_processor = text_processor
        self.add_instance_indices()

    def __len__(self):
        return len(self.annotation)

    @staticmethod
    def collator(samples):
        samples = [sample for sample in samples if sample is not None]

        if not samples:
            return {}

        collated_dict = {}
        keys = samples[0].keys()

        for key in keys:
            values = [sample[key] for sample in samples]
            collated_dict[key] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return collated_dict

    def set_processors(self, vision_processor, text_processor):
        self.vision_processor = vision_processor
        self.text_processor = text_processor

    def add_instance_indices(self, key="instance_id"):
        for index, annotation in enumerate(self.annotation):
            annotation[key] = str(index)


class ConcatDataset(data.ConcatDataset):
    def __init__(self, datasets: Iterable[data.Dataset]) -> None:
        super().__init__(datasets)

    def collator(self, samples):
        all_keys = set()

        for sample in samples:
            all_keys.update(sample)

        shared_keys = all_keys

        for sample in samples:
            shared_keys = shared_keys & set(sample.keys())

        samples_shared_keys = []

        for sample in samples:
            samples_shared_keys.append({key: sample[key] for key in sample.keys() if key in shared_keys})

        return self.datasets[0].collator(samples_shared_keys)
