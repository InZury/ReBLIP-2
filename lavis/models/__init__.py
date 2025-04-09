# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import torch

from omegaconf import OmegaConf
from lavis.models.base_model import BaseModel
from lavis.utils.registry import registry
from lavis.utils.logger import logger
from lavis.processors.base_processor import BaseProcessor

__all__ = [
    "BaseModel"
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)
    if is_eval:
        model.eval()
    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    def build_proc_from_config(configs):
        return (
            registry.get_processor_class(configs.name).from_config(configs)
            if configs is not None else BaseProcessor()
        )

    vision_processors = dict()
    text_processors = dict()

    vision_processor_config = config.get("vision_processor")
    text_processor_config = config.get("text_processor")

    if vision_processor_config is not None:
        vision_train_config = vision_processor_config.get("train")
        vision_eval_config = vision_processor_config.get("test")
    else:
        vision_train_config = None
        vision_eval_config = None

    vision_processors["train"] = build_proc_from_config(vision_train_config)
    vision_processors["eval"] = build_proc_from_config(vision_eval_config)

    if text_processor_config is not None:
        text_train_config = text_processor_config.get("train")
        text_eval_config = text_processor_config.get("eval")
    else:
        text_train_config = None
        text_eval_config = None

    text_processors["train"] = build_proc_from_config(text_train_config)
    text_processors["eval"] = build_proc_from_config(text_eval_config)

    return vision_processors, text_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    model_class = registry.get_model_class(name)

    model = model_class.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    config = OmegaConf.load(model_class.default_config_path(model_type))

    if config is not None:
        preprocess_config = config.preprocess
        vision_processors, text_processors = load_preprocess(preprocess_config)
    else:
        vision_processors, text_processors = None, None
        logger.warning(
            f"No default preprocess for model {name} ({model_type}). "
            f"This can happen if the model is not finetuned on downstream datasets, "
            f"or it is not intended for direct use without fine-tuning"
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vision_processors, text_processors


class ModelZoo:
    def __init__(self) -> None:
        self.model_zoo = {
            key: list(value.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for key, value in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        data = "\n".join([f"{name:30}, {types}" for name, types in self.model_zoo.items()])

        return (f"{'=' * 50}\n"
                f"{'Architectures':<30} {'Types'}\n"
                f"{'=' * 50}\n"
                f"{data}")

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(value) for value in self.model_zoo.values()])


model_zoo = ModelZoo()
