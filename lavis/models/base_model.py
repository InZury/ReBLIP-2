# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import torch
import numpy as np
import torch.nn as nn

from typing import List
from torch import Tensor
from omegaconf import OmegaConf
from lavis.utils.logger import logger
from lavis.utils.dist_utils import download_cached_file, is_dist_avail_and_initialized
from lavis.utils.utils import get_abs_path, is_url


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_file_name):
        if is_url(url_or_file_name):
            cached_file = download_cached_file(url_or_file_name, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_file_name):
            checkpoint = torch.load(url_or_file_name, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid!")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        message = self.load_state_dict(state_dict, strict=False)

        logger.info(f"Missing keys {message.missing_keys}.")
        logger.info(f"load checkpoint from {url_or_file_name}.")

        return message

    @classmethod
    def from_pretrained(cls, model_type):
        model_config = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_config)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
                model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), f"Unknown model type {model_type}."

        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, config, **kwargs):
        load_finetuned = config.get("load_finetuned", True)

        if load_finetuned:
            finetune_path = config.get("finetuned", None)

            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."

            self.load_checkpoint(url_or_file_name=finetune_path)
        else:
            load_pretrained = config.get("load_pretrained", True)

            if load_pretrained:
                pretrained_path = config.get("pretrained", None)

                assert "Found load_finetuned is False, but pretrain_path is None."

                self.load_from_pretrained(url_or_file_name=pretrained_path, **kwargs)  # TODO - What is this?

    def before_training(self, **kwargs):
        pass

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        param_weight_decay, param_non_weight_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "ln" in name or "bn" in name:
                param_non_weight_decay.append(param)
            else:
                param_weight_decay.append(param)

        optim_params = [
            {"params": param_weight_decay, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": param_non_weight_decay, "weight_decay": 0, "lr_scale": lr_scale}
        ]

        return optim_params

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        total = 0

        for param in self.parameters():
            weight = 1

            for x in param.shape:
                weight *= x

            total += weight

        if return_str:
            if total > 1e6:
                return f"{(total / 1e6):.1f}M"
            else:
                return f"{(total / 1e3):.1f}K"
        else:
            return total


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class SharedQueueMixin:
    queue_pointer: Tensor
    queue_size: int
    image_queue: Tensor
    text_queue: Tensor
    index_queue: Tensor

    @torch.no_grad()
    def dequeue_and_enqueue(self, image_feature, text_feature, indices=None):
        image_feature = concat_all_gather(image_feature)
        text_feature = concat_all_gather(text_feature)
        batch_size = image_feature.shape[0]
        pointer = int(self.queue_pointer)

        assert self.queue_size % batch_size == 0

        self.image_queue[:, pointer: pointer + batch_size] = image_feature.T
        self.text_queue[:, pointer: pointer + batch_size] = text_feature.T

        if indices is not None:
            indices = concat_all_gather(indices)
            self.index_queue[:, pointer: pointer + batch_size] = indices.T

        pointer = (pointer + batch_size) % self.queue_size
        self.queue_pointer[0] = pointer


class MomentumDistillationMixin:
    model_pairs: List
    momentum: float

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_momentum in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_momentum.data.copy_(param.data)
                param_momentum.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_momentum in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_momentum.data = param_momentum.data * self.momentum + param.data * (1.0 - self.momentum)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(context, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)

        return tuple(output)

    @staticmethod
    def backward(context, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)

        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return torch.cat(tensors_gather, dim=0)


def tile(x, dim, num_tile):
    init_dim = x.size(dim)
    repeat_index = [1] * x.dim()
    repeat_index[dim] = num_tile
    x = x.repeat(*repeat_index)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(num_tile) + index for index in range(init_dim)])
    )

    return torch.index_select(x, dim, order_index.to(x.device))
