# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import math
import torch
import torch.nn.functional as func
import collections.abc

from collections import OrderedDict
from itertools import repeat
from torch import nn
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from lavis.models.eva_vit import convert_weights_to_float16
from lavis.utils.logger import logger


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
            nn.Conv2d(out_planes, out_planes * self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_planes * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or in_planes != out_planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(in_planes, out_planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(out_planes * self.expansion))])
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.bottleneck(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.context_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x += self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = func.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.query_proj.weight,
            k_proj_weight=self.key_proj.weight,
            v_proj_weight=self.value_proj.weight,
            in_proj_bias=None,
            in_proj_weight=torch.cat([self.query_proj.bias, self.key_proj.bias, self.value_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.context_proj.weight,
            out_proj_bias=self.context_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class LayerNorm(nn.Module):
    def forward(self, x: torch.Tensor):
        origin_type = x.dtype

        return super().forward(x.type(torch.float32)).type(origin_type)


class QuickGELU(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, model, num_heads: int, attention_mask: torch.Tensor = None, use_gradient_checkpointing=False):
        super().__init__()

        self.attention = nn.MultiheadAttention(model, num_heads)
        self.ln1 = LayerNorm(model)
        self.ln2 = LayerNorm(model)
        self.mlp = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(model, model * 4)),
            ("gelu", QuickGELU()),
            ("context_proj", nn.Linear(model * 4, model))])
        )
        self.attention_mask = attention_mask

        if use_gradient_checkpointing:
            self.attention = checkpoint_wrapper(self.attention)
            self.mlp = checkpoint_wrapper(self.mlp)

    def _attention(self, x: torch.Tensor):
        self.attention_mask = self.attention_mask.to(
            dtype=x.dtype, device=x.device
        ) if self.attention_mask is not None else None

        return self.attention(x, x, x, need_weights=False, attention_mask=self.attention_mask)[0]

    def forward(self, x: torch.Tensor):
        x += self._attention(self.ln1(x))
        x += self.mlp(self.ln2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attention_mask: torch.Tensor = None, use_gradient_checkpointing=False):
        super().__init__()

        self.width = width
        self.layers = layers
        self.residual_blocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attention_mask, use_gradient_checkpointing and index > 12)
            for index in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.residual_blocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, use_gradient_checkpointing: bool):
        super().__init__()

        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, use_gradient_checkpointing=use_gradient_checkpointing)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )
        x += self.positional_embedding.to(x.dtype)
        x = self.ln(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        return x


def n_tuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x

        return tuple(repeat(x, n))

    return parse


to_2tuple = n_tuple(2)


def interpolate_position_embed(model, state_dict, interpolation: str = "bicubic", sequence_dim=1):
    _ = sequence_dim

    old_position_embed = state_dict.get("positional_embedding", None)
    grid_size = round((model.positional_embedding.shape[0] - 1) ** 0.5)

    if old_position_embed is None:
        return

    grid_size = to_2tuple(grid_size)
    extra_tokens = 1
    new_sequence_length = grid_size[0] * grid_size[1] + extra_tokens

    if new_sequence_length == old_position_embed.shape[0]:
        return

    if extra_tokens:
        position_embed_token = old_position_embed[:extra_tokens]
        position_embed_image = old_position_embed[extra_tokens:]
    else:
        position_embed_token = None
        position_embed_image = old_position_embed

    old_grid_size = to_2tuple(int(math.sqrt(len(position_embed_image))))

    logger.info(f"Resizing position embedding grid-size from {old_grid_size} to {grid_size}")

    position_embed_image = position_embed_image.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    position_embed_image = func.interpolate(
        position_embed_image, size=grid_size, mode=interpolation, align_corners=True
    )
    position_embed_image = position_embed_image.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]

    if position_embed_token is not None:
        new_position_embed = torch.cat([position_embed_token, position_embed_image], dim=0)
    else:
        new_position_embed = position_embed_image

    state_dict["positional_embedding"] = new_position_embed


def create_clip_vit_l(image_size=224, use_checkpoint=False, precision="fp16"):
    model = VisionTransformer(
        input_resolution=image_size,
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        use_gradient_checkpointing=use_checkpoint
    )

    model_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources"), "models")
    state_dict = torch.load(os.path.join(model_path, "clip_vit_l.pth"), map_location="cpu")

    interpolate_position_embed(model, state_dict)

    _ = model.load_state_dict(state_dict, strict=False)

    if precision == "fp16":
        convert_weights_to_float16(model)

    return model
