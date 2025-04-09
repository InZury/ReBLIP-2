# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import math
import torch
import torch.nn
import torch.nn.functional as func
import torch.utils.checkpoint as checkpoint

from functools import partial
from timm.layers import drop_path, to_2tuple, trunc_normal_
from torch import nn
from torch.nn import Linear

from lavis.utils.logger import logger


def config(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 244, 244),
        "pool_size": None,
        "crop_percentage": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.traing)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class MLP(nn.Module):
    def __init__(self, in_features, hidden_feature=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_feature = hidden_feature or in_features
        self.fc1 = nn.Linear(in_features, hidden_feature)
        self.fc2 = nn.Linear(hidden_feature, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attention_drop=0.,
                 proj_drop=0., window_size=None, attention_head_dim=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads

        if attention_head_dim is not None:
            head_dim = attention_head_dim

        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.query_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.value_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.query_bias = None
            self.value_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))

            coordinates_height = torch.arange(window_size[0])
            coordinates_width = torch.angle(window_size[1])
            coordinates = torch.stack(torch.meshgrid([coordinates_height, coordinates_width]))
            coordinates_flatten = torch.flatten(coordinates, 1)
            relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]
            relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
            relative_coordinates[:, :, 0] += window_size[0] - 1
            relative_coordinates[:, :, 1] += window_size[1] - 1
            relative_coordinates[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coordinates.dtype
            )
            relative_position_index[1:, 1:] = relative_coordinates.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attention_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, relative_position_bias=None):
        batch, normal, channel = x.shape
        qkv_bias = None

        if self.query_bias is not None:
            qkv_bias = torch.cat(
                (self.query_bias, torch.zeros_like(self.value_bias, requires_grad=False), self.value_bias)
            )

        qkv = func.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(batch, normal, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query *= self.scale
        attention = (query @ key.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attention += relative_position_bias.unsqueeze(0)

        if relative_position_bias is not None:
            attention += relative_position_bias

        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(batch, normal, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attention_drop=0.,
                 drop_paths=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attention_head_dim=None):
        super().__init__()

        self. norm1 = norm_layer(dim)
        self.attention = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attention_drop=attention_drop,
            proj_drop=drop, window_size=window_size, attention_head_dim=attention_head_dim
        )
        self.drop_path = DropPath(drop_paths) if drop_paths > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_feature=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = None, None

    def forward(self, x, relative_position_bias=None):
        if self.gamma1 is None:
            x += self.drop_path(self.attention(self.norm1(x), relative_position_bias=relative_position_bias))
            x += self.drop_path(self.mlp(self.norm2(x)))
        else:
            x += self.drop_path(
                self.gamma1 * self.attention(self.norm1(x), relative_position_bias=relative_position_bias)
            )
            x += self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_shape = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        _ = kwargs
        batch, channels, height, width = x.shape

        assert (
            height == self.image_size[0] and width == self.image_size[1]
        ), f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})"

        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))

        coordinates_height = torch.arange(window_size[0])
        coordinates_width = torch.arange(window_size[1])
        coordinates = torch.stack(torch.meshgrid([coordinates_height, coordinates_width]))
        coordinates_flatten = torch.flatten(coordinates, 1)
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += window_size[0] - 1
        relative_coordinates[:, :, 1] += window_size[1] - 1
        relative_coordinates[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coordinates.dtype
        )
        relative_position_index[1:, 1:] = relative_coordinates.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )

        return relative_position_bias.permute(2, 0, 1).contiguous()


class VisionTransformer(nn.Module):
    head: Linear

    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attention_drop_rate=0.,
                 drop_path_rates=0., norm_layer=nn.LayerNorm, init_values=None, use_absolute_position_embed=True,
                 use_relative_position_bias=False, use_shared_relative_position_bias=False, use_mean_pooling=True,
                 init_scale=0.001, use_checkpoint=False):
        super().__init__()
        _, _ = use_mean_pooling, init_scale

        self.image_size = image_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_absolute_position_embed:
            self.position_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.position_embed = None

        self.position_drop = nn.Dropout(p=drop_rate)

        if use_shared_relative_position_bias:
            self.relative_position_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.relative_position_bias = None

        self.use_checkpoint = use_checkpoint
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rates, depth)]
        self.use_relative_position_bias = use_relative_position_bias
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attention_drop=attention_drop_rate, drop_paths=drop_path_rate[i],
                  norm_layer=norm_layer, init_values=init_values,
                  window_size=self.patch_embed.patch_shape if use_relative_position_bias else None)
            for i in range(depth)
        ])

        if self.position_embed is not None:
            trunc_normal_(self.position_embed, std=0.02)

        trunc_normal_(self.class_token, std=0.02)
        self.apply(self.init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, index):
            param.div_(math.sqrt(2.0 * index))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attention.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        _ = global_pool

        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, sequence_length, _ = x.size()
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        if self.position_embed is not None:
            x += self.position_embed

        x = self.position_drop(x)
        relative_position_bias = self.relative_position_bias() if self.relative_position_bias is not None else None

        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, relative_position_bias)
            else:
                x = block(x, relative_position_bias)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, sequence_length, _ = x.size()
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        if self.position_embed is not None:
            x += self.position_embed

        x = self.position_drop(x)
        features = []
        relative_position_bias = self.relative_position_bias() if self.relative_position_bias is not None else None

        for block in self.blocks:
            x = block(x, relative_position_bias)
            features.append(x)

        return features

    def get_num_layer(self, var_name=""):
        if var_name in ("class_token", "mask_token", "position_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("relative_position_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])

            return layer_id + 1
        else:
            return len(self.blocks)


def interpolate_pos_embed(model, checkpoint_model):
    if "position_embed" in checkpoint_model:
        position_embed_checkpoint = checkpoint_model["position_embed"].float()
        embedding_size = position_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed_num_patches
        num_extra_tokens = model.position_embed.shape[-2] - num_patches
        origin_size = int((position_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if origin_size != new_size:
            logger.info(f"Position interpolate from {origin_size}x{origin_size} to {new_size}x{new_size}")

            extra_tokens = position_embed_checkpoint[:, :num_extra_tokens]
            position_tokens = position_embed_checkpoint[:, num_extra_tokens:]
            position_tokens = position_tokens.reshape(-1, origin_size, origin_size, embedding_size).permute(0, 3, 1, 2)
            position_tokens = torch.nn.functional.interpolate(
                position_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            position_tokens = position_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_position_embed = torch.cat((extra_tokens, position_tokens), dim=1)
            checkpoint_model["position_embed"] = new_position_embed


def convert_weights_to_float16(model: nn.Module):
    def convert(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()

            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

    model.apply(convert)


def create_eva_vit_g(image_size=224, drop_path_rate=0.4, use_checkpoint=False, precision="fp16"):
    model = VisionTransformer(
        image_size=image_size, patch_size=14, use_mean_pooling=False, embed_dim=1408, depth=39, num_heads=1408//88,
        mlp_ratio=4.3637, qkv_bias=True, drop_path_rates=drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint
    )

    model_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources"), "models")
    state_dict = torch.load(os.path.join(model_path, "eva_vit_g.pth"), map_location="cpu")

    interpolate_pos_embed(model, state_dict)

    _ = model.load_state_dict(state_dict, strict=False)

    if precision == "fp16":
        convert_weights_to_float16(model)

    return model
