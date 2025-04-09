# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import math
import copy
import warnings
import torch

from typing import Optional, Tuple, Union
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    find_pruneable_heads_and_indices,
    prune_linear_layer
)
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from lavis.utils.logger import logger

CONFIG_FOR_DOC = "T5Config"
TOKENIZER_FOR_DOC = "T5Tokenizer"
CHECKPOINT_FOR_DOC = "t5-small"

T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b"
]


def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    _ = config

    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in Pytorch, requires TensorFlow to be installed."
            "Please see https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)

    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split('/')

        if any(_name in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
               for _name in name):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue

        pointer = model
        array = tf_weights[txt_name]
        scope_names = []

        for module_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", module_name):
                scope_names = re.split(r"_(\d+)", module_name)
            else:
                scope_names = [module_name]

            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif (
                    scope_names[0] == "wi"
                    and len(scope_names) > 1
                    and scope_names[1].isdigit()
            ):
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue

            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if (
                scope_names[0] not in ["kernel", "scale", "embedding"]):
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)

        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as error:
            error.args += (pointer.shape, array.shape)
            raise

        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to Pytorch model: {', '.join(tf_weights.keys())}.")

    return model
