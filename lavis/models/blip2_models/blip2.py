# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import contextlib
import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributed as dist
import lavis.utils.dist_utils as dist_utils

from transformers import BertTokenizer
from lavis.utils.dist_utils import download_cached_file
from lavis.utils.utils import is_url
from lavis.utils.logger import MetricLogger, logger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.q_former import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_l


class Blip2Base(BaseModel):
    vit_name: str

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_q_former(cls, num_query_token, vision_width, cross_attention_frequency=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_frequency
        encoder_config.query_length = num_query_token
        q_former = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return q_former, query_tokens

    def init_vision_encoder(self, model, image_size, drop_path_rate, use_gradient_checkpoint, precision):
        assert model in [
            "eva_clip_g", "clip_l"
        ], "vit model must be eva_clip_g or clip_l"

        if model == "eva_clip_g":
            visual_encoder = create_eva_vit_g(image_size, drop_path_rate, use_gradient_checkpoint, precision)
        elif model == "clip_l":
            visual_encoder = create_clip_vit_l(image_size, use_gradient_checkpoint, precision)
        else:
            ValueError("Model name is not valid! You should use [\"eva_clip_g\", \"clip_L\"]")
            return

        layer_norm_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model

        return visual_encoder, layer_norm_vision

    def load_from_pretrained(self, url_or_file):
        if is_url(url_or_file):
            cached_file = download_cached_file(url_or_file, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_file):
            checkpoint = torch.load(url_or_file, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        message = self.load_state_dict(state_dict, strict=False)

        logger.info(f"load checkpoint from {url_or_file}")

        return message

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - index) for index in range(vit_num_layers + 2))
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if "visual_encoder" in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace("visual_encoder.", ''))
                group_name = f"vit_layer_{layer_id}_{group_name}"
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        optimizer_params = list(parameter_group_vars.values())

        return optimizer_params

    def lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []

            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)

            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                return spacy.load("en_core_web_sm")
            except ImportError:
                logger.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        origin_type = x.dtype

        return super().forward(x.type(torch.float32)).type(origin_type)


def disabled_train(self, mode=True):
    _ = mode

    return self


def compute_similarity_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logger.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_batches = 256
    text_indices = []
    text_embeds = []
    text_attentions = []

    for index in range(0, num_text, text_batches):
        text = texts[index: min(num_text, index + text_batches)]
        text_input = model.tokenizer(
            text, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(model.device)
        text_feature = model.forward_text(text_input)
        text_embed = func.normalize(model.text_proj(text_feature))
        text_embeds.append(text_embed)
        text_indices.append(text_input.input_indices)
        text_attentions.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_indices = torch.cat(text_indices, dim=0)
    text_attentions = torch.cat(text_attentions, dim=0)

    vit_features = []
    image_embeds = []

    for samples in data_loader:
        image = samples["image"]
        image = image.to(model.device)
        image_feature, vit_feature = model.forward_image(image)
        image_embed = model.vision_proj(image_feature)
        image_embed = func.normalize(image_embed, dim=-1)

        vit_features.append(vit_feature.cpu())
        image_embeds.append(image_embed)

    vit_features = torch.cat(vit_features, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    similarities_matrix = []

    for image_embed in image_embeds:
        similarity_query2text = image_embed @ text_embeds.t()
        similarity_image2text = similarity_query2text.max(0)
        similarities_matrix.append(similarity_image2text)

    similarities_matrix = torch.stack(similarities_matrix, dim=0)
    score_matrix_image2text = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = similarities_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(similarities_matrix.size(0), start + step)

    for index, similarities in enumerate(metric_logger.log_every(similarities_matrix[start:end], 50, header)):
        top_k_similarity, top_k_index = similarities.topk(k=k_test, dim=0)
        image_inputs = vit_features[start + index].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_image_text_matching(
            image_inputs=image_inputs,
            text_indices=text_indices[top_k_index],
            text_attentions=text_attentions[top_k_index]
        ).float()
        score_matrix_image2text[start + index, top_k_index] = score + top_k_similarity

    similarities_matrix = similarities_matrix.t()
    score_matrix_text2image = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(model.device)

    step = similarities_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(similarities_matrix.size(0), start + step)

    for index, similarities in enumerate(metric_logger.log_every(similarities_matrix[start:end], 50, header)):
        top_k_similarity, top_k_index = similarities.topk(k=k_test, dim=0)
        image_inputs = vit_features[top_k_index.cpu()].to(model.device)
        score = model.compute_image_text_matching(
            image_inputs=image_inputs,
            text_indices=text_indices[start + index].repeat(k_test, 1),
            text_attentions=text_attentions[start + index].repeat(k_test, 1)
        ).float()
        score_matrix_text2image[start + index, top_k_index] = score + top_k_similarity

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_image2text, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_text2image, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time

    logger.info(f"Evaluation time {datetime.timedelta(seconds=int(total_time))}")

    return score_matrix_image2text.cpu().numpy(), score_matrix_text2image.cpu().numpy()
