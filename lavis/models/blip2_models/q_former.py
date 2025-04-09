# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import math
import torch
import torch.utils.checkpoint

from typing import Tuple
from torch.nn import CrossEntropyLoss
from torch import Tensor, device, nn

from lavis.utils.logger import logger

from transformers.activations import ACT2FN
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer
)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_indices", torch.arange(config.max_position_embeddings).expand(1, -1))

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.config = config

    def forward(self, input_indices=None, position_indices=None, query_embeds=None, past_key_values_length=0):
        if input_indices is not None:
            sequence_length = input_indices.size()[1]
        else:
            sequence_length = 0

        if position_indices is None:
            position_indices = self.position_indices[
                               :, past_key_values_length: sequence_length + past_key_values_length
                               ].clone()

        if input_indices is not None:
            embeddings = self.word_embeddings(input_indices)

            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_indices)
                embeddings += position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    attention_gradients: Tensor
    attention_map: Tensor

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)

        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.save_attention = False

    def save_attention_gradients(self, attention_gradients):
        self.attention_gradients = attention_gradients

    def get_attention_gradients(self):
        return self.attention_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states,
                attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            sequence_length = hidden_states.size()[1]
            position_indices_left = torch.arange(
                sequence_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_indices_right = torch.arange(
                sequence_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_indices_left - position_indices_right
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores += relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores += relative_position_scores_query + relative_position_scores_key

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            if is_cross_attention and self.save_attention:
                self.save_attention_map(attention_probs)
                attention_probs.register_hook(self.save_attention_gradients)

            attention_probs_dropped = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs_dropped = attention_probs_dropped * head_mask

            context_layer = torch.matmul(attention_probs_dropped, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            outputs += (past_key_value,)

            return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_morm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states, input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states,
                attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_output = self.self(hidden_states, attention_mask, head_mask,
                                encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_output[0], hidden_states)
        outputs = (attention_output,) + self_output[1:]

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_func = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_func = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_func(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.sequence_length_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num

        if self.config.add_cross_attention and layer_num % self.config.cross_attention_frequency == 0:
            self.cross_attention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(self, hidden_states,
                attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, query_length=0):
        self_attention_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attention_past_key_value
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1: -1]
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                assert (
                        encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"

                cross_attention_outputs = self.cross_attention(
                    query_attention_output, attention_mask, head_mask,
                    encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1: -1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.sequence_length_dim,
                query_attention_output
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forawrd_chunk,
                    self.chunk_size_feed_forward,
                    self.sequence_length_dim,
                    attention_output[:, query_length:, :]
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.sequence_length_dim,
                attention_output
            )

        outputs = (layer_output,) + outputs
        outputs += (present_key_value,)

        return outputs

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, index) for index in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None,
                output_attentions=False, output_hidden_states=False, return_dict=True, query_length=0):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for index in range(self.config.num_hidden_layers):
            layer_module = self.layer[index]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_head_mask = head_mask[index] if head_mask is not None else None
            past_key_value = past_key_values[index] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn("\"use_cache=True\" is incompatible with gradient checkpointing."
                                "Setting \"use_cache=False\"...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, layer_head_mask, encoder_hidden_states,
                    encoder_attention_mask, past_key_value, output_attentions, query_length
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1,])
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                value for value in [
                    hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions
                ] if value is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_func = ACT2FN[config.hidden_act]
        else:
            self.transform_act_func = config.hidden_act

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_func(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vacab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_indices"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.proune_heads(heads)

    def _get_extended_attention_mask(
            self, attention_mask: Tensor, input_shape: Tuple[int],
            mask_device: device, is_decoder: bool, has_query: bool = False
    ) -> Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if is_decoder:
                batch_size, *sequence_length = input_shape  # TODO Check sequence_length dimension
                sequence_indices = torch.arange(sequence_length, device=mask_device)
                causal_mask = (
                        sequence_indices[None, None, :].repeat(batch_size, sequence_length, 1)
                        <= sequence_indices[None, :, None]
                )

                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_sequence_len = attention_mask.shape[1] - causal_mask.shape[1]

                    if has_query:
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_sequence_len, sequence_length),
                                    device=mask_device,
                                    dtype=causal_mask.dtype
                                ),
                                causal_mask
                            ],
                            dim=1
                        )

                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_sequence_len),
                                device=mask_device,
                                dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        dim=-1
                    )
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_indices (shape {input_shape}) or attention_mask (shape {attention_mask.shape})."
            )

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, input_indices=None, attention_mask=None, position_indices=None,
                head_mask=None, query_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, is_decoder=False):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_indices is None:
            assert query_embeds is not None, "You have to specify query_embeds when input_indices is None."

        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        embedding_output = self.embeddings(
            input_indices=input_indices,
            position_indices=position_indices,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, sequence_length = input_shape
        embedding_device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length + past_key_values_length), device=embedding_device
            )

        if is_decoder:
            extended_attention_mask = self._get_extended_attention_mask(
                attention_mask, input_indices.shape, embedding_device, is_decoder, has_query=(query_embeds is not None)
            )
        else:
            extended_attention_mask = self._get_extended_attention_mask(
                attention_mask, input_shape, embedding_device, is_decoder
            )

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()

            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=embedding_device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions
        )


class BertLMHeadModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_indices", r"predictions.decoders.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_indices=None, attention_mask=None, position_indices=None,
                head_mask=None, query_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                labels=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, return_logits=False, is_decoder=False, reduction="mean"):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_indices,
            attention_mask=attention_mask,
            position_indices=position_indices,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder
        )

        sequence_output = outputs[0]

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1], :, :]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        language_model_loss = None

        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_func = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            language_model_loss = loss_func(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if reduction == "none":
                language_model_loss = language_model_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]

            return ((language_model_loss, ) + outputs) if language_model_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=language_model_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions
        )

    @staticmethod
    def _prepare_inputs_for_generation(
        input_indices, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        if attention_mask is None:
            attention_mask = input_indices.new_ones(input_indices.shape)

        query_mask = input_indices.new_ones(query_embeds.shape[:-1])
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        if past is not None:
            input_indices = input_indices[:, -1:]

        return {
            "input_indices": input_indices,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None)
        }

    def _reorder_cache(self, past, beam_index):
        reorder_past = ()

        for layer_past in past:
            reorder_past += tuple(past_state.index_select(0, beam_index) for past_state in layer_past, )

        return reorder_past


class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_indices", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_indices=None, attention_mask=None, position_indices=None, head_mask=None, query_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, return_logits=False, is_decoder=False):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_indices,
            attention_mask=attention_mask,
            position_indices=position_indices,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]
        else:
            sequence_output = None
            logger.warn(f"sequence_output is None: {sequence_output}")

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_language_model_loss = None

        if labels is not None:
            loss_func = CrossEntropyLoss()
            masked_language_model_loss = loss_func(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]

            return ((masked_language_model_loss, ) + output) if masked_language_model_loss is not None else output

        return MaskedLMOutput(
            loss=masked_language_model_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
