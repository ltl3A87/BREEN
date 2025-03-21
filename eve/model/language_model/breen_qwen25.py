from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from.configuration_qwen2 import Qwen2Config
from .modeling_breen import Qwen2ImgForCausalLM, Qwen2ImgModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from ..eve_arch import EVEMetaForCausalLM, EVEMetaModel
from ...constants import QUERY_TOKEN_INDEX, IMAGE_TOKEN_INDEX
import torch.nn.functional as F


class BREENConfig(Qwen2Config):
    model_type = "breen"
    shared = False
    clip_init = False
    clip_hidden_size = 1024
    clip_intermediate_size = 4096
    clip_hidden_act = "quick_gelu"


class BREENQwen25Model(EVEMetaModel, Qwen2ImgModel):
    config_class = BREENConfig

    def __init__(self, config):
        super(BREENQwen25Model, self).__init__(config)


class BREENForCausalLM(Qwen2ImgForCausalLM, EVEMetaForCausalLM):
    config_class = BREENConfig

    def __init__(self, config):
        super(Qwen2ImgForCausalLM, self).__init__(config)
        self.model = BREENQwen25Model(config)
        self.auto_clip = getattr(config, "auto_clip", False)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        add_learnable_query = getattr(config, "add_learnable_query", False)
        self.multi_concat = getattr(config, "multi_concat", False)
        if add_learnable_query:
            if not self.multi_concat:
                num_token = (24 // config.query_stride)**2
                self.add_learnable_query = nn.Parameter(torch.randn(num_token, config.hidden_size))
            else:
                if config.query_stride == 2:
                    num_token = (24 // config.query_stride)**2 + (24 // (config.query_stride*2))**2
                elif config.query_stride == 3:
                    num_token = (24 // config.query_stride) ** 2 + (24 // (config.query_stride * 4 // 3)) ** 2
                self.add_learnable_query = nn.Parameter(torch.randn(num_token, config.hidden_size))
        else:
            self.add_learnable_query = None
        self.multi_align = getattr(config, "multi_align", False)
        self.reverse = getattr(config, "reverse", False)
        self.query_stride = getattr(config, "query_stride", None)
        self.woclip = getattr(config, "woclip", None)
        self.pre_text = getattr(config, "pre_text_fitu", None)
        self.add_binary_mask = getattr(config, "add_binary_mask", None)
        self.aggregate_mask = getattr(config, "aggregate_mask", None)
        if self.add_binary_mask:
            self.query_mask_predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()  # Outputs probabilities between 0 and 1
            )
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _input_ids = input_ids
        if self.auto_clip and self.training and labels is not None:
            clip_token_query = self.get_clip_query()(images_clip, self.multi_align, self.query_stride)
        else:
            clip_token_query = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, patch_hw, modi_input_ids, problem = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, clip_token_query=clip_token_query, add_learnable_query=self.add_learnable_query,
            pre_text=self.pre_text)

        input_indicators = torch.zeros_like(modi_input_ids, dtype=torch.long).to(modi_input_ids.device)

        # Set IMAGE_TOKEN_INDEX to 1
        input_indicators[modi_input_ids == IMAGE_TOKEN_INDEX] = 1

        # Set QUERY_TOKEN_INDEX to 2
        input_indicators[modi_input_ids == QUERY_TOKEN_INDEX] = 2

        print("input_indicators: ", input_indicators)

        if position_ids is not None:
            _modi_input_ids = modi_input_ids
            position_ids, past_key_values, attention_mask, cache_position = (
                self.modi_prepare_inputs_for_generation(_modi_input_ids, past_key_values=past_key_values,
                                                        attention_mask=attention_mask,
                                                        cache_position=None,
                                                        use_cache=True))
        print("input_ids: ", input_ids)

        outputs = self.model(
            input_ids=input_ids,
            modi_input_ids=modi_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        print("outputs: ", outputs)


        loss = None
        if labels is not None:
            print("labels is not None: ")
            hidden_states = outputs[0]
            hidden_dim = hidden_states.size(-1)

            shift_labels = labels[..., 1:].contiguous().reshape(-1)
            shift_hidden_states = hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
            assert shift_labels.size(0) == shift_hidden_states.size(0)
            mask = shift_labels > -1

            seen_tokens = mask.float().sum().item()
            if not seen_tokens > 0:
                logits = self.lm_head(shift_hidden_states[0:2])
                loss = logits.sum() * 0
            else:
                shift_labels_text = shift_labels[mask]
                shift_hidden_states_text = shift_hidden_states[mask, :]
                logits = self.lm_head(shift_hidden_states_text)
                logits = logits.float()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, shift_labels_text)

            if self.training:
                print('llm_loss', loss)

            if self.config.auto_clip and self.training and (not self.woclip):
                mask_clip = shift_labels == QUERY_TOKEN_INDEX
                clip_hidden_states = shift_hidden_states[mask_clip, :]
                if clip_hidden_states.shape[0]:
                    if type(clip_token_query) == tuple:
                        temp = clip_token_query[0]
                    else:
                        temp = clip_token_query
                    if temp.reshape(-1, hidden_dim).shape[0] == clip_hidden_states.shape[0] or self.multi_concat:
                        if self.config.earlier_align:
                            previous_hidden_states = outputs.hidden_states
                            for i in range(1, 9):
                                cur_hidden_states = previous_hidden_states[i]
                                cur_shift_hidden_states = cur_hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
                                cur_clip_hidden_states = cur_shift_hidden_states[mask_clip, :]
                                if i == 1:
                                    auto_clip_loss = self.get_clip_query().compute_mse_loss(cur_clip_hidden_states,
                                                                                            clip_token_query.reshape(-1,
                                                                                                                     hidden_dim),
                                                                                            cos_loss=self.config.cos_loss)
                                else:
                                    auto_clip_loss += self.get_clip_query().compute_mse_loss(cur_clip_hidden_states,
                                                                                            clip_token_query.reshape(-1,
                                                                                                                     hidden_dim),
                                                                                            cos_loss=self.config.cos_loss)
                            auto_clip_loss = auto_clip_loss / 8
                        else:
                            print("========== aligned ==========")
                            if self.multi_align:
                                clip_token_query_str2, clip_token_query_str4, = clip_token_query
                                bs = hidden_states.size(0)
                                token_length = clip_hidden_states.size(0) // bs
                                if not self.multi_concat:
                                    clip_hidden_states_batch = clip_hidden_states.view(bs, token_length, hidden_dim).transpose(1, 2).unflatten(-1, (12, 12))
                                    clip_hidden_states_str4 = F.avg_pool2d(clip_hidden_states_batch,
                                                                         kernel_size=2,
                                                                         stride=2)
                                    clip_hidden_states_str4 = clip_hidden_states_str4.flatten(2).transpose(1, 2).reshape(-1,
                                                                                                            hidden_dim)
                                    clip_loss_str2 = self.get_clip_query().compute_mse_loss(clip_hidden_states,
                                                                                            clip_token_query_str2.reshape(-1,
                                                                                                                     hidden_dim),
                                                                                            cos_loss=self.config.cos_loss)
                                    clip_loss_str4 = self.get_clip_query().compute_mse_loss(clip_hidden_states_str4,
                                                                                            clip_token_query_str4.reshape(
                                                                                                -1,
                                                                                                hidden_dim),
                                                                                            cos_loss=self.config.cos_loss)
                                else:
                                    clip_hidden_states_batch = clip_hidden_states.view(bs, token_length, hidden_dim)
                                    # print("clip_hidden_states_batch.shape: ", clip_hidden_states_batch.shape)
                                    if not self.reverse:
                                        clip_hidden_states_str4 = clip_hidden_states_batch[:, :(24 // (self.query_stride*2))**2, :].reshape(-1, hidden_dim)
                                        clip_hidden_states_str2 = clip_hidden_states_batch[:, (24 // (self.query_stride*2))**2:, :].reshape(-1, hidden_dim)
                                    else:
                                        clip_hidden_states_str2 = clip_hidden_states_batch[:,
                                                                  :(24 // self.query_stride) ** 2, :].reshape(-1, hidden_dim)
                                        clip_hidden_states_str4 = clip_hidden_states_batch[:,
                                                                  (24 // self.query_stride) ** 2:, :].reshape(-1,
                                                                                                              hidden_dim)
                                    if self.add_binary_mask:
                                        seq_len = shift_hidden_states.shape[0] // bs  # Assuming shift_hidden_states has shape (batch_size, seq_len, hidden_dim)

                                        # Get the first occurrence of QUERY_TOKEN_INDEX for each sequence
                                        mask_clip_rs = mask_clip.reshape(bs, -1)
                                        shift_hidden_states_rs = shift_hidden_states.reshape(bs, seq_len, -1)
                                        first_query_positions = mask_clip_rs.float().argmax(
                                            dim=1)  # Shape: (batch_size,)

                                        # Ensure QUERY_TOKEN_INDEX exists in each sequence
                                        assert (mask_clip_rs.sum(
                                            dim=1) > 0).all(), "Each sequence must contain at least one QUERY_TOKEN_INDEX."

                                        # Compute the positions just before the first QUERY_TOKEN_INDEX
                                        pre_query_positions = first_query_positions - 1

                                        # Ensure no negative indices (for sequences where QUERY_TOKEN_INDEX is at the first position)
                                        assert (
                                                pre_query_positions >= 0).all(), "QUERY_TOKEN_INDEX cannot be at the first position in any sequence."

                                        # Gather the hidden states just before the first QUERY_TOKEN_INDEX

                                        if self.aggregate_mask:
                                            pre_query_hidden_states = torch.stack([
                                                torch.mean(shift_hidden_states_rs[i, :pre_query_positions[i] + 1, :],
                                                           dim=0)
                                                for i in range(bs)
                                            ]).to(shift_hidden_states_rs.device)
                                        else:
                                            batch_indices = torch.arange(bs)  # Shape: (batch_size,)
                                            pre_query_hidden_states = shift_hidden_states_rs[batch_indices,
                                                                      pre_query_positions,
                                                                      :]  # Shape: (batch_size, hidden_dim)
                                        mask_prob = self.query_mask_predictor(pre_query_hidden_states)
                                        batch_size = mask_prob.shape[0]
                                        mask_stride2_shape = (batch_size, 64)
                                        mask_stride4_shape = (batch_size, 36)

                                        # Initialize masks with ones
                                        mask_stride2 = torch.ones(mask_stride2_shape, dtype=pre_query_hidden_states.dtype, device=pre_query_hidden_states.device)
                                        mask_stride4 = torch.ones(mask_stride4_shape, dtype=pre_query_hidden_states.dtype, device=pre_query_hidden_states.device)

                                        # Conditionally set mask values to 0
                                        # mask_stride2[
                                        #     mask_prob < 0.5] = 0  # For mask_prob < 0.5, set all 64 elements to 0
                                        # mask_stride4[
                                        #     mask_prob >= 0.5] = 0  # For mask_prob >= 0.5, set all 36 elements to 0
                                        # Broadcast mask_prob to match the shapes of mask_stride2 and mask_stride4
                                        mask_prob_expanded_stride2 = (mask_prob < 0.5).expand_as(
                                            mask_stride2)  # Shape: [batch_size, 64]
                                        mask_prob_expanded_stride4 = (mask_prob >= 0.5).expand_as(
                                            mask_stride4)  # Shape: [batch_size, 36]

                                        # Apply the conditions
                                        mask_stride2[
                                            mask_prob_expanded_stride2] = 0  # Set all 64 elements to 0 for mask_prob < 0.5
                                        mask_stride4[
                                            mask_prob_expanded_stride4] = 0  # Set all 36 elements to 0 for mask_prob >= 0.5
                                        mask_stride2 = mask_stride2.reshape(-1)
                                        mask_stride4 = mask_stride4.reshape(-1)
                                        # print("mask_prob < 0.5: ", mask_prob < 0.5)
                                        # print("mask_prob: ",mask_prob)
                                        if not mask_stride2.bool().any():
                                            clip_loss_str2 = torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
                                        else:
                                            clip_loss_str2 = self.get_clip_query().compute_mse_loss(clip_hidden_states_str2[mask_stride2.bool()],
                                                                                                    clip_token_query_str2.reshape(
                                                                                                        -1,
                                                                                                        hidden_dim)[mask_stride2.bool()],
                                                                                                    cos_loss=self.config.cos_loss)
                                        if not mask_stride4.bool().any():
                                            clip_loss_str4 = torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
                                        else:
                                            clip_loss_str4 = self.get_clip_query().compute_mse_loss(clip_hidden_states_str4[mask_stride4.bool()],
                                                                                                    clip_token_query_str4.reshape(
                                                                                                        -1,
                                                                                                        hidden_dim)[mask_stride4.bool()],
                                                                                                    cos_loss=self.config.cos_loss)
                                    else:
                                        clip_loss_str2 = self.get_clip_query().compute_mse_loss(clip_hidden_states_str2,
                                                                                                clip_token_query_str2.reshape(
                                                                                                    -1,
                                                                                                    hidden_dim),
                                                                                                cos_loss=self.config.cos_loss)
                                        clip_loss_str4 = self.get_clip_query().compute_mse_loss(clip_hidden_states_str4,
                                                                                                clip_token_query_str4.reshape(
                                                                                                    -1,
                                                                                                    hidden_dim),
                                                                                            cos_loss=self.config.cos_loss)
                            else:
                                auto_clip_loss = self.get_clip_query().compute_mse_loss(clip_hidden_states,
                                                                                        clip_token_query.reshape(-1, hidden_dim),
                                                                                        cos_loss=self.config.cos_loss)
                    else:
                        shift_labels_batch = labels[..., 1:].contiguous()
                        shift_hidden_states_batch = hidden_states[..., :-1, :].contiguous()
                        for batch_idx in range(shift_labels_batch.size(0)):
                            # Get mask for QUERY_TOKEN_INDEX in current batch example
                            mask_clip = shift_labels_batch[batch_idx] == QUERY_TOKEN_INDEX
                            clip_hidden_states = shift_hidden_states_batch[batch_idx][mask_clip, :]

                            # print(f'clip_hidden_states.shape for batch {batch_idx}:', clip_hidden_states.shape)

                            # Check if padding/truncating is needed for this example
                            num_query_tokens_in_labels = clip_hidden_states.shape[0]
                            clip_token_query_padded = clip_token_query[batch_idx, :num_query_tokens_in_labels, :]

                            # Compute MSE loss for this example and accumulate
                            mse_loss_example = self.get_clip_query().compute_mse_loss(
                                clip_hidden_states, clip_token_query_padded, cos_loss=self.config.cos_loss
                            )
                            if batch_idx == 0:
                                auto_clip_loss = mse_loss_example
                            else:
                                auto_clip_loss += mse_loss_example
                        auto_clip_loss = auto_clip_loss / shift_labels_batch.size(0)

                    if self.multi_align:
                        print(f'auto_clip_loss2: {self.config.clip_loss_scale * clip_loss_str2}, '
                              f'4: {self.config.clip_loss_scale * clip_loss_str4}')
                        auto_clip_loss = clip_loss_str2 + clip_loss_str4
                        print(f'sum_auto_clip_loss: {auto_clip_loss}')
                        loss = loss + self.config.clip_loss_scale * auto_clip_loss
                    else:
                        loss = loss + self.config.clip_loss_scale * auto_clip_loss
                        print('auto_clip_loss', self.config.clip_loss_scale * auto_clip_loss)
                else:
                    if self.add_binary_mask:
                        print("psuedo self.add_binary_mask")
                        temp = torch.zeros(32, hidden_dim, device=loss.device, dtype=inputs_embeds.dtype)
                        mask_prob = self.query_mask_predictor(temp)
                        loss = loss + torch.sum(mask_prob)*0
                print('all_loss', loss)
        else:
            print("labels is None: ")
            print('debug 1')
            hidden_states = outputs[0]
            print('debug 2')
            # if self.auto_clip:
            #     if num_logits_to_keep == 0:
            #         logits = self.lm_head(hidden_states[:, clip_token_query.shape[1]:, :]).float()
            #     else:
            #         logits = self.lm_head(hidden_states[:, -(num_logits_to_keep-clip_token_query.shape[1]):, :]).float()
            # else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
            print('debug 3')
            print('logits', logits)


        if self.config.requires_cliploss and self.training:
            clip_loss = self.get_clip_loss()(_input_ids,
                                             images_clip,
                                             torch.stack(outputs.hidden_states, dim=2),
                                             patch_hw)
            print('lvm_loss', clip_loss)
            loss = loss + clip_loss
            print('all_loss', loss)


        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def modi_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            # if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            #     input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            # elif past_length < input_ids.shape[1]:
            #     input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]
        return position_ids, past_key_values, attention_mask, cache_position


AutoConfig.register("breen", BREENConfig)
AutoModelForCausalLM.register(BREENConfig, BREENForCausalLM)
