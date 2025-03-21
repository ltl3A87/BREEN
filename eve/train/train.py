# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import math
import logging
import os
import pathlib
import warnings
import io
import base64
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import deepspeed

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoConfig

from eve import conversation as conversation_lib
from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, QUERY_TOKEN_INDEX)
from eve.mm_utils import tokenizer_image_token
from eve.model import BREENForCausalLM, EVEQwen2ForCausalLM, BREENConfig
from eve.train.eve_trainer import EVETrainer

from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from transformers import CLIPVisionModel, Qwen2ForCausalLM



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    requires_cliploss: bool = field(default=False)
    tokenizer_path: Optional[str] = field(default=None)
    vision_tower_clip: Optional[str] = field(default=None)
    tune_vision_tower: bool = field(default=False)
    moe: bool = field(default=False)
    image_expert: bool = field(default=False)
    img_expert_divider: Optional[int] = field(default=4)
    top_k_image: Optional[int] = field(default=4)
    representation_learning: bool = field(default=False)
    first_layer_number: Optional[int] = field(default=8)
    layer_align: bool = field(default=False)
    init_from_nothing: bool = field(default=False)
    qwen25: bool = field(default=False)
    shared: bool = field(default=False)
    clip_init: bool = field(default=False)
    clip_hidden_act: Optional[str] = field(default="quick_gelu")
    clip_hidden_size: Optional[int] = field(default=1024)
    clip_intermediate_size: Optional[int] = field(default=4096)
    inter_feature_map: bool = field(default=False)
    input_emb_align: bool = field(default=False)
    auto_clip: bool = field(default=False)
    clip_loss_scale: Optional[float] = field(default=1.0)
    query_stride: Optional[int] = field(default=3)
    cos_loss: bool = field(default=False)
    wo_layer_norm: bool = field(default=False)
    add_learnable_query: bool = field(default=False)
    earlier_align: bool = field(default=False)
    query_token_shared: bool = field(default=False)
    multi_align: bool = field(default=False)
    multi_concat: bool = field(default=False)
    reverse: bool = field(default=False)
    woclip: bool = field(default=False)
    pre_text_fitu: bool = field(default=False)
    add_binary_mask: bool = field(default=False)
    aggregate_mask: bool = field(default=False)
    linear_tokenizer: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    json_path: str = field(default=None,
                           metadata={"help": "Path to the split json data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    pre_text: bool = False
    add_text: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

# @contextlib.contextmanager
# def temporarily_disable_deepspeed_zero3(training_arguments: TrainingArguments):
#     if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
#         unset_hf_deepspeed_config()
#         yield
#         set_hf_deepspeed_config(training_arguments.hf_deepspeed_config)
#     else:
#         yield


def merge_lora(model_path, use_moe, use_image_expert, qwen25=False, shared=False, **kwargs):
    with open(os.path.join(model_path, 'adapter_config.json'), 'r') as f:
        model_base = json.load(f)['base_model_name_or_path']
    if 'lora' in model_path.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided.')
    if 'lora' in model_path.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        print('Loading BREEN from base model...')
        
        model = BREENForCausalLM.from_pretrained(
            model_base, config=lora_cfg_pretrained, shared=shared, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(
                token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(
                token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional BREEN weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(
                model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise ValueError
        non_lora_trainables = {(k[11:] if k.startswith(
            'base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith(
                'model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    return model


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        print("has ds_id")
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with (zero.GatheredParameters[param]):
            param = param.data.detach().cpu().clone()
    else:
        print("no ds_id")
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k,
                     t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if (
        "lora_" not in k) and ("vision_tower" not in k)}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    has_image: bool = False,
    pre_text: bool = False,
    add_text: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    im_start = 151644
    im_end = 151645
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        # print("len source: ", len(source))
        # print("source: ", source)
        for j, sentence in enumerate(source):
            if sentence["from"] == 'Answer':
                role = roles["gpt"]
            else:
                role = roles[sentence["from"]]
            if has_image:
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer_image_token(sentence["value"], tokenizer) + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            if role == '<|im_start|>user' and pre_text and has_image and DEFAULT_IMAGE_TOKEN in sentence["value"] and j==0:
                if not add_text:
                    _input_id = _input_id + [QUERY_TOKEN_INDEX]
                else:
                    _input_id_temp = _input_id + [QUERY_TOKEN_INDEX] + tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")).input_ids + [im_end] + nl_tokens
                    if len(_input_id_temp) > max_len:
                        _input_id[len(tokenizer(role).input_ids + nl_tokens)+1] = QUERY_TOKEN_INDEX
                    else:
                        _input_id = _input_id + [QUERY_TOKEN_INDEX] + tokenizer(role).input_ids + nl_tokens + \
                                         tokenizer(sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")).input_ids + [
                                             im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                if pre_text and has_image and DEFAULT_IMAGE_TOKEN in sentence["value"] and j == 0:
                    if not add_text:
                        _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 4) + [im_end] + nl_tokens + [QUERY_TOKEN_INDEX]
                    else:
                        _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
                else:
                    _target = [im_start] + [IGNORE_INDEX] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        temp_list = []
        for j, id in enumerate(input_id):
            if id == QUERY_TOKEN_INDEX:
                temp_list.append(j)
        if len(temp_list) > 1:
            print("Warning: multiple queries detected: ", temp_list)
            input_id = [input_id[i] for i in range(len(input_id)) if i not in temp_list[1:]]
            target = [target[i] for i in range(len(target)) if i not in temp_list[1:]]
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(
            source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    pre_text: bool = False,
    add_text: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN:
        return preprocess_qwen(sources, tokenizer, tokenizer.model_max_length, has_image=has_image, pre_text=pre_text, add_text=add_text)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Formatting {len(self.list_data_dict)} inputs...Skip in lazy mode")

        self.data_is_index = type(self.list_data_dict[0]) == str
        self.json_path = data_args.json_path
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if self.data_is_index:
                sample = json.load(open(os.path.join(self.json_path, f'{sample}.json'), 'r'))
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if self.data_is_index:
                sample = json.load(open(os.path.join(self.json_path, f'{sample}.json'), 'r'))
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if self.data_is_index:
            sources = json.load(open(os.path.join(self.json_path, f'{sources}.json'), 'r'))
        sources_has_image = 'image' in sources

        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if sources_has_image:
            image_file = sources[0]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            processor_clip = self.data_args.image_processor_clip
            raw_image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            
            def compute_basehw(raw_image, processor):
                width, height = raw_image.size
                max_size = processor.image_size_clip

                if width > height:
                    new_width = max_size
                    val_patch = int(math.ceil(max_size * height / width / processor.patch_stride_clip))
                    new_height = val_size = val_patch * processor.patch_stride_clip
                else:
                    new_height = max_size
                    val_patch = int(math.ceil(max_size * width / height / processor.patch_stride_clip))
                    new_width = val_size = val_patch * processor.patch_stride_clip
                
                max_patch = max_size // processor.patch_stride_clip
                assert max_size % processor.patch_stride_clip == 0
                pre_size = (max_patch - val_patch) // 2 * processor.patch_stride_clip
                post_size = max_size - pre_size - val_size
                assert post_size % processor.patch_stride_clip == 0

                return [new_width, new_height, pre_size, val_size, post_size]
            
            def resize2require(raw_image, all_sizes, background_color):
                width, height = raw_image.size
                max_size = max(width, height)
                assert max_size == sum(all_sizes)

                image = Image.new(raw_image.mode, (max_size, max_size), background_color)
                if width > height:
                    image.paste(raw_image, (0, all_sizes[0]))
                    image_mask = torch.cat([torch.zeros(1, all_sizes[0], max_size),
                                            torch.ones(1, all_sizes[1], max_size),
                                            torch.zeros(1, all_sizes[2], max_size)], dim=1)
                else:
                    image.paste(raw_image, (all_sizes[0], 0))
                    image_mask = torch.cat([torch.zeros(1, max_size, all_sizes[0]),
                                            torch.ones(1, max_size, all_sizes[1]),
                                            torch.zeros(1, max_size, all_sizes[2])], dim=2)
                return image, image_mask

            ratio = processor.image_size // processor.image_size_clip
            assert ratio >= 1
            assert processor.image_size % processor.image_size_clip == 0
            background_color = tuple(int(x*255) for x in processor.image_mean)
            
            basehw = compute_basehw(raw_image, processor)
            needhw = [value * ratio for value in basehw]

            image = raw_image.resize((needhw[0], needhw[1]))
            image_clip = None
            if processor_clip is not None:
                image_clip = raw_image.resize((basehw[0], basehw[1]))
            
            del raw_image

            if needhw[0] == needhw[1]:
                image_mask = torch.ones(1, needhw[1], needhw[0])
                if processor_clip is not None:
                    image_mask_clip = torch.ones(1, basehw[1], basehw[0])
            else:
                image, image_mask = resize2require(image, needhw[2:], background_color)
                if processor_clip is not None:
                    image_clip, image_mask_clip = resize2require(image_clip, basehw[2:], background_color)
    
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image = torch.cat([image, image_mask], dim=0)
            if processor_clip is not None:
                image_clip = processor_clip.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
                image_clip = torch.cat([image_clip, image_mask_clip], dim=0)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=sources_has_image,
            pre_text=self.data_args.pre_text,
            add_text=self.data_args.add_text)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if sources_has_image:
            data_dict['image'] = image
            data_dict['image_clip'] = image_clip
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            processor = self.data_args.image_processor
            image_size, image_size_clip = processor.image_size, processor.image_size_clip
            background_color = tuple(int(x*255) for x in processor.image_mean)

            image = Image.new('RGB', (image_size, image_size), background_color)
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_mask = torch.zeros(1, image_size, image_size)
            data_dict['image'] = torch.cat([image, image_mask], dim=0)
            
            if self.data_args.image_processor_clip is not None:
                processor_clip = self.data_args.image_processor_clip
                
                image_clip = Image.new('RGB', (image_size_clip, image_size_clip), background_color)
                image_clip = processor_clip.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
                image_mask_clip = torch.zeros(1, image_size_clip, image_size_clip)
                data_dict['image_clip'] = torch.cat([image_clip, image_mask_clip], dim=0)
            else:
                data_dict['image_clip'] = None

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

            if instances[0]['image_clip'] is not None:
                images_clip = [instance['image_clip'] for instance in instances]
                if all(x is not None and x.shape == images_clip[0].shape for x in images_clip):
                    batch['images_clip'] = torch.stack(images_clip)
                else:
                    batch['images_clip'] = images_clip
            else:
                batch['images_clip'] = None

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))
    
    if model_args.model_name_or_path is not None:
        config_path = os.path.join(model_args.model_name_or_path, 'config.json')
        config_file = json.load(open(config_path))
        if 'mm_vision_tower' in config_file and config_file['mm_vision_tower'] != model_args.vision_tower:
            print('change mm_vision_tower from ({}) to ({})'.format(config_file['mm_vision_tower'], model_args.vision_tower))
            config_file['mm_vision_tower'] = model_args.vision_tower
            json.dump(config_file, open(config_path, 'w'))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'lora' in model_args.model_name_or_path:
            model = merge_lora(model_args.model_name_or_path,
                               model_args.moe,
                               model_args.image_expert,
                               model_args.qwen25,
                               model_args.shared,
                               cache_dir=training_args.cache_dir,
                               **bnb_model_from_pretrained_args)
        else:
            if 'breen' not in model_args.model_name_or_path.lower():
                if training_args.deepspeed and is_deepspeed_zero3_enabled():
                    unset_hf_deepspeed_config()
                pretrained_model = EVEQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args,
                )
                if model_args.clip_init:
                    clip_model = CLIPVisionModel.from_pretrained(model_args.vision_tower_clip)
                set_hf_deepspeed_config(training_args.hf_deepspeed_config)
                pretrained_config = pretrained_model.config
                pretrained_config.architectures = ["BREENForCausalLM"]
                pretrained_config.model_type = "breen"
                pretrained_config.shared = model_args.shared
                pretrained_config.clip_init = model_args.clip_init
                pretrained_config.representation_learning = model_args.representation_learning
                pretrained_config.first_layer_number = model_args.first_layer_number
                pretrained_config.layer_align = model_args.layer_align
                pretrained_config.clip_init = model_args.clip_init
                pretrained_config.clip_hidden_act = model_args.clip_hidden_act
                pretrained_config.clip_hidden_size = model_args.clip_hidden_size
                pretrained_config.clip_intermediate_size = model_args.clip_intermediate_size
                pretrained_config.inter_feature_map = model_args.inter_feature_map
                pretrained_config.input_emb_align = model_args.input_emb_align
                pretrained_config.auto_clip = model_args.auto_clip
                pretrained_config.clip_loss_scale = model_args.clip_loss_scale
                pretrained_config.query_stride = model_args.query_stride
                pretrained_config.cos_loss = model_args.cos_loss
                pretrained_config.wo_layer_norm = model_args.wo_layer_norm
                pretrained_config.add_learnable_query = model_args.add_learnable_query
                pretrained_config.earlier_align = model_args.earlier_align
                pretrained_config.query_token_shared = model_args.query_token_shared
                pretrained_config.multi_align = model_args.multi_align
                pretrained_config.multi_concat = model_args.multi_concat
                pretrained_config.reverse = model_args.reverse
                pretrained_config.woclip = model_args.woclip
                pretrained_config.pre_text_fitu = model_args.pre_text_fitu
                pretrained_config.add_binary_mask = model_args.add_binary_mask
                pretrained_config.aggregate_mask = model_args.aggregate_mask
                pretrained_config.linear_tokenizer = model_args.linear_tokenizer
                model = BREENForCausalLM(pretrained_model.config)

                state_dict = pretrained_model.state_dict()
                if model_args.clip_init:
                    clip_state_dict = clip_model.state_dict()
                for name, param in model.state_dict().items():
                    target_device = param.device
                    if name in state_dict:
                        pretrain_param = state_dict[name].data.detach().cpu().clone().to(target_device)
                        model.state_dict()[name].copy_(pretrain_param)
                    else:
                        print("not existed param in pretrained model: ", name)
                        if not model_args.init_from_nothing:
                            if 'image_expert.' in name:
                                new_name = name.replace('image_expert.', '')
                            elif 'text_expert.' in name:
                                new_name = name.replace('text_expert.', '')
                            elif 'shared_expert.' in name:
                                new_name = name.replace('shared_expert.', '')
                            else:
                                new_name = 'not existed'
                                print("not existed name: ", name)
                            print('new_name: ', new_name)
                            if new_name in state_dict:
                                print("init with name in pretrained model: ", new_name)
                                pretrain_param = state_dict[new_name].data.detach().cpu().clone().to(target_device)
                                model.state_dict()[name].copy_(pretrain_param)
                            if model_args.clip_init:
                                layer_index = [f'layers.{str(idx)}' for idx in range(24)]
                                print("layer_index: ", layer_index)
                                for layer_name in layer_index:
                                    if layer_name in name and "fc1" in name and "pre_fc1" not in name:
                                        suffix = name.split('.')[-1]
                                        new_name = f"vision_model.encoder.{layer_name}.mlp.fc1.{suffix}"
                                        clip_param = clip_state_dict[new_name].data.detach().cpu().clone().to(
                                            target_device)
                                        model.state_dict()[name].copy_(clip_param)
                                    if layer_name in name and "fc2" in name and "post_fc2" not in name:
                                        suffix = name.split('.')[-1]
                                        new_name = f"vision_model.encoder.{layer_name}.mlp.fc2.{suffix}"
                                        clip_param = clip_state_dict[new_name].data.detach().cpu().clone().to(
                                            target_device)
                                        model.state_dict()[name].copy_(clip_param)
            else:
                config = BREENConfig.from_pretrained(model_args.model_name_or_path)
                config.shared = model_args.shared  # Set 'shared' parameter
                config.clip_init = model_args.clip_init
                config.representation_learning = model_args.representation_learning
                config.first_layer_number = model_args.first_layer_number
                config.layer_align = model_args.layer_align
                config.clip_init = model_args.clip_init
                config.clip_hidden_act = model_args.clip_hidden_act
                config.clip_hidden_size = model_args.clip_hidden_size
                config.clip_intermediate_size = model_args.clip_intermediate_size
                config.inter_feature_map = model_args.inter_feature_map
                config.input_emb_align = model_args.input_emb_align
                config.auto_clip = model_args.auto_clip
                config.clip_loss_scale = model_args.clip_loss_scale
                config.query_stride = model_args.query_stride
                config.cos_loss = model_args.cos_loss
                config.wo_layer_norm = model_args.wo_layer_norm
                config.add_learnable_query = model_args.add_learnable_query
                config.earlier_align = model_args.earlier_align
                config.query_token_shared = model_args.query_token_shared
                config.multi_align = model_args.multi_align
                config.multi_concat = model_args.multi_concat
                config.reverse = model_args.reverse
                config.woclip = model_args.woclip
                config.pre_text_fitu = model_args.pre_text_fitu
                config.add_binary_mask = model_args.add_binary_mask
                config.aggregate_mask = model_args.aggregate_mask
                config.linear_tokenizer = model_args.linear_tokenizer
                model = BREENForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if 'qwen' in model_args.model_name_or_path.lower() or 'moe' in model_args.model_name_or_path.lower():
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643


    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if 'qwen' not in model_args.model_name_or_path.lower() and 'moe' not in model_args.model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:  # Start the vision encoder initialization
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor

        if model_args.requires_cliploss or model_args.auto_clip:
            vision_clip = model.get_vision_clip()
            vision_clip.to(
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.image_processor_clip = vision_clip.image_processor
        else:
            data_args.image_processor_clip = None

        for name, param in model.named_parameters():
            if model_args.tune_vision_tower:
                if ('vision_tower' in name or 'mm_projector' in name or 'image_expert' in name or 'text_expert_gate' in name
                        or 'shared_expert' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if 'clip' in name:
                param.requires_grad = False

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.requires_cliploss = model_args.requires_cliploss
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.representation_learning = model_args.representation_learning
        model.config.first_layer_number = model_args.first_layer_number
        model.config.layer_align = model_args.layer_align
        model.config.clip_init = model_args.clip_init
        model.config.clip_hidden_act = model_args.clip_hidden_act
        model.config.clip_hidden_size = model_args.clip_hidden_size
        model.config.clip_intermediate_size = model_args.clip_intermediate_size
        model.config.inter_feature_map = model_args.inter_feature_map

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = EVETrainer(model=model,
                             tokenizer=tokenizer,
                             args=training_args,
                             **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
