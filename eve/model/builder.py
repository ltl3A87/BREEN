import os
import warnings

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_PATCH_TOKEN)
from eve.model import (BREENForCausalLM, BREENConfig)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", moe=True, image_expert=True, qwen25=True, shared=False, clip_init=False):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'breen' in model_name.lower():
        # Load BREEN model

        if model_base is not None:
            # this may be mm projector only
            print('Loading BREEN from base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            config = BREENConfig.from_pretrained(model_path)
            model = BREENForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True,
                                                           **kwargs)

            mm_projector_weights = torch.load(os.path.join(
                model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16)
                                    for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            config = BREENConfig.from_pretrained(model_path)
            model = BREENForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True, **kwargs)
            # model = BREENForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, config=config)
            # model = BREENForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'breen' in model_name.lower():
        mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(
            model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 4096
    return tokenizer, model, image_processor, context_len
