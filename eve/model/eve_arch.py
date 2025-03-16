from abc import ABC, abstractmethod

import torch

from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                               IMAGE_TOKEN_INDEX, QUERY_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_encoder.vision_tokenizer import VisionTokenizer, VisionLinearTokenizer, VisionCompressor, VisionRepresentation, VisionTokenQuery


class EVEMetaModel:

    # def __init__(self, config, shared=False, clip_init=False, clip_hidden_size=1024,
    #              clip_intermediate_size=4096, clip_hidden_act="quick_gelu"):
    def __init__(self, config, **kwargs):
        # if config.architectures == ["Qwen2ImgForCausalLM"]:
        #     super(EVEMetaModel, self).__init__(config, shared=shared, clip_init=clip_init, clip_hidden_size=clip_hidden_size,
        #          clip_intermediate_size=clip_intermediate_size, clip_hidden_act=clip_hidden_act)
        # else:
        super(EVEMetaModel, self).__init__(config)
        print("EVEMetaModel init")
        self.linear_tokenizer = getattr(config, 'linear_tokenizer', None)

        if hasattr(config, "mm_vision_tower"):
            if self.linear_tokenizer:
                self.vision_tower = VisionLinearTokenizer(config.mm_hidden_size,
                                                    config.hidden_size,
                                                    config.mm_vision_tower)
            else:
                self.vision_tower = VisionTokenizer(config.mm_hidden_size,
                                                    config.hidden_size,
                                                    config.mm_vision_tower)
            self.mm_projector = build_vision_projector(config)

            
            if self.training and config.requires_cliploss:
                self.vision_clip = build_vision_tower(config)
                self.vision_clip.requires_grad_(False)
                if hasattr(config, "representation_learning") and config.representation_learning:
                    self.vision_tower_compressor = VisionRepresentation(self.vision_clip,
                                                                        self.vision_tower.image_processor,
                                                                        llm_size=config.hidden_size,
                                                                        num_layer=config.first_layer_number,
                                                                        layer_align=config.layer_align,
                                                                        inter_feature_map=config.inter_feature_map,
                                                                        input_emb_align=config.input_emb_align)
                else:
                    self.vision_tower_compressor = VisionCompressor(self.vision_clip,
                                                                    self.vision_tower.image_processor,
                                                                    llm_size=config.hidden_size,
                                                                    num_layer=config.num_hidden_layers)

            elif self.training and hasattr(config, "auto_clip") and config.auto_clip:
                self.vision_clip = build_vision_tower(config)
                self.vision_clip.requires_grad_(False)
                self.vision_clip_query = VisionTokenQuery(self.vision_clip, self.vision_tower.image_processor,
                                                          llm_size=config.hidden_size, query_stride=config.query_stride,
                                                          wo_layer_norm=config.wo_layer_norm)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_clip(self):
        return self.vision_clip
    
    def get_vision_tower_compressor(self):
        return self.vision_tower_compressor

    def get_vision_clip_query(self):
        return self.vision_clip_query

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_tower_clip = model_args.vision_tower_clip
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_clip = vision_tower_clip
        self.config.requires_cliploss = model_args.requires_cliploss

        if self.get_vision_tower() is None:
            if self.linear_tokenizer:
                vision_tower = VisionLinearTokenizer(1024,
                                               self.config.hidden_size,
                                               vision_tower)
            else:
                vision_tower = VisionTokenizer(1024,
                                               self.config.hidden_size,
                                               vision_tower)
            if model_args.tokenizer_path is not None:
                def get_tw(weights, keyword):
                    return {k.split(keyword + '.')[-1]: v for k, v in weights.items() if keyword in k}
                tokenizer_weights = torch.load(model_args.tokenizer_path, map_location='cpu')
                vision_tower.load_state_dict(get_tw(tokenizer_weights, 'vision_tower'), strict=False)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
        
        # In case it is frozen by LoRA
        for p in self.vision_tower.parameters():
            p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print(f'Load mm_mlp_adapter from {pretrain_mm_mlp_adapter}')
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location='cpu')

            def get_aw(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(
                get_aw(mm_projector_weights, 'mm_projector'))

        if model_args.requires_cliploss:
            if getattr(self, 'vision_clip', None) is None:
                self.vision_clip = build_vision_tower(model_args)
                self.vision_clip.requires_grad_(False)


            if getattr(self, 'vision_tower_compressor', None) is None:
                if hasattr(self.config, "representation_learning") and model_args.representation_learning:
                    self.vision_tower_compressor = VisionRepresentation(self.vision_clip,
                                                                    self.vision_tower.image_processor,
                                                                    llm_size=self.config.hidden_size,
                                                                    num_layer=model_args.first_layer_number,
                                                                    layer_align=model_args.layer_align,
                                                                    inter_feature_map=model_args.inter_feature_map,
                                                                    input_emb_align=model_args.input_emb_align)
                else:
                    self.vision_tower_compressor = VisionCompressor(self.vision_clip,
                                                                    self.vision_tower.image_processor,
                                                                    llm_size=self.config.hidden_size,
                                                                    num_layer=self.config.num_hidden_layers)
        if model_args.auto_clip:
            if getattr(self, 'vision_clip', None) is None:
                self.vision_clip = build_vision_tower(model_args)
                self.vision_clip.requires_grad_(False)
            if getattr(self, 'vision_clip_query', None) is None:
                self.vision_clip_query = VisionTokenQuery(self.vision_clip, self.vision_tower.image_processor,
                                                          llm_size=self.config.hidden_size,
                                                          query_stride=model_args.query_stride,
                                                          wo_layer_norm=model_args.wo_layer_norm)
    

class EVEMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_clip(self):
        return self.get_model().get_vision_clip()
    
    def get_clip_loss(self):
        return self.get_model().get_vision_tower_compressor()

    def get_clip_query(self):
        return self.get_model().get_vision_clip_query()

    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        return vision_tower(images, self.get_model().mm_projector)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, clip_token_query=None, add_learnable_query=None, pre_text=False
    ):
        problem = False
        if add_learnable_query is not None:
            clip_token_query = None
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, input_ids, problem

        if type(images) is list or images.ndim == 5:
            exit(0)
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features, patch_hw = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        modi_input_ids = []  # New variable to store modified input_ids
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_modi_input_ids = []  # Modified input_ids for current batch
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:])
                # print("cur_image_features[0:0].shape: ", cur_image_features[0:0].shape)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                modi_input_ids.append(cur_input_ids)  # Add unchanged input_ids
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(
                cur_input_ids == IMAGE_TOKEN_INDEX)[0]


            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            if clip_token_query is not None:
                cur_clip_token_query = clip_token_query[batch_idx]
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)


                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[image_token_start+1:image_token_start+2]))
                    if add_learnable_query is not None:
                        if not pre_text:
                            cur_new_input_embeds.append(add_learnable_query)

                    # Update modi_input_ids to reflect changes
                    cur_modi_input_ids.extend(cur_input_ids[:image_token_start - 1])
                    cur_modi_input_ids.append(cur_input_ids[image_token_start - 1])  # Add original IMAGE_TOKEN_INDEX
                    cur_modi_input_ids.extend([IMAGE_TOKEN_INDEX] * cur_image_features.shape[
                        0])  # Replace image token with multiple IMAGE_TOKEN_INDEX
                    cur_modi_input_ids.extend(cur_input_ids[image_token_start + 1:image_token_start + 2])
                    if add_learnable_query is not None:
                        if not pre_text:
                            cur_modi_input_ids.extend([QUERY_TOKEN_INDEX] * add_learnable_query.shape[0])

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        if clip_token_query is not None:
                            if cur_image_features.shape[0] - cur_clip_token_query.shape[0] > 0:
                                cur_new_labels.append(torch.full(
                                    (cur_image_features.shape[0]-cur_clip_token_query.shape[0],),
                                    IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                                cur_new_labels.append(torch.full((cur_clip_token_query.shape[0],),
                                                                 QUERY_TOKEN_INDEX, device=labels.device, dtype=labels.dtype))
                            else:
                                cur_new_labels.append(torch.full((cur_image_features.shape[0],),
                                                                 QUERY_TOKEN_INDEX, device=labels.device,
                                                                 dtype=labels.dtype))
                        else:
                            cur_new_labels.append(torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(
                            cur_labels[image_token_start+1:image_token_start+2])
                        if add_learnable_query is not None:
                            if not pre_text:
                                cur_new_labels.append(torch.full(
                                    (add_learnable_query.shape[0],), QUERY_TOKEN_INDEX, device=labels.device,
                                    dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if add_learnable_query is not None:
                        if not pre_text:
                            cur_new_input_embeds.append(add_learnable_query)

                    # Update modi_input_ids for cases without the extra start-end token logic
                    cur_modi_input_ids.extend(cur_input_ids[:image_token_start])
                    cur_modi_input_ids.extend([IMAGE_TOKEN_INDEX] * cur_image_features.shape[
                        0])  # Replace image token with multiple IMAGE_TOKEN_INDEX
                    if add_learnable_query is not None:
                        if not pre_text:
                            cur_modi_input_ids.extend([QUERY_TOKEN_INDEX] * add_learnable_query.shape[0])

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        if clip_token_query is not None:
                            if cur_image_features.shape[0]-cur_clip_token_query.shape[0] > 0:
                                cur_new_labels.append(torch.full(
                                    (cur_image_features.shape[0]-cur_clip_token_query.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                                cur_new_labels.append(torch.full(
                                    (cur_clip_token_query.shape[0],), QUERY_TOKEN_INDEX,
                                    device=labels.device, dtype=labels.dtype))
                            else:
                                cur_new_labels.append(torch.full(
                                    (cur_image_features.shape[0],), QUERY_TOKEN_INDEX,
                                    device=labels.device, dtype=labels.dtype))
                        else:
                            cur_new_labels.append(torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        if add_learnable_query is not None:
                            if not pre_text:
                                cur_new_labels.append(torch.full(
                                    (add_learnable_query.shape[0],), QUERY_TOKEN_INDEX, device=labels.device,
                                    dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(
                    cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if pre_text and QUERY_TOKEN_INDEX in cur_input_ids:
                    query_token_index = torch.where(
                        cur_input_ids == QUERY_TOKEN_INDEX)[0][0]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    if not pre_text: # or QUERY_TOKEN_INDEX not in cur_input_ids:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids).detach())
                    elif pre_text and QUERY_TOKEN_INDEX not in cur_input_ids:
                        # print("cur_input_ids: ", cur_input_ids)
                        cur_new_input_embeds.append(add_learnable_query.detach())
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:-1]).detach())
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:query_token_index]).detach())
                        cur_new_input_embeds.append(add_learnable_query.detach())
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[query_token_index+1:]).detach())
                else:
                    if not pre_text: # or QUERY_TOKEN_INDEX not in cur_input_ids:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids))
                    elif pre_text and QUERY_TOKEN_INDEX not in cur_input_ids:
                        cur_new_input_embeds.append(add_learnable_query)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:-1]))
                    else:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:query_token_index]))
                        cur_new_input_embeds.append(add_learnable_query)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[query_token_index+1:]))
                if not pre_text:# or QUERY_TOKEN_INDEX not in cur_input_ids:
                    cur_modi_input_ids.extend(cur_input_ids)  # Append remaining tokens
                elif pre_text and QUERY_TOKEN_INDEX not in cur_input_ids:
                    cur_modi_input_ids.extend([QUERY_TOKEN_INDEX] * add_learnable_query.shape[0])
                    cur_modi_input_ids.extend(cur_input_ids[:-1])
                else:
                    cur_modi_input_ids.extend(cur_input_ids[:query_token_index])
                    cur_modi_input_ids.extend([QUERY_TOKEN_INDEX] * add_learnable_query.shape[0])
                    cur_modi_input_ids.extend(cur_input_ids[query_token_index+1:])
                if labels is not None:
                    if not pre_text: # or QUERY_TOKEN_INDEX not in cur_input_ids:
                        cur_new_labels.append(cur_labels)
                    elif pre_text and QUERY_TOKEN_INDEX not in cur_input_ids:
                        cur_new_labels.append(torch.full(
                            (add_learnable_query.shape[0],), QUERY_TOKEN_INDEX, device=labels.device,
                            dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[:-1])
                    else:
                        cur_new_labels.append(cur_labels[:query_token_index])
                        cur_new_labels.append(torch.full(
                            (add_learnable_query.shape[0],), QUERY_TOKEN_INDEX, device=labels.device,
                            dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[query_token_index+1:])
            cur_new_input_embeds = [x.to(device=self.device)
                                    for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            modi_input_ids.append(
                torch.tensor(cur_modi_input_ids, device=self.device))  # Add to the final modi_input_ids
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros(
                    (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Align modi_input_ids
            modi_input_ids_align = []
            for cur_modi_id in modi_input_ids:
                cur_modi_id = torch.cat((cur_modi_id, torch.full(
                    (max_len - cur_modi_id.shape[0],), 151643, dtype=cur_modi_id.dtype, device=cur_modi_id.device)),
                                        dim=0)
                modi_input_ids_align.append(cur_modi_id)
            modi_input_ids = torch.stack(modi_input_ids_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full(
                        (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if modi_input_ids:
                modi_input_ids = torch.stack(modi_input_ids, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, patch_hw, modi_input_ids, problem

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
