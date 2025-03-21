import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from .configuration_evaclip import EvaCLIPVisionConfig
from .modeling_evaclip import EvaCLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(
            args, 'mm_vision_select_feature', 'patch')
        self.inter_feature_map = getattr(
            args, 'inter_feature_map', False)
        self.input_emb_align = getattr(
            args, 'input_emb_align', False)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                self.vision_tower_name)

    def load_model(self):
        print(f'Load vision tower from {self.vision_tower_name}')
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)
        if 'eva' in self.vision_tower_name.lower():
            vision_cfg = EvaCLIPVisionConfig.from_pretrained(
                self.vision_tower_name)
            self.backbone = EvaCLIPVisionModel.from_pretrained(
                self.vision_tower_name, config=vision_cfg)
        else:
            self.backbone = CLIPVisionModel.from_pretrained(
                self.vision_tower_name)
        self.backbone.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, inter_feature_map=False, input_emb_align=False):
        if not inter_feature_map:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        else:
            if input_emb_align:
                image_features = image_forward_outs.hidden_states
            else:
                image_features = image_forward_outs.hidden_states[1:]
        if self.select_feature == 'patch':
            if not inter_feature_map:
                image_features = image_features[:, 1:]
            else:
                image_features = [image_feature[:, 1:] for image_feature in image_features]
        elif self.select_feature == 'cls_patch':
            if not inter_feature_map:
                image_features = image_features
            else:
                image_features = [image_feature for image_feature in image_features]
        else:
            raise ValueError(
                f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad() comment to enable fine-tune vit
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.backbone(image.to(
                    device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(
                    image_forward_out, self.inter_feature_map, self.input_emb_align).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.backbone(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(
                image_forward_outs, self.inter_feature_map, self.input_emb_align) # .to(images.dtype)
            if isinstance(image_features, list):
                image_features = [image_feature.to(images.dtype) for image_feature in image_features]
            else:
                image_features = image_features.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    @property
    def config(self):
        if self.is_loaded:
            return self.backbone.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
