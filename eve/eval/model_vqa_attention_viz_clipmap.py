import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from eve.conversation import SeparatorStyle, conv_templates
from eve.mm_utils import (get_model_name_from_path, process_images,
                              tokenizer_image_token)
from eve.model.builder import load_pretrained_model
from eve.utils import disable_torch_init

from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # print("pre_text_fitu: ", getattr(self.model_config, "pre_text_fitu", False))
        prompt = conv.get_prompt(getattr(self.model_config, "pre_text_fitu", False))
        # print("prompt: ", prompt)

        image = Image.open(os.path.join(
            self.image_folder, image_file)).convert('RGB')
        image_tensor, image_raw, img_token_length = process_images([image], self.image_processor, None, True)
        image_tensor = image_tensor[0]
        image_raw = image_raw[0]
        transform = transforms.ToTensor()
        image_raw = transform(image_raw)
        print("image_tensor shape:", image_tensor.shape)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image_raw, img_token_length

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder,
                            tokenizer, image_processor, model_config)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print("args.shared: ", args.shared)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, moe=args.moe, image_expert=args.image_expert, qwen25=args.qwen25,
        shared=args.shared, clip_init=args.clip_init)

    questions = [json.loads(q) for q in open(
        os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_raw, img_token_length), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        print("idx: ", idx)
        if idx != 3 and idx != 17 and idx != 20:
        # if idx > 20:
            continue
        cur_prompt = line["text"]
        print("image_tensor.shape: ", image_tensor.shape)
        print("cur_prompt: ", cur_prompt)
        transform_to_pil = transforms.ToPILImage()
        print("image_raw.shape: ", image_raw.shape)

        image = transform_to_pil(image_raw[0])

        image_size = image.size
        grid_width = image_size[0] // 14
        grid_height = image_size[1] // 14
        image_token_length = img_token_length[0][0].cpu().clone()
        print("image_token_length: ", image_token_length)
        print("image_size: ", image_size)
        # print("image_shape: ", image.shape)
        image_ratio = image_size[0] / image_size[1]

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[
            args.conv_mode].sep_style != SeparatorStyle.TWO and conv_templates[
            args.conv_mode].sep_style != SeparatorStyle.QWEN else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        print("input_ids: ", input_ids)
        print("input_ids.shape: ", input_ids.shape)

        if -300 in input_ids:
            print("-300 in input_ids")
        else:
            print("-300 not in input_ids")

        with torch.inference_mode():
            outputs_dict = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )


        # print("outputs_dict: ", outputs_dict)
        # break
        output_ids = outputs_dict["sequences"]

        input_token_len = input_ids.shape[1]

        # identify length or index of tokens
        # input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1  # -1 for the <image> token
        vision_token_start = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
        print("vision_token_start: ", vision_token_start)
        vision_token_end = vision_token_start + image_token_length
        query_token_start_str3 = (input_ids[0] == -300).nonzero(as_tuple=True)[0] + image_token_length - 1
        query_token_end_str3 = query_token_start_str3  + 64
        query_token_start_str4 = query_token_end_str3
        query_token_end_str4 = query_token_start_str4 + 36
        print("vision_token_end: ", vision_token_end)
        print("query_token_start_str3: ", query_token_start_str3)

        print("input_token_len: ", input_token_len)
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        print("output_ids.shape: ", output_ids.shape)
        print("output_ids: ", output_ids)
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print("outputs: ", outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")


        for i, layer in enumerate(outputs_dict["attentions"][0]):
            # print("layer: ", layer.shape)
            layer_attns = layer.squeeze(0)
            # print("layer_attns: ", layer_attns.shape)
            attns_per_head = layer_attns.mean(dim=0)
            # print("attns_per_head: ", attns_per_head.shape)
            # cur = attns_per_head[:-1].cpu().clone()
            cur = attns_per_head.cpu().clone()
            # print("cur.shape: ", cur.shape)


            # cur[1:, 0] = 0.
            # cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)

            llm_attn_matrix = heterogenous_stack(
                [torch.tensor([1])]
                + list(cur)
                + list(map(aggregate_llm_attention, outputs_dict["attentions"]))
            )
            # print("llm_attn_matrix: ", llm_attn_matrix.shape)
            # ans_file.flush()

            gamma_factor = 3
            # enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)
            enhanced_attn_m = np.power(cur.numpy(), 1 / gamma_factor)

            # fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
            # ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
            # plt.savefig(f"att_viz_twosample/{idx}_try_layer{i}.jpg")
            if i == 0:
                print("cur: ", cur.shape)
            output_token_end = cur.shape[0]
            # if i == 26:
            cur_attn = cur

            # if i == 1 or i == 13 or i ==26:
            if i < 28:
                layer_idx = i
                image_ratio = image_size[0] / image_size[1]
                output_token_len = output_ids[:, input_token_len:].shape[1]

                num_image_per_row = min(output_token_len, 4)
                num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)
                fig, axes = plt.subplots(
                    num_rows, num_image_per_row,
                    figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows),
                    dpi=150
                )
                plt.subplots_adjust(wspace=0.05, hspace=0.2)
                output_token_start = output_token_end - output_token_len

                # whether visualize the attention heatmap or
                # the image with the attention heatmap overlayed
                vis_overlayed_with_attn = True
                output_token_inds = list(range(output_token_start, output_token_end))
                for i, ax in enumerate(axes.flatten()):
                    if i >= output_token_len:
                        ax.axis("off")
                        continue

                    target_token_ind = output_token_inds[i]
                    attn_weights_over_vis_tokens = cur_attn[target_token_ind][query_token_start_str4:query_token_end_str4]
                    attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

                    # attn_over_image = []
                    # for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
                    #     vis_attn = vis_attn.reshape(grid_size, grid_size)
                        # vis_attn = vis_attn / vis_attn.max()
                        # attn_over_image.append(vis_attn * weight)
                    attn_over_image = attn_weights_over_vis_tokens
                    # attn_over_image = attn_over_image.reshape(grid_width, grid_height)
                    attn_over_image = attn_over_image.reshape(6, 6)
                    print("attn_over_image.shape: ", attn_over_image.shape)
                    print("image.size: ", image.size)
                    # attn_over_image = attn_over_image.sum(dim=0)
                    # print("attn_over_image.shape: ", attn_over_image.shape)
                    attn_over_image = attn_over_image / attn_over_image.max()

                    attn_over_image = F.interpolate(
                        attn_over_image.unsqueeze(0).unsqueeze(0).float(),
                        size=(image.size[1], image.size[0]),
                        mode='nearest',
                        # mode='bicubic', align_corners=False
                    ).squeeze().transpose(1, 0)
                    print("attn_over_image.shape: ", attn_over_image.shape)
                    print("image.size: ", image.size)
                    np_img = np.array(image)[:, :, ::-1]
                    print("np_img.shape: ", np_img.shape)
                    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
                    ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
                    ax.set_title(
                        tokenizer.decode(outputs_dict["sequences"][0, input_token_len:][i], add_special_tokens=False).strip(),
                        fontsize=7,
                        pad=1
                    )
                    ax.axis("off")
                plt.savefig(f"att_viz_twosample_clip_str4/{idx}_try_image_viz_layer{layer_idx}_v3_clipstride_str4_v2.jpg")
            # break
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="eve_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--image-expert", action="store_true")
    parser.add_argument("--qwen25", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--clip_init", action="store_true")
    args = parser.parse_args()

    eval_model(args)
