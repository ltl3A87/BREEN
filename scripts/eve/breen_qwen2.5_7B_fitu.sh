#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=4
export MASTER_PORT=23456
export CPUS_PER_TASK=32
export QUOTA=reserved
export DS_SKIP_CUDA_CHECK=1


export DATA_PATH=playground/data/breen_fitu.json
export IMAGE_PATH=playground/data/BREEN-Finetune
export VIT_PATH=openai/eve-patch14-anypixel-672
export VIT_PATH_CLIP=openai/clip-vit-large-patch14-336
export BASE_LR=2e-5
export LEARNIG_RATE=2e-5

export CKPT_PATH=checkpoints/breen-qwen25-7B-prtr1
export SAVE_PATH=BREEN


torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$INDEX --master_addr=$CHIEF_IP --master_port=$MASTER_PORT \
    eve/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero3.json \
    --version qwen \
    --moe True \
    --image_expert True \
    --qwen25 True \
    --auto_clip True \
    --cos_loss True \
    --add_learnable_query True \
    --query_stride 3 \
    --multi_align True \
    --multi_concat True \
    --reverse True \
    --pre_text True \
    --pre_text_fitu True \
    --add_text True \
    --linear_tokenizer True \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_PATH} \
    --vision_tower ${VIT_PATH} \
    --vision_tower_clip ${VIT_PATH_CLIP} \
    --requires_cliploss False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate ${BASE_LR} \
    --mm_projector_lr ${LEARNIG_RATE} \
    --vision_tower_lr ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH} \
    --save_safetensors False \
    2>&1 | tee logs/${SAVE_PATH}-rank$1-$(date "+%Y-%m-%d|%H:%M:%S").log
