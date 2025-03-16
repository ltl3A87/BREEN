#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=4
export NNODES=1
export MASTER_PORT=23456
export CPUS_PER_TASK=32
export QUOTA=reserved
export DS_SKIP_CUDA_CHECK=1

CKPT_NAME='checkpoints/BREEN'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model eve_moe \
    --model_args pretrained=${CKPT_PATH} \
    --tasks mmmu,mmstar,ai2d,chartqa,hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix breen \
    --verbosity=DEBUG \
    --output_path ./logs/


