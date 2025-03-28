#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2

python -m eve.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/pope/eve_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode qwen \
    --moe \
    --image-expert \
    --qwen25

python eve/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/eve_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${CKPT_NAME}.jsonl
