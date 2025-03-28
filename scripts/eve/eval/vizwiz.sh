#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2

python -m eve.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/vizwiz/eve_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode qwen \
    --moe \
    --image-expert \
    --qwen25

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/eve_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT_NAME}.json
