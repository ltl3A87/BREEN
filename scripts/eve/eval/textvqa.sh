#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT_NAME=$1
CKPT_PATH=$2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eve.eval.model_vqa_loader \
        --model-path ${CKPT_PATH}/${CKPT_NAME} \
        --question-file ./playground/data/eval/textvqa/eve_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/images \
        --answers-file ./playground/data/eval/textvqa/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode qwen \
        --moe \
        --image-expert \
        --qwen25 &
done

wait

output_file=./playground/data/eval/textvqa/answers/$CKPT_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m eve.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT_NAME}/merge.jsonl
