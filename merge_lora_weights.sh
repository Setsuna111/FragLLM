#!/bin/bash

MODEL_PATH="/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260120_GroundingAll_4e_lora32"
POS_DECODER_TYPE="Simple"
MODEL_BASE="/home/dataset-local/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct"

python merge_lora_weights.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --pos-decoder-type $POS_DECODER_TYPE