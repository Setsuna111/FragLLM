#!/bin/bash

MODEL_PATH="/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260120_GroundingAll_4e_lora32/checkpoint-1924"
SAVE_MODEL_PATH="/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260120_GroundingAll_4e_lora32_c1924_merge"

python merge_lora_weights_checkpoint.py \
    --model-path $MODEL_PATH \
    --save-model-path $SAVE_MODEL_PATH