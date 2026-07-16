#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

PYTHON_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/python"
ROOT_DIR="/home/dataset-local/projects_dir/FragLLM/data_70"
MODEL_PATH="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-315000_merge/"
MODEL_IDENTIFIER="0529_all_315000"
RESULTS_DIR="./eval_results"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GPU_ID=0

INPUT_CSV="$ROOT_DIR/Pro2Text/test_frag_test.csv"
INPUT_NAME="test_frag_test"

# Step 1: predict fragment regions for the test_frag_test subset only.
"$PYTHON_BIN" eval/eval_task4_prot2text_grounding.py \
  --model_path "$MODEL_PATH" \
  --root_dir "$ROOT_DIR" \
  --input_csv "$INPUT_CSV" \
  --save_results_dir "$RESULTS_DIR/task4_grounding" \
  --model_identifier "$MODEL_IDENTIFIER" \
  --batch_per_device "$BATCH_PER_DEVICE" \
  --temperature "$TEMPERATURE" \
  --gpu_id "$GPU_ID" \
  --candidate_splits "test"

GROUNDING_PATH="$RESULTS_DIR/task4_grounding/$MODEL_IDENTIFIER/${INPUT_NAME}_grounding_results.csv"

# Step 2: classify each predicted region; descriptions are not generated.
"$PYTHON_BIN" eval/eval_task4_prot2text_region_reference.py \
  --model_path "$MODEL_PATH" \
  --grounding_results_path "$GROUNDING_PATH" \
  --save_results_dir "$RESULTS_DIR/task4_region_ref" \
  --model_identifier "$MODEL_IDENTIFIER" \
  --tasks "cls" \
  --batch_per_device "$BATCH_PER_DEVICE" \
  --temperature "$TEMPERATURE" \
  --gpu_id "$GPU_ID"

REGION_REF_PATH="$RESULTS_DIR/task4_region_ref/$MODEL_IDENTIFIER/${INPUT_NAME}_grounding_results_region_ref_results.csv"

# Step 3: use only the predicted fragment classes for function generation.
"$PYTHON_BIN" eval/eval_task4_prot2text_function_with_fragments.py \
  --model_path "$MODEL_PATH" \
  --root_dir "$ROOT_DIR" \
  --input_csv "$INPUT_CSV" \
  --fragment_mode "predicted" \
  --fragment_text "cls" \
  --predicted_regions_path "$REGION_REF_PATH" \
  --save_results_dir "$RESULTS_DIR/task4_function" \
  --model_identifier "$MODEL_IDENTIFIER" \
  --batch_per_device "$BATCH_PER_DEVICE" \
  --temperature "$TEMPERATURE" \
  --gpu_id "$GPU_ID" \
  --evaluate_exact_match true \
  --evaluate_bleu true \
  --evaluate_rouge true \
  --evaluate_bert_score true \
  --verbose true
