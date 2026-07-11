#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

PYTHON_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/python"
ROOT_DIR="/home/dataset-local/projects_dir/FragLLM/data_70"
MODEL_PATH="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/"
MODEL_IDENTIFIER="0529_all_215000"
RESULTS_DIR="./eval_results"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GPU_ID=0

INPUT_CSVS=(
  "$ROOT_DIR/Pro2Text/test_frag_test.csv"
  "$ROOT_DIR/Pro2Text/test_frag_no_train.csv"
)

for INPUT_CSV in "${INPUT_CSVS[@]}"; do
  INPUT_NAME="$(basename "$INPUT_CSV" .csv)"

  "$PYTHON_BIN" eval/eval_task4_prot2text_grounding.py \
    --model_path "$MODEL_PATH" \
    --root_dir "$ROOT_DIR" \
    --input_csv "$INPUT_CSV" \
    --save_results_dir "$RESULTS_DIR/task4_grounding" \
    --model_identifier "$MODEL_IDENTIFIER" \
    --batch_per_device "$BATCH_PER_DEVICE" \
    --temperature "$TEMPERATURE" \
    --gpu_id "$GPU_ID" \
    --candidate_splits "test,train"

  GROUNDING_PATH="$RESULTS_DIR/task4_grounding/$MODEL_IDENTIFIER/${INPUT_NAME}_grounding_results.csv"

  "$PYTHON_BIN" eval/eval_task4_prot2text_region_reference.py \
    --model_path "$MODEL_PATH" \
    --grounding_results_path "$GROUNDING_PATH" \
    --save_results_dir "$RESULTS_DIR/task4_region_ref" \
    --model_identifier "$MODEL_IDENTIFIER" \
    --tasks "cls,desc" \
    --batch_per_device "$BATCH_PER_DEVICE" \
    --temperature "$TEMPERATURE" \
    --gpu_id "$GPU_ID"

  REGION_REF_PATH="$RESULTS_DIR/task4_region_ref/$MODEL_IDENTIFIER/${INPUT_NAME}_grounding_results_region_ref_results.csv"

  for FRAGMENT_TEXT in cls desc; do
    for MODE in none predicted random truth; do
      EXTRA_ARGS=()
      if [ "$MODE" = "predicted" ]; then
        EXTRA_ARGS+=(--predicted_regions_path "$REGION_REF_PATH")
      fi

      "$PYTHON_BIN" eval/eval_task4_prot2text_function_with_fragments.py \
        --model_path "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --input_csv "$INPUT_CSV" \
        --fragment_mode "$MODE" \
        --fragment_text "$FRAGMENT_TEXT" \
        "${EXTRA_ARGS[@]}" \
        --truth_splits "test,train" \
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
    done
  done
done
