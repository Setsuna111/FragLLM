#!/bin/bash

set -e

PYTHON_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/python"
export PYTHONPATH="./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/"
ROOT_DIR="./data_70"
SPLIT="test"
BATCH_PER_DEVICE=4
TEMPERATURE=0.0
MODEL_IDENTIFIER="grounding_lora_0529_all_215000"
POS_DECODER_TYPE="ProteinSAM"
USE_DETAILED_TEMPLATE=true

DATASETS="ActGroundGroup,BindIGroundGroup,MotifGroundGroup,EvoGroundGroup,DomGroundGroup"

GROUNDING_RESULTS_ROOT="./eval_results"
GROUNDING_RESULTS_DIR="${GROUNDING_RESULTS_ROOT}/grounding_group/${MODEL_IDENTIFIER}"
REGION_REF_RESULTS_DIR="./eval_results/grounding_group_ref"
REGION_REF_MODEL_IDENTIFIER="0529_all_215000"
REGION_REF_TASKS="cls,desc"

echo "========================================="
echo "Group Grounding Full Pipeline"
echo "========================================="
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model Path: $MODEL_PATH"
echo "Data Root: $ROOT_DIR"
echo "Datasets: $DATASETS"
echo "Split: $SPLIT"
echo "Grounding Results: $GROUNDING_RESULTS_DIR"
echo "Region Ref Results: ${REGION_REF_RESULTS_DIR}/${REGION_REF_MODEL_IDENTIFIER}"
echo "========================================="

mkdir -p "$GROUNDING_RESULTS_ROOT" "$REGION_REF_RESULTS_DIR"

"$PYTHON_BIN" eval/evaluate_grounding_addtoken_lfj.py \
    --model_path "$MODEL_PATH" \
    --root_dir "$ROOT_DIR" \
    --datasets "$DATASETS" \
    --split "$SPLIT" \
    --batch_per_device "$BATCH_PER_DEVICE" \
    --save_results_dir "$GROUNDING_RESULTS_ROOT" \
    --model_identifier "$MODEL_IDENTIFIER" \
    --temperature "$TEMPERATURE" \
    --gpu_id 0 \
    --pos_decoder_type "$POS_DECODER_TYPE" \
    $([ "$USE_DETAILED_TEMPLATE" = true ] && echo "--use_detailed_template")

"$PYTHON_BIN" eval/evaluate_grounding_group_reference_lfj.py \
    --model_path "$MODEL_PATH" \
    --root_dir "$ROOT_DIR" \
    --grounding_results_dir "$GROUNDING_RESULTS_DIR" \
    --datasets "$DATASETS" \
    --split "$SPLIT" \
    --batch_per_device "$BATCH_PER_DEVICE" \
    --save_results_dir "$REGION_REF_RESULTS_DIR" \
    --model_identifier "$REGION_REF_MODEL_IDENTIFIER" \
    --temperature "$TEMPERATURE" \
    --tasks "$REGION_REF_TASKS" \
    --gpu_id 0

echo "========================================="
echo "Pipeline completed."
echo "Grounding outputs: $GROUNDING_RESULTS_DIR"
echo "Region-level outputs: ${REGION_REF_RESULTS_DIR}/${REGION_REF_MODEL_IDENTIFIER}"
echo "========================================="
