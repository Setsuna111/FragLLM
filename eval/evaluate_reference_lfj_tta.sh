#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

FRAG_PYTHON="${FRAG_PYTHON:-/home/dataset-local/anaconda3/envs/fragllm/bin/python}"
EMBEDDING_PYTHON="${EMBEDDING_PYTHON:-/home/dataset-local/anaconda3/envs/pika_for_qwen/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/0529_all/checkpoint-315000_merge}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data_70}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/eval_results}"
MODEL_IDENTIFIER="${MODEL_IDENTIFIER:-0529_all_315000_tta3_mean}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
DATASETS="${DATASETS:-ActRefClass,BindIRefClass,DomRefClass,EvoRefClass,MotifRefClass}"

RUN_DIR="$RESULTS_ROOT/referring_cls/$MODEL_IDENTIFIER"
VIEW_ROOT="$RUN_DIR/views"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
cd "$ROOT_DIR"

echo "========================================="
echo "Reference-Class Three-View TTA Inference"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Physical GPU: $PHYSICAL_GPU"
echo "Seeds: 42,43,44"
echo "Datasets: $DATASETS"
echo "Output directory: $RUN_DIR"

"$FRAG_PYTHON" "$SCRIPT_DIR/evaluate_reference_lfj_tta.py" generate \
    --model_path "$MODEL_PATH" \
    --root_dir "$DATA_ROOT" \
    --split test \
    --datasets "$DATASETS" \
    --output_dir "$RUN_DIR" \
    --seeds "42,43,44" \
    --batch_per_device "$BATCH_PER_DEVICE" \
    --gpu_id 0

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"
for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="${DATASET//[[:space:]]/}"
    FINAL_RESULT="$RUN_DIR/${DATASET}_results.csv"

    echo "Aggregating dataset: $DATASET"
    "$EMBEDDING_PYTHON" "$SCRIPT_DIR/evaluate_reference_lfj_tta.py" aggregate \
        --inputs \
            "$VIEW_ROOT/seed42/${DATASET}_results.csv" \
            "$VIEW_ROOT/seed43/${DATASET}_results.csv" \
            "$VIEW_ROOT/seed44/${DATASET}_results.csv" \
        --output_path "$FINAL_RESULT" \
        --details_path "$RUN_DIR/${DATASET}_tta_details.csv" \
        --config_path "$RUN_DIR/${DATASET}_tta_aggregation_config.json" \
        --dataset_name "$DATASET" \
        --device cuda
done

echo "========================================="
echo "Three-view TTA inference completed"
echo "Final results directory: $RUN_DIR"
echo "========================================="
