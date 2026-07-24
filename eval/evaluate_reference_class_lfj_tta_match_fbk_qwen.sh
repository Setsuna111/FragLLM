#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

FRAG_PYTHON="${FRAG_PYTHON:-/home/dataset-local/anaconda3/envs/fragllm/bin/python}"
EMBEDDING_PYTHON="${EMBEDDING_PYTHON:-/home/dataset-local/anaconda3/envs/pika_for_qwen/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/0529_all/checkpoint-330000_merge}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data_70}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/eval_results}"
MODEL_IDENTIFIER="${MODEL_IDENTIFIER:-0529_all_330000_tta3_match_fbk_qwen}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
DATASETS="${DATASETS:-ActRefClass,BindIRefClass,DomRefClass,EvoRefClass,MotifRefClass}"
# Reuse the already generated seed42/43/44 views by default. Set
# RUN_GENERATE=1 to run the unchanged three-view inference stage first.
RUN_GENERATE="${RUN_GENERATE:-0}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$RESULTS_ROOT/referring_cls/0529_all_330000_tta3_match_fbk_qwen}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$RESULTS_ROOT/referring_cls/0529_all_315000_tta3_mean}"

RUN_DIR="$RESULTS_ROOT/referring_cls/$MODEL_IDENTIFIER"
if [[ "$RUN_GENERATE" == "1" ]]; then
    VIEW_ROOT="$RUN_DIR/views"
else
    VIEW_ROOT="$SOURCE_RUN_DIR/views"
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
cd "$ROOT_DIR"

echo "==============================================="
echo "Reference-Class TTA: exact match + Qwen fallback"
echo "==============================================="
echo "Model: $MODEL_PATH"
echo "Physical GPU: $PHYSICAL_GPU"
echo "Seeds: 42,43,44"
echo "Datasets: $DATASETS"
echo "Output directory: $RUN_DIR"
echo "Input view directory: $VIEW_ROOT"

mkdir -p "$RUN_DIR"

if [[ "$RUN_GENERATE" == "1" ]]; then
    "$FRAG_PYTHON" "$SCRIPT_DIR/evaluate_reference_class_lfj_tta_match_fbk_qwen.py" generate \
        --model_path "$MODEL_PATH" \
        --root_dir "$DATA_ROOT" \
        --split test \
        --datasets "$DATASETS" \
        --output_dir "$RUN_DIR" \
        --seeds "42,43,44" \
        --batch_per_device "$BATCH_PER_DEVICE" \
        --gpu_id 0
fi

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"
for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="${DATASET//[[:space:]]/}"
    for SEED in 42 43 44; do
        VIEW_PATH="$VIEW_ROOT/seed${SEED}/${DATASET}_results.csv"
        if [[ ! -f "$VIEW_PATH" ]]; then
            echo "Missing three-view input: $VIEW_PATH" >&2
            echo "Set SOURCE_RUN_DIR to the run containing seed42/43/44 views, or RUN_GENERATE=1." >&2
            exit 1
        fi
    done
    BASELINE_METRICS="$BASELINE_RUN_DIR/${DATASET}_metrics.json"
    if [[ ! -f "$BASELINE_METRICS" ]]; then
        echo "Missing baseline metrics: $BASELINE_METRICS" >&2
        exit 1
    fi

    FINAL_RESULT="$RUN_DIR/${DATASET}_results.csv"
    echo "Aggregating dataset: $DATASET"
    "$EMBEDDING_PYTHON" "$SCRIPT_DIR/evaluate_reference_class_lfj_tta_match_fbk_qwen.py" aggregate \
        --inputs \
            "$VIEW_ROOT/seed42/${DATASET}_results.csv" \
            "$VIEW_ROOT/seed43/${DATASET}_results.csv" \
            "$VIEW_ROOT/seed44/${DATASET}_results.csv" \
        --output_path "$FINAL_RESULT" \
        --details_path "$RUN_DIR/${DATASET}_tta_match_fbk_qwen_details.csv" \
        --config_path "$RUN_DIR/${DATASET}_tta_match_fbk_qwen_aggregation_config.json" \
        --metrics_path "$RUN_DIR/${DATASET}_metrics.json" \
        --comparison_path "$RUN_DIR/${DATASET}_match_fbk_qwen_comparison.json" \
        --baseline_metrics_path "$BASELINE_METRICS" \
        --dataset_name "$DATASET" \
        --device cuda
done

"$EMBEDDING_PYTHON" "$SCRIPT_DIR/evaluate_reference_class_lfj_tta_match_fbk_qwen.py" compare \
    --new_dir "$RUN_DIR" \
    --baseline_dir "$BASELINE_RUN_DIR" \
    --datasets "$DATASETS" \
    --output_path "$RUN_DIR/tta_match_fbk_qwen_comparison.csv"

echo "==============================================="
echo "Exact-match + Qwen fallback TTA completed"
echo "Final results directory: $RUN_DIR"
echo "Comparison: $RUN_DIR/tta_match_fbk_qwen_comparison.csv"
echo "==============================================="
