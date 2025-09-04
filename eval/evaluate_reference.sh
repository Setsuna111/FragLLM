#!/bin/bash

# Unified evaluation script for function and reference datasets
# Supports both single GPU and multi-GPU distributed evaluation

# ======================================
# Configuration Section - Modify as needed
# ======================================

# Model and data paths
MODEL_PATH="/home/lfj/projects_dir/FragLLM/checkpoints/test_load_stage1_only_motifdesc_lora_fast_save_after_data_fix/checkpoint-1100_merge/"
ROOT_DIR="./data"
RESULTS_DIR="./results/results_reference_fast_save_test_0904"

# Evaluation parameters
SPLIT="test"
BATCH_PER_DEVICE=4
TEMPERATURE=0.0

# GPU configuration
USE_SINGLE_GPU=false  # Set to true for single GPU mode, false for multi-GPU
SINGLE_GPU_ID=0       # GPU ID to use in single GPU mode
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Specify visible GPUs for multi-GPU mode
NUM_GPUS=8           # Number of GPUs for distributed training
MASTER_PORT=24989     # Master port for distributed training

# Dataset selection - modify as needed
# Available function datasets: Pro2Text
# Available reference datasets: ActRefClass, ActRefDesc, BindIRefClass, BindIRefDesc, 
#                              DomRefClass, DomRefDesc, EvoRefClass, EvoRefDesc, 
#                              MotifRefClass, MotifRefDesc

# Examples of dataset combinations:
# DATASETS="Pro2Text"                                    # Single function dataset
# DATASETS="ActRefClass,ActRefDesc"                      # Multiple reference datasets  
# DATASETS="Pro2Text,ActRefClass,MotifRefDesc"          # Mixed function and reference datasets
# DATASETS="MotifRefDesc,ActRefClass"                      # Default: two reference datasets
DATASETS="MotifRefDesc"


# ======================================
# Script Execution - Do not modify below unless needed
# ======================================

# Set environment variables
export PYTHONPATH="./:$PYTHONPATH"
# Uncomment the line below if you want to disable user site packages
# export PYTHONNOUSERSITE=True

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Unified Evaluation Script"
echo "========================================="
echo "Model Path: $MODEL_PATH"
echo "Data Root: $ROOT_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Datasets: $DATASETS"
echo "Split: $SPLIT"
echo "Batch per device: $BATCH_PER_DEVICE"
echo "Temperature: $TEMPERATURE"

if [ "$USE_SINGLE_GPU" = true ]; then
    echo "Mode: Single GPU (GPU ID: $SINGLE_GPU_ID)"
    echo "========================================="
    
    # Single GPU execution
    python eval/evaluate_reference.py \
        --model_path "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --datasets "$DATASETS" \
        --split "$SPLIT" \
        --batch_per_device "$BATCH_PER_DEVICE" \
        --save_results_dir "$RESULTS_DIR" \
        --temperature "$TEMPERATURE" \
        --single_gpu \
        --gpu_id "$SINGLE_GPU_ID"
else
    echo "Mode: Multi-GPU Distributed (GPUs: $NUM_GPUS)"
    echo "Master Port: $MASTER_PORT"
    echo "========================================="
    
    # Multi-GPU distributed execution
    torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        eval/evaluate_reference.py \
        --model_path "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --datasets "$DATASETS" \
        --split "$SPLIT" \
        --batch_per_device "$BATCH_PER_DEVICE" \
        --save_results_dir "$RESULTS_DIR" \
        --temperature "$TEMPERATURE"
fi

echo "========================================="
echo "Evaluation completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Check the following files for results:"
for dataset in $(echo "$DATASETS" | tr ',' ' '); do
    echo "  - $RESULTS_DIR/${dataset}_results.csv"
done
echo "========================================="