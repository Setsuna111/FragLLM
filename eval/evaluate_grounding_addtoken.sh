#!/bin/bash

# Unified evaluation script for function and reference datasets
# Supports both single GPU and multi-GPU distributed evaluation

# ======================================
# Configuration Section - Modify as needed
# ======================================

# Model and data paths
MODEL_PATH="/home/lfj/projects_dir/FragLLM/checkpoints/grounding_lora_only_act_trainable_iousam_merge_addtoken/"
ROOT_DIR="./data"
RESULTS_DIR="./eval_results"
MODEL_IDENTIFIER="grounding_lora_0918_trainable_iousam"  # Identifier for this model configuration
POS_DECODER_TYPE="Simple"
# Evaluation parameters
SPLIT="test"
BATCH_PER_DEVICE=4
TEMPERATURE=0.0

# GPU configuration
USE_SINGLE_GPU=false  # Set to true for single GPU mode, false for multi-GPU
SINGLE_GPU_ID=0       # GPU ID to use in single GPU mode
export CUDA_VISIBLE_DEVICES=0,1  # Specify visible GPUs for multi-GPU mode
NUM_GPUS=2           # Number of GPUs for distributed training
MASTER_PORT=24989     # Master port for distributed training

# Dataset selection - modify as needed
# Available function datasets: Pro2Text
# Available reference datasets: ActRefClass, ActRefDesc, BindIRefClass, BindIRefDesc, 
#                              DomRefClass, DomRefDesc, EvoRefClass, EvoRefDesc, 
#                              MotifRefClass, MotifRefDesc

# whether use detailed template
USE_DETAILED_TEMPLATE=true

# Examples of dataset combinations:
# DATASETS="Pro2Text"                                    # Single function dataset
# DATASETS="ActRefClass,ActRefDesc"                      # Multiple reference datasets  
# DATASETS="Pro2Text,ActRefClass,MotifRefDesc"          # Mixed function and reference datasets
# DATASETS="MotifRefDesc,ActRefClass"                      # Default: two reference datasets
DATASETS="ActGroundSingle"
# DATASETS="ActGroundSingle,BindIGroundSingle,MotifGroundSingle"


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
    python eval/evaluate_grounding_addtoken.py \
        --model_path "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --datasets "$DATASETS" \
        --split "$SPLIT" \
        --batch_per_device "$BATCH_PER_DEVICE" \
        --save_results_dir "$RESULTS_DIR" \
        --model_identifier "$MODEL_IDENTIFIER" \
        --temperature "$TEMPERATURE" \
        --single_gpu \
        --gpu_id "$SINGLE_GPU_ID" \
        --pos_decoder_type "$POS_DECODER_TYPE" \
        $([ "$USE_DETAILED_TEMPLATE" = true ] && echo "--use_detailed_template")
else
    echo "Mode: Multi-GPU Distributed (GPUs: $NUM_GPUS)"
    echo "Master Port: $MASTER_PORT"
    echo "========================================="
    
    # Multi-GPU distributed execution
    torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        eval/evaluate_grounding_addtoken.py \
        --model_path "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --datasets "$DATASETS" \
        --split "$SPLIT" \
        --batch_per_device "$BATCH_PER_DEVICE" \
        --save_results_dir "$RESULTS_DIR" \
        --model_identifier "$MODEL_IDENTIFIER" \
        --temperature "$TEMPERATURE" \
        --pos_decoder_type "$POS_DECODER_TYPE" \
        $([ "$USE_DETAILED_TEMPLATE" = true ] && echo "--use_detailed_template")
fi

echo "========================================="
echo "Evaluation completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Check the following files for results:"
for dataset in $(echo "$DATASETS" | tr ',' ' '); do
    if [[ "$dataset" == *"GroundSingle" ]]; then
        echo "  - $RESULTS_DIR/grounding_single/$MODEL_IDENTIFIER/${dataset}_results.csv"
    elif [[ "$dataset" == *"GroundGroup" ]]; then
        echo "  - $RESULTS_DIR/grounding_group/$MODEL_IDENTIFIER/${dataset}_results.csv"
    else
        echo "  - $RESULTS_DIR/${dataset}_results.csv"
    fi
done
echo "========================================="