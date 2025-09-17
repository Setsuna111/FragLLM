#!/bin/sh

## USAGE

## bash eval/region_captioning/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE

# export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="./:$PYTHONPATH"
# export PYTHONNOUSERSITE=True
# export PYTHONPATH=$(pwd):$PYTHONPATH
MASTER_PORT=24999
NUM_GPUS=8  # Adjust it as per the available #GPU

Model_PATH=/data/djy/FragLLM_git/checkpoints/fragment_training_only_stage2_bw_stage1_lora32_epoch5_function_0902_merge
# Path to the GranD-f evaluation dataset images directory
Root_DIR=/home/djy/projects/Documents/FragLLM/data

Data_NAME=Pro2Text
RESULT_PATH=./eval_results/function_desc/fragment_training_only_stage2_bw_stage1_lora32_epoch5_function_0902_merge_512_test1.csv

split=test
Batch_Per_Device=8

# generate params
Temperature=1.0
Num_Beams=1
Length_Penalty=1.0
Max_New_Tokens=512
Top_P=1.0
Top_K=50

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval_local/evaluate_function_desc.py --model_path "$Model_PATH" --root_dir "$Root_DIR" --data_name "$Data_NAME" --split "$split" --batch_per_device "$Batch_Per_Device" --save_results_path "$RESULT_PATH" --temperature $Temperature --num_beams $Num_Beams --length_penalty $Length_Penalty --max_new_tokens $Max_New_Tokens --top_p $Top_P --top_k $Top_K


# Evaluate
# python eval/region_captioning/evaluate.py --annotation_file "$ANNOTATION_FILE" --results_dir "$RESULT_PATH"
