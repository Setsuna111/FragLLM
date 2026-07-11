#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
# Results_Path=./eval_results/single_grounding/fragment_training_only_stage2_bw_stage1_lora32_epoch1_0907_grounding_single_merge/DomGroundSingle_results.csv
Path="/home/dataset-local/projects_dir/FragLLM/eval_results/grounding_single/grounding_lora_0529_all_215000/ActGroundSingle_results.csv"
IOU_Threshold=0.5

python eval/metric_single_grounding_lfj.py --results_path $Path --iou_threshold $IOU_Threshold
