#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=./eval_results/single_grounding/fragment_training_only_stage2_bw_stage1_lora32_epoch1_0907_grounding_single_merge/DomGroundSingle_results.csv
IOU_Threshold=0.5


python eval_local/metric_single_grounding.py --results_path $Results_Path --iou_threshold $IOU_Threshold
