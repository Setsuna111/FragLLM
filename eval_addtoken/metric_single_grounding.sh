#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=./eval_results/fragment_training_stage2_bw_stage1_concat_lora32_0917_epoch30_actsingle_lr2e4_stage1_checkpoint_12_addtoken_adapterfea_L_merge/single_grounding/ActGroundSingle_results.csv
IOU_Threshold=0.5


python eval_local_addtoken/metric_single_grounding.py --results_path $Results_Path --iou_threshold $IOU_Threshold
