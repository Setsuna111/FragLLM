#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=./eval_results/group_grounding/fragment_training_stage2_bw_stage1_concat_lora32_0912_epoch10_allgroundinggroup_test_lr2e4_stage1_checkpoint_6_merge/ActGroundGroup_results.csv
IOU_Threshold=0.5
BERT_Threshold=1.0
Embedding_Path=/home/djy/projects/Documents/FragLLM_git/eval_local/vectors_val_test/VenusX_Act.npz

python eval_local/metric_group_grounding.py --results_path $Results_Path --iou_threshold $IOU_Threshold --bert_threshold $BERT_Threshold --embedding_path $Embedding_Path
