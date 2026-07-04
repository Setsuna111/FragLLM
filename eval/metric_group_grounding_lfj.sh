#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=./eval_results/grounding_group/grounding_lora_0529_all_215000/DomGroundGroup_results.csv
IOU_Threshold=0.5
BERT_Threshold=1.0
Embedding_Path=/home/dataset-local/projects_dir/FragLLM/eval/cache/VenusX_Act.npz
DATA_ROOT="/home/dataset-local/projects_dir/FragLLM/data_70"
BATCH_SIZE=32

python eval/metric_group_grounding_lfj.py \
    --results_path $Results_Path \
    --iou_threshold $IOU_Threshold \
    --bert_threshold $BERT_Threshold \
    --embedding_path $Embedding_Path \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE
