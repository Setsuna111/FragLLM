#!/bin/bash

# Usage: ./metric_single_grounding_blast.sh [dataset] [iou_threshold]

DATASET=${1:-"Act"}
IOU_Threshold=${2:-0.5}
Results_Path="baselines/blast_cls_results/VenusX_${DATASET}/single_localization_results.csv"

python eval/metric_single_grounding_blast.py --results_path $Results_Path --iou_threshold $IOU_Threshold
