#!/bin/bash

# Simplified script to evaluate BLAST classification results

export PYTHONPATH="./:$PYTHONPATH"

# Default parameters
RESULTS_PATH=${1:-"./baselines/blast_cls_results/VenusX_Act/multiple_localization_results.csv"}
IOU_THRESHOLD=${2:-0.5}

# Run evaluation
python eval/metric_group_grounding_blast.py \
    --results_path "$RESULTS_PATH" \
    --iou_threshold "$IOU_THRESHOLD"