#!/bin/bash

# Foldseek baseline for VenusX datasets
# Usage: ./run_foldseek.sh [dataset] [threads] [alignment_type]

DATASET=${1:-"all"}
# DATASET=${1:-"Dom"}

THREADS=${2:-8}
ALIGNMENT_TYPE=${3:-0}  # 0: 3Di, 1: TMalign, 2: 3Di+AA

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Evo Motif Dom; do
        echo "Running Foldseek on VenusX_$ds..."
        python baselines/foldseek.py --dataset $ds --num_threads $THREADS --alignment_type $ALIGNMENT_TYPE
    done
else
    echo "Running Foldseek on VenusX_$DATASET..."
    python baselines/foldseek.py --dataset $DATASET --num_threads $THREADS --alignment_type $ALIGNMENT_TYPE
fi