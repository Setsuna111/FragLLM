#!/bin/bash

# TM-align baseline for VenusX datasets
# Usage: ./run_tmalign.sh [dataset]

# DATASET=${1:-"all"}
DATASET=${1:-"Act"}

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Dom Evo Motif; do
        echo "Running TM-align on VenusX_$ds..."
        python baselines/tm_align.py --dataset $ds
    done
else
    echo "Running TM-align on VenusX_$DATASET..."
    python baselines/tm_align.py --dataset $DATASET
fi