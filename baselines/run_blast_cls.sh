#!/bin/bash

# BLAST classification baseline for VenusX datasets
# Usage: ./run_blast_cls.sh [dataset] [task]

DATASET=${1:-"Act"}
TASK=${2:-"single"}

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Evo Motif Dom; do
        echo "Running BLAST classification on VenusX_$ds with task $TASK..."
        python baselines/blast_grounding.py --dataset $ds --task $TASK
    done
else
    echo "Running BLAST classification on VenusX_$DATASET with task $TASK..."
    python baselines/blast_grounding.py --dataset $DATASET --task $TASK
fi