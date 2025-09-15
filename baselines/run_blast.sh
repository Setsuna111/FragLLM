#!/bin/bash

# BLAST baseline for VenusX datasets
# Usage: ./run_blast.sh [dataset] [threads]

DATASET=${1:-"all"}
# DATASET=${1:-"Motif"}

THREADS=${2:-16}

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Dom Evo Motif; do
        echo "Running BLAST on VenusX_$ds..."
        python baselines/blast.py --dataset $ds --num_threads $THREADS
    done
else
    echo "Running BLAST on VenusX_$DATASET..."
    python baselines/blast.py --dataset $DATASET --num_threads $THREADS
fi