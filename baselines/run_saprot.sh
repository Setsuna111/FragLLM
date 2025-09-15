#!/bin/bash

# SaProt baseline for VenusX datasets - combines protein sequences with structure information
# Usage: ./run_saprot.sh [dataset] [batch_size] [model_path]

DATASET=${1:-"all"}
BATCH_SIZE=${2:-16}

MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/SaProt_35M_AF2/"}  # saprot 35M
# MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/SaProt_650M_AF2/"}  # saprot 650M

# Structure-related paths
PDB_BASE_PATH="/home/lfj/database/VenusX_AFDB"
CORRECTION_FILE="/home/lfj/database/VenusX_AFDB/pdb_fragment_name_corrections.json"

export CUDA_VISIBLE_DEVICES=6  # Specify visible GPUs for multi-GPU mode

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Motif Evo Dom; do
        echo "Running SaProt on VenusX_$ds..."
        python baselines/saprot.py \
            --dataset $ds \
            --batch_size $BATCH_SIZE \
            --model_path "$MODEL_PATH" \
            --pdb_base_path "$PDB_BASE_PATH" \
            --correction_file "$CORRECTION_FILE"
    done
else
    echo "Running SaProt on VenusX_$DATASET..."
    python baselines/saprot.py \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --model_path "$MODEL_PATH" \
        --pdb_base_path "$PDB_BASE_PATH" \
        --correction_file "$CORRECTION_FILE"
fi