#!/bin/bash

# Protein Language Model baseline for VenusX datasets
# Usage: ./run_plm.sh [dataset] [batch_size] [model_path]

DATASET=${1:-"all"}
BATCH_SIZE=${2:-16}

# MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/esm2_t30_150M_UR50D/"}  # ESM2 t30 150M
# MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/"}  # ESM2 t33 650M
# MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/prot_t5_xl_uniref50/"}  # proT5 3B
# MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/prot_bert_bfd /"}  # protbert 420M
MODEL_PATH=${3:-"/home/lfj/projects_dir/pretrained_model/ankh-base/"}  # ankhbase 450M


export CUDA_VISIBLE_DEVICES=6  # Specify visible GPUs for multi-GPU mode

if [[ "$DATASET" == "all" ]]; then
    for ds in Act BindI Motif Evo Dom; do
        echo "Running PLM on VenusX_$ds..."
        python baselines/plm.py --dataset $ds --batch_size $BATCH_SIZE --model_path "$MODEL_PATH"
    done
else
    echo "Running PLM on VenusX_$DATASET..."
    python baselines/plm.py --dataset $DATASET --batch_size $BATCH_SIZE --model_path "$MODEL_PATH"
fi