#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"

PYTHON_BIN="/home/dataset-local/anaconda3/envs/pika_for_qwen/bin/python"
Results_Path="./eval_results/referring_desc/0529_all_123500/ActRefDesc_results.csv"
Embedding_Model="/home/dataset-local/projects_dir/pretrained_model/Qwen3-Embedding-0.6B/"
Interpro_DB_Path="/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json"
Cache_Dir="/home/dataset-local/projects_dir/FragLLM/eval/cache/"
Batch_Size=16
Top_K=5
Device="cuda"
Verbose=True

if [ "$Verbose" = "True" ]; then
    "$PYTHON_BIN" eval/metric_ref_desc_recall_lfj.py \
        --results_path "$Results_Path" \
        --embedding_model "$Embedding_Model" \
        --interpro_db_path "$Interpro_DB_Path" \
        --cache_dir "$Cache_Dir" \
        --batch_size "$Batch_Size" \
        --top_k "$Top_K" \
        --device "$Device" \
        --verbose
else
    "$PYTHON_BIN" eval/metric_ref_desc_recall_lfj.py \
        --results_path "$Results_Path" \
        --embedding_model "$Embedding_Model" \
        --interpro_db_path "$Interpro_DB_Path" \
        --cache_dir "$Cache_Dir" \
        --batch_size "$Batch_Size" \
        --top_k "$Top_K" \
        --device "$Device"
fi
