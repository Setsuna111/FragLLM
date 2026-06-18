#!/bin/sh
export PYTHONPATH="./:$PYTHONPATH"

DATASETS="ActRefClass"
# DATASETS="DomRefClass"

# RESULTS_DIR="./eval_results/referring_cls"
RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/referring_cls"  # prot2text

# MODEL_IDENTIFIER="0512_ref_small_4_4000"  # Identifier for this model configuration
# MODEL_IDENTIFIER="0529_all_123500"  # Identifier for this model configuration
MODEL_IDENTIFIER="prot2text_v2_11b"  # prot2text

EMBEDDING_MODEL="/home/dataset-local/projects_dir/pretrained_model/Qwen3-Embedding-0.6B/"
INTERPRO_DB_PATH="/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json"
CACHE_DIR="/home/dataset-local/projects_dir/FragLLM/eval/cache/"
BATCH_SIZE=16
BERT_MODEL_TYPE="/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1"
CSV_PATH="${RESULTS_DIR}/${MODEL_IDENTIFIER}/${DATASETS}_results.csv"

python ./eval/metric_ref_cls_lfj.py \
    --results_path $CSV_PATH \
    --embedding_model $EMBEDDING_MODEL \
    --interpro_db_path $INTERPRO_DB_PATH \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --bert_model_type $BERT_MODEL_TYPE