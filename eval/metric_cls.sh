#!/bin/sh
export PYTHONPATH="./:$PYTHONPATH"

DATASETS="ActRefClass"

RESULTS_DIR="./eval_results/referring_cls"

MODEL_IDENTIFIER="FragLLM_260120_ReferringAll_1e_bs16_4gpu_lr2e4_lora32_merge"  # Identifier for this model configuration

EMBEDDING_MODEL="/home/dataset-local/projects/Data/HF_models/Qwen3-Embedding-0.6B"
INTERPRO_DB_PATH="./data/new_interpro_metadata_short_with_fragment_type_v4.json"
CACHE_DIR="/home/dataset-local/projects/Data/FragLLM_git_v1_2512/eval/cache"
BATCH_SIZE=16
BERT_MODEL_TYPE="/home/dataset-local/projects/Data/HF_models/biobert-large-cased-v1.1"
CSV_PATH="${RESULTS_DIR}/${MODEL_IDENTIFIER}/${DATASETS}_results.csv"

python ./eval/metric_cls.py \
    --results_path $CSV_PATH \
    --embedding_model $EMBEDDING_MODEL \
    --interpro_db_path $INTERPRO_DB_PATH \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --bert_model_type $BERT_MODEL_TYPE