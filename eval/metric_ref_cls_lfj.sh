#!/bin/sh
export PYTHONPATH="./:$PYTHONPATH"

DATASETS="ActRefClass"
# DATASETS="BindIRefClass"
# DATASETS="DomRefClass"
# DATASETS="EvoRefClass"
# DATASETS="MotifRefClass"
# DATASETS="ActRefClass,BindIRefClass,DomRefClass,EvoRefClass,MotifRefClass"

# RESULTS_DIR="./eval_results/referring_cls"  # ours
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_70/referring_cls"  # prot2text no frag token
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_70/referring_cls_fragment_emb"  # prot2text all seqs
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_30/ref_frag_only_cls"  # prot2text frag only
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/instructbiomol_results/data_30/referring_cls_fragment_struct"  # instructbiomol all seq
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/instructbiomol_results/data_30/ref_frag_only_cls"  # instructbiomol frag only
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/ref_frag_only_cls_v6"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/referring_cls_fragment_emb_v6"
RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/referring_cls_fragment_struct_v6"

# MODEL_IDENTIFIER="0512_ref_small_4_4000"  # Identifier for this model configuration
# MODEL_IDENTIFIER="0529_all_315000"  # Identifier for this model configuration
# MODEL_IDENTIFIER="prot2text_v2_11b"  # prot2text
# MODEL_IDENTIFIER="instructbiomol_instruct"  # instructbiomol
# MODEL_IDENTIFIER="prot2text_active_site_v6"  # prot2text v6
MODEL_IDENTIFIER="instructbiomol_active_site_v6"  # instructbiomol v6

EMBEDDING_MODEL="/home/dataset-local/projects_dir/pretrained_model/Qwen3-Embedding-0.6B/"
INTERPRO_DB_PATH="/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json"
CACHE_DIR="/home/dataset-local/projects_dir/FragLLM/eval/cache/"
BATCH_SIZE=16
BERT_MODEL_TYPE="/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1"

for DATASET in $(echo "$DATASETS" | tr ',' ' '); do
    CSV_PATH="${RESULTS_DIR}/${MODEL_IDENTIFIER}/${DATASET}_results.csv"

    echo "Evaluating dataset: ${DATASET}"
    echo "CSV path: ${CSV_PATH}"

    python ./eval/metric_ref_cls_lfj.py \
        --results_path "$CSV_PATH" \
        --embedding_model "$EMBEDDING_MODEL" \
        --interpro_db_path "$INTERPRO_DB_PATH" \
        --cache_dir "$CACHE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --bert_model_type "$BERT_MODEL_TYPE"
done
