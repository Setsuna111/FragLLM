#!/bin/sh

## USAGE
## Configure DATASETS with comma-separated dataset names, then run this script.

export PYTHONPATH="./:$PYTHONPATH"

DATASETS="ActRefDesc"
# DATASETS="BindIRefDesc"
# DATASETS="DomRefDesc"
# DATASETS="EvoRefDesc"
# DATASETS="MotifRefDesc"
# DATASETS="ActRefDesc,BindIRefDesc,DomRefDesc,EvoRefDesc,MotifRefDesc"

# ours
# MODEL_IDENTIFIER="0529_all_315000"
# RESULTS_DIR="./eval_results/referring_desc"

# prot2text
# MODEL_IDENTIFIER="prot2text_v2_11b"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_70/referring_desc_fragment_emb"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_30/ref_frag_only_desc"
# MODEL_IDENTIFIER="instructbiomol_instruct"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/instructbiomol_results/data_30/referring_desc_fragment_struct"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/instructbiomol_results/data_30/ref_frag_only_desc"

# prot2text/instructbiomol v6
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/ref_frag_only_desc_v6"
RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/referring_desc_fragment_emb_v6"
RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/analysis/0712analysis_2/active_site_v6_results/data_70/referring_desc_fragment_struct_v6"
# MODEL_IDENTIFIER="prot2text_active_site_v6"  # prot2text v6
MODEL_IDENTIFIER="instructbiomol_active_site_v6"  # instructbiomol v6

Evaluate_Exact_Match=True
Evaluate_Bleu=True
Evaluate_Rouge=True
Evaluate_Bert_Score=True
Verbose=True

for DATASET in $(echo "$DATASETS" | tr ',' ' '); do
    CSV_PATH="${RESULTS_DIR}/${MODEL_IDENTIFIER}/${DATASET}_results.csv"

    echo "Evaluating dataset: ${DATASET}"
    echo "CSV path: ${CSV_PATH}"

    python eval/metric_ref_desc_benchmark_lfj.py --results_path "$CSV_PATH" --evaluate_exact_match "$Evaluate_Exact_Match" --evaluate_bleu "$Evaluate_Bleu" --evaluate_rouge "$Evaluate_Rouge" --evaluate_bert_score "$Evaluate_Bert_Score" --verbose "$Verbose"
done
