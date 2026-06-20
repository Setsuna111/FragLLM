#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"

# DATASETS="ActRefDesc"
# DATASETS="BindIRefDesc"
# DATASETS="DomRefDesc"
# DATASETS="EvoRefDesc"
DATASETS="MotifRefDesc"

# prot2text
MODEL_IDENTIFIER="prot2text_v2_11b"
# RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_70/referring_desc_fragment_emb"
RESULTS_DIR="/home/dataset-local/projects_dir/FragLLM/baselines/prot2text_results/data_30/ref_frag_only_desc"


Evaluate_Exact_Match=True
Evaluate_Bleu=True
Evaluate_Rouge=True
Evaluate_Bert_Score=True
Verbose=True

CSV_PATH="${RESULTS_DIR}/${MODEL_IDENTIFIER}/${DATASETS}_results.csv"

python eval/metric_ref_desc_benchmark_lfj.py --results_path $CSV_PATH --evaluate_exact_match $Evaluate_Exact_Match --evaluate_bleu $Evaluate_Bleu --evaluate_rouge $Evaluate_Rouge --evaluate_bert_score $Evaluate_Bert_Score --verbose $Verbose
