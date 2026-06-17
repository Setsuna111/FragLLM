#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=/home/dataset-local/projects_dir/FragLLM/baselines/prot2text/referring_desc/prot2text_v2_11b/EvoRefDesc_results.csv
Evaluate_Exact_Match=True
Evaluate_Bleu=True
Evaluate_Rouge=True
Evaluate_Bert_Score=True
Verbose=True

python eval/metric_ref_desc_benchmark_lfj.py --results_path $Results_Path --evaluate_exact_match $Evaluate_Exact_Match --evaluate_bleu $Evaluate_Bleu --evaluate_rouge $Evaluate_Rouge --evaluate_bert_score $Evaluate_Bert_Score --verbose $Verbose
