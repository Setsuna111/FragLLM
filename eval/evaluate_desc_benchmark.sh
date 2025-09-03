#!/bin/sh

## USAGE

export PYTHONPATH="./:$PYTHONPATH"
Results_Path=./eval_results/function_desc/fragment_training_only_stage2_bw_stage1_lora32_epoch3_function_0901_merge_512.csv
Evaluate_Exact_Match=True
Evaluate_Bleu=True
Evaluate_Rouge=True
Evaluate_Bert_Score=True
Verbose=True

python eval/evaluate_desc_benchmark.py --results_path $Results_Path --evaluate_exact_match $Evaluate_Exact_Match --evaluate_bleu $Evaluate_Bleu --evaluate_rouge $Evaluate_Rouge --evaluate_bert_score $Evaluate_Bert_Score --verbose $Verbose

