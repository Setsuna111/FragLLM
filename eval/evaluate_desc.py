import evaluate
import pandas as pd
import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("--save_results_path",default='./eval_results/function_desc/fragment_training_only_stage2_lora32_epoch1_0831_merge_512.csv', help="path to save the generated description")
args = argParser.parse_args()

bleu = evaluate.load(path="./eval/metrics/bleu")
rouge = evaluate.load(path="./eval/metrics/rouge")
bert_score = evaluate.load(path="./eval/metrics/bertscore")
res = pd.read_csv(args.save_results_path)
print("len(res): ", len(res))
# res = res.drop_duplicates()
# print("len(res.drop_duplicates()): ", len(res))
res_bleu = bleu.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
res_rouge = rouge.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
res_bertscore = bert_score.compute(predictions=res['generated'].tolist(), references=res['function'].tolist(), model_type="/home/djy/projects/Data/HF_models/biobert-large-cased-v1.1", num_layers=24)
print(res_bleu)
print(res_rouge)
def Average(lst):
    return sum(lst) / len(lst)
print('Bert Score: ', Average(res_bertscore['f1']))