from transformers import AutoTokenizer
from models.protein_llama import ProteinLlamaForCausalLM
# import evaluate
from dataset.dataloader_function import FunctionDataset
from dataset.dataloader_frag import FragDataCollator
from dataset.templates import *
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from transformers.utils import logging
from tqdm import tqdm
import torch
import os
import argparse
from eval.ddp import *
from scripts import utils_argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--model_path", help="path to the prot2text model")
argParser.add_argument("--temperature", default=1.0, type=float)
argParser.add_argument("--num_beams", default=1, type=int)
argParser.add_argument("--top_p", default=1.0, type=float)
argParser.add_argument("--top_k", default=50, type=int)
argParser.add_argument("--length_penalty", default=1.0, type=float)
argParser.add_argument("--max_new_tokens", default=512, type=int)
argParser.add_argument("--root_dir", help="root folder of the data")
argParser.add_argument("--data_name", help="data name")
argParser.add_argument("--split", help="train, test or eval split?")
argParser.add_argument("--batch_per_device", help="batch size for each device")
argParser.add_argument("--save_results_path", help="path to save the generated description")
argParser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
argParser.add_argument('--local_rank', default=-1, type=int)
argParser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')



args = argParser.parse_args()
init_distributed_mode(args)
device = torch.device(f"cuda:{args.rank}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|reserved_special_token_0|>')
model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
model.config.pad_token_id = tokenizer.pad_token_id
sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)
eval_dataset = FunctionDataset(
    root_dir=args.root_dir,
    data_name=args.data_name,
    split=args.split,
    task_type="function",
    question_template=ProteinFunction,
    answer_template=None,
    max_sequence_length=1021
)
print('eval set loaded')
model.eval()
batch_size = int(args.batch_per_device)
model = model.bfloat16().to(device)


data_collator = FragDataCollator(
        sequence_tokenizer=sequence_tokenizer,
        llm_tokenizer=tokenizer,
        mode="inference",
        max_sequence_length=1021,
        max_description_length=512,
        use_max_desc_length=True
    )
distributed_sampler = DistributedSampler(eval_dataset, rank=args.rank, shuffle=False)
dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0,
                            sampler=distributed_sampler, collate_fn=data_collator)
print("length of eval_dataset: ", len(eval_dataset))
if torch.distributed.is_initialized():
    if torch.distributed.get_rank()==0:
        if os.path.exists(args.save_results_path):
            os.remove(args.save_results_path)
else:
    if os.path.exists(args.save_results_path):
        os.remove(args.save_results_path)

generated = list()
functions = list()
dataset_idxs = list()
for inputs in tqdm(dataloader):
    # inputs = inputs.to_dict()
    functions += tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)
    inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
    dataset_idxs += inputs["dataset_idxs"]

    tok_ids = model.generate(inputs=None, 
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            protein_input_ids=inputs["protein_input_ids"],
                            protein_attention_mask=inputs["protein_attention_mask"],
                            protein_inputs_embeds=None,
                            position_refs=inputs["position_refs"],
                            num_beams=args.num_beams,
                            early_stopping=False,
                            no_repeat_ngram_size=None,
                            eos_token_id=128009, 
                            pad_token_id=128002,
                            length_penalty=args.length_penalty,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            top_p=args.top_p,
                            top_k=args.top_k 
                        )
    generated += tokenizer.batch_decode(tok_ids, skip_special_tokens=True)

data= {'generated': generated, 'function':functions, 'dataset_idxs':dataset_idxs}
df = pd.DataFrame(data)
print("length of data: ", len(df))
df.to_csv(args.save_results_path, index=False, mode='a')

# if torch.distributed.is_initialized():  
#     torch.distributed.barrier() 
#     if torch.distributed.get_rank() > 0:
#         exit(0)   
# res = pd.read_csv(args.save_results_path).drop_duplicates()
# res = res.drop(res[res['name'] == 'name'].index)   

# res_bleu = bleu.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
# res_rouge = rouge.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
# res_bertscore = bert_score.compute(predictions=res['generated'].tolist(), references=res['function'].tolist(),
#                                   model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24)
# print(res_bleu)
# print(res_rouge)
# def Average(lst):
#     return sum(lst) / len(lst)
# print('Bert Score: ', Average(res_bertscore['f1']))