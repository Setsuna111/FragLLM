import sys
sys.path.append('.')
from transformers import AutoTokenizer
from models.protein_llama import ProteinLlamaForCausalLM
# import evaluate
from dataset.dataloader_refferring import *
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

argParser = argparse.ArgumentParser()
argParser.add_argument("--model_path", help="path to the trained model", default="/home/lfj/projects_dir/FragLLM/checkpoints/fragment_training_only_stage2_lora16_save_test_merge/")
argParser.add_argument("--temperature", default=0.0, type=float)
argParser.add_argument("--root_dir", default='./data', help="root folder of the data")
argParser.add_argument("--dataset_name", default='MotifRefDesc', help="reference dataset name (e.g., ActRefClass, DomRefDesc, etc.)")
argParser.add_argument("--split", help="train, test or eval split?", default="test")
argParser.add_argument("--batch_per_device", help="batch size for each device", type=int, default=1)
argParser.add_argument("--save_results_path", help="path to save the generated results", default="./results/motifref_desc_results.csv")
argParser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
argParser.add_argument('--local_rank', default=-1, type=int)
argParser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

# Mapping of dataset names to dataset classes
REFERENCE_DATASETS = {
    'ActRefClass': ActRefClass,
    'ActRefDesc': ActRefDesc,
    'BindIRefClass': BindIRefClass,
    'BindIRefDesc': BindIRefDesc,
    'DomRefClass': DomainRefClass,
    'DomRefDesc': DomainRefDesc,
    'EvoRefClass': EvoRefClass,
    'EvoRefDesc': EvoRefDesc,
    'MotifRefClass': MotifRefClass,
    'MotifRefDesc': MotifRefDesc,
}

args = argParser.parse_args()

# Validate dataset name
if args.dataset_name not in REFERENCE_DATASETS:
    raise ValueError(f"Dataset {args.dataset_name} not supported. Available datasets: {list(REFERENCE_DATASETS.keys())}")

# Initialize distributed mode (support both single-card and multi-card)
init_distributed_mode(args)

# Set device based on distributed mode
if hasattr(args, 'distributed') and args.distributed:
    device = torch.device(f"cuda:{args.rank}")
    print(f"Using distributed mode with rank {args.rank}")
else:
    # Single-card mode
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:7")
    args.rank = 0
    args.world_size = 1
    print("Using single-card mode")

# Load model and tokenizers
tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|reserved_special_token_0|>')
model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
model.config.pad_token_id = tokenizer.pad_token_id
sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)

# Create reference dataset
dataset_class = REFERENCE_DATASETS[args.dataset_name]
eval_dataset = dataset_class(
    root_dir=args.root_dir,
    split=args.split,
    max_sequence_length=1021,
)

print(f'Reference dataset {args.dataset_name} loaded with {len(eval_dataset)} samples')
model.eval()
batch_size = int(args.batch_per_device)
model = model.bfloat16().to(device)

# Data collator for inference
data_collator = FragDataCollator(
    sequence_tokenizer=sequence_tokenizer,
    llm_tokenizer=tokenizer,
    mode="inference",
    max_sequence_length=1021,
    max_description_length=512,
    use_max_desc_length=True
)

# Create dataloader with appropriate sampler
if hasattr(args, 'distributed') and args.distributed:
    # Multi-card mode: use DistributedSampler
    distributed_sampler = DistributedSampler(eval_dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0,
                            sampler=distributed_sampler, collate_fn=data_collator)
else:
    # Single-card mode: use standard DataLoader
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0,
                            shuffle=False, collate_fn=data_collator)

# Clean existing results file (only on rank 0 or single-card mode)
should_clean_file = False
if hasattr(args, 'distributed') and args.distributed:
    # Multi-card mode: only rank 0 cleans the file
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        should_clean_file = True
else:
    # Single-card mode: always clean the file
    should_clean_file = True

if should_clean_file and os.path.exists(args.save_results_path):
    os.remove(args.save_results_path)

generated = list()
references = list()

print(f"Starting evaluation on {args.dataset_name} dataset...")
for inputs in tqdm(dataloader):
    # Extract reference answers
    references += tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)
    
    # Move inputs to device
    inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Generate responses
    with torch.no_grad():
        tok_ids = model.generate(
            inputs=None, 
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            protein_input_ids=inputs["protein_input_ids"],
            protein_attention_mask=inputs["protein_attention_mask"],
            protein_inputs_embeds=None,
            position_refs=inputs["position_refs"],
            num_beams=1,
            early_stopping=False,
            no_repeat_ngram_size=None,
            length_penalty=1.0,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=512,
            use_cache=True
        )
    generated += tokenizer.batch_decode(tok_ids, skip_special_tokens=True)

# Save results as CSV
data = {'generated': generated, 'reference': references, 'dataset': [args.dataset_name] * len(generated)}
df = pd.DataFrame(data)
df.to_csv(args.save_results_path, index=False, mode='a')

print(f"Results saved to {args.save_results_path}")
print(f"Generated {len(generated)} responses for {args.dataset_name} dataset")