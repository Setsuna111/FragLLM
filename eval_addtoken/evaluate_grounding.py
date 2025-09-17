import sys
sys.path.append('.')
from transformers import AutoTokenizer
from models.protein_llama import ProteinLlamaForCausalLM
from dataset.dataloader_grounding import *
from dataset.dataloader_frag import FragDataCollator
from dataset.templates import *
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from transformers.utils import logging
from tqdm import tqdm
import torch
import os
import argparse
import re
from eval.ddp import *

GROUNDING_DATASETS = {
    'ActGroundSingle': ActGroundingSingle,
    'BindIGroundSingle': BindIGroundingSingle,
    'DomGroundSingle': DomainGroundingSingle,
    'EvoGroundSingle': EvoGroundingSingle,
    'MotifGroundSingle': MotifGroundingSingle,
    'ActGroundGroup': ActGroundingGroup,
    'BindIGroundGroup': BindIGroundingGroup,
    'DomGroundGroup': DomainGroundingGroup,
    'EvoGroundGroup': EvoGroundingGroup,
    'MotifGroundGroup': MotifGroundingGroup,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Unified evaluation script for function and reference datasets')
    parser.add_argument("--model_path", default="/home/lfj/projects_dir/FragLLM/checkpoints/test_load_stage1_only_motifdesc_lora_fast_save/checkpoint-600_merge/", help="path to the trained model")
    parser.add_argument("--temperature", default=0.0, type=float, help="generation temperature")
    parser.add_argument("--root_dir", default='./data', help="root folder of the data")
    parser.add_argument("--datasets", default="MotifRefDesc", help="comma-separated list of datasets to evaluate")
    parser.add_argument("--split", default="test", help="data split to use (train, test, eval)")
    parser.add_argument("--batch_per_device", type=int, default=2, help="batch size for each device")
    parser.add_argument("--save_results_dir", default="./eval_results", help="directory to save results")
    parser.add_argument("--single_gpu", action="store_true", help="use single GPU mode instead of distributed")
    # parser.add_argument("--single_gpu", default=True, help="use single GPU mode instead of distributed")
    parser.add_argument("--gpu_id", type=int, default=7, help="GPU ID to use in single GPU mode")
    
    # Distributed training arguments
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser.parse_args()

def create_dataset(dataset_name, root_dir, split, max_sequence_length=1021):
    """Create dataset based on dataset name"""
    if dataset_name in GROUNDING_DATASETS:
        dataset_class = GROUNDING_DATASETS[dataset_name]
        return dataset_class(
            root_dir=root_dir,
            split=split,
            max_sequence_length=max_sequence_length
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available datasets: {list(GROUNDING_DATASETS.keys())}")
    

def list_nested_elements_recursive(data):
        """
        使用递归方法提取多层嵌套列表中的每个元素。
        """
        data_list = []
        # 遍历列表中的每一个元素
        for element in data:
            # 如果元素是列表，则递归调用函数并将结果累加
            if isinstance(element, list):
                data_list.extend(list_nested_elements_recursive(element))
            # 如果元素不是列表，说明它是一个最里层的元素
            else:
                data_list.append(element)
        return data_list

def replace_matches_sequentially(
    input_string,
    pattern,
    replacements,
):
    """
    Finds all matches for a regex pattern in a string and replaces each 
    match sequentially with an item from the replacements list.

    Args:
        input_string: The text to perform replacements on.
        pattern: A regex pattern (string or compiled) to find matches.
        replacements: A list of items to use as replacements. Each item
                      will be formatted into a string like '(item[0],item[1])'.

    Returns:
        The modified string with all replacements made.

    Raises:
        ValueError: If the number of matches is greater than the number of 
                    available items in the replacements list.
    """
    # Create an iterator from the list to pull values one by one
    replacements_iter = iter(replacements)

    def replacer(match):
        """Inner function called by re.sub() for each match."""
        try:
            # Get the next tuple from our iterator
            next_val = next(replacements_iter)
            # Format it into the desired string and return it
            if next_val[0] > next_val[1]:
                return f"({next_val[1]},{next_val[0]})"
            return f"({next_val[0]},{next_val[1]})"
        except StopIteration:
            # This error occurs if we run out of replacement items.
            raise ValueError("Not enough replacement items for the number of matches found.")

    # Use re.sub with the replacer function and return the result
    return re.sub(pattern, replacer, input_string)

def evaluate_dataset(dataset_name, model, tokenizer, data_collator, 
                    args, device, save_results_path):
    """Evaluate a single dataset"""
    print(f"\n=== Evaluating {dataset_name} dataset ===")
    
    # Create dataset
    eval_dataset = create_dataset(dataset_name, args.root_dir, args.split)
    print(f'Dataset {dataset_name} loaded with {len(eval_dataset)} samples')
    
    # Create dataloader
    if args.single_gpu:
        # Single GPU mode
        dataloader = DataLoader(eval_dataset, batch_size=args.batch_per_device, 
                              num_workers=0, shuffle=False, collate_fn=data_collator)
    else:
        # Multi-GPU mode
        distributed_sampler = DistributedSampler(eval_dataset, rank=args.rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(eval_dataset, batch_size=args.batch_per_device, 
                              num_workers=0, sampler=distributed_sampler, collate_fn=data_collator)
        print("DEBUG:", list(distributed_sampler))  # 0904 debug
    # Clean existing results file (only on rank 0 or single GPU mode)
    should_clean_file = args.single_gpu or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
    if should_clean_file and os.path.exists(save_results_path):
        os.remove(save_results_path)
        print(f"Removed existing results file: {save_results_path}")
    
    generated = []
    references = []
    dataset_idx_list = []
    gt_position_grds = []
    pred_position_grds = []
    ignore_tokens = [128009, 128002]
    pattern = re.compile(r'\(\s*\<frag_start\>\s*,\s*\<frag_end\>\s*\)')
    print(f"Starting evaluation on {dataset_name}...")
    for inputs in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        # Extract reference answers
        # references += tokenizer.batch_decode(inputs['answer_input_ids'])
        
        answers_gt = tokenizer.batch_decode(inputs['answer_input_ids'])
        answers_gt_replace = []
        for i, answer_gt in enumerate(answers_gt):
            position_grds = list_nested_elements_recursive(inputs['position_grds'][i])
            position_grds = list(zip(position_grds[::2], position_grds[1::2]))
            gt_position_grds.append(position_grds)
            # 提取answer_gt中的(<frag_start>,<frag_end>)或者(<frag_start>, <frag_end>)元素
            answer_gt_replace = replace_matches_sequentially(answer_gt, pattern, position_grds)
            answers_gt_replace.append(answer_gt_replace.replace("<|reserved_special_token_0|>", "").replace("<|eot_id|>", ""))
        references += answers_gt_replace
        
        # Move inputs to device
        inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        dataset_idx_list += inputs.get('dataset_idxs', [None]*inputs['input_ids'].size(0))

        # Generate responses
        # generated += tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)  # 0904 debug，代替实际生成过程
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            tok_ids, position_grds_pred = model.generate(
                inputs=None,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                protein_input_ids=inputs["protein_input_ids"],
                protein_attention_mask=inputs["protein_attention_mask"],
                protein_inputs_embeds=None,
                position_refs=inputs["position_refs"],
                grounding_inference=True,
                num_beams=1,
                early_stopping=False,
                no_repeat_ngram_size=None,
                length_penalty=1.0,
                eos_token_id=128009, 
                pad_token_id=128002,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=512,
                use_cache=True
            )
        # import pdb; pdb.set_trace()
        # generated += tokenizer.batch_decode(tok_ids)

        answers_pred = tokenizer.batch_decode(tok_ids)
        answers_pred_replace = []
        for i, answer_pred in enumerate(answers_pred):
            position_grd_pred = list(zip(position_grds_pred[i].tolist()[::2], position_grds_pred[i].tolist()[1::2]))
            pred_position_grds.append(position_grd_pred)
            answer_pred_replace = replace_matches_sequentially(answer_pred, pattern, position_grd_pred)
            answers_pred_replace.append(answer_pred_replace.replace("<|reserved_special_token_0|>", "").replace("<|eot_id|>", ""))
        generated += answers_pred_replace
        # import pdb; pdb.set_trace()
    
    # Handle multi-GPU result collection
    if args.single_gpu:
        # Single GPU mode: directly save results
        data = {
            'generated': generated,
            'reference': references,
            'dataset_idx': dataset_idx_list,
            'gt_position_grds': gt_position_grds,
            'pred_position_grds': pred_position_grds
        }
        df = pd.DataFrame(data)
        df.to_csv(save_results_path, index=False)
        total_samples = len(generated)
    else:
        # Multi-GPU mode: collect results from all GPUs
        if torch.distributed.is_initialized():
            # Save partial results with rank suffix first
            partial_save_path = save_results_path.replace('.csv', f'_rank{args.rank}.csv')
            data = {
                'generated': generated,
                'reference': references,
                'dataset_idx': dataset_idx_list,
                'gt_position_grds': gt_position_grds,
                'pred_position_grds': pred_position_grds
            }
            df = pd.DataFrame(data)
            df.to_csv(partial_save_path, index=False)
            
            # Wait for all processes to finish saving partial results
            torch.distributed.barrier()
            
            # Only rank 0 merges all results
            if torch.distributed.get_rank() == 0:
                print(f"Rank 0: Merging results from all GPUs...")
                all_data = {'generated': [], 'reference': [], 'dataset_idx': [], 'gt_position_grds': [], 'pred_position_grds': []}
                
                # Collect results from all ranks
                for rank in range(args.world_size):
                    rank_file = save_results_path.replace('.csv', f'_rank{rank}.csv')
                    if os.path.exists(rank_file):
                        rank_df = pd.read_csv(rank_file)
                        all_data['generated'].extend(rank_df['generated'].tolist())
                        all_data['reference'].extend(rank_df['reference'].tolist())
                        all_data['dataset_idx'].extend(rank_df['dataset_idx'].tolist())
                        all_data['gt_position_grds'].extend(rank_df['gt_position_grds'].tolist())
                        all_data['pred_position_grds'].extend(rank_df['pred_position_grds'].tolist())
                        # Clean up partial file
                        os.remove(rank_file)

                # Save merged results
                merged_df = pd.DataFrame(all_data)
                # drop duplicates based on dataset_idx
                merged_df = merged_df.drop_duplicates(subset=['dataset_idx'], keep='first')
                merged_df.to_csv(save_results_path, index=False)
                total_samples = len(all_data['generated'])
                print(f"Merged results from {args.world_size} GPUs: {total_samples} total samples")
            
            # All processes wait for merging to complete
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                total_samples = len(all_data['generated']) if 'all_data' in locals() else len(generated)
            else:
                total_samples = len(generated)  # Local count for return value
        else:
            # Fallback to single GPU behavior if distributed not initialized
            data = {
                'generated': generated,
                'reference': references,
                'dataset_idx': dataset_idx_list,
                'gt_position_grds': gt_position_grds,
                'pred_position_grds': pred_position_grds
            }
            df = pd.DataFrame(data)
            df.to_csv(save_results_path, index=False)
            total_samples = len(generated)
    
    if args.single_gpu or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
        print(f"Results saved to {save_results_path}")
        print(f"Generated {total_samples} responses for {dataset_name} dataset")
    
    return len(generated)  # Return local count for progress tracking

def main():
    args = parse_args()
    
    # Parse dataset list
    dataset_list = [d.strip() for d in args.datasets.split(',')]
    print(f"Will evaluate datasets: {dataset_list}")
    
    # Validate all datasets exist
    all_datasets = {**GROUNDING_DATASETS}
    for dataset_name in dataset_list:
        if dataset_name not in all_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available datasets: {list(all_datasets.keys())}")
    
    # Setup device and distributed training
    if args.single_gpu:
        # Single GPU mode
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        args.rank = 0
        args.world_size = 1
        print(f"Using single GPU mode on device: {device}")
    else:
        # Multi-GPU distributed mode
        init_distributed_mode(args)
        device = torch.device(f"cuda:{args.rank}")
        print(f"Using distributed mode with rank {args.rank}, world_size {args.world_size}")
    
    # Load model and tokenizers
    print("Loading model and tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|reserved_special_token_0|>')
    model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)
    
    model.eval()
    model = model.bfloat16().to(device)
    print("Model loaded and moved to device")
    
    # Create data collator
    data_collator = FragDataCollator(
        sequence_tokenizer=sequence_tokenizer,
        llm_tokenizer=tokenizer,
        mode="inference",
        max_sequence_length=1021,
        max_description_length=512,
        use_max_desc_length=True
    )
    
    # Create results directory
    os.makedirs(args.save_results_dir, exist_ok=True)
    
    # Evaluate each dataset
    for dataset_name in dataset_list:
        save_results_path = os.path.join(args.save_results_dir, f"{dataset_name}_results.csv")
        samples_count = evaluate_dataset(
            dataset_name, model, tokenizer, 
            data_collator, args, device, save_results_path
        )
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total datasets evaluated: {len(dataset_list)}")
    print(f"Total samples evaluated: {samples_count}")
    print(f"Results saved in directory: {args.save_results_dir}")

if __name__ == "__main__":
    main()