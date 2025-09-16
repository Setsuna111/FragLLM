import sys
sys.path.append('.')
from transformers import AutoTokenizer
from models.protein_llama_addtoken import ProteinLlamaForCausalLM
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
    parser.add_argument("--model_path", default="/home/lfj/projects_dir/FragLLM/checkpoints/grounding_lora_0916test_merge_addtoken/", help="path to the trained model")
    parser.add_argument("--temperature", default=0.0, type=float, help="generation temperature")
    parser.add_argument("--root_dir", default='./data', help="root folder of the data")
    parser.add_argument("--datasets", default="ActGroundSingle", help="comma-separated list of datasets to evaluate")
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
    ignore_tokens = [128009, 128002]
    print(f"Starting evaluation on {dataset_name}...")
    for inputs in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        # Extract reference answers and add fragment position information
        batch_references = tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)
        
        # Process reference texts to add position information from position_grds
        for i, ref_text in enumerate(batch_references):
            if "position_grds" in inputs and inputs["position_grds"] is not None and i < len(inputs["position_grds"]):
                position_grd = inputs["position_grds"][i]
                if position_grd is not None and len(position_grd) > 0:
                    # Process each group of positions in position_grd
                    modified_text = ref_text
                    
                    # Replace position placeholders with actual positions
                    # The text should contain patterns like "name:." where we need to insert positions
                    for group_positions in position_grd:
                        if group_positions and len(group_positions) > 0:
                            # Format position pairs as "start-end"
                            position_pairs = []
                            for start, end in group_positions:
                                position_pairs.append(f"{start}-{end}")
                            position_str = ", ".join(position_pairs)
                            
                            # Find and replace position placeholder patterns
                            # Look for ":." or ":," patterns and insert position before the punctuation
                            if ":." in modified_text:
                                modified_text = modified_text.replace(":.", f":{position_str}.", 1)
                            elif ":," in modified_text:
                                modified_text = modified_text.replace(":,", f":{position_str},", 1)
                            elif modified_text.endswith(":"):
                                modified_text = modified_text + position_str + "."
                    
                    references.append(modified_text)
                else:
                    references.append(ref_text)
            else:
                references.append(ref_text)
        
        # Remove the original line that was adding references
        # references += tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)
        
        # Move inputs to device
        inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        dataset_idx_list += inputs.get('dataset_idxs', [None]*inputs['input_ids'].size(0))

        # Generate responses
        # generated += tokenizer.batch_decode(inputs['answer_input_ids'], skip_special_tokens=True)  # 0904 debug，代替实际生成过程

        with torch.no_grad():
            # Generate text with grounding inference enabled
            generation_result = model.generate(
                inputs=None,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                protein_input_ids=inputs["protein_input_ids"],
                protein_attention_mask=inputs["protein_attention_mask"],
                protein_inputs_embeds=None,
                position_refs=inputs["position_refs"],
                grounding_inference=True,  # Enable grounding inference
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
            
            # Extract text and position predictions
            if isinstance(generation_result, tuple) and len(generation_result) == 2:
                # grounding_inference=True now returns (generate_output_ids, position_grds_batch)
                tok_ids, position_grds_batch = generation_result
                
                # Decode the generated text
                batch_generated_text = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
                
                # Process position predictions and integrate with text
                for i, text in enumerate(batch_generated_text):
                    if position_grds_batch is not None and i < len(position_grds_batch):
                        positions = position_grds_batch[i]
                        if positions is not None:
                            # Extract start and end positions
                            start_positions = positions.get("start_positions", [])
                            end_positions = positions.get("end_positions", [])
                            
                            if len(start_positions) > 0 and len(end_positions) > 0:
                                # Format position pairs as "start-end"
                                position_pairs = []
                                for start, end in zip(start_positions, end_positions):
                                    position_pairs.append(f"{start.item()}-{end.item()}")
                                position_str = ", ".join(position_pairs)
                                
                                # Replace the placeholder with actual position
                                if ":" in text and text.endswith(":"):
                                    # Format: "The position of X is Y:." -> "The position of X is Y:123-456."
                                    text = text[:-1] + position_str + "."
                                elif "is located at" in text and text.endswith(":."):
                                    # Format: "X is located at Y:." -> "X is located at Y:123-456."
                                    text = text[:-2] + ":" + position_str + "."
                    generated.append(text)
            else:
                # Fallback: treat as regular generation output
                tok_ids = generation_result
                generated += tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
        # import pdb; pdb.set_trace()
    
    # Handle multi-GPU result collection
    if args.single_gpu:
        # Single GPU mode: directly save results
        data = {
            'generated': generated,
            'reference': references,
            'dataset_idx': dataset_idx_list
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
                'dataset_idx': dataset_idx_list
            }
            df = pd.DataFrame(data)
            df.to_csv(partial_save_path, index=False)
            
            # Wait for all processes to finish saving partial results
            torch.distributed.barrier()
            
            # Only rank 0 merges all results
            if torch.distributed.get_rank() == 0:
                print(f"Rank 0: Merging results from all GPUs...")
                all_data = {'generated': [], 'reference': [], 'dataset_idx': []}
                
                # Collect results from all ranks
                for rank in range(args.world_size):
                    rank_file = save_results_path.replace('.csv', f'_rank{rank}.csv')
                    if os.path.exists(rank_file):
                        rank_df = pd.read_csv(rank_file)
                        all_data['generated'].extend(rank_df['generated'].tolist())
                        all_data['reference'].extend(rank_df['reference'].tolist())
                        all_data['dataset_idx'].extend(rank_df['dataset_idx'].tolist())
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
                'dataset_idx': dataset_idx_list
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
    
    # Initialize proteinSAM and other modules if needed
    # 这些args都不是proteinSAM用到的，也不会被里面的任何模块加载，只是传进去防止报错，从而偷懒利用initialize_modules函数初始化
    class MockModelArgs:
        def __init__(self):
            self.esm_path = model.config.esm_path
            self.perceiver_latent_size = getattr(model.config, 'perceiver_latent_size', 1)
            self.num_perceiver_heads = getattr(model.config, 'num_perceiver_heads', 8)
            self.num_perceiver_layers = getattr(model.config, 'num_perceiver_layers', 2)
            self.intermediate_dim = getattr(model.config, 'intermediate_dim', 2048)
            self.dropout_rate = getattr(model.config, 'dropout_rate', 0.3)
            self.freeze_backbone = getattr(model.config, 'freeze_backbone', False)
            self.load_adapter_checkpoint_dir = None
            self.load_fragment_checkpoint_dir = None
        
    mock_model_args = MockModelArgs()
    model.get_model().initialize_modules(model_args=mock_model_args, fsdp=None)
    print("ProteinSAM and modules initialized successfully")
    
    # Set position placeholder ID for grounding inference
    model.config.position_placeholder_id = 128256  # Use the addtoken position placeholder ID
    print(f"Position placeholder ID set to: {model.config.position_placeholder_id}")
    
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