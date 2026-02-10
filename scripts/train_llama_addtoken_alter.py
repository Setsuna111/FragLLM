"""
Training script for Esm2LlamaInstructForCausalLM using transformers.Trainer.

This script implements training of the fragment-aware protein language model
using the HuggingFace Trainer framework with LoRA fine-tuning support.

Based on train_trainer_refer.py but adapted for Esm2LlamaInstructForCausalLM.
"""

import sys

from transformers.models import grounding_dino
sys.path.append("..")
sys.path.append(".")

import wandb
import time
import pathlib
import transformers
import random
import torch
import os
import json
import argparse
import deepspeed
import numpy as np
from torch.utils.data import random_split, DataLoader
from functools import partial
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers import Trainer, AutoTokenizer
from transformers.trainer import get_parameter_names, is_sagemaker_mp_enabled, ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import EsmModel, LlamaForCausalLM
from peft import get_peft_model, LoraConfig, PeftModel
import logging
from models.protein_llama_addtoken_lfj import *
from models.protein_llama_addtoken_djy import *
from dataset.dataloader_referring import FragRefDataset
from dataset.dataloader_frag import FragDataCollator, make_multitask_dataset, HybridFuncDataset, HybridReferringDataset, HybridGroundingDataset, HybridValidDataset
from utils import  (AverageMeter, ProgressMeter, dict_to_cuda, Summary)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
import tqdm
from transformers.trainer import _is_peft_model
import re
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
                      will be formatted into a string like 'start,end'.

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
                return f"{next_val[1]},{next_val[0]}"
            return f"{next_val[0]},{next_val[1]}"
        except StopIteration:
            # This error occurs if we run out of replacement items.
            raise ValueError("Not enough replacement items for the number of matches found.")

    # Use re.sub with the replacer function and return the result
    return re.sub(pattern, replacer, input_string)
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def parse_args(args):
    parser = argparse.ArgumentParser(description="FragLLM Model Training")
    # model arguments
    parser.add_argument("--esm_path", type=str, default="/home/dataset-local/projects/Data/HF_models/esm2_t36_3B_UR50D", help="Path to ESM model")
    parser.add_argument("--llama_path", type=str, default="/home/dataset-local/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct", help="Path to LLaMA model")
    parser.add_argument("--load_adapter_checkpoint_dir", type=str, default=None, help="Path to load adapter checkpoint")
    parser.add_argument("--load_fragment_checkpoint_dir", type=str, default=None, help="Path to load fragment checkpoint")
    parser.add_argument("--load_pro2text_checkpoint_dir", type=str, default='/home/dataset-local/projects/Data/HF_models/Prot2Text-V2-11B-Instruct-hf', help="Path to load fragment checkpoint")
    ## grounding model arguments
    parser.add_argument("--protein_sam_checkpoint_path", type=str, default='./model_grounding_segformer/checkpoints_grounding_3B/checkpoint_epoch_0.pt', help="Path to ProteinSAM checkpoint (.pt file). Required for training.")
    ## model architecture arguments
    parser.add_argument("--fix_modality_adapter", type=bool, default=False, help="Whether to fix modality adapter")
    ## fragment adapter arguments
    parser.add_argument("--perceiver_latent_size", type=int, default=1, help="Perceiver latent size")
    parser.add_argument("--num_perceiver_heads", type=int, default=8, help="Number of perceiver heads")
    parser.add_argument("--num_perceiver_layers", type=int, default=2, help="Number of perceiver layers")
    parser.add_argument("--intermediate_dim", type=int, default=2048, help="Intermediate dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Whether to freeze backbone")
    ## placeholder ids
    parser.add_argument("--sequence_placeholder_id", type=int, default=128003, help="Sequence placeholder id")
    parser.add_argument("--fragment_placeholder_id", type=int, default=128005, help="Fragment placeholder id")
    ## ce loss weight
    parser.add_argument("--ce_loss_weight", type=float, default=1.0, help="ce loss weight")
    ## position loss weight
    parser.add_argument("--position_loss_weight", type=float, default=0.1, help="position loss weight")
    ## fragment adapter type
    parser.add_argument("--frag_adapter_type", type=str, default="multilevel", help="Fragment adapter type: 'qformer' or 'multilevel'")
    ## pos decoder type
    parser.add_argument("--pos_decoder_type", type=str, default="Simple", help="Pos decoder type: 'ProteinSAM', 'Simple'")

    # data arguments
    parser.add_argument("--root_dir", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--use_detailed_template", action="store_true", default=False, help="Whether to use detailed template")
    parser.add_argument("--dataset_func_config", type=str, default="ProFunction", help="Dataset config for training")
    parser.add_argument("--dataset_grounding_config", type=str, default="ActGroundSingle", help="Dataset config for training")
    parser.add_argument("--dataset_referring_config", type=str, default="ActRefClass", help="Dataset config for training")
    parser.add_argument("--dataset_valid_config", type=str, default="ActGroundSingle", help="Dataset config for evaluation")
    parser.add_argument("--max_sequence_length", type=int, default=1021, help="Maximum sequence length")
    parser.add_argument("--filter_sequence", action="store_true", default=False, help="Whether to filter sequence")
    parser.add_argument("--dataset_size", type=int, default=-1, help="Dataset size for function dataset. -1 means use full dataset, otherwise truncate to this size")
    # special tokens
    parser.add_argument("--sequence_placeholder", type=str, default="<|reserved_special_token_1|>", help="Sequence placeholder")
    parser.add_argument("--fragment_placeholder", type=str, default="<|reserved_special_token_2|>", help="Fragment placeholder")
    ## For ProteinSAM
    parser.add_argument("--position_placeholder", type=str, default="<frag_position>", help="Position placeholder for fragment grounding")
    ## For SimpleDecoder
    parser.add_argument("--pos_start_placeholder", type=str, default="<frag_start>", help="Position start placeholder")
    parser.add_argument("--pos_end_placeholder", type=str, default="<frag_end>", help="Position end placeholder")
    parser.add_argument("--phrase_start_placeholder", type=str, default="<p>", help="Phrase start placeholder")
    parser.add_argument("--phrase_end_placeholder", type=str, default="</p>", help="Phrase end placeholder")
    ## system message
    parser.add_argument("--system_message", type=str, default="You are a scientific assistant specializing in protein sequence analysis. Based on protein sequence embeddings and other related information, please answer the relevant questions using professional language.", help="System message")

    # training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/fragment_training_debug", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--evaluation_strategy", type=str, default=None, help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False, help="Remove unused columns")
    parser.add_argument("--bf16", action="store_true", default=True, help="Whether to use bf16")
    parser.add_argument("--tf32", action="store_true", default=True, help="Whether to use tf32")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Whether to use gradient checkpointing")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer")
    parser.add_argument("--tune_fragment_adapter", action="store_true", default=False, help="Whether to tune fragment adapter")
    parser.add_argument("--tune_adapter", action="store_true", default=False, help="Whether to tune adapter")
    parser.add_argument("--freeze_adapter", action="store_true", default=False, help="Whether to freeze adapter")
    parser.add_argument("--freeze_fragment_adapter", action="store_true", default=False, help="Whether to freeze fragment adapter")
    parser.add_argument("--bits", type=int, default=16, help="How many bits to use.")
    parser.add_argument("--lora_enable", action="store_true", default=False, help="Whether to enable lora")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_bias", type=str, default="none", help="LoRA bias")
    parser.add_argument("--lora_target_modules", type=str, default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj", help="LoRA target modules")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)

    parser.add_argument("--grounding_validation", action="store_true", default=False)
    parser.add_argument("--run_name", type=str, default="alter_test")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument("--steps_per_epoch", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Attention implementation(flash_attention_2)")
    return parser.parse_args(args)


def initialize_wandb(args):
    if args.report_to == "wandb":
        wandb.init(project="FragLLM_2601_alter", name=args.run_name)
        wandb.config.update(args)
        return None
    elif args.report_to == "tensorboard" and args.local_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.output_dir)
    else:
        return None
    return writer

def initialize_model(args):
    if args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    if args.pos_decoder_type == "ProteinSAM":
        model = ProteinLlamaForCausalLM.from_pretrained(
            args.llama_path, 
            torch_dtype=torch_dtype,
            cache_dir=args.cache_dir, 
            attn_implementation=args.attn_implementation)
    else:
        model = ProteinLlamaForCausalLM_Simple.from_pretrained(args.llama_path, torch_dtype=torch_dtype, cache_dir=args.cache_dir, attn_implementation=args.attn_implementation)
    if  args.load_pro2text_checkpoint_dir is not None:
        pro2text_model = AutoModelForCausalLM.from_pretrained(args.load_pro2text_checkpoint_dir, trust_remote_code=True)
        pro2text_param_dict = pro2text_model.state_dict()
        pro2text_llama_weights = {k.split('llama_decoder.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.' in k) and ('embed_tokens' not in k) and ('lm_head' not in k)}
        model.load_state_dict(pro2text_llama_weights, strict=False)
        rank0_print("Loaded Prot2Text llama_decoder weights from ", args.load_pro2text_checkpoint_dir)
    else:
        pro2text_model = None
    # add special tokens
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args.llama_path,
        pad_token='<|reserved_special_token_0|>'
    )
    if  args.pos_decoder_type == "ProteinSAM":
        llama_tokenizer.add_tokens([
            args.position_placeholder,
            args.phrase_start_placeholder,
            args.phrase_end_placeholder
        ], special_tokens=True)
    else:
        llama_tokenizer.add_tokens([
            args.pos_start_placeholder,
            args.pos_end_placeholder,
            args.phrase_start_placeholder,
            args.phrase_end_placeholder
        ], special_tokens=True)
    return model, llama_tokenizer, esm_tokenizer, torch_dtype, pro2text_model

def prepare_model_for_training(model, llama_tokenizer, torch_dtype, pro2text_model, args):
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # Set requires_grad based on LoRA training
    lora_enable = args.lora_enable
    if  not lora_enable:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().adapter.parameters():
            p.requires_grad_(True)
        for p in model.get_model().fragment_adapter.parameters():
            p.requires_grad_(True)
    if  lora_enable:
        target_modules = args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            init_lora_weights=True,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        if args.bits == 16:
            if args.bf16:
                model.to(torch.bfloat16)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    # initial FragLLM
    model.config.use_cache = False
    model.resize_token_embeddings(len(llama_tokenizer))
    model.lm_head.weight.requires_grad_(True)
    model.get_model().embed_tokens.weight.requires_grad_(True)
    if args.esm_path is not None:
        model.config.max_sequence_length = args.max_sequence_length
        model.get_model().initialize_modules(model_args=args)
        
        # 覆盖pro2text权重（2/2）
        if args.load_pro2text_checkpoint_dir is not None and pro2text_model is not None:
            pro2text_param_dict = pro2text_model.state_dict()
            pro2text_adapter_weights = {k.split('adapter.')[1]: v for k, v in pro2text_param_dict.items() if 'adapter.' in k}
            model.get_model().adapter.load_state_dict(pro2text_adapter_weights, strict=False)
            rank0_print("Loaded Prot2Text adapter weights from ", args.load_pro2text_checkpoint_dir)
            del pro2text_model

        # add special tokens ids
        model.config.sequence_placeholder_id = args.sequence_placeholder_id
        model.config.fragment_placeholder_id = args.fragment_placeholder_id
        if args.pos_decoder_type == "ProteinSAM":
            model.config.position_placeholder_id = llama_tokenizer.convert_tokens_to_ids(args.position_placeholder)
        else:
            model.config.pos_start_placeholder_id = llama_tokenizer.convert_tokens_to_ids(args.pos_start_placeholder)
            model.config.pos_end_placeholder_id = llama_tokenizer.convert_tokens_to_ids(args.pos_end_placeholder)
        model.config.phrase_start_placeholder_id = llama_tokenizer.convert_tokens_to_ids(args.phrase_start_placeholder)
        model.config.phrase_end_placeholder_id = llama_tokenizer.convert_tokens_to_ids(args.phrase_end_placeholder)
        
        # add ce loss weight and position loss weight
        model.config.ce_loss_weight = args.ce_loss_weight
        model.config.position_loss_weight = args.position_loss_weight

        # add proteinsam path
        model.config.protein_sam_checkpoint_path = args.protein_sam_checkpoint_path

        rank0_print("model.config.sequence_placeholder_id: ", model.config.sequence_placeholder_id)
        rank0_print("model.config.fragment_placeholder_id: ", model.config.fragment_placeholder_id)
        if args.pos_decoder_type == "ProteinSAM":
            rank0_print("model.config.position_placeholder_id: ", model.config.position_placeholder_id)
        else:
            rank0_print("model.config.pos_start_placeholder_id: ", model.config.pos_start_placeholder_id)
            rank0_print("model.config.pos_end_placeholder_id: ", model.config.pos_end_placeholder_id)
        rank0_print("model.config.phrase_start_placeholder_id: ", model.config.phrase_start_placeholder_id)
        rank0_print("model.config.phrase_end_placeholder_id: ", model.config.phrase_end_placeholder_id)
        
        esm_encoder = model.get_model().get_esm_encoder()
        esm_encoder.to(dtype=torch_dtype, device=args.local_rank)
        model.config.tune_fragment_adapter = args.tune_fragment_adapter
        model.config.freeze_adapter = args.freeze_adapter
        model.config.freeze_fragment_adapter = args.freeze_fragment_adapter
        model.config.tune_adapter = args.tune_adapter
    rank0_print("ModelArchitecture:")
    rank0_print(model)
    if args.tune_adapter: # DONE: 只优化adapter
        model.requires_grad_(False)
        for p in model.get_model().adapter.parameters():
            p.requires_grad_(True)
    if args.freeze_adapter:
        for p in model.get_model().adapter.parameters():
            p.requires_grad_(False)
    if args.tune_fragment_adapter: # DONE: 只优化fragment_adapter
        model.requires_grad_(False)
        for p in model.get_model().fragment_adapter.parameters():
            p.requires_grad_(True)
    if args.freeze_fragment_adapter:
        for p in model.get_model().fragment_adapter.parameters():
            p.requires_grad_(False)
    rank0_print("ModelTrainable:")
    rank0_print([n for n, p in model.named_parameters() if p.requires_grad])
    rank0_print(model.device)


def initialize_datasets(args):
    # Common dataset arguments
    common_ds_args = {
        "root_dir": args.root_dir,
        "split": "train",
        "max_sequence_length": args.max_sequence_length,
        "perceiver_latent_size": args.perceiver_latent_size,
        "use_detailed_template": args.use_detailed_template,
        "pos_decoder_type": args.pos_decoder_type,
        }
    func_train_dataset = HybridFuncDataset(**common_ds_args, data_func=args.dataset_func_config) if args.dataset_func_config is not None else None
    referring_train_dataset = HybridReferringDataset(**common_ds_args, data_referring=args.dataset_referring_config)  if args.dataset_referring_config is not None else None
    grounding_train_dataset = HybridGroundingDataset(**common_ds_args, data_grounding=args.dataset_grounding_config)  if args.dataset_grounding_config is not None else None

    valid_dataset = HybridValidDataset(**common_ds_args, data_valid=args.dataset_valid_config) if args.dataset_valid_config is not None else None

    return func_train_dataset, referring_train_dataset, grounding_train_dataset, valid_dataset

def setup_data_loaders(args, func_train_dataset, referring_train_dataset, grounding_train_dataset, valid_dataset, llama_tokenizer, esm_tokenizer):
    train_sampler_args = {"shuffle": True, "drop_last": False}
    eval_sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.per_device_train_batch_size, "num_workers": args.dataloader_num_workers, "pin_memory": False}
    val_loader_args = {"batch_size": args.per_device_eval_batch_size, "shuffle": False, "num_workers": args.dataloader_num_workers, "pin_memory": False}
    data_collator_train = FragDataCollator(
        sequence_tokenizer=esm_tokenizer,
        llm_tokenizer=llama_tokenizer,
        mode="train",
        max_sequence_length=args.max_sequence_length,
    )
    # data_collator_valid = FragDataCollator(
    #     sequence_tokenizer=esm_tokenizer,
    #     llm_tokenizer=llama_tokenizer,
    #     mode="inference",
    #     max_sequence_length=args.max_sequence_length,
    # )
    # Traning dataloaders
    func_train_dataloader = torch.utils.data.DataLoader(
        func_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            func_train_dataset, **train_sampler_args
        ), collate_fn=data_collator_train, **train_loader_args
    , ) if func_train_dataset is not None else None

    referring_train_dataloader = torch.utils.data.DataLoader(
        referring_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            referring_train_dataset, **train_sampler_args
        ), collate_fn=data_collator_train, **train_loader_args
    , ) if referring_train_dataset is not None else None

    grounding_train_dataloader = torch.utils.data.DataLoader(
            grounding_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
                grounding_train_dataset, **train_sampler_args
            ), collate_fn=data_collator_train, **train_loader_args
        , ) if grounding_train_dataset is not None else None

    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
                valid_dataset, **eval_sampler_args
            ), collate_fn=data_collator_train, **val_loader_args
        , ) if valid_dataset is not None else None
    return func_train_dataloader, referring_train_dataloader, grounding_train_dataloader, valid_dataloader
    

def initialize_deepspeed(model, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
                 "gradient_accumulation_steps": args. gradient_accumulation_steps,
                #  "optimizer": {
                #     "type": "AdamW", 
                #     "params": 
                #     {"lr": args.learning_rate, 
                #     "weight_decay": args.weight_decay, 
                #     "betas": (args.adam_beta1, args.adam_beta2)}
                    # },
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.num_train_epochs * args.steps_per_epoch, "warmup_min_lr": 0,"warmup_max_lr": args.learning_rate, "warmup_num_steps": 120, "warmup_type": "linear"}},
                 "fp16": {
                    "enabled": False,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1}, 
                "bf16": {"enabled": True},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": False, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        eps=args.adam_epsilon,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),optimizer=optimizer, config=ds_config
    )

    return model_engine, optimizer, scheduler

def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.output_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    global local_rank
    local_rank = args.local_rank
    model, llama_tokenizer, esm_tokenizer, torch_dtype, pro2text_model = initialize_model(args)
    prepare_model_for_training(model, llama_tokenizer, torch_dtype, pro2text_model, args)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, args)
    resume_training_from_checkpoint(model_engine, args)
    rank0_print("-------------------------init model done-------------------------")
    rank0_print(model_engine)
    
    func_train_dataset, referring_train_dataset, grounding_train_dataset, valid_dataset = initialize_datasets(args)
    func_train_dataloader, referring_train_dataloader, grounding_train_dataloader, valid_dataloader = setup_data_loaders(args, func_train_dataset, referring_train_dataset, grounding_train_dataset, valid_dataset, llama_tokenizer, esm_tokenizer)


    # Determine active datasets
    # active_dataloaders = []
    use_func_data = False
    use_referring_data = False
    use_grounding_data = False
    if args.dataset_func_config is not None:
        use_func_data = True
    if args.dataset_referring_config is not None:
        use_referring_data = True
    if args.dataset_grounding_config is not None:
        use_grounding_data = True
    active_dataloaders = {
        'func': func_train_dataloader,
        'referring': referring_train_dataloader,
        'grounding': grounding_train_dataloader,
    }
    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (func, referring, or grounding) must be active."

    dataset_iters = {'func': iter(func_train_dataloader) if use_func_data else None,
                     'referring': iter(referring_train_dataloader) if use_referring_data else None,
                     'grounding': iter(grounding_train_dataloader) if use_grounding_data else None,}

    
    writer = initialize_wandb(args)

    save_args_path = os.path.join(args.output_dir, "args.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_args_path, "w") as f:
        json.dump({"args": vars(args), "model_configs": model.config.to_dict()}, f, indent=4)
    if args.eval_only:
        cur_val_loss = validate_model_performance(valid_dataloader, model_engine, 0, writer, args)[0]
        exit()
    epoch_seeds = [random.randint(0, 100000) for _ in range(args.num_train_epochs)]
    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.num_train_epochs):
        random.seed(epoch_seeds[epoch])

        dataset_iters = train(
            active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, torch_dtype, args
        )

        if args.grounding_validation:
            giou, ciou = validate_model_performance(valid_dataloader, model_engine, epoch, writer, llama_tokenizer, args)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.local_rank == 0:  # Log the progress
                rank0_print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(valid_dataloader, model_engine, epoch, writer, llama_tokenizer, args)
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if args.local_rank == 0:  # Log the progress
                rank0_print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)



def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.output_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)

def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, torch_dtype, args):
    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    # def log_progress():
    #     """Log training progress."""
    #     if global_step % args.logging_steps == 0:
    #         if args.distributed:
    #             for tracker in trackers.values():
    #                 tracker.all_reduce()

    #         if args.local_rank == 0:
    #             progress.display(global_step + 1)
    #             if writer is not None:
    #                 for key, tracker in trackers.items():
    #                     writer.add_scalar(f"train/{key}", tracker.avg, global_step)
    #                 writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
    #                 writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
    #                 for key, tracker in trackers.items():
    #                     writer.add_scalar(f"train/{key}_all_epoch", tracker.avg, global_step+epoch*args.steps_per_epoch)
    #             else:
    #                 for key, tracker in trackers.items():
    #                     wandb.log({f"train/{key}": tracker.avg},step=global_step)
    #                 wandb.log({"metrics/total_secs_per_batch": batch_time.avg,
    #                 "metrics/data_secs_per_batch": data_time.avg
    #                 }, step=global_step)
    #                 for key, tracker in trackers.items():
    #                     wandb.log({f"train/{key}_all_epoch": tracker.avg}, step=global_step+epoch*args.steps_per_epoch)
    #         for tracker in trackers.values():
    #             tracker.reset()

    def log_progress():
        """Log training progress."""
        if global_step % args.logging_steps == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                # progress.display(global_step + 1)
                if writer is not None:
                    writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step+epoch*args.steps_per_epoch)
                    writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step+epoch*args.steps_per_epoch)
                else:
                    wandb.log({"metrics/total_secs_per_batch": batch_time.avg,
                    "metrics/data_secs_per_batch": data_time.avg
                    }, step=global_step)
            for tracker in trackers.values():
                tracker.reset()

    def log_progress_task(task_type):
        """Log training progress."""
        if global_step % args.logging_steps == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                if writer is not None:
                    # for key, tracker in trackers.items():
                    if task_type == "grounding":
                        writer.add_scalar("train/loss_grounding", trackers["loss_grounding"].avg, global_step+epoch*args.steps_per_epoch)
                        writer.add_scalar("train/ce_loss", trackers["ce_loss"].avg, global_step+epoch*args.steps_per_epoch)
                        writer.add_scalar("train/pos_loss", trackers["pos_loss"].avg, global_step+epoch*args.steps_per_epoch)
                    elif task_type == "referring":
                        writer.add_scalar("train/loss_referring", trackers["loss_referring"].avg, global_step+epoch*args.steps_per_epoch)
                    else:
                        writer.add_scalar("train/loss_func", trackers["loss_func"].avg, global_step)
                else:
                    if task_type == "grounding":
                        wandb.log({"train/loss_grounding": trackers["loss_grounding"].avg},step=global_step+epoch*args.steps_per_epoch)
                        wandb.log({"train/ce_loss": trackers["ce_loss"].avg},step=global_step+epoch*args.steps_per_epoch)
                        wandb.log({"train/pos_loss": trackers["pos_loss"].avg},step=global_step+epoch*args.steps_per_epoch)
                    elif task_type == "referring":
                        wandb.log({"train/loss_referring": trackers["loss_referring"].avg},step=global_step+epoch*args.steps_per_epoch)
                    else:
                        wandb.log({"train/loss_func": trackers["loss_func"].avg},step=global_step+epoch*args.steps_per_epoch)
            for tracker in trackers.values():
                tracker.reset()
    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss_func": AverageMeter("LossFunc", ":.4f"),
                "loss_referring": AverageMeter("LossRef", ":.4f"),
                "loss_grounding": AverageMeter("LossGround", ":.4f"),
                "ce_loss": AverageMeter("CELoss", ":.4f"),
                "pos_loss": AverageMeter("PositionLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        if epoch <= 2: # warmup for segm
            for _ in range(args.gradient_accumulation_steps):
                # Select data loader based on step choice
                dataset_type = "grounding"
                data_loader = active_datasets[dataset_type]
                data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                dataset_iters[dataset_type] = new_iter

                data_time.update(time.time() - end)
                # Prepare data and convert relevant tensors to bfloat16
                data_batch = dict_to_cuda(data_batch)
                # for key in ["protein_input_ids", "input_ids"]:
                #     if data_batch[key] is not None:
                #         data_batch[key] = data_batch[key].to(dtype=torch_dtype)

                output_dict = model(**data_batch)

                # # Update training metrics
                # for key, tracker in trackers.items():
                #     if key in output_dict:
                #         tracker.update(output_dict[key].item(), data_batch["input_ids"].size(0))
                trackers["loss_grounding"].update(output_dict["loss"].item(), data_batch["input_ids"].size(0))
                trackers["ce_loss"].update(output_dict["ce_loss"].item(), data_batch["input_ids"].size(0))
                trackers["pos_loss"].update(output_dict["position_loss"].item(), data_batch["input_ids"].size(0))
                # import pdb;pdb.set_trace()
                model.backward(output_dict["loss"])
                model.step()
            log_progress_task(task_type=dataset_type)
                # import pdb;pdb.set_trace()
        else:
            if global_step % 2 == 0:
                for _ in range(args.gradient_accumulation_steps):
                    # Select data loader based on step choice
                    dataset_type = "referring"
                    # import pdb;pdb.set_trace()
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    # for key in ["protein_input_ids", "input_ids"]:
                    #     if data_batch[key] is not None:
                    #         data_batch[key] = data_batch[key].to(dtype=torch_dtype)

                    output_dict = model(**data_batch)

                    # Update training metrics
                    # for key, tracker in trackers.items():
                    #     if key in output_dict:
                    #         tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))
                    trackers["loss_referring"].update(output_dict["loss"].item(), data_batch["input_ids"].size(0))
                    # import pdb;pdb.set_trace()
                    model.backward(output_dict["loss"])
                    model.step()
                log_progress_task(task_type=dataset_type)
            if global_step % 4 == 0:
                for _ in range(args.gradient_accumulation_steps):
                    # import pdb;pdb.set_trace()
                    # Select data loader based on step choice
                    dataset_type = "func"
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    # for key in ["protein_input_ids", "input_ids"]:
                    #     if data_batch[key] is not None:
                    #         data_batch[key] = data_batch[key].to(dtype=torch_dtype)

                    output_dict = model(**data_batch)

                    # Update training metrics
                    # for key, tracker in trackers.items():
                    #     if key in output_dict:
                    #         tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))
                    trackers["loss_func"].update(output_dict["loss"].item(), data_batch["input_ids"].size(0))
                    model.backward(output_dict["loss"])
                    model.step()
                log_progress_task(task_type=dataset_type)
            if global_step % 1 == 0:
                for _ in range(args.gradient_accumulation_steps):
                    # import pdb;pdb.set_trace()
                    # Select data loader based on step choice
                    dataset_type = "grounding"
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    # for key in ["protein_input_ids", "input_ids"]:
                    #     if data_batch[key] is not None:
                    #         data_batch[key] = data_batch[key].to(dtype=torch_dtype)

                    output_dict = model(**data_batch)

                    trackers["loss_grounding"].update(output_dict["loss"].item(), data_batch["input_ids"].size(0))
                    trackers["ce_loss"].update(output_dict["ce_loss"].item(), data_batch["input_ids"].size(0))
                    trackers["pos_loss"].update(output_dict["position_loss"].item(), data_batch["input_ids"].size(0))

                    model.backward(output_dict["loss"])
                    model.step()
                log_progress_task(task_type=dataset_type)

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()
        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                if writer is not None:
                    writer.add_scalar("train/lr", curr_lr[0], global_step)
                    writer.add_scalar("train/lr_all_epoch", curr_lr[0], global_step+epoch*args.steps_per_epoch) # 记录时间
                else:
                    wandb.log(
                        {"train/lr":curr_lr[0],
                        "train/lr_all_epoch": curr_lr[0]},
                        step=global_step)
    return dataset_iters


def compute_iou_single(pos_pre, pos_ref):
    inter_left = max(pos_pre[0], pos_ref[0])
    inter_right = min(pos_pre[1], pos_ref[1])
    intersection = max(0, inter_right - inter_left + 1)
    union_left = min(pos_pre[0], pos_ref[0])
    union_right = max(pos_pre[1], pos_ref[1])
    union = union_right - union_left + 1
    if union == 0:
        iou = 0
    else:
        iou = intersection / union
    return intersection, union, iou

def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, llama_tokenizer, args):
    if args.grounding_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {
            "intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
            "union": AverageMeter("Union", ":.4f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            torch.cuda.empty_cache()
            with torch.no_grad():
                position_grds_pred, position_grds = training_model(**data_batch, inference=True)
            # import pdb;pdb.set_trace()
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for batch_position_pred,  batch_position_grd in zip(position_grds_pred, position_grds):
                num_positions = 0
                if batch_position_grd is None:
                    intersection += 0
                    union += 0
                    accuracy_iou += 0
                    num_positions = 1
                else:
                    position_pred = batch_position_pred.argmax(dim=-1)
                    position_pred[1::2] += 1 # [num_pos]
                    position_labels = []
                    for position_grd in batch_position_grd: # group层
                        for position in position_grd: # position层
                            position_labels.append(position[0])
                            position_labels.append(position[1]) # 
                            num_positions += 2
                    for pos_i in range(0, num_positions, 2):
                        position_pred_i = [position_pred[pos_i], position_pred[pos_i+1]]
                        position_label_i = [position_labels[pos_i], position_labels[pos_i+1]]
                        intersection_i, union_i, accuracy_iou_i = compute_iou_single(position_pred_i, position_label_i)
                        intersection += intersection_i
                        union += union_i
                        accuracy_iou += accuracy_iou_i
                    
            accuracy_iou = accuracy_iou.cpu().numpy() / num_positions
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=num_positions)
        for meter in trackers.values():
            meter.all_reduce()
        class_iou = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        global_iou = trackers["gIoU"].avg

        if args.local_rank == 0:
            rank0_print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
                tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            else:
                wandb.log(
                    {"val/giou": global_iou,
                    "val/ciou": class_iou
                    },
                    step=current_epoch
                )
        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), 
                    "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "position_loss": AverageMeter("PositionLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            # for key in ["protein_input_ids", "input_ids"]:
            #     if data_batch[key] is not None:
            #         data_batch[key] = data_batch[key].to(dtype=torch_dtype)
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            # import pdb;pdb.set_trace()
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["input_ids"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_loss = trackers["loss"].avg
        avg_ce_loss = trackers["ce_loss"].avg
        avg_pos_loss = trackers["position_loss"].avg

        # Tensorboard logging for primary process
        if args.local_rank == 0:
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("val/loss", avg_loss, current_epoch)
                tensorboard_writer.add_scalar("val/ce_loss", avg_ce_loss, current_epoch)
                tensorboard_writer.add_scalar("val/pos_loss", avg_pos_loss, current_epoch)
            else:
                wandb.log({
                    "val/loss": avg_loss,
                    "val/ce_loss": avg_ce_loss,
                    "val/pos_loss": avg_pos_loss
                }, step=current_epoch)

        return avg_loss

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)