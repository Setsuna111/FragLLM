"""
Training script for Esm2LlamaInstructForCausalLM using transformers.Trainer.

This script implements training of the fragment-aware protein language model
using the HuggingFace Trainer framework with LoRA fine-tuning support.

Based on train_trainer_refer.py but adapted for Esm2LlamaInstructForCausalLM.
"""

import pathlib
import transformers
import random
import torch
import os
import json

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
from models import (
    ModalityAdapter, 
    ModalityAdapterConfig, 
    Esm2LlamaInstructForCausalLM
)
from dataset.dataloader_refferring import FragRefDataset
from dataset.dataloader_frag import FragDataCollator, make_multitask_dataset

@dataclass
class FragModelArguments:
    """Model arguments for fragment training."""
    esm_path: Optional[str] = field(default="", metadata={"help": "Path to ESM model"})
    llama_path: Optional[str] = field(default="", metadata={"help": "Path to LLaMA model"})
    load_model_checkpoint_path: Optional[str] = field(default="", metadata={"help": "Path to load model checkpoint"})
    load_adapter_checkpoint_dir: Optional[str] = field(default="", metadata={"help": "Path to load adapter checkpoint"})
    
    # Model architecture arguments
    fix_modality_adapter: Optional[bool] = field(default=False, metadata={"help": "Whether to fix modality adapter"})
    
    # Fragment adapter arguments
    perceiver_latent_size: Optional[int] = field(default=1, metadata={"help": "Perceiver latent size"})
    num_perceiver_heads: Optional[int] = field(default=8, metadata={"help": "Number of perceiver heads"})
    num_perceiver_layers: Optional[int] = field(default=2, metadata={"help": "Number of perceiver layers"})
    

@dataclass
class FragDataArguments:
    """Data arguments for fragment training."""
    root_dir: Optional[str] = field(default="./data", metadata={"help": "Root directory for datasets"})
    dataset_train_config: Optional[str] = field(default="ProFunction||ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass", metadata={"help": "Dataset config for training"})
    sample_rate_train: Optional[str] = field(default="1,1,1,1,1,1", metadata={"help": "Sample rate for training"})
    dataset_valid_config: Optional[str] = field(default="ProFunction", metadata={"help": "Dataset config for evaluation"})
    sample_rate_valid: Optional[str] = field(default="1", metadata={"help": "Sample rate for evaluation"})
    max_sequence_length: Optional[int] = field(default=1021, metadata={"help": "Maximum sequence length"})
    filter_sequence: Optional[bool] = field(default=False, metadata={"help": "Whether to filter sequence"})
    sequence_placeholder: Optional[str] = field(default="<|reserved_special_token_1|>", metadata={"help": "Sequence placeholder"})
    fragment_placeholder: Optional[str] = field(default="<|reserved_special_token_2|>", metadata={"help": "Fragment placeholder"})
    pos_start_placeholder: Optional[str] = field(default="<|reserved_special_token_3|>", metadata={"help": "Position start placeholder"})
    pos_end_placeholder: Optional[str] = field(default="<|reserved_special_token_4|>", metadata={"help": "Position end placeholder"})
    system_message: Optional[str] = field(default="You are a scientific assistant specializing in protein sequence analysis. Based on protein sequence embeddings and other related information, please answer the relevant questions using professional language. ", metadata={"help": "System message"})
    def __repr__(self):
        # 获取 dataclass 默认字段
        fields = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        # 获取动态添加的字段
        dynamic_fields = {k: v for k, v in self.__dict__.items() if k not in fields}
        # 合并
        all_fields = {**fields, **dynamic_fields}
        return f"DataArguments({all_fields})"


@dataclass
class FragTrainingArguments(TrainingArguments):
    """Extended training arguments for fragment training."""
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: str = "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# 自定义 JSON 序列化器，处理无法序列化的对象
def custom_serializer(obj):
    try:
        return vars(obj)  # 尝试转换为字典
    except TypeError:
        return str(obj)  # 直接转换为字符串

def get_frag_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

class FragTrainer(Trainer):
    """Custom Trainer for Esm2LlamaInstructForCausalLM with fragment support."""
    def create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        
        opt_model = self.model
        
        # Separate parameters by component
        protein_params = []
        llm_params = []
        protein_params_wo_decay = []
        llm_params_wo_decay = []
        
        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]
        
        for k, v in opt_model.named_parameters():
            if v.requires_grad:
                if any(component in k for component in ["esm_encoder", "adapter", "fragment_adapter"]):
                    if k in decay_params:
                        protein_params.append(v)
                    else:
                        protein_params_wo_decay.append(v)
                else:
                    if k in decay_params:
                        llm_params.append(v)
                    else:
                        llm_params_wo_decay.append(v)
        
        optimizer_grouped_parameters = [
            {"params": protein_params + llm_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
            {"params": protein_params_wo_decay + llm_params_wo_decay, "lr": self.args.learning_rate, "weight_decay": 0.0},
        ]
        
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_frag_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['fragment_adapter']
            weight_to_save = get_frag_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'fragment_adapter.bin'))
        else:
            super(FragTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_frag_adapter', False):
            pass
        else:
            super(FragTrainer, self)._save(output_dir, state_dict)



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['fragment_adapter']

        weight_to_save = get_frag_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                fragment_adapter_folder = os.path.join(parent_folder, "fragment_adapter")
                os.makedirs(fragment_adapter_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(fragment_adapter_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'fragment_adapter.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa




def create_datasets_and_collator(data_args: FragDataArguments, model_args: FragModelArguments):
    """Create training and evaluation datasets with data collator."""
    
    # Load tokenizers
    esm_tokenizer = AutoTokenizer.from_pretrained(model_args.esm_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(
        model_args.llama_path,
        pad_token='<|reserved_special_token_0|>'
    )
    
    # Create datasets
    train_dataset = FragRefDataset(
        root_dir=data_args.root_dir,
        data_name=data_args.data_name,
        split=data_args.train_split,
        task_type=data_args.task_type,
        max_sequence_length=data_args.max_sequence_length,
    )
    
    eval_dataset = FragRefDataset(
        root_dir=data_args.root_dir,
        data_name=data_args.data_name,
        split=data_args.eval_split,
        task_type=data_args.task_type,
        max_sequence_length=data_args.max_sequence_length,
    )
    
    # Create data collator
    data_collator = FragDataCollator(
        sequence_tokenizer=esm_tokenizer,
        llm_tokenizer=llama_tokenizer,
        mode="train",
        max_sequence_length=data_args.max_sequence_length,
    )
    
    return train_dataset, eval_dataset, data_collator, llama_tokenizer


def train(attn_implementation=None):
    """Main training function."""
    parser = transformers.HfArgumentParser(
        (FragModelArguments, FragDataArguments, FragTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Determine torch dtype
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    # Load base models
    esm_encoder = EsmModel.from_pretrained(
        model_args.esm_path,
        add_pooling_layer=False,
        torch_dtype=torch_dtype,
        cache_dir=training_args.cache_dir,
    )
    
    llama_decoder = LlamaForCausalLM.from_pretrained(
        model_args.llama_path,
        torch_dtype=torch_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
    )
    
    # Create adapter
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch_dtype)
    
    # Create main model
    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        llama_decoder=llama_decoder,
        perceiver_latent_size=model_args.perceiver_latent_size,
        num_perceiver_heads=model_args.num_perceiver_heads,
        num_perceiver_layers=model_args.num_perceiver_layers,
    )

    # Load pretrained weights if specified
    if model_args.load_model_checkpoint_path:
        print(f"Loading model checkpoint from {model_args.load_model_checkpoint_path}")
        model_state_dict = torch.load(
            model_args.load_model_checkpoint_path,
            weights_only=True,
            map_location="cpu"
        )
        model.load_state_dict(model_state_dict)
    
    # Apply LoRA
    if model_args.load_adapter_checkpoint_dir:
        print(f"Loading LoRA adapter from {model_args.load_adapter_checkpoint_dir}")
        model = PeftModel.from_pretrained(
            model,
            model_args.load_adapter_checkpoint_dir,
            is_trainable=True
        )
    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if training_args.lora_enable:
        print("Initializing LoRA adapter")
        target_modules = training_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            init_lora_weights=True,
            target_modules=target_modules,
            modules_to_save=(
                ["adapter.fc1", "adapter.fc2", "fragment_adapter"]
                if not model_args.fix_modality_adapter
                else ["fragment_adapter"]
            )
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()


    args_dict = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": vars(training_args)
    }
    save_path = f"{training_args.output_dir}/training_config.json"
    # 创建路径（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存为 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False, default=custom_serializer)
    
    # Set random seeds
    transformers.trainer_utils.set_seed(training_args.seed)
    

    # Create datasets and data collator
    #Load tokenizers
    esm_tokenizer = AutoTokenizer.from_pretrained(model_args.esm_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(
        model_args.llama_path,
        pad_token='<|reserved_special_token_0|>'
    )
    data_args.sequence_tokenizer = esm_tokenizer
    data_args.llm_tokenizer = llama_tokenizer
    data_module = make_multitask_dataset(data_args)
    
    # Create trainer
    trainer = FragTrainer(
        model=model,
        args=training_args,
        **data_module,
    )

    print("Model:")
    print(model)
    print(f"Training dataset size: {len(data_module['train_dataset'])}")
    print(f"Evaluation dataset size: {len(data_module['eval_dataset'])}")
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    model.config.gradient_checkpointing = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    train()
