"""
Training script for Esm2LlamaInstructForCausalLM using transformers.Trainer.

This script implements training of the fragment-aware protein language model
using the HuggingFace Trainer framework with LoRA fine-tuning support.

Based on train_trainer_refer.py but adapted for Esm2LlamaInstructForCausalLM.
"""

import sys
sys.path.append("..")
sys.path.append(".")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
from models.protein_llama_addtoken_lfj import *
from dataset.dataloader_referring import FragRefDataset
from dataset.dataloader_frag import FragDataCollator, make_multitask_dataset
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class FragModelArguments:
    """Model arguments for fragment training."""
    esm_path: Optional[str] = field(default="/home/lfj/projects_dir/pretrained_model/esm2_t36_3B_UR50D", metadata={"help": "Path to ESM model"})
    llama_path: Optional[str] = field(default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct", metadata={"help": "Path to LLaMA model"})
    load_adapter_checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "Path to load adapter checkpoint"})
    load_fragment_checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "Path to load fragment checkpoint"})
    load_pro2text_checkpoint_dir: Optional[str] = field(default='/home/lfj/projects_dir/pretrained_model/Prot2Text-V2-11B-Instruct-hf', metadata={"help": "Path to load fragment checkpoint"})
    
    # Grounding model arguments
    protein_sam_checkpoint_path: Optional[str] = field(default='/home/lfj/projects_dir/FragLLM/model_grounding_segformer/checkpoints_grounding_3B/checkpoint_epoch_0.pt', metadata={"help": "Path to ProteinSAM checkpoint (.pt file). Required for training."})
    
    # Model architecture arguments
    fix_modality_adapter: Optional[bool] = field(default=False, metadata={"help": "Whether to fix modality adapter"})
    
    # Fragment adapter arguments
    perceiver_latent_size: Optional[int] = field(default=1, metadata={"help": "Perceiver latent size"})
    num_perceiver_heads: Optional[int] = field(default=8, metadata={"help": "Number of perceiver heads"})
    num_perceiver_layers: Optional[int] = field(default=2, metadata={"help": "Number of perceiver layers"})
    intermediate_dim: Optional[int] = field(default=2048, metadata={"help": "Intermediate dimension"})
    dropout_rate: Optional[float] = field(default=0.3, metadata={"help": "Dropout rate"})
    freeze_backbone: Optional[bool] = field(default=False, metadata={"help": "Whether to freeze backbone"})
    # placeholder ids
    sequence_placeholder_id: int = 128003
    fragment_placeholder_id: int = 128005
    ce_loss_weight: Optional[float] = field(default=1.0, metadata={"help": "ce loss weight"})
    position_loss_weight: Optional[float] = field(default=0.1, metadata={"help": "position loss weight"})

    # Fragment adapter type
    frag_adapter_type: Optional[str] = field(default="qformer", metadata={"help": "Fragment adapter type: 'qformer' or 'multilevel'"})

    def __repr__(self):
        fields = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        dynamic_fields = {k: v for k, v in self.__dict__.items() if k not in fields}
        all_fields = {**fields, **dynamic_fields}
        return f"ModelArguments({all_fields})"

@dataclass
class FragDataArguments:
    """Data arguments for fragment training."""
    root_dir: Optional[str] = field(default="./data", metadata={"help": "Root directory for datasets"})
    use_detailed_template: Optional[bool] = field(default=False, metadata={"help": "Whether to use detailed template"})
    # dataset_train_config: Optional[str] = field(default="ActGroundSingle||ActRefClass", metadata={"help": "Dataset config for training"})
    dataset_train_config: Optional[str] = field(default="ActRefClass", metadata={"help": "Dataset config for training"})
    dataset_valid_config: Optional[str] = field(default=None, metadata={"help": "Dataset config for evaluation"})
    max_sequence_length: Optional[int] = field(default=1021, metadata={"help": "Maximum sequence length"})
    filter_sequence: Optional[bool] = field(default=False, metadata={"help": "Whether to filter sequence"})
    dataset_size: Optional[int] = field(default=-1, metadata={"help": "Dataset size for function dataset. -1 means use full dataset, otherwise truncate to this size"})
    # special tokens
    sequence_placeholder: Optional[str] = field(default="<|reserved_special_token_1|>", metadata={"help": "Sequence placeholder"})
    fragment_placeholder: Optional[str] = field(default="<|reserved_special_token_2|>", metadata={"help": "Fragment placeholder"})
    position_placeholder: Optional[str] = field(default="<frag_position>", metadata={"help": "Position placeholder for fragment grounding"})
    phrase_start_placeholder: Optional[str] = field(default="<p>", metadata={"help": "Phrase start placeholder"})
    phrase_end_placeholder: Optional[str] = field(default="</p>", metadata={"help": "Phrase end placeholder"})

    system_message: Optional[str] = field(default="You are a scientific assistant specializing in protein sequence analysis. Based on protein sequence embeddings and other related information, please answer the relevant questions using professional language. ", metadata={"help": "System message"})
    def __repr__(self):
        fields = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        dynamic_fields = {k: v for k, v in self.__dict__.items() if k not in fields}
        all_fields = {**fields, **dynamic_fields}
        return f"DataArguments({all_fields})"


@dataclass
class FragTrainingArguments(TrainingArguments):
    """Extended training arguments for fragment training."""
    output_dir: Optional[str] = field(default="./checkpoints/fragment_training_debug", metadata={"help": "Output directory"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size per device for training"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "Batch size per device for evaluation"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "Gradient accumulation steps"})
    evaluation_strategy: Optional[str] = field(default=None, metadata={"help": "Evaluation strategy"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "Save strategy"})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Maximum number of checkpoints to save"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "Learning rate"})
    weight_decay: Optional[float] = field(default=0., metadata={"help": "Weight decay"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "Warmup ratio"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "Logging steps"})
    dataloader_num_workers: Optional[int] = field(default=4, metadata={"help": "Number of workers for dataloader"})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "Remove unused columns"})
    bf16: bool = field(default=True, metadata={"help": "Whether to use bf16"})
    tf32: bool = field(default=True, metadata={"help": "Whether to use tf32"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    tune_fragment_adapter: bool = field(default=False, metadata={"help": "Whether to tune fragment adapter"})
    freeze_adapter: bool = field(default=False, metadata={"help": "Whether to freeze adapter"})
    freeze_fragment_adapter: bool = field(default=False, metadata={"help": "Whether to freeze fragment adapter"})
    tune_adapter: bool = field(default=False, metadata={"help": "Whether to tune adapter"})
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    # lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: str = "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj"

    max_grad_norm: float = field(
        default=0.5, 
        metadata={"help": "Max gradient norm (for gradient clipping)."}
    )

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

def get_protein_sam_state_maybe_zero_3(named_params):
    """Extract ProteinSAM parameters excluding the ESM encoder"""
    to_return = {}
    for k, t in named_params:
        if "protein_sam" in k and "protein_sam.esm_model" not in k:
            to_return[k] = maybe_zero_3(t, ignore_status=True, name=k).cpu()
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

def get_peft_state_non_lora_with_protein_sam_maybe_zero_3(named_params, require_grad_only=True):
    """Extract non-LoRA parameters including ProteinSAM trainable parameters (excluding ESM encoder)"""
    to_return = {}
    for k, t in named_params:
        # Include non-LoRA parameters
        if "lora_" not in k:
            # For ProteinSAM, exclude ESM encoder but include other trainable parts
            if "protein_sam" in k:
                if "protein_sam.esm_model" not in k:
                    if not require_grad_only or t.requires_grad:
                        to_return[k] = maybe_zero_3(t, ignore_status=True).cpu()
            else:
                if not require_grad_only or t.requires_grad:
                    to_return[k] = maybe_zero_3(t, ignore_status=True).cpu()
    return to_return


class FragTrainer(Trainer):
    """Custom Trainer for Esm2LlamaInstructForCausalLM with fragment support."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss_history = []
        self.position_loss_history = []

    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """        
        # 如果存在自定义损失，则将其暂存
        if self.ce_loss_history:
            # 计算梯度累积步骤内的平均损失
            avg_ce_loss = sum(self.ce_loss_history) / len(self.ce_loss_history)
            logs["ce_loss"] = avg_ce_loss
            # 清空历史记录，为下一个 logging interval 做准备
            self.ce_loss_history.clear()

        if self.position_loss_history:
            avg_position_loss = sum(self.position_loss_history) / len(self.position_loss_history)
            logs["position_loss"] = avg_position_loss
            self.position_loss_history.clear()

        # 调用父类log方法，确保所有callbacks正常工作
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        ce_loss = outputs.get("ce_loss")
        position_loss = outputs.get("position_loss") 
         # 如果存在自定义损失，则将其暂存
        if ce_loss is not None:
            # .detach() 防止梯度信息和计算图被存储
            # .item() 获取纯粹的 Python 数字，释放 GPU 内存
            self.ce_loss_history.append(ce_loss.detach().item())
        if position_loss is not None:
            self.position_loss_history.append(position_loss.detach().item())

        return (loss, outputs) if return_outputs else loss
    

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
            # Explicitly exclude frozen ProteinSAM parameters (only ESM encoder should be frozen)
            if "protein_sam.esm_model" in k:
                continue
            if v.requires_grad:
                if any(component in k for component in ["esm_encoder", "adapter", "fragment_adapter", "protein_sam"]):
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

def train(attn_implementation=None):
    global local_rank
    """Main training function."""
    parser = transformers.HfArgumentParser(
        (FragModelArguments, FragDataArguments, FragTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load base models
    model = ProteinLlamaForCausalLM.from_pretrained(
        model_args.llama_path,
        torch_dtype=torch_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
    )

    # 覆盖pro2text权重 (1/2)
    # 只需要加载adapter和llama_decoder部分的权重，prot2text训练没调llama的embedding和lm_head（已确认），不加载它们以避免和我们reshape过的模型的冲突
    # 此处仅加载llama权重，后面实例化adapter后再加载adapter权重。这里先加载llama为了避开后面注入lora后的复杂变量名。
    if model_args.load_pro2text_checkpoint_dir is not None:
        pro2text_model = AutoModelForCausalLM.from_pretrained(model_args.load_pro2text_checkpoint_dir, trust_remote_code=True)
        pro2text_param_dict = pro2text_model.state_dict()
        pro2text_llama_weights = {k.split('llama_decoder.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.' in k) and ('embed_tokens' not in k) and ('lm_head' not in k)}
        
        model.load_state_dict(pro2text_llama_weights, strict=False)
        rank0_print("Loaded Prot2Text llama_decoder weights from ", model_args.load_pro2text_checkpoint_dir)

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    args_dict = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": vars(training_args)
    }
    save_path = f"{training_args.output_dir}/training_config.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False, default=custom_serializer)
        
    # add special tokens
    esm_tokenizer = AutoTokenizer.from_pretrained(model_args.esm_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(
        model_args.llama_path,
        pad_token='<|reserved_special_token_0|>'
    )
    llama_tokenizer.add_tokens([
        data_args.position_placeholder,
        data_args.phrase_start_placeholder,
        data_args.phrase_end_placeholder
    ], special_tokens=True)
    model.resize_token_embeddings(len(llama_tokenizer))
    model.lm_head.weight.requires_grad_(True)
    model.get_model().embed_tokens.weight.requires_grad_(True)
    if model_args.esm_path is not None:
        model.config.max_sequence_length = data_args.max_sequence_length
        model.get_model().initialize_modules(model_args=model_args, fsdp= training_args.fsdp)
        
        # 覆盖pro2text权重（2/2）
        if model_args.load_pro2text_checkpoint_dir is not None and pro2text_model is not None:
            pro2text_param_dict = pro2text_model.state_dict()
            pro2text_adapter_weights = {k.split('adapter.')[1]: v for k, v in pro2text_param_dict.items() if 'adapter.' in k}

            model.get_model().adapter.load_state_dict(pro2text_adapter_weights, strict=False)
            rank0_print("Loaded Prot2Text adapter weights from ", model_args.load_pro2text_checkpoint_dir)
            del pro2text_model

        # add special tokens ids
        model.config.sequence_placeholder_id = model_args.sequence_placeholder_id
        model.config.fragment_placeholder_id = model_args.fragment_placeholder_id
        model.config.position_placeholder_id = llama_tokenizer.convert_tokens_to_ids(data_args.position_placeholder)
        model.config.phrase_start_placeholder_id = llama_tokenizer.convert_tokens_to_ids(data_args.phrase_start_placeholder)
        model.config.phrase_end_placeholder_id = llama_tokenizer.convert_tokens_to_ids(data_args.phrase_end_placeholder)
        
        # add ce loss weight and position loss weight
        model.config.ce_loss_weight = model_args.ce_loss_weight
        model.config.position_loss_weight = model_args.position_loss_weight

        # add proteinsam path
        model.config.protein_sam_checkpoint_path = model_args.protein_sam_checkpoint_path

        rank0_print("model.config.sequence_placeholder_id: ", model.config.sequence_placeholder_id)
        rank0_print("model.config.fragment_placeholder_id: ", model.config.fragment_placeholder_id)
        rank0_print("model.config.position_placeholder_id: ", model.config.position_placeholder_id)
        rank0_print("model.config.phrase_start_placeholder_id: ", model.config.phrase_start_placeholder_id)
        rank0_print("model.config.phrase_end_placeholder_id: ", model.config.phrase_end_placeholder_id)
        
        esm_encoder = model.get_model().get_esm_encoder()
        esm_encoder.to(dtype=torch_dtype, device=training_args.device)
        model.config.tune_fragment_adapter = training_args.tune_fragment_adapter
        model.config.freeze_adapter = training_args.freeze_adapter
        model.config.freeze_fragment_adapter = training_args.freeze_fragment_adapter
        model.config.tune_adapter = training_args.tune_adapter
        if training_args.tune_adapter:
            model.requires_grad_(False)
            for p in model.get_model().adapter.parameters():
                p.requires_grad_(True)
        if training_args.freeze_adapter:
            for p in model.get_model().adapter.parameters():
                p.requires_grad_(False)
        if training_args.tune_fragment_adapter:
            model.requires_grad_(False)
            for p in model.get_model().fragment_adapter.parameters():
                p.requires_grad_(True)
        if training_args.freeze_fragment_adapter:
            for p in model.get_model().fragment_adapter.parameters():
                p.requires_grad_(False)
        
    rank0_print("ModelTrainable:")
    rank0_print([n for n, p in model.named_parameters() if p.requires_grad])
    rank0_print(model.device)
    data_args.sequence_tokenizer = esm_tokenizer
    data_args.llm_tokenizer = llama_tokenizer
    data_args.perceiver_latent_size = model_args.perceiver_latent_size
    data_module = make_multitask_dataset(data_args)
    
    # Create trainer
    trainer = FragTrainer(
        model=model,
        tokenizer=llama_tokenizer,
        args=training_args,
        **data_module,
    )

    # rank0_print("Model:")
    # rank0_print(model)
    rank0_print("device:", model.device)
    rank0_print("training_args.device:", training_args.device)
    rank0_print(f"Training dataset size: {len(data_module['train_dataset'])}")
    if data_args.dataset_valid_config is not None:
        rank0_print(f"Evaluation dataset size: {len(data_module['eval_dataset'])}")
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch...")
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    model.config.gradient_checkpointing = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_with_protein_sam_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
