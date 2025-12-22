import os
import torch
import argparse
import warnings
from peft import get_peft_model
from models.protein_llama_addtoken import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.esm.modeling_esm import EsmModel


def load_pretrained_model_fragllm_addtoken(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    from models.protein_llama_addtoken import ProteinLlamaConfig
    lora_cfg_pretrained = ProteinLlamaConfig.from_pretrained(model_path)
    
    # Load tokenizer and add special tokens (same as training)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, pad_token='<|reserved_special_token_0|>')
    
    # Add the same special tokens as in training
    special_tokens = [
        "<frag_position>",  # position_placeholder
        "<p>",             # phrase_start_placeholder  
        "</p>"             # phrase_end_placeholder
    ]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    # Temporarily set vocab_size to base model size for loading
    base_tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    original_vocab_size = lora_cfg_pretrained.vocab_size
    lora_cfg_pretrained.vocab_size = len(base_tokenizer)
    
    print('Loading FragLLM from base model...')
    model = ProteinLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
    
    # Restore original vocab_size for consistency
    lora_cfg_pretrained.vocab_size = original_vocab_size
    
    # Resize token embeddings to match training vocabulary size AFTER model loading
    model.resize_token_embeddings(len(tokenizer))

    print('Loading additional FragLLM weights...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    # import pdb; pdb.set_trace()
    
    # Then load the trained embeddings (which include the expanded vocabulary)
    model.load_state_dict(non_lora_trainables, strict=False)
    # import pdb;pdb.set_trace()
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    esm_encoder = EsmModel.from_pretrained(model.config.esm_path, add_pooling_layer=False)
    model.get_model().esm_encoder = esm_encoder
    
    return tokenizer, model, esm_encoder

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def merge_lora_addtoken(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, esm_encoder = load_pretrained_model_fragllm_addtoken(args.model_path, args.model_base, model_name, device_map='cpu')
    
    # Save the merged model and tokenizer (with expanded vocabulary)
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    
    print(f"Merged model saved to: {args.save_model_path}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 虽然这样设置有些奇怪
    # 这一版本删减掉了一些不必要的代码
    model_path = "/home/lfj/projects_dir/FragLLM/checkpoints/hybrid_tasks_base"

    model_path = model_path.rstrip('/')
    merged_path = model_path + "_merge"

    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct")
    parser.add_argument("--save-model-path", type=str, default=merged_path)

    args = parser.parse_args()

    merge_lora_addtoken(args)
