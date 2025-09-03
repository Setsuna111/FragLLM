import os
import torch
import argparse
import warnings
from peft import get_peft_model
from models.protein_llama import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.esm.modeling_esm import EsmModel


def load_pretrained_model_fragllm(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
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


    if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        from models.protein_llama import ProteinLlamaConfig
        lora_cfg_pretrained = ProteinLlamaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading FragLLM from base model...')
        model = ProteinLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

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
        model.load_state_dict(non_lora_trainables, strict=False)
        # import pdb;pdb.set_trace()
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading FragLLM from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = ProteinLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        adapter_weights = torch.load(os.path.join(model_path, 'adapter.bin'), map_location='cpu')
        adapter_weights = {k: v.to(torch.float16) for k, v in adapter_weights.items()}
        model.load_state_dict(adapter_weights, strict=False)
        fragment_adapter_weights = torch.load(os.path.join(model_path, 'fragment_adapter.bin'), map_location='cpu')
        fragment_adapter_weights = {k: v.to(torch.float16) for k, v in fragment_adapter_weights.items()}
        model.load_state_dict(fragment_adapter_weights, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model =ProteinLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

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

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, esm_encoder = load_pretrained_model_fragllm(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/home/djy/projects/Documents/FragLLM_git/checkpoints/fragment_training_only_stage2_lora32_epoch1_0831")
    # parser.add_argument("--model-base", type=str, default="/home/djy/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct")
    # parser.add_argument("--save-model-path", type=str, default="/home/djy/projects/Documents/FragLLM_git/checkpoints/fragment_training_only_stage2_lora32_epoch1_0831_merge_initesm")

    # 0902 test
    # model_path = "/home/lfj/projects_dir/FragLLM/checkpoints/test_load_stage1_only_motifdesc_lora"
    # model_path = "/home/lfj/projects_dir/FragLLM/checkpoints/test_load_stage1_both_motifdesc_motifcls_lora"
    model_path = "/home/lfj/projects_dir/FragLLM/checkpoints/test_from_scratch_only_motifdesc_lora"

    # 0903 test

    merged_path = model_path + "_merge"

    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct")
    parser.add_argument("--save-model-path", type=str, default=merged_path)


    args = parser.parse_args()

    merge_lora(args)
