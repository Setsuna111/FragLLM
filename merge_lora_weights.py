import os
import json
import torch
import argparse
import warnings
from peft import get_peft_model
from models.protein_llama_addtoken_lfj import *
from models.protein_llama_addtoken_djy import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.esm.modeling_esm import EsmModel
from types import SimpleNamespace


def load_pretrained_model_fragllm(model_path, model_base, model_name, pos_decoder_type, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
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
        with open(os.path.join(model_path, "training_config.json"), "r") as f:
            args_dict = json.load(f)
        model_args = SimpleNamespace(**args_dict["model_args"])
        training_args = SimpleNamespace(**args_dict["training_args"])
        data_args = SimpleNamespace(**args_dict["data_args"])
        if pos_decoder_type == "ProteinSAM":
            from models.protein_llama_addtoken_lfj import ProteinLlamaConfig
        else:
            from models.protein_llama_addtoken_djy import ProteinLlamaConfig
        lora_cfg_pretrained = ProteinLlamaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        model_path_vocab_size = lora_cfg_pretrained.vocab_size
        lora_cfg_pretrained.vocab_size = len(tokenizer)
        print('Loading FragLLM from base model...')
        if pos_decoder_type == "ProteinSAM":
            # 这一步已经会根据config配置自动初始化一些模块的参数，如fragment_adapter和adapter
            model = ProteinLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            if model.lm_head.weight.shape[0] != model_path_vocab_size:
                tokenizer.add_tokens([
                "<frag_position>",
                "<p>",
                "</p>"
                ], special_tokens=True)
                assert model_path_vocab_size == len(tokenizer)
                model.resize_token_embeddings(len(tokenizer))
        else:
            model = ProteinLlamaForCausalLM_Simple.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            if model.lm_head.weight.shape[0] != model_path_vocab_size:
                tokenizer.add_tokens([
                "<frag_start>",
                "<frag_end>",
                "<p>",
                "</p>"
                ], special_tokens=True)
                assert model_path_vocab_size == len(tokenizer)
                model.resize_token_embeddings(len(tokenizer))
        
        # 加载prot2text权重
        if model_args.load_pro2text_checkpoint_dir is not None:
            pro2text_model = AutoModelForCausalLM.from_pretrained(model_args.load_pro2text_checkpoint_dir, trust_remote_code=True)
            pro2text_param_dict = pro2text_model.state_dict()
            pro2text_llama_weights = {k.split('llama_decoder.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.' in k) and ('embed_tokens' not in k) and ('lm_head' not in k)}
            
            model.load_state_dict(pro2text_llama_weights, strict=False)
            print("Loaded Prot2Text llama_decoder weights from ", model_args.load_pro2text_checkpoint_dir) 
            pro2text_adapter_weights = {k.split('adapter.')[1]: v for k, v in pro2text_param_dict.items() if 'adapter.' in k}
            model.get_model().adapter.load_state_dict(pro2text_adapter_weights, strict=False)
            print("Loaded Prot2Text adapter weights from ", model_args.load_pro2text_checkpoint_dir)
            del pro2text_model

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
        model.load_state_dict(non_lora_trainables, strict=False)
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        with open(os.path.join(model_path, "training_config.json"), "r") as f:
            args_dict = json.load(f)
        model_args = SimpleNamespace(**args_dict["model_args"])
        training_args = SimpleNamespace(**args_dict["training_args"])
        data_args = SimpleNamespace(**args_dict["data_args"])
        # this may be mm projector only
        print('Loading FragLLM from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = ProteinLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)


        # 加载prot2text权重
        if model_args.load_pro2text_checkpoint_dir is not None:
            pro2text_model = AutoModelForCausalLM.from_pretrained(model_args.load_pro2text_checkpoint_dir, trust_remote_code=True)
            pro2text_param_dict = pro2text_model.state_dict()
            pro2text_llama_weights = {k.split('llama_decoder.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.' in k) and ('embed_tokens' not in k) and ('lm_head' not in k)}
            
            model.load_state_dict(pro2text_llama_weights, strict=False)
            print("Loaded Prot2Text llama_decoder weights from ", model_args.load_pro2text_checkpoint_dir) 
            pro2text_adapter_weights = {k.split('adapter.')[1]: v for k, v in pro2text_param_dict.items() if 'adapter.' in k}
            model.get_model().adapter.load_state_dict(pro2text_adapter_weights, strict=False)
            print("Loaded Prot2Text adapter weights from ", model_args.load_pro2text_checkpoint_dir)
            del pro2text_model

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
    tokenizer, model, esm_encoder = load_pretrained_model_fragllm(args.model_path, args.model_base, model_name, args.pos_decoder_type, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default= "/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260120_GroundingAll_4e_lora32")
    parser.add_argument("--model-base", type=str, default="/home/dataset-local/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--pos-decoder-type", type=str, default="Simple")
    # parser.add_argument("--save-model-path", type=str, default=merged_path)
    args = parser.parse_args()
    model_path = args.model_path.rstrip('/')
    merged_path = model_path + "_merge"
    args.save_model_path = merged_path

    merge_lora(args)
