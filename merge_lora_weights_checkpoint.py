import os
import torch
import argparse
import types
import json
import wget
import collections
from peft import get_peft_model, PeftConfig, PeftModel
from models.protein_llama_addtoken_djy import *
from models.protein_llama_addtoken_lfj import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.esm.modeling_esm import EsmModel
from types import SimpleNamespace



def parse_args():
    parser = argparse.ArgumentParser(description="FragLLM: Merge lora weights and save model in hf format")

    parser.add_argument("--model-path", type=str, default="/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260118_GroundingAll_1e_lora32/checkpoint-1924")
    parser.add_argument("--save-model-path", type=str, default="/home/dataset-local/projects/Documents/FragLLM_git_v1_2512/checkpoints/FragLLM_260118_GroundingAll_1e_lora32_c1924_merge")

    return parser.parse_args()


def main():
    args = parse_args()
    # 从args.model_path的上一层目录中读取training_config.json
    with open(os.path.join(os.path.dirname(args.model_path) if "checkpoint-" in args.model_path else args.model_path, "training_config.json"), "r") as f:
        args_dict = json.load(f)
    model_args = SimpleNamespace(**args_dict["model_args"])
    training_args = SimpleNamespace(**args_dict["training_args"])
    data_args = SimpleNamespace(**args_dict["data_args"])
    # Create output directory if not exists already
    os.makedirs(args.save_model_path, exist_ok=True)
    weight_path = os.path.join(args.model_path, "pytorch_model.bin")
    peft_config = PeftConfig.from_pretrained(args.model_path)
    if model_args.pos_decoder_type == "ProteinSAM":
        from models.protein_llama_addtoken_lfj import ProteinLlamaConfig
    else:
        from models.protein_llama_addtoken_djy import ProteinLlamaConfig
    cfg_pretrained = ProteinLlamaConfig.from_pretrained(os.path.dirname(args.model_path) if "checkpoint-" in args.model_path else args.model_path)
    if model_args.pos_decoder_type == "ProteinSAM":
        model = ProteinLlamaForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    else:
        model = ProteinLlamaForCausalLM_Simple.from_pretrained(peft_config.base_model_name_or_path)
    model.config = cfg_pretrained
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        pad_token='<|reserved_special_token_0|>'
    )
    # 覆盖pro2text权重 (1/2)
    # 只需要加载adapter和llama_decoder部分的权重，prot2text训练没调llama的embedding和lm_head（已确认），不加载它们以避免和我们reshape过的模型的冲突
    # 此处仅加载llama权重，后面实例化adapter后再加载adapter权重。这里先加载llama为了避开后面注入lora后的复杂变量名。
    if model_args.load_pro2text_checkpoint_dir is not None:
        pro2text_model = AutoModelForCausalLM.from_pretrained(model_args.load_pro2text_checkpoint_dir, trust_remote_code=True)
        pro2text_param_dict = pro2text_model.state_dict()
        pro2text_llama_weights = {k.split('llama_decoder.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.' in k) and ('embed_tokens' not in k) and ('lm_head' not in k)}
        
        model.load_state_dict(pro2text_llama_weights, strict=False)
        print("Loaded Prot2Text llama_decoder weights from ", model_args.load_pro2text_checkpoint_dir)
    # Initializing LoRA adapter
    model = get_peft_model(model, peft_config)

    model.config.ce_loss_weight = model_args.ce_loss_weight
    model.config.position_loss_weight = model_args.position_loss_weight
    model.config.use_cache = True
    if model.lm_head.weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if model_args.esm_path is not None:
            model.config.max_sequence_length = data_args.max_sequence_length
            model.get_model().initialize_modules(model_args=model_args, fsdp=training_args.fsdp)
            
            # 覆盖pro2text权重（2/2）
            if model_args.load_pro2text_checkpoint_dir is not None and pro2text_model is not None:
                pro2text_param_dict = pro2text_model.state_dict()
                pro2text_adapter_weights = {k.split('adapter.')[1]: v for k, v in pro2text_param_dict.items() if 'adapter.' in k}
                model.get_model().adapter.load_state_dict(pro2text_adapter_weights, strict=False)
                print("Loaded Prot2Text adapter weights from ", model_args.load_pro2text_checkpoint_dir)
                del pro2text_model
            # add special tokens ids
            model.config.sequence_placeholder_id = model_args.sequence_placeholder_id
            model.config.fragment_placeholder_id = model_args.fragment_placeholder_id
            if model_args.pos_decoder_type == "ProteinSAM":
                model.config.position_placeholder_id = tokenizer.convert_tokens_to_ids(data_args.position_placeholder)
            else:
                model.config.pos_start_placeholder_id = tokenizer.convert_tokens_to_ids(data_args.pos_start_placeholder)
                model.config.pos_end_placeholder_id = tokenizer.convert_tokens_to_ids(data_args.pos_end_placeholder)
            model.config.phrase_start_placeholder_id = tokenizer.convert_tokens_to_ids(data_args.phrase_start_placeholder)
            model.config.phrase_end_placeholder_id = tokenizer.convert_tokens_to_ids(data_args.phrase_end_placeholder)
            # add proteinsam path
            model.config.protein_sam_checkpoint_path = model_args.protein_sam_checkpoint_path
            model.config.tune_fragment_adapter = training_args.tune_fragment_adapter
            model.config.freeze_adapter = training_args.freeze_adapter
            model.config.freeze_fragment_adapter = training_args.freeze_fragment_adapter
            model.config.tune_adapter = training_args.tune_adapter

        
    # Load the state-dict from --weights
    state_dict = torch.load(weight_path, map_location="cpu")
    updated_state_dict = {}
    for key in state_dict.keys():
        if "esm_encoder" in key:
            continue
        else:
            updated_state_dict[key] = state_dict[key]
    model.load_state_dict(updated_state_dict, strict=False)
    # Merge and save
    model = model.merge_and_unload()
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    main()