import os
import torch
import argparse
import types
import json
import wget
import collections
from peft import get_peft_model, PeftConfig, PeftModel
from models.protein_llama import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.esm.modeling_esm import EsmModel
from types import SimpleNamespace



def parse_args():
    parser = argparse.ArgumentParser(description="FragLLM: Merge lora weights and save model in hf format")

    parser.add_argument("--model-path", type=str, default="/data/djy/FragLLM_git/checkpoints/fragment_training_only_stage2_bw_stage1_lora32_epoch5_function_0902/checkpoint-11640")
    parser.add_argument("--model-base", type=str, default="/home/djy/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--save-model-path", type=str, default="/data/djy/FragLLM_git/checkpoints/fragment_training_only_stage2_bw_stage1_lora32_epoch5_function_reuselora_0902_checkpoint-11640_merge")

    return parser.parse_args()


def main():
    args = parse_args()
    # 从args.model_path的上一层目录中读取training_config.json
    with open(os.path.join(os.path.dirname(args.model_path), "training_config.json"), "r") as f:
        args_dict = json.load(f)
    model_args = SimpleNamespace(**args_dict["model_args"])
    training_args = SimpleNamespace(**args_dict["training_args"])
    # Create output directory if not exists already
    os.makedirs(args.save_model_path, exist_ok=True)
    weight_path = os.path.join(args.model_path, "pytorch_model.bin")
    peft_config = PeftConfig.from_pretrained(args.model_path)
    model = ProteinLlamaForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        pad_token='<|reserved_special_token_0|>'
    )
    model = get_peft_model(model, peft_config)
    # Load the state-dict from --weights
    state_dict = torch.load(weight_path, map_location="cpu")
    updated_state_dict = {}
    for key in state_dict.keys():
        if "esm_encoder" in key:
            continue
        else:
            updated_state_dict[key] = state_dict[key]
    if model_args.esm_path is not None:
        model.get_model().initialize_modules(model_args=model_args)
        model.config.sequence_placeholder_id = model_args.sequence_placeholder_id
        model.config.fragment_placeholder_id = model_args.fragment_placeholder_id
        model.config.pos_start_placeholder_id = model_args.pos_start_placeholder_id
        model.config.pos_end_placeholder_id = model_args.pos_end_placeholder_id
        model.config.freeze_adapter = training_args.freeze_adapter
        model.config.freeze_fragment_adapter = training_args.freeze_fragment_adapter
        model.config.tune_adapter = training_args.tune_adapter
    model.load_state_dict(updated_state_dict, strict=False)

    # Merge and save
    model = model.merge_and_unload()
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    main()