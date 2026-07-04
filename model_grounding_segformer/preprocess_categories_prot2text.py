"""
Preprocess categories to generate text embeddings cache.
This script extracts all unique categories from datasets and pre-computes their Llama embeddings.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import torch
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer
from typing import Dict, List, Set
from tqdm import tqdm
import argparse


def extract_categories_from_dataset(data_path: str) -> Set[str]:
    """Extract all unique categories from a dataset file."""
    if not os.path.exists(data_path):
        return set()
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    categories = set()
    for item in data:
        for fragment in item.get("fragments", []):
            category = fragment.get("category", "")
            if category:
                categories.add(category)
    
    return categories


def parse_data_names(data_name_str: str) -> List[str]:
    """Parse data_name string to extract multiple dataset names."""
    return [name.strip() for name in data_name_str.split("||")] if "||" in data_name_str else [data_name_str]


def get_all_categories(data_root: str, data_names: List[str]) -> Set[str]:
    """Get all unique categories from multiple datasets."""
    all_categories = set()
    
    for data_name in data_names:
        for split in ["train", "valid", "test"]:
            data_path = os.path.join(data_root, data_name, f"{split}.json")
            categories = extract_categories_from_dataset(data_path)
            all_categories.update(categories)
            if categories:  # Only print if categories found
                print(f"Found {len(categories)} categories in {data_name}/{split}")
    
    print(f"Total unique categories: {len(all_categories)}")
    return all_categories


def get_categories_from_data_name_string(data_root: str, data_name_str: str) -> Set[str]:
    """Get all categories from data_name string (supports multi-dataset format)."""
    data_names = parse_data_names(data_name_str)
    return get_all_categories(data_root, data_names)


def encode_categories_with_llama(
    categories: List[str],
    llama_model_path: str,
    output_llama_layer: int = 16,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Encode categories using Llama model.
    
    Args:
        categories: List of category names
        llama_model_path: Path to Llama model
        output_llama_layer: Which layer to use for embeddings
        batch_size: Batch size for encoding
        device: Device to use
        
    Returns:
        Dictionary mapping category names to embeddings
    """
    print("Loading Llama model...")
    model = LlamaModel.from_pretrained(llama_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)

    # 覆盖prot2text中的llama权重
    load_pro2text_checkpoint_dir = "/home/dataset-local/projects_dir/pretrained_model/Prot2Text-V2-11B-Instruct-hf/"
    from transformers import AutoModelForCausalLM
    pro2text_model = AutoModelForCausalLM.from_pretrained(load_pro2text_checkpoint_dir, trust_remote_code=True)
    pro2text_param_dict = pro2text_model.state_dict()
    pro2text_llama_weights = {k.split('llama_decoder.model.')[1]: v for k, v in pro2text_param_dict.items() if ('llama_decoder.model.' in k)}
    model.load_state_dict(pro2text_llama_weights, strict=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|reserved_special_token_0|>'
    
    model.eval()
    category_embeddings = {}
    
    print("Encoding categories...")
    for i in tqdm(range(0, len(categories), batch_size)):
        batch_categories = categories[i:i + batch_size]
        
        # Tokenize batch
        tokenized = tokenizer(
            batch_categories,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Encode with Llama
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get embeddings from specified layer
            hidden_states = outputs.hidden_states[output_llama_layer]
            
            # Mean pooling over valid tokens
            masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            valid_tokens = attention_mask.sum(dim=1, keepdim=True)
            embeddings = sum_embeddings / valid_tokens
        
        # Store embeddings
        for j, category in enumerate(batch_categories):
            category_embeddings[category] = embeddings[j].cpu()
    
    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    print(f"Encoded {len(category_embeddings)} categories")
    return category_embeddings


def main():
    parser = argparse.ArgumentParser(description="Preprocess categories for ProteinSAM")
    
    parser.add_argument("--data_root", type=str, default="../data_frag_70",
                       help="Root directory for datasets")
    parser.add_argument("--data_name", type=str, default="VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo",
                       help="Alternative: single data_name string supporting multiple datasets with || separator")
    parser.add_argument("--llama_model_path", type=str,
                       default="/home/dataset-local/projects_dir/pretrained_model/Llama-3.1-8B-Instruct/",
                       help="Path to Llama model")
    parser.add_argument("--output_llama_layer", type=int, default=16,
                       help="Which Llama layer to use for embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for encoding")
    parser.add_argument("--output_base_dir", type=str, default=".",
                       help="Base directory under which category_embeddings/<data>/ will be created")
        
    args = parser.parse_args()
    
    data_name = os.path.basename(os.path.normpath(args.data_root))
    output_path = os.path.join(args.output_base_dir, "category_embeddings", data_name, "category_embeddings.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=== Category Preprocessing for ProteinSAM ===")
    print(f"Data root: {args.data_root}")
    print(f"Llama model: {args.llama_model_path}")
    print(f"Output layer: {args.output_llama_layer}")
    print(f"Device: {args.device}")
    print()
    
    # Step 1: Extract all categories
    print("Step 1: Extracting categories from datasets...")
    
    # Use single data_name string (supports || separator)
    categories = get_categories_from_data_name_string(args.data_root, args.data_name)
    print(f"Processing datasets from data_name: {args.data_name}")
    
    categories_list = sorted(list(categories))
    
    print(f"Categories to encode: {categories_list[:10]}...")  # Show first 10
    print()
    
    # Step 2: Encode categories
    print("Step 2: Encoding categories with Llama...")
    category_embeddings = encode_categories_with_llama(
        categories=categories_list,
        llama_model_path=args.llama_model_path,
        output_llama_layer=args.output_llama_layer,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Step 3: Save embeddings
    print("Step 3: Saving category embeddings...")
    
    torch.save({
        "category_embeddings": category_embeddings,
        "categories_list": categories_list,
        "llama_model_path": args.llama_model_path,
        "output_llama_layer": args.output_llama_layer,
        "embedding_dim": list(category_embeddings.values())[0].shape[0]
    }, output_path)
    
    print(f"Category embeddings saved to: {output_path}")
    print(f"Embedding dimension: {list(category_embeddings.values())[0].shape[0]}")
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()