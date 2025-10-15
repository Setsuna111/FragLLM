"""Example script for ProteinSAM inference on datasets.
Loads model and runs inference on specified test datasets with llama cache support.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from protein_sam import ProteinSAM
from dataset import get_datasets_and_collator
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm


def load_model_for_inference(
    checkpoint_path: str,
    esm_model_path: str,
    llama_model_path: str,
    device: str = "cuda",
    use_category_cache: bool = True,
    category_embeddings_path: str = "./category_embeddings.pt"
) -> ProteinSAM:
    """Load trained ProteinSAM model for inference with llama cache support."""
    
    model = ProteinSAM(
        esm_model_path=esm_model_path,
        llama_model_path=llama_model_path,
        output_llama_layer=16,
        decoder_num_heads=8,
        decoder_num_layers=2,
        decoder_intermediate_size=512,
        max_sequence_length=1021,
        dropout_rate=0.1,
        device=device,
        use_category_cache=use_category_cache,
        category_embeddings_path=category_embeddings_path if use_category_cache else None
    )
    
    model.load_model(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    return model


def run_dataset_inference(
    model: ProteinSAM,
    test_loader: DataLoader,
    device: str = "cuda"
) -> list:
    """Run inference on test dataset."""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Handle point positions
            # point_mask = None
            point_mask = batch["point_mask"]
            point_positions = batch["point_positions"] if torch.any(point_mask) else None
            
            # Forward pass
            outputs = model(
                protein_input_ids=batch["protein_input_ids"],
                protein_attention_mask=batch["protein_attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
                categories=batch["categories"],
                point_positions=point_positions
            )
            
            # Store results
            batch_results = {
                "start_predictions": outputs["start_predictions"].cpu(),
                "end_predictions": outputs["end_predictions"].cpu(),
                "start_labels": batch["start_labels"].cpu(),
                "end_labels": batch["end_labels"].cpu(),
                "categories": batch["categories"]
            }
            all_results.append(batch_results)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="ProteinSAM Dataset Inference")
    
    parser.add_argument("--checkpoint_path", type=str, 
                       default="./checkpoints_grounding/best_model.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--esm_model_path", type=str,
                       default="/home/lfj/projects_dir/pretrained_model/esm2_t30_150M_UR50D",
                       help="Path to ESM model")
    parser.add_argument("--llama_model_path", type=str,
                       default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct",
                       help="Path to Llama model")
    parser.add_argument("--data_root", type=str, default="../data",
                       help="Root directory for datasets")
    parser.add_argument("--data_name", type=str, default='VenusX_BindI',
                       help="Dataset name(s). Single: 'VenusX_Dom' or Multiple: 'VenusX_Dom||VenusX_Act||VenusX_BindI'")
    parser.add_argument("--use_category_cache", action="store_true", default=True,
                       help="Use llama cache (pre-computed category embeddings)")
    parser.add_argument("--category_embeddings_path", type=str, 
                       default="./category_embeddings.pt",
                       help="Path to pre-computed category embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Inference batch size")
    parser.add_argument("--output_file", type=str, default="dataset_predictions.json",
                       help="Output file for predictions")
    
    args = parser.parse_args()
    
    print(f"Loading datasets: {args.data_name}")
    datasets, collator = get_datasets_and_collator(
        root_dir=args.data_root,
        data_name=args.data_name,
        esm_model_path=args.esm_model_path,
        llama_model_path=args.llama_model_path if not args.use_category_cache else None,
        max_sequence_length=1021,
        max_text_length=128,
        use_category_cache=args.use_category_cache
    )
    
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    print("Loading model...")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint_path,
        esm_model_path=args.esm_model_path,
        llama_model_path=args.llama_model_path,
        device=args.device,
        use_category_cache=args.use_category_cache,
        category_embeddings_path=args.category_embeddings_path
    )
    print("Model loaded successfully!")
    
    print("Running inference on test dataset...")
    results = run_dataset_inference(model, test_loader, args.device)
    
    print(f"Inference completed! Processed {len(results)} batches.")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()