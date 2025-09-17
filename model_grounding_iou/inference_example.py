"""Example script for ProteinSAM inference on datasets.
Loads model and runs inference on specified test datasets with llama cache support.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from protein_sam import ProteinSAM
from dataset import get_datasets_and_collator
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
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


def mask_to_positions(mask: torch.Tensor) -> tuple:
    """Convert binary mask to start/end positions."""
    indices = torch.where(mask == 1)[0]
    if len(indices) > 0:
        return indices[0].item(), indices[-1].item()
    else:
        # If no positive predictions, find the position with highest foreground probability
        return 0, 0


def calculate_metrics(pred_masks: torch.Tensor, true_masks: torch.Tensor, attention_masks: torch.Tensor) -> dict:
    """Calculate evaluation metrics for mask predictions."""
    batch_size = pred_masks.shape[0]
    total_accuracy = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    for b in range(batch_size):
        pred = pred_masks[b]
        true = true_masks[b]
        mask = attention_masks[b].bool() if attention_masks is not None else torch.ones_like(pred, dtype=torch.bool)
        
        # Apply attention mask
        pred = pred[mask]
        true = true[mask]
        
        # Accuracy
        accuracy = (pred == true).float().mean().item()
        total_accuracy += accuracy
        
        # IoU
        intersection = (pred & true).float().sum()
        union = (pred | true).float().sum()
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0 if intersection == 0 else 0.0
        total_iou += iou
        
        # Precision and Recall
        tp = (pred & true).float().sum()
        fp = (pred & ~true).float().sum()
        fn = (~pred & true).float().sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        total_precision += precision.item() if torch.is_tensor(precision) else precision
        total_recall += recall.item() if torch.is_tensor(recall) else recall
    
    return {
        'accuracy': total_accuracy / batch_size,
        'iou': total_iou / batch_size,
        'precision': total_precision / batch_size,
        'recall': total_recall / batch_size
    }


def run_dataset_inference(
    model: ProteinSAM,
    test_loader: DataLoader,
    device: str = "cuda"
) -> tuple:
    """Run inference on test dataset and return results with metrics."""
    model.eval()
    all_results = []
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Handle point positions
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
            
            # Create ground truth masks from start/end labels
            batch_size, seq_len = batch["protein_input_ids"].shape

            adjusted_seq_len = seq_len - 2  # Adjust for BOS/EOS tokens
            adjusted_mask = batch["protein_attention_mask"][:, 1:-1]  # Adjust attention mask

            true_masks = torch.zeros(batch_size, adjusted_seq_len, dtype=torch.long)
            
            for b in range(batch_size):
                start = batch["start_labels"][b].item()
                end = batch["end_labels"][b].item()
                if start >= 0 and end >= 0 and start <= end:
                    true_masks[b, start:end+1] = 1
            
            # Calculate metrics
            metrics = calculate_metrics(
                outputs["mask_predictions"].cpu(), 
                true_masks, 
                adjusted_mask.cpu()
            )
            all_metrics.append(metrics)
            
            # Convert masks to start/end positions for compatibility
            pred_starts = []
            pred_ends = []
            for b in range(batch_size):
                start, end = mask_to_positions(outputs["mask_predictions"][b].cpu())
                pred_starts.append(start)
                pred_ends.append(end)
            
            # Store results - only predicted and actual start/end positions
            batch_results = {
                "pred_start_positions": pred_starts,
                "pred_end_positions": pred_ends,
                "true_start_positions": batch["start_labels"].cpu().tolist(),
                "true_end_positions": batch["end_labels"].cpu().tolist()
            }
            all_results.append(batch_results)
    
    return all_results, all_metrics


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
    results, metrics = run_dataset_inference(model, test_loader, args.device)
    
    print(f"Inference completed! Processed {len(results)} batches.")
    
    # Calculate overall metrics
    if metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics]),
            'iou': np.mean([m['iou'] for m in metrics]),
            'precision': np.mean([m['precision'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics])
        }
        
        print(f"\n=== Overall Metrics ===")
        print(f"Mask Accuracy: {avg_metrics['accuracy']:.4f}")
        print(f"Mask IoU: {avg_metrics['iou']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall: {avg_metrics['recall']:.4f}")
    
    # Save results
    output_data = {
        'results': results,
        'overall_metrics': avg_metrics if metrics else None,
        'batch_metrics': metrics
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()