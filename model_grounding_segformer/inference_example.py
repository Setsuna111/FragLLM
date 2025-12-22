"""Example script for ProteinSAM inference on datasets.
Loads model and runs inference on specified test datasets with llama cache support.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    params_file: str = None,
    device: str = "cuda"
) -> ProteinSAM:
    """Load trained ProteinSAM model for inference with hyperparameters from JSON file.

    Args:
        checkpoint_path: Path to model checkpoint file
        params_file: Path to JSON file containing model hyperparameters.
                    If None, will look for 'protein_sam_init_params.json'
                    in the same directory as checkpoint_path
        device: Device to load model on

    Returns:
        Loaded ProteinSAM model in eval mode
    """

    # If params_file not provided, try to find it in checkpoint directory
    if params_file is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        params_file = os.path.join(checkpoint_dir, "protein_sam_init_params.json")

    # Load hyperparameters from JSON file
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Model parameters file not found: {params_file}")

    with open(params_file, 'r') as f:
        params = json.load(f)

    print(f"Loading model hyperparameters from: {params_file}")
    print(f"Model configuration: {json.dumps(params, indent=2)}")

    # Create model with all loaded hyperparameters
    model = ProteinSAM(**params)

    print(f"Loading checkpoint from: {checkpoint_path}")
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

            # Get ground truth residue-level labels from collator
            true_masks = batch["residue_labels"]  # Shape: (batch_size, max_seq_len)
            adjusted_mask = batch["protein_attention_mask"][:, 1:-1]  # Remove BOS/EOS from attention mask

            # Calculate metrics
            metrics = calculate_metrics(
                outputs["mask_predictions"].cpu(),
                true_masks.cpu(),
                adjusted_mask.cpu()
            )
            all_metrics.append(metrics)

            batch_size = outputs["mask_predictions"].shape[0]

            # Convert masks to start/end positions for comparison
            pred_starts = []
            pred_ends = []
            for b in range(batch_size):
                start, end = mask_to_positions(outputs["mask_predictions"][b].cpu())
                pred_starts.append(start)
                pred_ends.append(end)

            # Extract ground truth start/end positions from residue labels
            true_starts = []
            true_ends = []
            for b in range(batch_size):
                label_indices = torch.where(true_masks[b] == 1)[0]
                if len(label_indices) > 0:
                    true_starts.append(label_indices[0].item())
                    true_ends.append(label_indices[-1].item())
                else:
                    true_starts.append(-1)
                    true_ends.append(-1)

            # Store results - simplified format for testing
            # Only keep predicted vs actual start/end positions for comparison
            batch_results = {
                "predictions": [
                    {
                        "pred_start": pred_starts[b],
                        "pred_end": pred_ends[b],
                        "true_start": true_starts[b],
                        "true_end": true_ends[b]
                    }
                    for b in range(batch_size)
                ]
            }
            all_results.append(batch_results)

    return all_results, all_metrics


def main():
    parser = argparse.ArgumentParser(description="ProteinSAM Dataset Inference")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints_grounding_1026_650M_1280dim_1e-5/best_model.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--params_file", type=str, default=None,
                       help="Path to model hyperparameters JSON file. "
                            "If not provided, will look for 'protein_sam_init_params.json' "
                            "in the checkpoint directory")
    parser.add_argument("--data_root", type=str, default="../data",
                       help="Root directory for datasets")
    parser.add_argument("--data_name", type=str, default='VenusX_BindI',
                       help="Dataset name(s). Single: 'VenusX_Dom' or Multiple: 'VenusX_Dom||VenusX_Act||VenusX_BindI'")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Inference batch size")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for predictions. If not provided, will save to checkpoint directory with name 'inference_results.json'")

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinSAM Inference")
    print("=" * 60)

    # Load model parameters from JSON file
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))

    if args.params_file is None:
        args.params_file = os.path.join(checkpoint_dir, "protein_sam_init_params.json")

    # Set output file path to checkpoint directory if not specified
    if args.output_file is None:
        args.output_file = os.path.join(checkpoint_dir, "inference_results.json")

    with open(args.params_file, 'r') as f:
        model_params = json.load(f)

    esm_model_path = model_params.get("esm_model_path")
    llama_model_path = model_params.get("llama_model_path")
    use_category_cache = model_params.get("use_category_cache", True)

    print(f"\n1. Loading datasets: {args.data_name}")
    datasets, collator = get_datasets_and_collator(
        root_dir=args.data_root,
        data_name=args.data_name,
        esm_model_path=esm_model_path,
        llama_model_path=llama_model_path if not use_category_cache else None,
        max_sequence_length=model_params.get("max_sequence_length", 1021),
        max_text_length=128,
        use_category_cache=use_category_cache
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    print(f"   Loaded test set with {len(datasets['test'])} samples")

    print(f"\n2. Loading model from checkpoint: {args.checkpoint_path}")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint_path,
        params_file=args.params_file,
        device=args.device
    )
    print("   Model loaded successfully!")

    print(f"\n3. Running inference on test dataset...")
    results, metrics = run_dataset_inference(model, test_loader, args.device)

    print(f"   Inference completed! Processed {len(results)} batches.")

    # Calculate overall metrics
    if metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics]),
            'iou': np.mean([m['iou'] for m in metrics]),
            'precision': np.mean([m['precision'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics])
        }

        print(f"\n=== Overall Metrics ===")
        print(f"Mask Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Mask IoU:       {avg_metrics['iou']:.4f}")
        print(f"Precision:      {avg_metrics['precision']:.4f}")
        print(f"Recall:         {avg_metrics['recall']:.4f}")

    # Save results
    output_data = {
        'results': results,
        'overall_metrics': avg_metrics if metrics else None,
        'batch_metrics': metrics
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\n4. Results saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()