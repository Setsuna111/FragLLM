"""
Protein-level inference for ProteinSAM.

Each batch contains all fragments of one protein. Every fragment is inferred
independently with its own point prompt (same as the fragment-level script),
but results are reported per protein: the predicted masks of all fragments are
OR-ed together and compared against the union of all ground-truth fragment masks.

Model loading and data processing mirror inference_example_3B_with_precalculated_emb.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from protein_sam import ProteinSAM
from dataset_protein_level import (
    ProteinGroupedSampler,
    get_datasets_and_collator_with_esm_cache_protein_level,
)
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
    if params_file is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        params_file = os.path.join(checkpoint_dir, "protein_sam_init_params.json")

    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Model parameters file not found: {params_file}")

    with open(params_file, 'r') as f:
        params = json.load(f)

    params['use_external_esm'] = True

    print(f"Loading model hyperparameters from: {params_file}")
    print(f"Model configuration: {json.dumps(params, indent=2)}")

    model = ProteinSAM(**params)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_model(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def calculate_protein_metrics(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    seq_mask: torch.Tensor
) -> dict:
    """
    Compute metrics for a single protein (1-D boolean tensors after masking).

    Args:
        pred_mask: (seq_len,) predicted binary mask
        true_mask: (seq_len,) ground truth binary mask
        seq_mask:  (seq_len,) valid residue mask (excludes padding)
    """
    pred = pred_mask[seq_mask]
    true = true_mask[seq_mask]

    accuracy = (pred == true).float().mean().item()

    intersection = (pred & true).float().sum()
    union = (pred | true).float().sum()
    iou = (intersection / union).item() if union > 0 else (1.0 if intersection == 0 else 0.0)

    tp = (pred & true).float().sum()
    fp = (pred & ~true).float().sum()
    fn = (~pred & true).float().sum()
    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0

    return {"accuracy": accuracy, "iou": iou, "precision": precision, "recall": recall}


def run_protein_level_inference(
    model: ProteinSAM,
    test_loader: DataLoader,
    device: str = "cuda",
) -> tuple:
    """
    Run inference where each batch = all fragments of one protein.

    For each protein:
      - Run forward() independently for every fragment (with its point prompt).
      - OR all predicted masks → protein-level prediction.
      - OR all ground-truth masks → protein-level ground truth.
      - Compute protein-level metrics.

    Returns:
        (all_results, all_metrics)
        all_results: list of per-protein dicts with uid, fragment details, metrics
        all_metrics: list of per-protein metric dicts (for easy averaging)
    """
    model.eval()
    all_results = []
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running protein-level inference"):
            uid = batch["uids"][0]  # all fragments share the same uid
            batch_size = len(batch["uids"])

            # Move tensors to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            point_mask = batch["point_mask"]
            point_positions = batch["point_positions"] if torch.any(point_mask) else None

            outputs = model(
                protein_input_ids=None,
                protein_attention_mask=batch["protein_attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
                categories=batch["categories"],
                point_positions=point_positions,
                external_esm_embeddings=batch["external_esm_embeddings"]
            )

            pred_masks = outputs["mask_predictions"].cpu()   # (batch_size, seq_len)
            true_masks = batch["residue_labels"].cpu()       # (batch_size, seq_len)
            # seq_mask: valid residues (remove virtual BOS/EOS from attention_mask)
            seq_mask_2d = batch["protein_attention_mask"][:, 1:-1].cpu().bool()  # (batch_size, seq_len)

            # All fragments belong to the same protein → same sequence length
            # OR across fragments to get protein-level masks
            protein_pred = pred_masks[0].clone()
            protein_true = true_masks[0].clone()
            protein_seq_mask = seq_mask_2d[0].clone()

            for b in range(1, batch_size):
                # Sequences are identical (same protein), so seq lengths match
                protein_pred = protein_pred | pred_masks[b]
                protein_true = protein_true | true_masks[b]
                # seq_mask is the same for all fragments of the same protein

            metrics = calculate_protein_metrics(protein_pred, protein_true, protein_seq_mask)
            all_metrics.append(metrics)

            # Per-fragment detail for debugging
            fragment_details = []
            for b in range(batch_size):
                true_indices = torch.where(true_masks[b])[0]
                pred_indices  = torch.where(pred_masks[b])[0]
                fragment_details.append({
                    "category": batch["categories"][b],
                    "point_position": batch["point_positions"][b].item(),
                    "true_start": true_indices[0].item()  if len(true_indices) > 0 else -1,
                    "true_end":   true_indices[-1].item() if len(true_indices) > 0 else -1,
                    "pred_start": pred_indices[0].item()  if len(pred_indices) > 0 else -1,
                    "pred_end":   pred_indices[-1].item() if len(pred_indices) > 0 else -1,
                })

            all_results.append({
                "uid": uid,
                "num_fragments": batch_size,
                "fragments": fragment_details,
                "protein_metrics": metrics,
            })

    return all_results, all_metrics


def main():
    parser = argparse.ArgumentParser(description="ProteinSAM Protein-Level Inference")

    parser.add_argument("--checkpoint_path", type=str,
                        default="./checkpoints_grounding_3B_cluster_70/checkpoint_epoch_40.pt")
    parser.add_argument("--params_file", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="../data_70")
    parser.add_argument("--data_name", type=str, default="VenusX_Dom")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file. Defaults to <checkpoint_dir>/inference_results_protein_level.json")
    parser.add_argument("--esm_embeddings_base_dir", type=str, default=".",
                        help="Base directory under which esm_embeddings/<model>/<data>/ exists.")

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinSAM Protein-Level Inference")
    print("=" * 60)

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))

    if args.params_file is None:
        args.params_file = os.path.join(checkpoint_dir, "protein_sam_init_params.json")
    if args.output_file is None:
        args.output_file = os.path.join(checkpoint_dir, "inference_results_protein_level.json")

    with open(args.params_file, 'r') as f:
        model_params = json.load(f)

    esm_model_path    = model_params.get("esm_model_path")
    llama_model_path  = model_params.get("llama_model_path")
    use_category_cache = model_params.get("use_category_cache", True)

    esm_model_name = os.path.basename(os.path.normpath(esm_model_path))
    data_root_name = os.path.basename(os.path.normpath(args.data_root))
    esm_embeddings_dir = os.path.join(
        args.esm_embeddings_base_dir, "esm_embeddings", esm_model_name, data_root_name
    )
    print(f"   Using ESM embeddings directory: {esm_embeddings_dir}")

    print(f"\n1. Loading datasets: {args.data_name}")
    datasets, collator = get_datasets_and_collator_with_esm_cache_protein_level(
        root_dir=args.data_root,
        data_name=args.data_name,
        esm_model_path=esm_model_path,
        esm_embeddings_dir=esm_embeddings_dir,
        llama_model_path=llama_model_path if not use_category_cache else None,
        max_sequence_length=model_params.get("max_sequence_length", 1021),
        max_text_length=128,
        use_category_cache=use_category_cache,
        null_position_prob=0.0,
        random_position_prob=0.0,
        position_noise_std=0.0,
    )

    test_dataset = datasets["test"]
    sampler = ProteinGroupedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=sampler,   # dynamic batch size: one protein per batch
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    print(f"   Test set: {len(test_dataset)} fragments across {len(sampler)} proteins")

    print(f"\n2. Loading model from checkpoint: {args.checkpoint_path}")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint_path,
        params_file=args.params_file,
        device=args.device,
    )
    print("   Model loaded successfully!")

    print(f"\n3. Running protein-level inference...")
    results, metrics = run_protein_level_inference(model, test_loader, args.device)
    print(f"   Done. Evaluated {len(results)} proteins.")

    avg_metrics = {
        "accuracy":  float(np.mean([m["accuracy"]  for m in metrics])),
        "iou":       float(np.mean([m["iou"]       for m in metrics])),
        "precision": float(np.mean([m["precision"] for m in metrics])),
        "recall":    float(np.mean([m["recall"]    for m in metrics])),
    }

    print(f"\n=== Protein-Level Metrics ===")
    print(f"Mask Accuracy:  {avg_metrics['accuracy']:.4f}")
    print(f"Mask IoU:       {avg_metrics['iou']:.4f}")
    print(f"Precision:      {avg_metrics['precision']:.4f}")
    print(f"Recall:         {avg_metrics['recall']:.4f}")

    output_data = {
        "overall_metrics": avg_metrics,
        "results": results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\n4. Results saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
