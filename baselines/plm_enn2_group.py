import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from plm_enn2_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDINGS_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_ROOT,
    build_label_map,
    get_architecture_from_metadata,
    get_embedding_dir,
    get_result_dir,
    load_combined_data,
    load_ensemble_models,
    load_metadata,
    load_planned_samples,
    load_sample_embedding,
    parse_dataset_names,
    predict_logits_ensemble,
    protein_fragment_groups_to_masks,
    residue_iou,
)


def evaluate_group(args):
    dataset_names = parse_dataset_names(args.datasets)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_dir = args.checkpoint_dir or get_result_dir(args.out_dir, args.data_dir, args.model_path, dataset_names, args.num_ensemble)
    metadata = load_metadata(result_dir)
    threshold = args.threshold if args.threshold is not None else metadata.get("threshold", 0.5)

    train_data = load_combined_data(dataset_names, "train", data_dir=args.data_dir)
    label_map = build_label_map(train_data)
    num_classes = len(label_map)
    emb_dir = get_embedding_dir(
        args.data_dir,
        args.model_path,
        dataset_names,
        args.max_seq_len,
        args.embeddings_root,
    )
    with open(os.path.join(emb_dir, "cache_metadata.json")) as handle:
        cache_metadata = json.load(handle)
    if int(cache_metadata["max_seq_len"]) != args.max_seq_len:
        raise ValueError("embedding cache max_seq_len does not match evaluation")
    if cache_metadata.get("datasets") != dataset_names:
        raise ValueError("embedding cache datasets do not match evaluation")
    if os.path.abspath(cache_metadata["data_root"]) != os.path.abspath(args.data_dir):
        raise ValueError("embedding cache data_root does not match evaluation")
    if os.path.abspath(cache_metadata["model_path"]) != os.path.abspath(args.model_path):
        raise ValueError("embedding cache model_path does not match evaluation")
    if metadata.get("cache_model_identity") not in {
        None,
        cache_metadata["model_identity"],
    }:
        raise ValueError("checkpoint and embedding cache use different ESM models")
    test_samples = load_planned_samples(emb_dir, "test_group", args.max_seq_len)
    if args.limit_test is not None:
        test_samples = test_samples[: args.limit_test]
    if not test_samples:
        raise RuntimeError("No retained Group Grounding test samples")
    first_emb = load_sample_embedding(emb_dir, test_samples[0])
    hidden_dim = first_emb.shape[1]

    checkpoint_max_len = int(metadata.get("max_seq_len", args.max_seq_len))
    if checkpoint_max_len != args.max_seq_len:
        raise ValueError(
            f"checkpoint max_seq_len={checkpoint_max_len} != requested {args.max_seq_len}"
        )

    models = load_ensemble_models(
        result_dir,
        hidden_dim,
        num_classes,
        get_architecture_from_metadata(metadata),
        device,
        metadata.get("ensemble_size", args.num_ensemble),
    )

    protein_rows = []
    matched_rows = []
    protein_ious_by_dataset = {dataset_name: [] for dataset_name in dataset_names}
    dataset_intersections = {dataset_name: 0 for dataset_name in dataset_names}
    dataset_unions = {dataset_name: 0 for dataset_name in dataset_names}
    for sample in tqdm(test_samples, desc="Evaluating group grounding"):
        uid = sample["uid"]
        dataset_name = sample["dataset_name"]
        emb = load_sample_embedding(emb_dir, sample)
        L = emb.shape[0]
        logits = predict_logits_ensemble(models, emb, device)
        pred_bin = (logits[:L].numpy() > threshold)
        true_masks = protein_fragment_groups_to_masks(
            sample, label_map, L, args.max_seq_len
        )

        pred_classes = set(np.where(pred_bin[:L].any(axis=0))[0].tolist())
        true_classes = set(true_masks.keys())
        evaluated_classes = sorted(pred_classes | true_classes)
        protein_intersection = 0
        protein_union = 0
        for cls_idx in evaluated_classes:
            true_mask = true_masks.get(cls_idx, np.zeros(L, dtype=bool))
            pred_mask = pred_bin[:L, cls_idx]
            iou = residue_iou(pred_mask, true_mask)
            if iou is None:
                continue
            intersection = int(np.logical_and(pred_mask, true_mask).sum())
            union = int(np.logical_or(pred_mask, true_mask).sum())
            protein_intersection += intersection
            protein_union += union
            matched_rows.append({
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "uid": uid,
                "class_idx": cls_idx,
                "residue_level_iou": iou,
                "residue_level_intersection": intersection,
                "residue_level_union": union,
                "has_prediction": bool(pred_mask.any()),
                "has_truth": bool(true_mask.any()),
            })

        protein_iou = float(protein_intersection / protein_union) if protein_union else 0.0
        dataset_intersections[dataset_name] += protein_intersection
        dataset_unions[dataset_name] += protein_union
        protein_ious_by_dataset[dataset_name].append(protein_iou)
        protein_rows.append({
            "sample_id": sample["sample_id"],
            "dataset": dataset_name,
            "uid": uid,
            "crop_start": sample["crop_start"],
            "crop_end": sample["crop_end"],
            "global_spans": json.dumps(sample["global_spans"]),
            "local_spans": json.dumps(sample["local_spans"]),
            "residue_level_iou": protein_iou,
            "residue_level_intersection": protein_intersection,
            "residue_level_union": protein_union,
            "num_true_domains": len(true_masks),
            "num_predicted_domains": len(pred_classes),
            "num_evaluated_domains": len(evaluated_classes),
        })

    total_intersection = sum(dataset_intersections.values())
    total_union = sum(dataset_unions.values())
    metrics = {
        "residue_level_iou": float(np.mean([row["residue_level_iou"] for row in protein_rows])) if protein_rows else 0.0,
        "global_residue_level_iou": float(total_intersection / total_union) if total_union else 0.0,
        "residue_level_intersection": int(total_intersection),
        "residue_level_union": int(total_union),
        "num_proteins": len(protein_rows),
        "num_evaluated_domains": len(matched_rows),
        "num_retained_manifest_samples": len(test_samples),
        "num_filtered_envelope": int(
            cache_metadata.get("stats", {})
            .get("test_group", {})
            .get("filtered_envelope", 0)
        ),
        "aggregation": "per-protein IoU over the union of predicted and true label-residue masks; global_residue_level_iou is summed intersection over summed union",
    }
    dataset_metrics = {}
    for dataset_name, values in protein_ious_by_dataset.items():
        dataset_union = dataset_unions[dataset_name]
        dataset_metrics[dataset_name] = {
            "residue_level_iou": float(np.mean(values)) if values else 0.0,
            "global_residue_level_iou": float(dataset_intersections[dataset_name] / dataset_union) if dataset_union else 0.0,
            "residue_level_intersection": int(dataset_intersections[dataset_name]),
            "residue_level_union": int(dataset_union),
            "num_proteins": len(values),
        }

    output_dir = os.path.join(result_dir, "group_grounding")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(protein_rows).to_csv(os.path.join(output_dir, "group_grounding_results.csv"), index=False)
    pd.DataFrame(matched_rows).to_csv(os.path.join(output_dir, "group_grounding_matched_domains.csv"), index=False)
    with open(os.path.join(output_dir, "group_grounding_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "group_grounding_dataset_metrics.json"), "w") as f:
        json.dump(dataset_metrics, f, indent=2)

    print(f"residue_level_iou: {metrics['residue_level_iou']}")
    print(f"global_residue_level_iou: {metrics['global_residue_level_iou']}")
    print(f"num_proteins: {metrics['num_proteins']}")
    print(f"num_evaluated_domains: {metrics['num_evaluated_domains']}")
    for dataset_name in dataset_names:
        dataset_iou = dataset_metrics[dataset_name]["residue_level_iou"]
        dataset_global_iou = dataset_metrics[dataset_name]["global_residue_level_iou"]
        print(f"{dataset_name}_residue_level_iou: {dataset_iou}")
        print(f"{dataset_name}_global_residue_level_iou: {dataset_global_iou}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PLM-ENN2 group grounding")
    parser.add_argument("--datasets", type=str, default="Act,BindI,Dom,Evo,Motif")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=1021)
    parser.add_argument("--embeddings_root", type=str, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit_test", type=int, default=None)
    evaluate_group(parser.parse_args())
