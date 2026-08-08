import argparse
import json
import os
from collections import defaultdict

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
    residue_iou,
)


def evaluate_single(args):
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
    test_samples = load_planned_samples(emb_dir, "test_single", args.max_seq_len)
    if args.limit_test is not None:
        test_samples = test_samples[: args.limit_test]
    if not test_samples:
        raise RuntimeError("No retained Single Grounding test samples")
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

    rows = []
    unknown_classes = 0
    ious_by_category = defaultdict(list)
    ious_by_dataset_category = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    for sample in tqdm(test_samples, desc="Evaluating single grounding"):
        uid = sample["uid"]
        dataset_name = sample["dataset_name"]
        emb = load_sample_embedding(emb_dir, sample)
        L = emb.shape[0]
        logits = predict_logits_ensemble(models, emb, device)
        pred_bin = (logits[:L].numpy() > threshold)
        cls_idx = label_map.get(sample["interpro_id"])
        if cls_idx is None:
            unknown_classes += 1
            continue
        true_mask = np.zeros(L, dtype=bool)
        for start, end in sample["local_spans"]:
            true_mask[int(start) : int(end)] = True
        pred_mask = pred_bin[:L, cls_idx]
        iou = residue_iou(pred_mask, true_mask)
        if iou is None:
            continue
        ious_by_category[sample["interpro_id"]].append(iou)
        ious_by_dataset_category[dataset_name][sample["interpro_id"]].append(iou)
        rows.append({
            "sample_id": sample["sample_id"],
            "dataset": dataset_name,
            "uid": uid,
            "category": sample["category"],
            "interpro_id": sample["interpro_id"],
            "crop_start": sample["crop_start"],
            "crop_end": sample["crop_end"],
            "global_spans": json.dumps(sample["global_spans"]),
            "local_spans": json.dumps(sample["local_spans"]),
            "residue_level_iou": iou,
        })

    category_scores = {
        interpro_id: float(np.mean(values))
        for interpro_id, values in ious_by_category.items()
        if values
    }
    metrics = {
        "residue_level_iou": float(np.mean(list(category_scores.values()))) if category_scores else 0.0,
        "num_samples": len(rows),
        "num_categories": len(category_scores),
        "num_retained_manifest_samples": len(test_samples),
        "num_unknown_class_samples": unknown_classes,
        "num_filtered_envelope": int(
            cache_metadata.get("stats", {})
            .get("test_single", {})
            .get("filtered_envelope", 0)
        ),
        "aggregation": "mean over domain categories",
    }
    dataset_metrics = {}
    for dataset_name, category_values in ious_by_dataset_category.items():
        category_scores_for_dataset = [
            float(np.mean(values))
            for values in category_values.values()
            if values
        ]
        dataset_metrics[dataset_name] = {
            "residue_level_iou": float(np.mean(category_scores_for_dataset)) if category_scores_for_dataset else 0.0,
            "num_categories": len(category_scores_for_dataset),
        }

    output_dir = os.path.join(result_dir, "single_grounding")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "single_grounding_results.csv"), index=False)
    with open(os.path.join(output_dir, "single_grounding_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "single_grounding_category_metrics.json"), "w") as f:
        json.dump(category_scores, f, indent=2)
    with open(os.path.join(output_dir, "single_grounding_dataset_metrics.json"), "w") as f:
        json.dump(dataset_metrics, f, indent=2)

    print(f"residue_level_iou: {metrics['residue_level_iou']}")
    print(f"num_samples: {metrics['num_samples']}")
    print(f"num_categories: {metrics['num_categories']}")
    for dataset_name in dataset_names:
        dataset_iou = dataset_metrics[dataset_name]["residue_level_iou"]
        print(f"{dataset_name}_residue_level_iou: {dataset_iou}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PLM-ENN2 single grounding")
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
    evaluate_single(parser.parse_args())
