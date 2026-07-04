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
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_ROOT,
    ProteinDatasetPrecomputed,
    build_label_map,
    get_architecture_from_metadata,
    get_embedding_dir,
    get_result_dir,
    load_combined_data,
    load_ensemble_models,
    load_metadata,
    parse_dataset_names,
    predict_logits_ensemble,
    residue_iou,
)


def load_test_data_with_dataset_names(dataset_names, data_dir):
    combined = []
    sample_datasets = []
    for dataset_name in dataset_names:
        data = load_combined_data([dataset_name], "test", data_dir=data_dir)
        combined.extend(data)
        sample_datasets.extend([dataset_name] * len(data))
    return combined, sample_datasets


def evaluate_single(args):
    dataset_names = parse_dataset_names(args.datasets)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_dir = args.checkpoint_dir or get_result_dir(args.out_dir, args.data_dir, args.model_path, dataset_names, args.num_ensemble)
    metadata = load_metadata(result_dir)
    threshold = args.threshold if args.threshold is not None else metadata.get("threshold", 0.5)

    train_data = load_combined_data(dataset_names, "train", data_dir=args.data_dir)
    test_data, sample_datasets = load_test_data_with_dataset_names(dataset_names, args.data_dir)
    if args.limit_test is not None:
        test_data = test_data[:args.limit_test]
        sample_datasets = sample_datasets[:args.limit_test]

    label_map = build_label_map(train_data)
    num_classes = len(label_map)
    emb_dir = get_embedding_dir(args.data_dir, args.model_path, dataset_names)
    test_dataset = ProteinDatasetPrecomputed(test_data, label_map, num_classes, os.path.join(emb_dir, "test"), args.max_seq_len)
    _, first_emb, _ = test_dataset[0]
    hidden_dim = first_emb.shape[1]

    models = load_ensemble_models(
        result_dir,
        hidden_dim,
        num_classes,
        get_architecture_from_metadata(metadata),
        device,
        metadata.get("ensemble_size", args.num_ensemble),
    )

    rows = []
    ious_by_category = defaultdict(list)
    ious_by_dataset_category = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating single grounding"):
        uid, emb, _ = test_dataset[idx]
        protein = test_data[idx]
        dataset_name = sample_datasets[idx]
        L = emb.shape[0]
        logits = predict_logits_ensemble(models, emb, device)
        pred_bin = (logits[:L].numpy() > threshold)

        for fg in protein["fragments"]:
            cls_idx = label_map.get(fg["interpro_id"])
            if cls_idx is None:
                continue
            true_mask = np.zeros(L, dtype=bool)
            for frag in fg["frags"]:
                s = frag["start_position"]
                e = min(frag["end_position"] + 1, args.max_seq_len, L)
                if s < L:
                    true_mask[s:e] = True
            pred_mask = pred_bin[:L, cls_idx]
            iou = residue_iou(pred_mask, true_mask)
            if iou is None:
                continue
            ious_by_category[fg["interpro_id"]].append(iou)
            ious_by_dataset_category[dataset_name][fg["interpro_id"]].append(iou)
            rows.append({
                "dataset": dataset_name,
                "uid": uid,
                "category": fg["category"],
                "interpro_id": fg["interpro_id"],
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
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--limit_test", type=int, default=None)
    evaluate_single(parser.parse_args())
