#!/usr/bin/env python3
"""Three-view reference-class inference with exact-match/Qwen fallback decoding.

The ``generate`` stage delegates to the existing reference-class TTA
implementation so that the model inference path remains unchanged.  The
``aggregate`` stage first extracts a category from each of the three generated
answers.  An InterPro ID is accepted directly only when all three normalized
categories are identical and map to one ontology entry.  Every other sample
uses the existing three-view mean Qwen embedding retrieval.  Metrics compare
the resulting InterPro IDs directly with the target IDs.
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate_reference_class_lfj_tta import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATA,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    SUPPORTED_DATASETS,
    calculate_direct_id_metrics,
    comma_separated_datasets,
    comma_separated_seeds,
    generate_views as generate_three_views,
)


DEFAULT_ONTOLOGY = Path(
    "/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json"
)

# These prefixes cover the answer templates used by the reference-class model.
# Matching is done after NFKC/case/whitespace normalization and terminal
# punctuation removal, so the exact stage is insensitive to harmless formatting.
ANSWER_PREFIXES = (
    "the appropriate category for this is ",
    "it is categorized as ",
    "its class designation is ",
    "this belongs to the ",
    "the category is ",
    "it is classified as ",
    "it belongs to the ",
    "it is an? ",
    "it is the ",
    "it is ",
)


def normalize_category(text: object) -> str:
    """Normalize generated/ontology category text for exact matching."""

    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n.?!;:\"'")


def extract_category(text: object) -> str:
    """Remove a known answer prefix and return the normalized category."""

    normalized = normalize_category(text)
    for prefix in ANSWER_PREFIXES:
        match = re.fullmatch(prefix + r"(.+)", normalized)
        if match:
            return match.group(1).strip(" \t\r\n.?!;:\"'")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-class TTA with exact-match then Qwen fallback"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    generate = subparsers.add_parser("generate", help="Generate three greedy views")
    generate.add_argument("--model_path", default=str(DEFAULT_MODEL))
    generate.add_argument("--root_dir", default=str(DEFAULT_DATA))
    generate.add_argument("--split", default="test")
    generate.add_argument(
        "--datasets",
        type=comma_separated_datasets,
        default=["ActRefClass"],
        help=f"Comma-separated datasets chosen from {list(SUPPORTED_DATASETS)}",
    )
    generate.add_argument("--output_dir", required=True)
    generate.add_argument("--seeds", type=comma_separated_seeds, default=[42, 43, 44])
    generate.add_argument("--batch_per_device", type=int, default=4)
    generate.add_argument("--gpu_id", type=int, default=0)
    generate.add_argument("--perceiver_latent_size", type=int, default=4)
    generate.add_argument("--max_new_tokens", type=int, default=512)
    generate.add_argument("--limit", type=int)
    # Keep the existing implementation's behavior: progress is enabled unless
    # callers explicitly alter the source implementation's argument handling.
    generate.add_argument("--show_progress", default=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="Exact-match three views, then Qwen fallback"
    )
    aggregate.add_argument("--inputs", nargs=3, required=True)
    aggregate.add_argument("--output_path", required=True)
    aggregate.add_argument("--details_path", required=True)
    aggregate.add_argument("--config_path", required=True)
    aggregate.add_argument(
        "--metrics_path",
        help="Optional direct-ID metrics JSON path; defaults beside output_path",
    )
    aggregate.add_argument(
        "--comparison_path",
        help="Optional JSON path comparing these metrics with a previous TTA run",
    )
    aggregate.add_argument("--baseline_metrics_path")
    aggregate.add_argument("--dataset_name", choices=SUPPORTED_DATASETS)
    aggregate.add_argument("--embedding_model", default=str(DEFAULT_EMBEDDING_MODEL))
    aggregate.add_argument("--cache_path", default=str(DEFAULT_CACHE))
    aggregate.add_argument("--ontology_path", default=str(DEFAULT_ONTOLOGY))
    aggregate.add_argument("--batch_size", type=int, default=32)
    aggregate.add_argument("--device", default="cuda")

    compare = subparsers.add_parser(
        "compare", help="Compare new and previous direct-ID metric JSON files"
    )
    compare.add_argument("--new_dir", required=True)
    compare.add_argument("--baseline_dir", required=True)
    compare.add_argument("--datasets", type=comma_separated_datasets, required=True)
    compare.add_argument("--output_path", required=True)

    return parser.parse_args()


def validate_view_frames(frames) -> None:
    required = {"generated", "reference", "dataset_idx", "interpro_ids"}
    for path, frame in frames:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    base = frames[0][1]
    for path, frame in frames[1:]:
        if not base["dataset_idx"].equals(frame["dataset_idx"]):
            raise ValueError(f"dataset_idx is not aligned in {path}")
        if not base["interpro_ids"].astype(str).equals(frame["interpro_ids"].astype(str)):
            raise ValueError(f"target interpro_ids are not aligned in {path}")


def load_unique_category_map(ontology_path: str, valid_ids) -> dict:
    """Return normalized category -> unique ID, omitting ambiguous aliases."""

    ontology = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    valid_ids = {str(value) for value in valid_ids}
    candidates = {}
    for interpro_id, entry in ontology.items():
        interpro_id = str(interpro_id)
        if interpro_id not in valid_ids:
            continue
        category = normalize_category(entry.get("category", ""))
        if category:
            candidates.setdefault(category, set()).add(interpro_id)
    # A normalized category mapping to multiple IDs is ambiguous and must use
    # the Qwen fallback rather than selecting an arbitrary label.
    return {
        category: next(iter(ids))
        for category, ids in candidates.items()
        if len(ids) == 1
    }


def qwen_mean_retrieval(
    frames,
    cache_path: str,
    embedding_model: str,
    batch_size: int,
    device: str,
):
    """Replicate the existing three-view mean Qwen retrieval stage."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    label_embeddings = np.asarray(cache["embeddings"])
    interpro_ids = [str(value) for value in cache["interpro_ids"]]
    if label_embeddings.ndim != 2 or label_embeddings.shape[0] != len(interpro_ids):
        raise ValueError("Cached label embeddings and InterPro IDs have different lengths")

    text_views = [frame["generated"].fillna("").astype(str).tolist() for _, frame in frames]
    all_texts = [text for view in text_views for text in view]
    unique_texts = list(dict.fromkeys(all_texts))
    model = SentenceTransformer(embedding_model, device=device)
    text_embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    similarities = model.similarity(text_embeddings, label_embeddings).cpu().numpy()
    scores_by_text = dict(zip(unique_texts, similarities))
    view_scores = [
        np.stack([scores_by_text[text] for text in texts])
        for texts in text_views
    ]
    mean_scores = np.stack(view_scores).mean(axis=0)
    winners = mean_scores.argmax(axis=1)
    fallback_ids = [interpro_ids[int(index)] for index in winners]
    top2 = np.partition(mean_scores, -2, axis=1)[:, -2:]
    top1_scores = mean_scores[np.arange(len(mean_scores)), winners]
    margins = top1_scores - top2.min(axis=1)
    return fallback_ids, top1_scores, margins, len(unique_texts), interpro_ids


def aggregate_views(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    started_at = time.perf_counter()
    frames = [(Path(path), pd.read_csv(path)) for path in args.inputs]
    validate_view_frames(frames)
    base = frames[0][1].copy()

    fallback_ids, fallback_scores, fallback_margins, unique_text_count, cache_ids = (
        qwen_mean_retrieval(
            frames,
            args.cache_path,
            args.embedding_model,
            args.batch_size,
            args.device,
        )
    )
    exact_map = load_unique_category_map(args.ontology_path, cache_ids)
    extracted_views = [
        [extract_category(value) for value in frame["generated"].fillna("")]
        for _, frame in frames
    ]

    predictions = []
    methods = []
    exact_ids = []
    exact_categories = []
    for row_index in range(len(base)):
        categories = [view[row_index] for view in extracted_views]
        category = categories[0] if len(set(categories)) == 1 else ""
        exact_id = exact_map.get(category) if category else None
        if exact_id is not None:
            predictions.append(exact_id)
            methods.append("exact_unanimous")
            exact_ids.append(exact_id)
            exact_categories.append(category)
        else:
            predictions.append(fallback_ids[row_index])
            methods.append("qwen_mean_fallback")
            exact_ids.append("")
            exact_categories.append(category)

    result = pd.DataFrame(
        {
            "dataset_idx": base["dataset_idx"],
            "predicted_interpro_id": predictions,
            "interpro_ids": base["interpro_ids"],
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    metrics_path = Path(args.metrics_path) if args.metrics_path else output_path.with_name(
        f"{args.dataset_name or output_path.stem.removesuffix('_results')}_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = calculate_direct_id_metrics(
        result["predicted_interpro_id"].tolist(),
        result["interpro_ids"].tolist(),
    )
    metrics_payload = {
        "stage": "direct_id_metrics",
        "dataset": args.dataset_name,
        "results_path": str(output_path.resolve()),
        "prediction_column": "predicted_interpro_id",
        "target_column": "interpro_ids",
        "uses_embedding_for_metric": False,
        "decoding": "unanimous_normalized_exact_category_else_three_view_mean_qwen",
        "metrics": metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    details = {
        "dataset_idx": base["dataset_idx"],
        "target_interpro_id": base["interpro_ids"],
        "predicted_interpro_id": predictions,
        "selection_method": methods,
        "exact_category": exact_categories,
        "exact_interpro_id": exact_ids,
        "qwen_fallback_interpro_id": fallback_ids,
        "qwen_fallback_top1_score": fallback_scores,
        "qwen_fallback_top1_margin": fallback_margins,
    }
    for view_index, ((path, frame), extracted) in enumerate(
        zip(frames, extracted_views)
    ):
        details[f"view_{view_index}_source"] = str(path)
        details[f"view_{view_index}_generated"] = frame["generated"]
        details[f"view_{view_index}_extracted_category"] = extracted
    details_path = Path(args.details_path)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(details_path, index=False)

    exact_count = methods.count("exact_unanimous")
    fallback_count = methods.count("qwen_mean_fallback")
    config = {
        "stage": "aggregate",
        "method": "unanimous_normalized_exact_category_else_three_view_mean_qwen",
        "dataset": args.dataset_name,
        "inputs": [str(path.resolve()) for path, _ in frames],
        "output_path": str(output_path.resolve()),
        "details_path": str(details_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "ontology_path": str(Path(args.ontology_path).resolve()),
        "embedding_model": str(Path(args.embedding_model).resolve()),
        "cache_path": str(Path(args.cache_path).resolve()),
        "batch_size": args.batch_size,
        "device": args.device,
        "num_samples": len(result),
        "unique_generated_texts": unique_text_count,
        "exact_unanimous_samples": exact_count,
        "qwen_fallback_samples": fallback_count,
        "exact_coverage": exact_count / len(result) if len(result) else 0.0,
        "uses_target_for_selection": False,
        "result_columns": list(result.columns),
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started_at,
    }
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.baseline_metrics_path:
        baseline_path = Path(args.baseline_metrics_path)
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_metrics = baseline_payload.get("metrics", baseline_payload)
        comparison = build_comparison(args.dataset_name, metrics, baseline_metrics)
        comparison_path = (
            Path(args.comparison_path)
            if args.comparison_path
            else output_path.with_name(
                f"{args.dataset_name or output_path.stem.removesuffix('_results')}_comparison.json"
            )
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved comparison to {comparison_path}")

    print(
        f"{args.dataset_name or output_path.stem}: exact={exact_count}, "
        f"qwen_fallback={fallback_count}, metrics={json.dumps(metrics, ensure_ascii=False)}"
    )
    print(f"Saved final result to {output_path}")


def build_comparison(dataset: str, new_metrics: dict, baseline_metrics: dict) -> dict:
    metric_names = ("accuracy", "recall", "precision", "f1", "mcc")
    return {
        "dataset": dataset,
        "baseline": {name: baseline_metrics.get(name) for name in metric_names},
        "new": {name: new_metrics.get(name) for name in metric_names},
        "delta_new_minus_baseline": {
            name: float(new_metrics[name] - baseline_metrics[name])
            for name in metric_names
            if new_metrics.get(name) is not None and baseline_metrics.get(name) is not None
        },
        "correct_delta": int(new_metrics["correct"] - baseline_metrics["correct"])
        if new_metrics.get("correct") is not None and baseline_metrics.get("correct") is not None
        else None,
        "samples": new_metrics.get("samples"),
    }


def compare_runs(args: argparse.Namespace) -> None:
    import pandas as pd

    rows = []
    for dataset in args.datasets:
        new_path = Path(args.new_dir) / f"{dataset}_metrics.json"
        baseline_path = Path(args.baseline_dir) / f"{dataset}_metrics.json"
        if not new_path.exists():
            raise FileNotFoundError(f"Missing new metrics file: {new_path}")
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline metrics file: {baseline_path}")
        new_payload = json.loads(new_path.read_text(encoding="utf-8"))
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = build_comparison(
            dataset,
            new_payload.get("metrics", new_payload),
            baseline_payload.get("metrics", baseline_payload),
        )
        row = {"dataset": dataset}
        for name, value in comparison["baseline"].items():
            row[f"baseline_{name}"] = value
        for name, value in comparison["new"].items():
            row[f"new_{name}"] = value
        for name, value in comparison["delta_new_minus_baseline"].items():
            row[f"delta_{name}"] = value
        row["baseline_correct"] = baseline_payload.get("metrics", baseline_payload).get("correct")
        row["new_correct"] = new_payload.get("metrics", new_payload).get("correct")
        row["correct_delta"] = comparison["correct_delta"]
        rows.append(row)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison CSV to {output_path}")
    print(f"Saved comparison JSON to {json_path}")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.stage == "generate":
        # This is the unchanged inference implementation from the established
        # TTA script; only aggregation/decoding is specialized in this file.
        generate_three_views(args)
    elif args.stage == "aggregate":
        aggregate_views(args)
    elif args.stage == "compare":
        compare_runs(args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
