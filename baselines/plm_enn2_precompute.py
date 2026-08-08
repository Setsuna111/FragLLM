#!/usr/bin/env python3
"""Plan aligned PLM-ENN2 samples and precompute their cropped ESM embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aligned_esm_cache import (
    base_cache_metadata,
    materialize_cache,
    write_cache_plan,
)
from plm_enn2_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDINGS_ROOT,
    DEFAULT_MODEL_PATH,
    LEGACY_EMBEDDINGS_ROOT,
    get_embedding_dir,
    load_combined_data,
    parse_dataset_names,
    plan_group_samples,
    plan_single_samples,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="Act,BindI,Dom,Evo,Motif")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embeddings_root", default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--legacy_embeddings_root", default=LEGACY_EMBEDDINGS_ROOT)
    parser.add_argument("--peer_cache_bases", default="")
    parser.add_argument("--max_seq_len", type=int, default=1021)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--link_only", action="store_true")
    parser.add_argument("--limit_raw_records", type=int, default=0)
    parser.add_argument("--limit_new_views", type=int, default=0)
    return parser.parse_args()


def add_legacy_candidates(records, args, dataset_names):
    legacy_base = (
        Path(args.legacy_embeddings_root)
        / Path(args.data_dir).resolve().name
        / Path(args.model_path).resolve().name
        / "+".join(dataset_names)
    )
    for record in records:
        if (
            int(record["crop_start"]) == 0
            and int(record["cropped_sequence_length"])
            == int(record["original_sequence_length"])
        ):
            record["legacy_embedding_path"] = str(
                legacy_base / record["split"] / f"{record['uid']}.pt"
            )


def main():
    args = parse_args()
    dataset_names = parse_dataset_names(args.datasets)
    cache_base = Path(
        get_embedding_dir(
            args.data_dir,
            args.model_path,
            dataset_names,
            args.max_seq_len,
            args.embeddings_root,
        )
    )

    train_data = load_combined_data(dataset_names, "train", args.data_dir)
    test_data = load_combined_data(dataset_names, "test", args.data_dir)
    if args.limit_raw_records > 0:
        train_data = train_data[: args.limit_raw_records]
        test_data = test_data[: args.limit_raw_records]

    train_group, filtered_train_group, train_stats = plan_group_samples(
        train_data, "train", args.max_seq_len, args.model_path
    )
    test_group, filtered_test_group, test_group_stats = plan_group_samples(
        test_data, "test", args.max_seq_len, args.model_path
    )
    test_single, filtered_test_single, test_single_stats = plan_single_samples(
        test_data, "test", args.max_seq_len, args.model_path
    )
    records_by_mode = {
        "train_group": train_group,
        "test_group": test_group,
        "test_single": test_single,
    }
    for records in records_by_mode.values():
        add_legacy_candidates(records, args, dataset_names)

    filtered = filtered_train_group + filtered_test_group + filtered_test_single
    metadata = {
        **base_cache_metadata(args.model_path, args.max_seq_len),
        "method": "plm_enn2",
        "data_root": str(Path(args.data_dir).resolve()),
        "datasets": dataset_names,
        "stats": {
            "train_group": train_stats,
            "test_group": test_group_stats,
            "test_single": test_single_stats,
        },
    }
    preflight = write_cache_plan(cache_base, records_by_mode, filtered, metadata)
    print(json.dumps({"preflight": preflight}, indent=2))
    if args.manifest_only:
        return
    if not args.link_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to encode missing ESM views")
    peer_bases = [value for value in args.peer_cache_bases.split(",") if value]
    result = materialize_cache(
        cache_base,
        records_by_mode,
        args.model_path,
        torch.device(args.device if torch.cuda.is_available() else "cpu"),
        batch_size=args.batch_size,
        peer_cache_bases=peer_bases,
        limit_new_views=args.limit_new_views,
        link_only=args.link_only,
    )
    print(json.dumps({"materialization": result}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
