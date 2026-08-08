"""Build and validate deduplicated residue-level ESM embedding caches."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from baselines.grounding_alignment import (
    CROP_POLICY_VERSION,
    EMBEDDING_FORMAT_VERSION,
    model_identity,
    write_json,
    write_jsonl,
)


def unique_views(records_by_mode: Dict[str, Sequence[Dict]]) -> Dict[str, Dict]:
    views: Dict[str, Dict] = {}
    for records in records_by_mode.values():
        for record in records:
            view_id = record["view_id"]
            previous = views.get(view_id)
            if previous is not None and previous["sequence"] != record["sequence"]:
                raise AssertionError(f"view hash collision: {view_id}")
            if previous is None:
                views[view_id] = record
            elif not previous.get("legacy_embedding_path") and record.get(
                "legacy_embedding_path"
            ):
                views[view_id] = record
    return views


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _valid_tensor(path: Path, expected_length: int, expected_hidden: int = 1280) -> bool:
    try:
        tensor = _load_tensor(path)
    except Exception:
        return False
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tensor.shape[0] == expected_length
        and tensor.shape[1] == expected_hidden
        and tensor.dtype == torch.float32
    )


def _link(source: Path, target: Path) -> bool:
    if source.resolve() == target.resolve():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        return True
    except OSError:
        return False
    return True


def write_cache_plan(
    cache_base: str | Path,
    records_by_mode: Dict[str, Sequence[Dict]],
    filtered_records: Sequence[Dict],
    metadata: Dict,
) -> Dict:
    cache_base = Path(cache_base)
    cache_base.mkdir(parents=True, exist_ok=True)
    for mode, records in records_by_mode.items():
        write_jsonl(cache_base / mode / "manifest.jsonl", records)
    write_jsonl(cache_base / "filtered_samples.jsonl", filtered_records)

    views = unique_views(records_by_mode)
    total_residues = sum(int(record["cropped_sequence_length"]) for record in views.values())
    preflight = {
        "cache_base": str(cache_base.resolve()),
        "num_samples_by_mode": {
            mode: len(records) for mode, records in records_by_mode.items()
        },
        "num_filtered": len(filtered_records),
        "num_unique_views": len(views),
        "total_unique_residues": total_residues,
        "estimated_fp32_gib_without_reuse": total_residues * 1280 * 4 / 1024**3,
    }
    write_json(cache_base / "storage_preflight.json", preflight)
    write_json(cache_base / "cache_metadata.json", {**metadata, **preflight})
    return preflight


def materialize_cache(
    cache_base: str | Path,
    records_by_mode: Dict[str, Sequence[Dict]],
    model_path: str | Path,
    device: torch.device,
    batch_size: int = 1,
    peer_cache_bases: Iterable[str | Path] = (),
    limit_new_views: int = 0,
    link_only: bool = False,
) -> Dict:
    """Reuse exact tensors where possible, then encode remaining unique views."""

    cache_base = Path(cache_base)
    peers = [Path(path) for path in peer_cache_bases]
    views = unique_views(records_by_mode)
    stats = defaultdict(int)
    pending: List[Dict] = []

    for record in tqdm(views.values(), desc="Resolving cached ESM views"):
        target = cache_base / record["embedding_relative_path"]
        expected_length = int(record["cropped_sequence_length"])
        if target.exists() and _valid_tensor(target, expected_length):
            stats["existing"] += 1
            continue
        if target.exists():
            raise ValueError(f"invalid existing cache object: {target}")

        linked = False
        for peer in peers:
            candidate = peer / record["embedding_relative_path"]
            if candidate.exists() and _valid_tensor(candidate, expected_length):
                linked = _link(candidate, target)
                if linked:
                    stats["linked_peer"] += 1
                    break
        if linked:
            continue

        legacy_value = record.get("legacy_embedding_path")
        if legacy_value:
            legacy_path = Path(legacy_value)
            if legacy_path.exists() and _valid_tensor(legacy_path, expected_length):
                linked = _link(legacy_path, target)
                if linked:
                    stats["linked_legacy"] += 1
        if not linked:
            pending.append(record)

    # Length bucketing substantially reduces padding while preserving a stable,
    # deterministic materialization order. View identity is independent of order.
    pending.sort(key=lambda record: (len(record["sequence"]), record["view_id"]))

    if link_only:
        stats["deferred_link_only"] = len(pending)
        pending = []
    elif limit_new_views > 0:
        deferred = max(0, len(pending) - limit_new_views)
        pending = pending[:limit_new_views]
        stats["deferred_by_limit"] = deferred

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True, add_pooling_layer=False
        ).to(device)
        model.eval()
        for batch_start in tqdm(
            range(0, len(pending), batch_size), desc="Encoding new ESM views"
        ):
            batch = pending[batch_start : batch_start + batch_size]
            sequences = [record["sequence"] for record in batch]
            tokens = tokenizer(
                sequences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            for index, (record, sequence) in enumerate(zip(batch, sequences)):
                residue = hidden[index, 1 : len(sequence) + 1].float().cpu()
                if residue.shape[0] != len(sequence):
                    raise AssertionError(
                        f"ESM residue length mismatch for {record['sample_id']}"
                    )
                target = cache_base / record["embedding_relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
                torch.save(residue, temp)
                os.replace(temp, target)
                stats["encoded"] += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stats["unique_views"] = len(views)
    stats["pending_before_limit"] = (
        len(pending)
        + int(stats["deferred_by_limit"])
        + int(stats["deferred_link_only"])
    )
    result = dict(stats)
    write_json(cache_base / "materialization_stats.json", result)
    return result


def base_cache_metadata(model_path: str | Path, max_seq_len: int) -> Dict:
    return {
        "format_version": EMBEDDING_FORMAT_VERSION,
        "crop_policy_version": CROP_POLICY_VERSION,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_identity": model_identity(model_path),
        "max_seq_len": int(max_seq_len),
        "embedding_dtype": "float32",
        "hidden_size": 1280,
        "special_tokens_stored": False,
    }
