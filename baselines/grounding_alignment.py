"""Shared deterministic data alignment and ESM-cache helpers for baselines."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


CROP_POLICY_VERSION = "envelope_center_v1"
EMBEDDING_FORMAT_VERSION = "residue_last_hidden_state_v1"


@dataclass(frozen=True)
class CropPlan:
    crop_start: int
    crop_end: int
    envelope_start: int
    envelope_end: int

    @property
    def crop_length(self) -> int:
        return self.crop_end - self.crop_start

    @property
    def envelope_length(self) -> int:
        return self.envelope_end - self.envelope_start + 1


def plan_center_crop(
    sequence_length: int,
    spans_inclusive: Sequence[Tuple[int, int]],
    max_length: int,
) -> CropPlan | None:
    """Return a fixed, envelope-centred crop or ``None`` if it cannot fit."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if not spans_inclusive:
        raise ValueError("at least one span is required")

    normalized = [(int(start), int(end)) for start, end in spans_inclusive]
    for start, end in normalized:
        if start < 0 or end < start or end >= sequence_length:
            raise ValueError(
                f"invalid inclusive span {(start, end)} for sequence length {sequence_length}"
            )

    envelope_start = min(start for start, _ in normalized)
    envelope_end = max(end for _, end in normalized)
    envelope_length = envelope_end - envelope_start + 1
    if envelope_length > max_length:
        return None

    if sequence_length <= max_length:
        crop_start = 0
        crop_end = sequence_length
    else:
        left_context = (max_length - envelope_length) // 2
        crop_start = envelope_start - left_context
        crop_start = max(0, min(crop_start, sequence_length - max_length))
        crop_end = crop_start + max_length

    return CropPlan(
        crop_start=crop_start,
        crop_end=crop_end,
        envelope_start=envelope_start,
        envelope_end=envelope_end,
    )


def localize_spans(
    spans_inclusive: Sequence[Tuple[int, int]], crop: CropPlan
) -> List[Tuple[int, int]]:
    """Convert global inclusive spans into local half-open spans."""

    localized = [
        (int(start) - crop.crop_start, int(end) - crop.crop_start + 1)
        for start, end in spans_inclusive
    ]
    for start, end in localized:
        if start < 0 or end <= start or end > crop.crop_length:
            raise AssertionError(
                f"localized span {(start, end)} falls outside crop length {crop.crop_length}"
            )
    return localized


def crop_sequence(sequence: str, crop: CropPlan) -> str:
    cropped = sequence[crop.crop_start : crop.crop_end]
    if len(cropped) != crop.crop_length:
        raise AssertionError("cropped sequence length does not match the crop plan")
    return cropped


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def model_identity(model_path: str | Path) -> str:
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json"
    config_hash = "missing"
    if config_path.exists():
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return f"{path}:{config_hash}"


def make_view_id(sequence: str, model_path: str | Path, dtype: str) -> str:
    payload = {
        "embedding_format": EMBEDDING_FORMAT_VERSION,
        "model_identity": model_identity(model_path),
        "dtype": dtype,
        "sequence_sha256": sequence_sha256(sequence),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def object_relative_path(view_id: str) -> str:
    return str(Path("objects") / view_id[:2] / f"{view_id}.pt")


def attach_view_identity(
    record: Dict, model_path: str | Path, dtype: str = "float32"
) -> Dict:
    sequence = record["sequence"]
    view_id = make_view_id(sequence, model_path, dtype)
    record = dict(record)
    record["view_id"] = view_id
    record["sequence_hash"] = sequence_sha256(sequence)
    record["embedding_relative_path"] = object_relative_path(view_id)
    record["embedding_dtype"] = dtype
    record["crop_policy_version"] = CROP_POLICY_VERSION
    return record


def cache_dir(
    cache_root: str | Path,
    data_root: str | Path,
    model_path: str | Path,
    dataset_names: Sequence[str],
    max_length: int,
) -> Path:
    data_tag = Path(data_root).resolve().name
    model_tag = Path(model_path).resolve().name
    dataset_tag = "+".join(dataset_names)
    return (
        Path(cache_root)
        / data_tag
        / model_tag
        / dataset_tag
        / f"max_len_{int(max_length)}"
    )


def write_jsonl(path: str | Path, records: Iterable[Dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            serializable = {key: value for key, value in record.items() if key != "sequence"}
            handle.write(json.dumps(serializable, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temp_path, path)
    return count


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def load_cached_embedding(cache_base: str | Path, record: Dict) -> torch.Tensor:
    path = Path(cache_base) / record["embedding_relative_path"]
    if not path.exists():
        raise FileNotFoundError(f"missing precomputed embedding: {path}")
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"invalid embedding tensor in {path}")
    expected_length = int(record["cropped_sequence_length"])
    if tensor.shape[0] != expected_length:
        raise ValueError(
            f"embedding length mismatch for {record['sample_id']}: "
            f"tensor={tensor.shape[0]} manifest={expected_length}"
        )
    return tensor.float()


def validate_manifest_record(record: Dict, max_length: int) -> None:
    original_length = int(record["original_sequence_length"])
    crop_start = int(record["crop_start"])
    crop_end = int(record["crop_end"])
    crop_length = int(record["cropped_sequence_length"])
    if crop_start < 0 or crop_end > original_length or crop_end <= crop_start:
        raise AssertionError(f"invalid crop for {record['sample_id']}")
    if crop_end - crop_start != crop_length:
        raise AssertionError(f"crop length mismatch for {record['sample_id']}")
    if original_length > max_length and crop_length != max_length:
        raise AssertionError(f"long sequence has a non-fixed crop for {record['sample_id']}")
    if original_length <= max_length and crop_length != original_length:
        raise AssertionError(f"short sequence was altered for {record['sample_id']}")
    for global_span, local_span in zip(record["global_spans"], record["local_spans"]):
        global_start, global_end = map(int, global_span)
        local_start, local_end = map(int, local_span)
        if local_start + crop_start != global_start:
            raise AssertionError(f"start coordinate mismatch for {record['sample_id']}")
        if local_end + crop_start - 1 != global_end:
            raise AssertionError(f"end coordinate mismatch for {record['sample_id']}")
        if not (0 <= local_start < local_end <= crop_length):
            raise AssertionError(f"local span outside crop for {record['sample_id']}")
