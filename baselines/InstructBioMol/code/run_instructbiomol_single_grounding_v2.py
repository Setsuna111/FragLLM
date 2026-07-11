import argparse
import csv
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from instructbiomol_reference_common_v2 import (  # noqa: E402
    FoldseekRunner,
    add_common_args,
    batched,
    full_structure_path,
    generation_kwargs,
    load_instructbiomol_model,
    make_output_dir,
    resolve_data_root,
)


SINGLE_DATASETS = (
    "ActGroundSingle",
    "BindIGroundSingle",
    "DomGroundSingle",
    "EvoGroundSingle",
    "MotifGroundSingle",
)

DATASET_TO_SOURCE = {
    "ActGroundSingle": ("VenusX_Act", "Act", "active site"),
    "BindIGroundSingle": ("VenusX_BindI", "BindI", "binding site"),
    "DomGroundSingle": ("VenusX_Dom", "Dom", "functional domain"),
    "EvoGroundSingle": ("VenusX_Evo", "Evo", "evolutionary conserved site"),
    "MotifGroundSingle": ("VenusX_Motif", "Motif", "motif domain"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InstructBioMol on single grounding datasets."
    )
    add_common_args(parser, default_datasets=",".join(SINGLE_DATASETS))
    parser.set_defaults(data_dir="data_70", max_new_tokens=256)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build datasets and metrics inputs without loading the model.",
    )
    return parser.parse_args()


def parse_dataset_list(datasets: str) -> List[str]:
    names = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = [name for name in names if name not in SINGLE_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed datasets: {list(SINGLE_DATASETS)}")
    if not names:
        raise ValueError("At least one dataset must be selected")
    return names


def sorted_regions(frags: Sequence[Dict], offset: int = 0) -> List[Tuple[int, int]]:
    regions = [
        (int(frag["start_position"]) - offset, int(frag["end_position"]) - offset)
        for frag in frags
    ]
    regions.sort(key=lambda pair: pair[0])
    return regions


def format_regions(regions: Sequence[Tuple[int, int]]) -> str:
    return "[" + ",".join(f"[{start},{end}]" for start, end in regions) + "]"


def build_single_prompt(sample: Dict) -> str:
    return (
        f"Locate every contiguous residue region corresponding to the {sample['task_name']} "
        f"category \"{sample['category']}\" in the protein. "
        f"The protein sequence has length {len(sample['sequence'])}. "
        f"Protein sequence: {sample['sequence']}. "
        "Use 0-based inclusive residue coordinates. Return only ASCII text in this format: "
        "regions: (start,end); (start,end). "
        "Use a single pair for one region. If no region is found, return exactly: regions: none. "
        "Do not use XML tags, JSON tags, angle brackets, or explanatory text."
    )


def build_single_reference(sample: Dict) -> str:
    if not sample["regions"]:
        return "regions: none"
    return "regions: " + "; ".join(f"({start},{end})" for start, end in sample["regions"])


def load_single_grounding_samples(
    data_root: Path,
    dataset_name: str,
    split: str,
    limit: Optional[int],
) -> List[Dict]:
    if dataset_name not in DATASET_TO_SOURCE:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    source_name, source_dataset, task_name = DATASET_TO_SOURCE[dataset_name]
    json_path = data_root / source_name / f"{split}.json"
    with json_path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    samples: List[Dict] = []
    for item in raw_items:
        for fragment_group in item["fragments"]:
            regions = sorted_regions(fragment_group["frags"])
            if not regions:
                continue
            sample = {
                "uid": item["uid"],
                "sequence": item["sequence"],
                "category": fragment_group["category"],
                "description": fragment_group.get("description", ""),
                "interpro_id": fragment_group["interpro_id"],
                "shortname": fragment_group.get("shortname", ""),
                "task_name": task_name,
                "source_dataset": source_dataset,
                "dataset_name": dataset_name,
                "regions": regions,
                "dataset_idx": len(samples),
            }
            sample["prompt_question"] = build_single_prompt(sample)
            sample["reference"] = build_single_reference(sample)
            sample["gt_positions"] = regions
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def prepare_single_batch(
    samples: Sequence[Dict],
    foldseek: FoldseekRunner,
) -> Tuple[Tuple[List[str], List[str]], Dict]:
    input_sequences: List[str] = []
    saprot_sequences: List[str] = []
    instructions: List[str] = []
    references: List[str] = []
    ids: List[str] = []
    structure_paths: List[str] = []

    for sample in samples:
        sequence = sample["sequence"]
        structure_path = full_structure_path(
            sample["source_dataset"], sample["interpro_id"], sample["uid"]
        )
        input_sequences.append(sequence)
        saprot_sequences.append(foldseek.sequence_for_structure(structure_path, sequence))
        instructions.append(sample["prompt_question"])
        references.append(sample["reference"])
        ids.append(str(sample["dataset_idx"]))
        structure_paths.append(str(structure_path))

    inputs = {
        "input_seqs": input_sequences,
        "target_seqs": references,
        "input_enc_seqs": input_sequences,
        "input_enc_fps": [[] for _ in input_sequences],
        "input_modality": "protein",
        "target_modality": "text",
        "instructions": instructions,
        "ids": ids,
        "data_name": "single_grounding",
        "structure_paths": structure_paths,
    }
    return (input_sequences, saprot_sequences), inputs


def extract_loc_payload(text: str) -> Optional[Dict]:
    loc_match = re.search(r"<loc>\s*(\{.*?\})\s*</loc>", text, flags=re.DOTALL)
    candidates = [loc_match.group(1)] if loc_match else []
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def valid_pair(value) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end < 0:
        return None
    if end < start:
        start, end = end, start
    return (start, end)


def extract_positions(text: str) -> List[Tuple[int, int]]:
    payload = extract_loc_payload(text)
    positions: List[Tuple[int, int]] = []
    if payload is not None:
        for pair in payload.get("regions", []):
            parsed = valid_pair(pair)
            if parsed is not None:
                positions.append(parsed)
    if positions:
        return positions

    for start, end in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text):
        parsed = valid_pair((start, end))
        if parsed is not None:
            positions.append(parsed)
    if positions:
        return positions

    for start, end in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text):
        parsed = valid_pair((start, end))
        if parsed is not None:
            positions.append(parsed)
    if positions:
        return positions
    for start, end in re.findall(
        r"(?:start|from)\D{0,12}(\d+)\D{0,20}(?:end|to)\D{0,12}(\d+)",
        text,
        flags=re.IGNORECASE,
    ):
        parsed = valid_pair((start, end))
        if parsed is not None:
            positions.append(parsed)
    return positions


def compute_iou_single(pos_pre: Tuple[int, int], pos_ref: Tuple[int, int]) -> float:
    inter_left = max(pos_pre[0], pos_ref[0])
    inter_right = min(pos_pre[1], pos_ref[1])
    intersection = max(0, inter_right - inter_left + 1)
    union_left = min(pos_pre[0], pos_ref[0])
    union_right = max(pos_pre[1], pos_ref[1])
    union = union_right - union_left + 1
    return intersection / union if union > 0 else 0


def match_positions(
    prediction_positions: Sequence[Tuple[int, int]],
    reference_positions: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if not prediction_positions or not reference_positions:
        return [], []
    dist_matrix = [[0.0 for _ in reference_positions] for _ in prediction_positions]
    for i, pred_pos in enumerate(prediction_positions):
        for j, ref_pos in enumerate(reference_positions):
            dist_matrix[i][j] = compute_iou_single(pred_pos, ref_pos)

    matched_pred_positions = []
    matched_ref_positions = []
    while any(value != -1 for row in dist_matrix for value in row):
        max_i, max_j, max_value = 0, 0, -1.0
        for i, row in enumerate(dist_matrix):
            for j, value in enumerate(row):
                if value > max_value:
                    max_i, max_j, max_value = i, j, value
        matched_pred_positions.append(prediction_positions[max_i])
        matched_ref_positions.append(reference_positions[max_j])
        for j in range(len(reference_positions)):
            dist_matrix[max_i][j] = -1
        for i in range(len(prediction_positions)):
            dist_matrix[i][max_j] = -1
    return matched_pred_positions, matched_ref_positions


def compute_interval_iou(
    positions_pre: Sequence[Tuple[int, int]],
    positions_ref: Sequence[Tuple[int, int]],
) -> Tuple[float, int, int]:
    iou = 0.0
    unions = 0
    intersections = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        inter_left = max(pos_pre[0], pos_ref[0])
        inter_right = min(pos_pre[1], pos_ref[1])
        intersection = max(0, inter_right - inter_left + 1)
        union_left = min(pos_pre[0], pos_ref[0])
        union_right = max(pos_pre[1], pos_ref[1])
        union = union_right - union_left + 1
        unions += union
        intersections += intersection
        iou += intersection / union if union > 0 else 0
    if positions_pre:
        iou /= len(positions_pre)
    return iou, unions, intersections


def compute_distance(
    positions_pre: Sequence[Tuple[int, int]],
    positions_ref: Sequence[Tuple[int, int]],
) -> float:
    distance = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        distance += abs(pos_pre[0] - pos_ref[0]) + abs(pos_pre[1] - pos_ref[1])
    return distance / len(positions_pre) if positions_pre else 0


def compute_tp(
    positions_pre: Sequence[Tuple[int, int]],
    positions_ref: Sequence[Tuple[int, int]],
    iou_threshold: float,
) -> int:
    return sum(
        1
        for pos_pre, pos_ref in zip(positions_pre, positions_ref)
        if compute_iou_single(pos_pre, pos_ref) >= iou_threshold
    )


def positions_to_residue_set(positions: Sequence[Tuple[int, int]]) -> set:
    residues = set()
    for start, end in positions:
        if end < start:
            start, end = end, start
        residues.update(range(start, end + 1))
    return residues


def compute_residue_level_iou(
    prediction_positions: Sequence[Tuple[int, int]],
    reference_positions: Sequence[Tuple[int, int]],
) -> float:
    pred_residues = positions_to_residue_set(prediction_positions)
    ref_residues = positions_to_residue_set(reference_positions)
    intersection = len(pred_residues & ref_residues)
    union = len(pred_residues | ref_residues)
    return intersection / union if union > 0 else 0


def compute_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    iou_threshold: float,
) -> Dict[str, float]:
    ious = []
    distances = []
    tp_nums = []
    unions_list = []
    intersections_list = []
    residue_ious = []
    pred_nums = []
    ref_nums = []

    for prediction, reference in zip(predictions, references):
        prediction_positions = extract_positions(prediction)
        reference_positions = extract_positions(reference)
        residue_ious.append(compute_residue_level_iou(prediction_positions, reference_positions))
        pred_nums.append(len(prediction_positions))
        ref_nums.append(len(reference_positions))

        interval_prediction_positions = list(prediction_positions)
        if len(interval_prediction_positions) < len(reference_positions):
            interval_prediction_positions += [(0, 0)] * (
                len(reference_positions) - len(interval_prediction_positions)
            )

        matched_pred_positions, matched_ref_positions = match_positions(
            interval_prediction_positions,
            reference_positions,
        )
        iou, unions, intersections = compute_interval_iou(
            matched_pred_positions, matched_ref_positions
        )
        ious.append(iou)
        unions_list.append(unions)
        intersections_list.append(intersections)
        distances.append(compute_distance(matched_pred_positions, matched_ref_positions))
        tp_nums.append(compute_tp(matched_pred_positions, matched_ref_positions, iou_threshold))

    total_union = sum(unions_list)
    total_ref = sum(ref_nums)
    total_pred = sum(pred_nums)
    recall = sum(tp_nums) / total_ref if total_ref > 0 else 0
    precision = sum(tp_nums) / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        "interval_level_avg_iou": sum(ious) / len(ious) if ious else 0,
        "interval_level_global_iou": sum(intersections_list) / total_union
        if total_union > 0
        else 0,
        "interval_level_avg_distance": sum(distances) / len(distances) if distances else 0,
        "interval_level_global_recall": recall,
        "interval_level_global_precision": precision,
        "interval_level_global_f1": f1,
        "residue_level_avg_iou": sum(residue_ious) / len(residue_ious)
        if residue_ious
        else 0,
    }


def evaluate_dataset(dataset_name: str, model, args: argparse.Namespace) -> Path:
    data_root = resolve_data_root(args)
    samples = load_single_grounding_samples(data_root, dataset_name, args.split, args.limit)
    if args.dry_run:
        dry_args = argparse.Namespace(**vars(args))
        dry_args.model_identifier = f"{args.model_identifier}_dry_run"
        out_dir = make_output_dir(dry_args, "grounding_single_v2")
    else:
        out_dir = make_output_dir(args, "grounding_single_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{dataset_name}_results.csv"
    metrics_path = out_dir / f"{dataset_name}_results_metrics.json"

    foldseek = FoldseekRunner(args.foldseek_path, args.missing_structure)
    generated: List[str] = []
    references: List[str] = []
    structure_paths: List[str] = []

    print(f"Evaluating {dataset_name}: {len(samples)} samples from {data_root}")
    total_batches = (len(samples) + args.batch_per_device - 1) // args.batch_per_device
    for batch_samples in tqdm(
        batched(samples, args.batch_per_device),
        total=total_batches,
        desc=f"Evaluating {dataset_name}",
    ):
        input_batch, inputs = prepare_single_batch(batch_samples, foldseek)
        references.extend(inputs["target_seqs"])
        structure_paths.extend(inputs["structure_paths"])
        if args.dry_run:
            generated.extend(inputs["target_seqs"])
            continue
        with torch.no_grad():
            generated.extend(model.generate(input_batch, inputs, "text", **generation_kwargs(args)))

    pred_positions = [extract_positions(text) for text in generated]
    metrics = compute_metrics(generated, references, args.iou_threshold)

    rows = []
    for sample, text, pred_pos, structure_path in zip(
        samples, generated, pred_positions, structure_paths
    ):
        rows.append(
            {
                "generated": text,
                "reference": sample["reference"],
                "dataset_idx": sample["dataset_idx"],
                "gt_positions": sample["gt_positions"],
                "pred_positions": pred_pos,
                "uid": sample["uid"],
                "category": sample["category"],
                "interpro_id": sample["interpro_id"],
                "shortname": sample["shortname"],
                "structure_path": structure_path,
            }
        )

    fieldnames = [
        "generated",
        "reference",
        "dataset_idx",
        "gt_positions",
        "pred_positions",
        "uid",
        "category",
        "interpro_id",
        "shortname",
        "structure_path",
    ]
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved {save_path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return save_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    dataset_names = parse_dataset_list(args.datasets)
    model = None if args.dry_run else load_instructbiomol_model(args)
    for dataset_name in dataset_names:
        evaluate_dataset(dataset_name, model, args)


if __name__ == "__main__":
    main()
