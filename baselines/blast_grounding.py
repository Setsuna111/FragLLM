# BLAST-based protein fragment localization baseline for single grounding.

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]
DEFAULT_DATA_DIR = os.path.join(project_root, "data_70")
DEFAULT_BLASTP = "/home/dataset-local/anaconda3/envs/fragllm/bin/blastp"
DEFAULT_MAKEBLASTDB = "/home/dataset-local/anaconda3/envs/fragllm/bin/makeblastdb"


def parse_dataset_names(dataset_arg):
    dataset_names = [name.strip() for name in dataset_arg.split(",") if name.strip()]
    invalid = [name for name in dataset_names if name not in DATASET_NAMES]
    if invalid:
        raise ValueError(f"Unknown dataset(s): {invalid}. Choices: {DATASET_NAMES}")
    if not dataset_names:
        raise ValueError("At least one dataset must be specified.")
    return dataset_names


def load_venusx_dataset(dataset_name, split, data_dir):
    dataset_path = os.path.join(data_dir, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        return json.load(f)


def extract_category_fragments(data):
    category_fragments = defaultdict(list)

    for protein in data:
        uid = protein["uid"]
        for fragment_group in protein["fragments"]:
            category = fragment_group["category"]
            interpro_id = fragment_group["interpro_id"]

            for i, frag in enumerate(fragment_group["frags"]):
                category_fragments[category].append({
                    "uid": uid,
                    "category": category,
                    "interpro_id": interpro_id,
                    "sequence": frag["sequence"],
                    "start_pos": frag["start_position"],
                    "end_pos": frag["end_position"],
                    "frag_id": f"{uid}_{interpro_id}_{i}",
                })

    return category_fragments


def create_blast_database(sequences, db_path, makeblastdb_path):
    temp_fasta = f"{db_path}.fasta"
    with open(temp_fasta, "w") as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n{sequence}\n")

    cmd = [makeblastdb_path, "-in", temp_fasta, "-dbtype", "prot", "-out", db_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating BLAST database: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)


def run_blastp_alignment(query_sequences, db_path, blastp_path, evalue_threshold=1e-2, num_threads=1):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as query_file:
        for seq_id, sequence in query_sequences:
            query_file.write(f">{seq_id}\n{sequence}\n")
        query_path = query_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    outfmt_fields = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
    cmd = [
        blastp_path,
        "-query", query_path,
        "-db", db_path,
        "-evalue", str(evalue_threshold),
        "-word_size", "2",
        "-max_target_seqs", "10000",
        "-seg", "no",
        "-out", output_path,
        "-outfmt", outfmt_fields,
        "-num_threads", str(num_threads),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running BLASTP: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    results = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 12:
                    continue
                results.append({
                    "query_id": parts[0],
                    "subject_id": parts[1],
                    "pident": float(parts[2]),
                    "length": int(parts[3]),
                    "mismatch": int(parts[4]),
                    "gapopen": int(parts[5]),
                    "qstart": int(parts[6]),
                    "qend": int(parts[7]),
                    "sstart": int(parts[8]),
                    "send": int(parts[9]),
                    "evalue": float(parts[10]),
                    "bitscore": float(parts[11]),
                })

    os.unlink(query_path)
    os.unlink(output_path)
    return results


def merge_overlapping_ranges(ranges, max_gap=5):
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + max_gap:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def predict_fragment_locations(
    test_protein_sequence,
    fragment_sequences,
    blastp_path,
    makeblastdb_path,
    identity_threshold=30.0,
    evalue_threshold=1e-2,
    coverage_threshold=0.5,
    merge_gap=5,
    num_threads=1,
):
    if not fragment_sequences:
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "fragment_database")
        fragment_db_sequences = [(f"frag_{i}", seq) for i, seq in enumerate(fragment_sequences)]
        create_blast_database(fragment_db_sequences, db_path, makeblastdb_path)
        blast_results = run_blastp_alignment(
            [("target_protein", test_protein_sequence)],
            db_path,
            blastp_path,
            evalue_threshold=evalue_threshold,
            num_threads=num_threads,
        )

    predicted_ranges = []
    for result in blast_results:
        subject_idx = int(result["subject_id"].split("_")[-1])
        fragment_len = len(fragment_sequences[subject_idx])
        coverage = result["length"] / fragment_len if fragment_len > 0 else 0.0
        if result["pident"] < identity_threshold:
            continue
        if result["evalue"] > evalue_threshold:
            continue
        if coverage < coverage_threshold:
            continue

        start = min(result["qstart"], result["qend"]) - 1
        end = max(result["qstart"], result["qend"]) - 1
        predicted_ranges.append((start, end))

    return merge_overlapping_ranges(predicted_ranges, max_gap=merge_gap)


def process_single_grounding(
    test_data,
    train_category_fragments,
    blastp_path,
    makeblastdb_path,
    identity_threshold=30.0,
    evalue_threshold=1e-2,
    coverage_threshold=0.5,
    merge_gap=5,
    num_threads=1,
):
    results = []

    for dataset_idx, protein in enumerate(tqdm(test_data, desc="Processing single grounding")):
        uid = protein["uid"]
        sequence = protein["sequence"]

        for fragment_group in protein["fragments"]:
            target_category = fragment_group["category"]
            interpro_id = fragment_group["interpro_id"]
            true_ranges = [
                (frag["start_position"], frag["end_position"])
                for frag in fragment_group["frags"]
            ]

            train_fragments = train_category_fragments.get(target_category, [])
            fragment_sequences = [frag["sequence"] for frag in train_fragments]
            predicted_ranges = predict_fragment_locations(
                sequence,
                fragment_sequences,
                blastp_path=blastp_path,
                makeblastdb_path=makeblastdb_path,
                identity_threshold=identity_threshold,
                evalue_threshold=evalue_threshold,
                coverage_threshold=coverage_threshold,
                merge_gap=merge_gap,
                num_threads=num_threads,
            )

            results.append({
                "dataset_idx": dataset_idx,
                "uid": uid,
                "category": target_category,
                "interpro_id": interpro_id,
                "true_ranges": true_ranges,
                "predicted_ranges": predicted_ranges,
                "sequence_length": len(sequence),
                "num_true_fragments": len(true_ranges),
                "num_predicted_fragments": len(predicted_ranges),
            })

    return results


def compute_iou_single(pos_pre, pos_ref):
    inter_left = max(pos_pre[0], pos_ref[0])
    inter_right = min(pos_pre[1], pos_ref[1])
    intersection = max(0, inter_right - inter_left + 1)
    union_left = min(pos_pre[0], pos_ref[0])
    union_right = max(pos_pre[1], pos_ref[1])
    union = union_right - union_left + 1
    return intersection / union if union > 0 else 0


def match_positions(prediction_positions, reference_positions):
    dist_matrix = [[0.0 for _ in reference_positions] for _ in prediction_positions]
    for i, pred_pos in enumerate(prediction_positions):
        for j, ref_pos in enumerate(reference_positions):
            dist_matrix[i][j] = compute_iou_single(pred_pos, ref_pos)

    matched_pred_positions = []
    matched_ref_positions = []
    while dist_matrix and any(value != -1 for row in dist_matrix for value in row):
        max_i, max_j = 0, 0
        max_value = -1
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


def compute_interval_iou(positions_pre, positions_ref):
    iou = 0
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


def compute_distance(positions_pre, positions_ref):
    distance = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        distance += abs(pos_pre[0] - pos_ref[0]) + abs(pos_pre[1] - pos_ref[1])
    if positions_pre:
        distance /= len(positions_pre)
    return distance


def compute_tp(positions_pre, positions_ref, iou_threshold):
    tp = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        if compute_iou_single(pos_pre, pos_ref) >= iou_threshold:
            tp += 1
    return tp


def positions_to_residue_set(positions):
    residues = set()
    for start, end in positions:
        if end < start:
            start, end = end, start
        residues.update(range(start, end + 1))
    return residues


def compute_residue_level_iou(prediction_positions, reference_positions):
    pred_residues = positions_to_residue_set(prediction_positions)
    ref_residues = positions_to_residue_set(reference_positions)
    intersection = len(pred_residues & ref_residues)
    union = len(pred_residues | ref_residues)
    return (intersection / union if union > 0 else 0), union, intersection


def compute_metrics(results, iou_threshold):
    ious = []
    distances = []
    tp_nums = []
    unions_list = []
    intersections_list = []
    residue_ious = []
    pred_nums = []
    ref_nums = []

    for result in results:
        prediction_positions = list(result["predicted_ranges"])
        reference_positions = list(result["true_ranges"])
        residue_iou, _, _ = compute_residue_level_iou(prediction_positions, reference_positions)
        residue_ious.append(residue_iou)
        pred_nums.append(len(prediction_positions))
        ref_nums.append(len(reference_positions))

        if len(prediction_positions) < len(reference_positions):
            prediction_positions = prediction_positions + [(0, 0)] * (len(reference_positions) - len(prediction_positions))

        matched_pred_positions, matched_ref_positions = match_positions(prediction_positions, reference_positions)
        iou, unions, intersections = compute_interval_iou(matched_pred_positions, matched_ref_positions)
        ious.append(iou)
        unions_list.append(unions)
        intersections_list.append(intersections)
        distances.append(compute_distance(matched_pred_positions, matched_ref_positions))
        tp_nums.append(compute_tp(matched_pred_positions, matched_ref_positions, iou_threshold))

    total_unions = sum(unions_list)
    total_ref = sum(ref_nums)
    total_pred = sum(pred_nums)
    global_recall = sum(tp_nums) / total_ref if total_ref > 0 else 0
    global_precision = sum(tp_nums) / total_pred if total_pred > 0 else 0
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if global_precision + global_recall != 0
        else 0
    )

    return {
        "interval_level_avg_iou": sum(ious) / len(ious) if ious else 0,
        "interval_level_global_iou": sum(intersections_list) / total_unions if total_unions > 0 else 0,
        "interval_level_avg_distance": sum(distances) / len(distances) if distances else 0,
        "interval_level_global_recall": global_recall,
        "interval_level_global_precision": global_precision,
        "interval_level_global_f1": global_f1,
        "residue_level_avg_iou": sum(residue_ious) / len(residue_ious) if residue_ious else 0,
    }


def format_ranges(ranges):
    return ";".join([f"{start}-{end}" for start, end in ranges])


def save_results_to_csv(results, output_path):
    csv_data = []
    for result in results:
        csv_data.append({
            "dataset_idx": result["dataset_idx"],
            "uid": result["uid"],
            "category": result["category"],
            "interpro_id": result["interpro_id"],
            "true_ranges": format_ranges(result["true_ranges"]),
            "predicted_ranges": format_ranges(result["predicted_ranges"]),
            "sequence_length": result["sequence_length"],
            "num_true_fragments": result["num_true_fragments"],
            "num_predicted_fragments": result["num_predicted_fragments"],
        })

    pd.DataFrame(csv_data).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def save_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def run_dataset(dataset_name, args):
    print(f"[*] Starting BLAST single grounding for VenusX_{dataset_name}")
    train_data = load_venusx_dataset(dataset_name, "train", args.data_dir)
    test_data = load_venusx_dataset(dataset_name, "test", args.data_dir)
    if args.limit_test is not None:
        test_data = test_data[:args.limit_test]

    train_category_fragments = extract_category_fragments(train_data)
    print(f"Train categories: {len(train_category_fragments)}")
    print(f"Test proteins: {len(test_data)}")

    results = process_single_grounding(
        test_data,
        train_category_fragments,
        blastp_path=args.blastp_path,
        makeblastdb_path=args.makeblastdb_path,
        identity_threshold=args.identity_threshold,
        evalue_threshold=args.evalue_threshold,
        coverage_threshold=args.coverage_threshold,
        merge_gap=args.merge_gap,
        num_threads=args.num_threads,
    )
    metrics = compute_metrics(results, args.iou_threshold)

    dataset_out_dir = os.path.join(args.output_dir, f"VenusX_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    results_path = os.path.join(dataset_out_dir, "single_grounding_results.csv")
    metrics_path = os.path.join(dataset_out_dir, "single_grounding_metrics.json")
    save_results_to_csv(results, results_path)
    save_metrics(metrics, metrics_path)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="BLAST-based single grounding baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Act,BindI,Dom,Evo,Motif",
        help="Comma-separated dataset names. Choices: Act,BindI,Dom,Evo,Motif",
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "baselines", "blast_grounding_results"),
    )
    parser.add_argument("--blastp_path", type=str, default=DEFAULT_BLASTP)
    parser.add_argument("--makeblastdb_path", type=str, default=DEFAULT_MAKEBLASTDB)
    parser.add_argument("--identity_threshold", type=float, default=30.0)
    parser.add_argument("--evalue_threshold", type=float, default=1e-2)
    parser.add_argument("--coverage_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--merge_gap", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--limit_test", type=int, default=None, help="Optional smoke-test limit for test proteins.")

    args = parser.parse_args()
    dataset_names = parse_dataset_names(args.dataset)

    all_metrics = {}
    for dataset_name in dataset_names:
        all_metrics[dataset_name] = run_dataset(dataset_name, args)

    if len(all_metrics) > 1:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "single_grounding_metrics_summary.json")
        save_metrics(all_metrics, summary_path)


if __name__ == "__main__":
    main()
