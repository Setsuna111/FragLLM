import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]
ESM2_650M_MODEL_PATH = "/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/"
ESMC_MODEL_PATH = "/home/dataset-local/projects_dir/pretrained_model/ESMC-600M/"
INTERPROT_REPO_ROOT = "/home/dataset-local/projects_dir/CAPSUL/interprot"
INTERPROT_ESM_MODEL_DIR = "/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D"
INTERPROT_SAE_CHECKPOINT = "/home/dataset-local/projects_dir/pretrained_model/InterProt-ESM2-SAEs/esm2_plm1280_l24_sae4096.safetensors"
INTERPROT_PLM_LAYER = 24
INTERPROT_ESM_DIM = 1280
INTERPROT_SAE_DIM = 4096


def resolve_data_dir(data_dir):
    data_dir = os.path.expanduser(data_dir)
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.join(project_root, data_dir)


def get_data_dir_name(data_dir):
    return os.path.basename(os.path.normpath(data_dir))


def load_venusx_dataset(dataset_name, split, data_dir="data"):
    data_root = resolve_data_dir(data_dir)
    dataset_path = os.path.join(data_root, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        return json.load(f)


def _find_all(haystack, needle):
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return positions
        positions.append(idx)
        start = idx + 1


def resolve_fragment_span(protein_sequence, frag):
    """Return a 0-based half-open span for a fragment.

    VenusX data normally uses 0-based inclusive start/end positions. Some
    filtered splits contain a small number of stale coordinates; for those, use
    a unique or nearest sequence match when available, then fall back to the
    declared coordinate span.
    """
    frag_sequence = frag["sequence"]
    raw_start = int(frag["start_position"])
    raw_end = int(frag["end_position"])
    frag_len = len(frag_sequence)
    seq_len = len(protein_sequence)

    candidates = [
        (raw_start, raw_end + 1, "coord_0_based_inclusive"),
        (raw_start, raw_start + frag_len, "coord_0_based_start_len"),
        (raw_start - 1, raw_end, "coord_1_based_inclusive"),
        (raw_start - 1, raw_start - 1 + frag_len, "coord_1_based_start_len"),
    ]
    seen = set()
    for start, end, status in candidates:
        if (start, end) in seen:
            continue
        seen.add((start, end))
        if 0 <= start < end <= seq_len and protein_sequence[start:end] == frag_sequence:
            return start, end, status

    matches = _find_all(protein_sequence, frag_sequence)
    if matches:
        start = min(matches, key=lambda pos: abs(pos - raw_start))
        return start, start + frag_len, "sequence_match_fallback"

    start = max(0, min(raw_start, seq_len))
    end = max(start, min(raw_end + 1, seq_len))
    if end == start and frag_len > 0:
        end = min(seq_len, start + frag_len)
    return start, end, "coordinate_mismatch_fallback"


def extract_fragment_records(data, dataset_name=None):
    records = []
    labels = []
    fragment_ids = []

    for protein in data:
        uid = protein["uid"]
        protein_sequence = protein["sequence"]
        for fragment_group in protein["fragments"]:
            interpro_id = fragment_group["interpro_id"]
            for i, frag in enumerate(fragment_group["frags"]):
                start, end, span_status = resolve_fragment_span(protein_sequence, frag)
                frag_id = f"{uid}_{interpro_id}_{i}"
                if dataset_name is not None:
                    frag_id = f"{dataset_name}:{frag_id}"
                records.append(
                    {
                        "protein_id": uid,
                        "protein_sequence": protein_sequence,
                        "fragment_sequence": frag["sequence"],
                        "start": start,
                        "end": end,
                        "span_status": span_status,
                    }
                )
                labels.append(interpro_id)
                fragment_ids.append(frag_id)

    return records, labels, fragment_ids


def _is_esmc_model(model_path):
    model_path_lower = model_path.lower().rstrip("/")
    return model_path_lower == "esmc" or "esmc" in os.path.basename(model_path_lower)


def _is_interprot_model(model_path):
    model_path_lower = model_path.lower().rstrip("/")
    model_basename = os.path.basename(model_path_lower)
    return (
        model_path_lower in {"interprot", "interprot_sae", "interprot-esm2-sae"}
        or "interprot" in model_path_lower
        or (model_basename.startswith("esm2_plm") and model_basename.endswith(".safetensors"))
    )


def _is_supported_esm2_model(model_path):
    normalized = os.path.abspath(os.path.expanduser(model_path.rstrip("/")))
    supported = os.path.abspath(ESM2_650M_MODEL_PATH.rstrip("/"))
    return normalized == supported


def get_model_name(model_path):
    if model_path.lower().rstrip("/") == "esmc":
        return "esmc"
    if _is_interprot_model(model_path):
        return "interprot"
    return model_path.rstrip("/").split("/")[-1]


def get_base_out_dir(out_dir, model_name, data_dir_name):
    if data_dir_name == "data":
        return os.path.join(out_dir, model_name)
    return os.path.join(out_dir, model_name, data_dir_name)


def _make_context_window(record, max_residues):
    sequence = record["protein_sequence"]
    seq_len = len(sequence)
    start = int(record["start"])
    end = int(record["end"])
    end = max(start + 1, end)

    if seq_len <= max_residues:
        window_start = 0
        window_end = seq_len
    else:
        frag_len = end - start
        if frag_len >= max_residues:
            window_start = min(start, max(0, seq_len - max_residues))
        else:
            left_context = (max_residues - frag_len) // 2
            window_start = start - left_context
            window_start = max(0, min(window_start, seq_len - max_residues))
        window_end = min(seq_len, window_start + max_residues)

    rel_start = max(0, start - window_start)
    rel_end = min(window_end - window_start, end - window_start)
    if rel_end <= rel_start:
        rel_start = min(rel_start, max(0, window_end - window_start - 1))
        rel_end = rel_start + 1

    windowed = dict(record)
    windowed.update(
        {
            "window_sequence": sequence[window_start:window_end],
            "window_start": window_start,
            "window_end": window_end,
            "relative_start": rel_start,
            "relative_end": rel_end,
            "used_centered_window": seq_len > max_residues,
        }
    )
    return windowed


def _mean_pool_fragment(sequence_embeddings, records):
    embeddings = []
    metadata = []
    for idx, record in enumerate(records):
        token_start = 1 + int(record["relative_start"])
        token_end = 1 + int(record["relative_end"])
        residue_embeddings = sequence_embeddings[idx, token_start:token_end]
        if residue_embeddings.numel() == 0:
            residue_embeddings = sequence_embeddings[idx, token_start : token_start + 1]
        embeddings.append(residue_embeddings.mean(dim=0).float().cpu().numpy())
        metadata.append(
            {
                "protein_id": record["protein_id"],
                "start": int(record["start"]),
                "end": int(record["end"]),
                "span_status": record["span_status"],
                "window_start": int(record["window_start"]),
                "window_end": int(record["window_end"]),
                "used_centered_window": bool(record["used_centered_window"]),
            }
        )
    return embeddings, metadata


class ESMCGlobalFragmentEncoder:
    def __init__(self, model_path, device):
        from transformers import AutoModelForMaskedLM

        load_kwargs = {}
        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, **load_kwargs).eval()
        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode_fragment_records(self, records, batch_size=16, max_length=1024):
        return _encode_fragment_records_with_model(
            self.model,
            self.tokenizer,
            records,
            self.device,
            batch_size,
            max_length,
            use_hidden_states=True,
        )


class InterProtGlobalFragmentEncoder:
    def __init__(self, esm_model_dir, sae_checkpoint, plm_layer, esm_dim, sae_dim, device):
        try:
            from safetensors.torch import load_file
            from transformers import EsmModel
        except ImportError as exc:
            raise ImportError("InterProt encoding requires transformers and safetensors.") from exc

        if INTERPROT_REPO_ROOT not in sys.path:
            sys.path.insert(0, INTERPROT_REPO_ROOT)
        from interprot.sae_model import SparseAutoencoder

        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_dir)
        self.esm_model = EsmModel.from_pretrained(esm_model_dir).to(device).eval()
        self.sae_model = SparseAutoencoder(esm_dim, sae_dim)
        self.sae_model.load_state_dict(load_file(sae_checkpoint))
        self.sae_model = self.sae_model.to(device).eval()

        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.sae_model.parameters():
            param.requires_grad = False

        self.plm_layer = plm_layer
        self.device = device

    @torch.no_grad()
    def encode_fragment_records(self, records, batch_size=16, max_length=1024):
        embeddings = []
        metadata = []
        max_residues = max_length - 2
        for i in tqdm(range(0, len(records), batch_size)):
            batch_records = [
                _make_context_window(record, max_residues)
                for record in records[i : i + batch_size]
            ]
            batch_sequences = [record["window_sequence"] for record in batch_records]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            esm_layer_acts = outputs.hidden_states[self.plm_layer]
            sae_acts = self.sae_model.get_acts(esm_layer_acts)
            batch_embeddings, batch_metadata = _mean_pool_fragment(sae_acts, batch_records)
            embeddings.extend(batch_embeddings)
            metadata.extend(batch_metadata)
        return np.array(embeddings), metadata


def _encode_fragment_records_with_model(
    model,
    tokenizer,
    records,
    device,
    batch_size=16,
    max_length=1024,
    use_hidden_states=False,
):
    embeddings = []
    metadata = []
    max_residues = max_length - 2

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size)):
            batch_records = [
                _make_context_window(record, max_residues)
                for record in records[i : i + batch_size]
            ]
            batch_sequences = [record["window_sequence"] for record in batch_records]
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if use_hidden_states:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                sequence_embeddings = outputs.hidden_states[-1]
            else:
                outputs = model(**inputs)
                sequence_embeddings = outputs.last_hidden_state

            batch_embeddings, batch_metadata = _mean_pool_fragment(
                sequence_embeddings, batch_records
            )
            embeddings.extend(batch_embeddings)
            metadata.extend(batch_metadata)

    return np.array(embeddings), metadata


def load_plm_model(model_path, device):
    print(f"Loading PLM model from {model_path}...")

    if _is_esmc_model(model_path):
        resolved_model_path = ESMC_MODEL_PATH if model_path.lower().rstrip("/") == "esmc" else model_path
        encoder = ESMCGlobalFragmentEncoder(resolved_model_path, device)
        print(f"Loaded ESMC model on device: {encoder.device}")
        return encoder, None

    if _is_interprot_model(model_path):
        sae_checkpoint = (
            model_path
            if model_path.lower().rstrip("/").endswith(".safetensors")
            else INTERPROT_SAE_CHECKPOINT
        )
        encoder = InterProtGlobalFragmentEncoder(
            esm_model_dir=INTERPROT_ESM_MODEL_DIR,
            sae_checkpoint=sae_checkpoint,
            plm_layer=INTERPROT_PLM_LAYER,
            esm_dim=INTERPROT_ESM_DIM,
            sae_dim=INTERPROT_SAE_DIM,
            device=device,
        )
        print(f"Loaded InterProt model on device: {encoder.device}")
        return encoder, None

    if not _is_supported_esm2_model(model_path):
        raise ValueError(
            "plm_ref_cls_global.py only supports the ESM2-650M local path, "
            "`esmc`, and `interprot`."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    print(f"Loaded ESM2 model on device: {device}")
    return model, tokenizer


def encode_fragment_records_batch(model, tokenizer, records, device, batch_size=16):
    print(f"Encoding {len(records)} fragment records with global context...")
    if hasattr(model, "encode_fragment_records"):
        return model.encode_fragment_records(records, batch_size=batch_size, max_length=1024)
    return _encode_fragment_records_with_model(
        model,
        tokenizer,
        records,
        device,
        batch_size=batch_size,
        max_length=1024,
        use_hidden_states=False,
    )


def find_most_similar(query_embeddings, train_embeddings, train_labels, train_ids):
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(query_embeddings, train_embeddings)
    predictions = []

    for query_similarities in tqdm(similarity_matrix, desc="Finding best matches"):
        best_match_idx = np.argmax(query_similarities)
        predictions.append(
            {
                "predicted_interpro_id": train_labels[best_match_idx],
                "matched_train_id": train_ids[best_match_idx],
                "similarity_score": float(query_similarities[best_match_idx]),
            }
        )

    return predictions


def save_embeddings_cache(embeddings, metadata, cache_path, metadata_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved embeddings cache to {cache_path}")
    print(f"Saved embedding metadata to {metadata_path}")


def load_embeddings_cache(cache_path, metadata_path, expected_count=None):
    if not os.path.exists(cache_path):
        return None, None
    embeddings = np.load(cache_path)
    if expected_count is not None and len(embeddings) != expected_count:
        print(
            f"Cache size mismatch for {cache_path}: "
            f"{len(embeddings)} != {expected_count}. Recomputing..."
        )
        return None, None
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        if expected_count is not None and len(metadata) != expected_count:
            metadata = None
    print(f"Loaded embeddings cache from {cache_path}")
    return embeddings, metadata


def compute_classification_metrics(results, label_to_idx=None):
    if not results:
        global_num_classes = len(label_to_idx) if label_to_idx is not None else 0
        return {
            "acc": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "total": 0,
            "correct": 0,
            "num_classes": 0,
            "macro_average": "test_true_labels",
            "observed_precision": 0.0,
            "observed_recall": 0.0,
            "observed_f1": 0.0,
            "observed_num_classes": 0,
            "global_precision": 0.0,
            "global_recall": 0.0,
            "global_f1": 0.0,
            "global_num_classes": global_num_classes,
        }

    true_labels = [r["true_interpro_id"] for r in results]
    pred_labels = [r["predicted_interpro_id"] for r in results]
    true_count = Counter(true_labels)
    pred_count = Counter(pred_labels)
    tp_count = Counter(
        true_label
        for true_label, pred_label in zip(true_labels, pred_labels)
        if true_label == pred_label
    )

    total = len(results)
    correct = sum(tp_count.values())

    def macro_scores(labels):
        labels = list(labels)
        if not labels:
            return 0.0, 0.0, 0.0, 0
        precisions = []
        recalls = []
        f1_scores = []
        for label in labels:
            tp = tp_count[label]
            precision = tp / pred_count[label] if pred_count[label] > 0 else 0.0
            recall = tp / true_count[label] if true_count[label] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        return (
            sum(precisions) / len(labels),
            sum(recalls) / len(labels),
            sum(f1_scores) / len(labels),
            len(labels),
        )

    true_label_set = sorted(true_count)
    observed_label_set = sorted(set(true_count) | set(pred_count))
    global_label_set = (
        observed_label_set
        if label_to_idx is None
        else sorted(set(label_to_idx) | set(true_count) | set(pred_count))
    )

    precision, recall, f1, num_classes = macro_scores(true_label_set)
    observed_precision, observed_recall, observed_f1, observed_num_classes = macro_scores(
        observed_label_set
    )
    global_precision, global_recall, global_f1, global_num_classes = macro_scores(
        global_label_set
    )

    active_labels = set(true_count) | set(pred_count)
    sum_row_col = sum(true_count[label] * pred_count[label] for label in active_labels)
    numerator = correct * total - sum_row_col
    denominator_left = total * total - sum(value * value for value in true_count.values())
    denominator_right = total * total - sum(value * value for value in pred_count.values())
    denominator = math.sqrt(denominator_left * denominator_right)
    mcc = numerator / denominator if denominator > 0 else 0.0

    return {
        "acc": correct / total,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "total": total,
        "correct": correct,
        "num_classes": num_classes,
        "macro_average": "test_true_labels",
        "observed_precision": observed_precision,
        "observed_recall": observed_recall,
        "observed_f1": observed_f1,
        "observed_num_classes": observed_num_classes,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_f1": global_f1,
        "global_num_classes": global_num_classes,
    }


def summarize_fragment_metadata(metadata):
    status_count = Counter(item["span_status"] for item in metadata)
    return {
        "span_status_counts": dict(status_count),
        "num_centered_windows": sum(1 for item in metadata if item["used_centered_window"]),
    }


def main(model_path, batch_size, out_dir, use_cuda, data_dir="data", eval_datasets=None):
    eval_datasets = eval_datasets or DATASET_NAMES
    model_name = get_model_name(model_path)
    data_root = resolve_data_dir(data_dir)
    data_dir_name = get_data_dir_name(data_root)
    print(f"[*] Processing VenusX datasets with global-context PLM ({model_name})...")
    print(f"Using dataset root: {data_root}")
    print(f"Training datasets: {', '.join(DATASET_NAMES)}")
    print(f"Evaluation datasets: {', '.join(eval_datasets)}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_out_dir = get_base_out_dir(out_dir, model_name, data_dir_name)
    os.makedirs(base_out_dir, exist_ok=True)

    train_records = []
    train_labels = []
    train_ids = []
    test_sets = {}
    global_labels = set()

    for dataset_name in DATASET_NAMES:
        train_data = load_venusx_dataset(dataset_name, "train", data_root)
        test_data = load_venusx_dataset(dataset_name, "test", data_root)

        ds_train_records, ds_train_labels, ds_train_ids = extract_fragment_records(
            train_data, dataset_name
        )
        ds_test_records, ds_test_labels, ds_test_ids = extract_fragment_records(
            test_data, dataset_name
        )

        train_records.extend(ds_train_records)
        train_labels.extend(ds_train_labels)
        train_ids.extend(ds_train_ids)
        test_sets[dataset_name] = {
            "records": ds_test_records,
            "labels": ds_test_labels,
            "ids": ds_test_ids,
        }
        global_labels.update(ds_train_labels)
        global_labels.update(ds_test_labels)

        print(
            f"VenusX_{dataset_name}: "
            f"train fragments={len(ds_train_records)}, test fragments={len(ds_test_records)}"
        )

    label_to_idx = {label: idx for idx, label in enumerate(sorted(global_labels))}
    label_index_file = os.path.join(base_out_dir, "plm_label_to_idx.json")
    with open(label_index_file, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"Combined train fragments: {len(train_records)}")
    print(f"Global label count: {len(label_to_idx)}")

    model, tokenizer = load_plm_model(model_path, device)

    train_cache_path = os.path.join(base_out_dir, "all_train_global_embeddings.npy")
    train_metadata_path = os.path.join(base_out_dir, "all_train_global_embedding_metadata.json")
    train_embeddings, train_metadata = load_embeddings_cache(
        train_cache_path, train_metadata_path, expected_count=len(train_records)
    )

    if train_embeddings is None or train_metadata is None:
        train_embeddings, train_metadata = encode_fragment_records_batch(
            model, tokenizer, train_records, device, batch_size
        )
        save_embeddings_cache(
            train_embeddings, train_metadata, train_cache_path, train_metadata_path
        )

    all_metrics = {}
    csv_outputs = {}

    for dataset_name in eval_datasets:
        if dataset_name not in test_sets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_out_dir = os.path.join(base_out_dir, f"VenusX_{dataset_name}")
        os.makedirs(dataset_out_dir, exist_ok=True)

        test_records = test_sets[dataset_name]["records"]
        test_labels = test_sets[dataset_name]["labels"]
        test_ids = test_sets[dataset_name]["ids"]

        test_cache_path = os.path.join(dataset_out_dir, "test_global_embeddings.npy")
        test_metadata_path = os.path.join(dataset_out_dir, "test_global_embedding_metadata.json")
        test_embeddings, test_metadata = load_embeddings_cache(
            test_cache_path, test_metadata_path, expected_count=len(test_records)
        )

        if test_embeddings is None or test_metadata is None:
            test_embeddings, test_metadata = encode_fragment_records_batch(
                model, tokenizer, test_records, device, batch_size
            )
            save_embeddings_cache(
                test_embeddings, test_metadata, test_cache_path, test_metadata_path
            )

        predictions = find_most_similar(
            test_embeddings, train_embeddings, train_labels, train_ids
        )

        results = []
        for test_id, true_label, pred_info, fragment_meta in zip(
            test_ids, test_labels, predictions, test_metadata
        ):
            results.append(
                {
                    "query_sequence": test_id,
                    "protein_id": fragment_meta["protein_id"],
                    "fragment_start": fragment_meta["start"],
                    "fragment_end": fragment_meta["end"],
                    "span_status": fragment_meta["span_status"],
                    "window_start": fragment_meta["window_start"],
                    "window_end": fragment_meta["window_end"],
                    "used_centered_window": fragment_meta["used_centered_window"],
                    "true_interpro_id": true_label,
                    "true_label_idx": label_to_idx[true_label],
                    "predicted_interpro_id": pred_info["predicted_interpro_id"],
                    "predicted_label_idx": label_to_idx[pred_info["predicted_interpro_id"]],
                    "matched_train_sequence": pred_info["matched_train_id"],
                    "similarity_score": pred_info["similarity_score"],
                }
            )

        results_df = pd.DataFrame(results)
        csv_output = os.path.join(dataset_out_dir, "plm_predictions.csv")
        results_df.to_csv(csv_output, index=False)
        csv_outputs[dataset_name] = csv_output

        metrics = compute_classification_metrics(results, label_to_idx)
        metrics["avg_similarity"] = (
            float(np.mean([r["similarity_score"] for r in results])) if results else 0.0
        )
        metrics["test_fragment_metadata"] = summarize_fragment_metadata(test_metadata)
        all_metrics[dataset_name] = metrics

        metrics_file = os.path.join(dataset_out_dir, "plm_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        metadata = {
            "dataset_name": dataset_name,
            "train_datasets": DATASET_NAMES,
            "data_dir": data_dir,
            "data_root": data_root,
            "model_path": model_path,
            "batch_size": batch_size,
            "device": str(device),
            "num_train_fragments": len(train_records),
            "num_test_fragments": len(test_records),
            "num_global_labels": len(label_to_idx),
            "long_sequence_strategy": "centered_window_if_protein_exceeds_1022_residues",
            "train_fragment_metadata": summarize_fragment_metadata(train_metadata),
            "test_fragment_metadata": summarize_fragment_metadata(test_metadata),
            "metrics": metrics,
        }

        metadata_file = os.path.join(dataset_out_dir, "plm_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {csv_output}")
        print(f"Metrics saved to {metrics_file}")
        print(f"Accuracy: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")

    all_metrics_file = os.path.join(base_out_dir, "plm_all_metrics.json")
    with open(all_metrics_file, "w") as f:
        json.dump(
            {
                "train_datasets": DATASET_NAMES,
                "eval_datasets": eval_datasets,
                "num_train_fragments": len(train_records),
                "num_global_labels": len(label_to_idx),
                "long_sequence_strategy": "centered_window_if_protein_exceeds_1022_residues",
                "train_fragment_metadata": summarize_fragment_metadata(train_metadata),
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    print(f"All metrics saved to {all_metrics_file}")
    return csv_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global-context PLM nearest-neighbor reference classification"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default=None,
        help="Deprecated alias for evaluating one test dataset. Training always uses all datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=DATASET_NAMES,
        help="VenusX test datasets to evaluate. Training always uses all datasets.",
    )
    parser.add_argument("--model_path", type=str, default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/")
    # parser.add_argument("--model_path", type=str, default="esmc")
    # parser.add_argument("--model_path", type=str, default="interprot")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(project_root, "baselines", "plm_ref_cls_global_results"),
    )
    parser.add_argument("--data_dir", type=str, default="data_70")
    # parser.add_argument("--data_dir", type=str, default="data_30")
    parser.add_argument("--cpu", default=False, action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_datasets = [args.dataset] if args.dataset is not None else args.datasets
    main(args.model_path, args.batch_size, args.out_dir, not args.cpu, args.data_dir, eval_datasets)
