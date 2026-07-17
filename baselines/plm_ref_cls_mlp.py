import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]


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


def extract_fragments_with_labels(data, dataset_name=None):
    fragments = []
    labels = []
    fragment_ids = []

    for protein in data:
        uid = protein["uid"]
        for fragment_group in protein["fragments"]:
            interpro_id = fragment_group["interpro_id"]
            for i, frag in enumerate(fragment_group["frags"]):
                frag_id = f"{uid}_{interpro_id}_{i}"
                if dataset_name is not None:
                    frag_id = f"{dataset_name}:{frag_id}"
                fragments.append(frag["sequence"])
                labels.append(interpro_id)
                fragment_ids.append(frag_id)

    return fragments, labels, fragment_ids


def get_model_name(model_path):
    model_path_lower = model_path.lower().rstrip("/")
    model_basename = os.path.basename(model_path_lower)
    if model_path_lower == "esmc":
        return "esmc"
    if (
        model_path_lower in {"interprot", "interprot_sae", "interprot-esm2-sae"}
        or "interprot" in model_path_lower
        or (model_basename.startswith("esm2_plm") and model_basename.endswith(".safetensors"))
    ):
        return "interprot"
    return model_path.rstrip("/").split("/")[-1]


def get_base_out_dir(out_dir, model_name, data_dir_name):
    if data_dir_name == "data":
        return os.path.join(out_dir, model_name)
    return os.path.join(out_dir, model_name, data_dir_name)


def load_embeddings_cache(cache_path, expected_count=None):
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if expected_count is not None and len(embeddings) != expected_count:
            print(
                f"Cache size mismatch for {cache_path}: "
                f"{len(embeddings)} != {expected_count}. Recomputing..."
            )
            return None
        print(f"Loaded embeddings cache from {cache_path}")
        return embeddings
    return None


def save_embeddings_cache(embeddings, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings cache to {cache_path}")


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
    accuracy = correct / total

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
    if label_to_idx is None:
        global_label_set = observed_label_set
    else:
        global_label_set = sorted(set(label_to_idx) | set(true_count) | set(pred_count))

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
        "acc": accuracy,
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


def load_plm_ref_cls_encoder_helpers():
    from plm_ref_cls import encode_sequences_batch, load_plm_model

    return load_plm_model, encode_sequences_batch


class FragmentMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed, seed_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_label_key(dataset_name, interpro_id):
    return f"{dataset_name}:{interpro_id}"


def split_label_key(label_key):
    if ":" not in label_key:
        return "", label_key
    return label_key.split(":", 1)


def parse_hidden_dims(hidden_dims):
    if not hidden_dims:
        return []
    return [int(dim.strip()) for dim in hidden_dims.split(",") if dim.strip()]


def class_weights_from_labels(label_indices, num_classes, mode, device):
    if mode == "none":
        return None

    counts = np.bincount(label_indices, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if mode == "inverse":
        weights[present] = counts[present].sum() / (counts[present] * present.sum())
    elif mode == "sqrt_inverse":
        weights[present] = np.sqrt(counts[present].sum() / (counts[present] * present.sum()))
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    weights[~present] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_all_fragment_data(data_root):
    train_sequences = []
    train_class_labels = []
    test_sets = {}
    global_class_labels = set()
    dataset_label_sets = {}

    for dataset_name in DATASET_NAMES:
        train_data = load_venusx_dataset(dataset_name, "train", data_root)
        test_data = load_venusx_dataset(dataset_name, "test", data_root)

        ds_train_sequences, ds_train_labels, ds_train_ids = extract_fragments_with_labels(
            train_data, dataset_name
        )
        ds_test_sequences, ds_test_labels, ds_test_ids = extract_fragments_with_labels(
            test_data, dataset_name
        )

        ds_train_class_labels = [make_label_key(dataset_name, label) for label in ds_train_labels]
        ds_test_class_labels = [make_label_key(dataset_name, label) for label in ds_test_labels]

        train_sequences.extend(ds_train_sequences)
        train_class_labels.extend(ds_train_class_labels)

        test_sets[dataset_name] = {
            "sequences": ds_test_sequences,
            "raw_labels": ds_test_labels,
            "class_labels": ds_test_class_labels,
            "ids": ds_test_ids,
        }

        dataset_label_sets[dataset_name] = set(ds_train_class_labels) | set(ds_test_class_labels)
        global_class_labels.update(dataset_label_sets[dataset_name])

        print(
            f"VenusX_{dataset_name}: train fragments={len(ds_train_sequences)}, "
            f"test fragments={len(ds_test_sequences)}, "
            f"independent labels={len(dataset_label_sets[dataset_name])}"
        )

    label_to_idx = {label: idx for idx, label in enumerate(sorted(global_class_labels))}
    dataset_label_counts = {
        dataset_name: len(dataset_label_sets[dataset_name]) for dataset_name in DATASET_NAMES
    }

    return {
        "train_sequences": train_sequences,
        "train_class_labels": train_class_labels,
        "test_sets": test_sets,
        "label_to_idx": label_to_idx,
        "dataset_label_counts": dataset_label_counts,
    }


def ensure_embeddings(
    cache_path,
    sequences,
    expected_count,
    model_state,
    model_path,
    device,
    encode_batch_size,
):
    embeddings = load_embeddings_cache(cache_path, expected_count=expected_count)
    if embeddings is not None:
        return embeddings, model_state

    model, tokenizer = model_state
    if model is None:
        load_plm_model, encode_sequences_batch = load_plm_ref_cls_encoder_helpers()
        model, tokenizer = load_plm_model(model_path, device)
    else:
        _, encode_sequences_batch = load_plm_ref_cls_encoder_helpers()

    embeddings = encode_sequences_batch(
        model,
        tokenizer,
        sequences,
        device,
        encode_batch_size,
        model_path,
    )
    save_embeddings_cache(embeddings, cache_path)
    return embeddings, (model, tokenizer)


def train_mlp(
    train_embeddings,
    train_label_indices,
    input_dim,
    num_classes,
    args,
    device,
):
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = FragmentMLPClassifier(input_dim, hidden_dims, num_classes, args.dropout).to(device)
    labels_np = np.asarray(train_label_indices, dtype=np.int64)
    dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(labels_np, dtype=torch.long),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    weights = class_weights_from_labels(labels_np, num_classes, args.class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_embeddings, batch_labels in pbar:
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            total += batch_size
            pbar.set_postfix(loss=f"{total_loss / max(total, 1):.6f}", acc=f"{correct / max(total, 1):.4f}")

        epoch_stats = {
            "epoch": epoch,
            "loss": total_loss / max(total, 1),
            "acc": correct / max(total, 1),
        }
        history.append(epoch_stats)
        print(
            f"Epoch {epoch}: train loss={epoch_stats['loss']:.6f}, "
            f"train acc={epoch_stats['acc']:.4f}"
        )

    return model, history


@torch.no_grad()
def predict(model, embeddings, batch_size, device):
    model.eval()
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_indices = []
    confidences = []

    for (batch_embeddings,) in tqdm(loader, desc="Predicting"):
        batch_embeddings = batch_embeddings.to(device)
        logits = model(batch_embeddings)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)
        pred_indices.extend(pred_idx.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())

    return pred_indices, confidences


def build_results(test_ids, raw_labels, class_labels, pred_indices, confidences, idx_to_label, label_to_idx):
    results = []
    for test_id, raw_label, true_class_label, pred_idx, confidence in zip(
        test_ids,
        raw_labels,
        class_labels,
        pred_indices,
        confidences,
    ):
        pred_class_label = idx_to_label[pred_idx]
        pred_dataset, pred_raw_label = split_label_key(pred_class_label)
        true_dataset, _ = split_label_key(true_class_label)
        results.append(
            {
                "query_sequence": test_id,
                "true_interpro_id": true_class_label,
                "true_raw_interpro_id": raw_label,
                "true_dataset": true_dataset,
                "true_label_idx": label_to_idx[true_class_label],
                "predicted_interpro_id": pred_class_label,
                "predicted_raw_interpro_id": pred_raw_label,
                "predicted_dataset": pred_dataset,
                "predicted_label_idx": pred_idx,
                "confidence": float(confidence),
            }
        )
    return results


def write_results_csv(results, csv_output):
    fieldnames = [
        "query_sequence",
        "true_interpro_id",
        "true_raw_interpro_id",
        "true_dataset",
        "true_label_idx",
        "predicted_interpro_id",
        "predicted_raw_interpro_id",
        "predicted_dataset",
        "predicted_label_idx",
        "confidence",
    ]
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main(args):
    set_seed(args.seed, seed_cuda=not args.cpu)

    eval_datasets = [args.dataset] if args.dataset is not None else args.datasets
    model_name = get_model_name(args.model_path)
    data_root = resolve_data_dir(args.data_dir)
    data_dir_name = get_data_dir_name(data_root)
    embedding_base_dir = get_base_out_dir(args.embedding_dir, model_name, data_dir_name)
    mlp_base_dir = get_base_out_dir(args.out_dir, model_name, data_dir_name)
    os.makedirs(embedding_base_dir, exist_ok=True)
    os.makedirs(mlp_base_dir, exist_ok=True)

    print(f"[*] Training PLM-embedding MLP baseline ({model_name})")
    print(f"Using dataset root: {data_root}")
    print(f"Embedding cache directory: {embedding_base_dir}")
    print(f"MLP output directory: {mlp_base_dir}")
    print(f"Training datasets: {', '.join(DATASET_NAMES)}")
    print(f"Evaluation datasets: {', '.join(eval_datasets)}")

    device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("[1] Loading datasets and building independent dataset-specific labels...")
    data = load_all_fragment_data(data_root)
    train_sequences = data["train_sequences"]
    train_class_labels = data["train_class_labels"]
    test_sets = data["test_sets"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    label_index_file = os.path.join(mlp_base_dir, "mlp_label_to_idx.json")
    with open(label_index_file, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    dataset_label_counts_file = os.path.join(mlp_base_dir, "mlp_dataset_label_counts.json")
    with open(dataset_label_counts_file, "w") as f:
        json.dump(data["dataset_label_counts"], f, indent=2)

    print(f"Combined train fragments: {len(train_sequences)}")
    print(f"Independent global class count: {len(label_to_idx)}")
    print(f"Per-dataset class counts: {data['dataset_label_counts']}")
    print(f"Label index saved to {label_index_file}")

    print("[2] Loading or computing PLM embeddings...")
    model_state = (None, None)
    train_cache_path = os.path.join(embedding_base_dir, "all_train_embeddings.npy")
    train_embeddings, model_state = ensure_embeddings(
        train_cache_path,
        train_sequences,
        len(train_sequences),
        model_state,
        args.model_path,
        device,
        args.encode_batch_size,
    )
    train_embeddings = train_embeddings.astype(np.float32, copy=False)
    train_label_indices = [label_to_idx[label] for label in train_class_labels]

    print("[3] Training MLP classifier...")
    classifier, train_history = train_mlp(
        train_embeddings,
        train_label_indices,
        train_embeddings.shape[1],
        len(label_to_idx),
        args,
        device,
    )

    checkpoint_file = os.path.join(mlp_base_dir, "mlp_classifier.pt")
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "input_dim": int(train_embeddings.shape[1]),
            "hidden_dims": parse_hidden_dims(args.hidden_dims),
            "num_classes": len(label_to_idx),
            "dropout": args.dropout,
            "label_to_idx": label_to_idx,
            "args": vars(args),
        },
        checkpoint_file,
    )
    print(f"Checkpoint saved to {checkpoint_file}")

    history_file = os.path.join(mlp_base_dir, "mlp_train_history.json")
    with open(history_file, "w") as f:
        json.dump(train_history, f, indent=2)

    all_metrics = {}
    csv_outputs = {}
    for dataset_name in eval_datasets:
        if dataset_name not in test_sets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_out_dir = os.path.join(mlp_base_dir, f"VenusX_{dataset_name}")
        os.makedirs(dataset_out_dir, exist_ok=True)

        test_sequences = test_sets[dataset_name]["sequences"]
        test_raw_labels = test_sets[dataset_name]["raw_labels"]
        test_class_labels = test_sets[dataset_name]["class_labels"]
        test_ids = test_sets[dataset_name]["ids"]

        print(f"[4:{dataset_name}] Loading or computing test embeddings...")
        test_cache_path = os.path.join(embedding_base_dir, f"VenusX_{dataset_name}", "test_embeddings.npy")
        test_embeddings, model_state = ensure_embeddings(
            test_cache_path,
            test_sequences,
            len(test_sequences),
            model_state,
            args.model_path,
            device,
            args.encode_batch_size,
        )
        test_embeddings = test_embeddings.astype(np.float32, copy=False)

        print(f"[5:{dataset_name}] Predicting with unified MLP classifier...")
        pred_indices, confidences = predict(classifier, test_embeddings, args.eval_batch_size, device)
        results = build_results(
            test_ids,
            test_raw_labels,
            test_class_labels,
            pred_indices,
            confidences,
            idx_to_label,
            label_to_idx,
        )

        print(f"[6:{dataset_name}] Saving results and metrics...")
        csv_output = os.path.join(dataset_out_dir, "mlp_predictions.csv")
        write_results_csv(results, csv_output)
        csv_outputs[dataset_name] = csv_output

        metrics = compute_classification_metrics(results, label_to_idx)
        metrics["avg_confidence"] = float(np.mean(confidences)) if confidences else 0.0
        metrics["predicted_out_of_dataset"] = int(
            sum(r["predicted_dataset"] != dataset_name for r in results)
        )
        metrics["predicted_out_of_dataset_rate"] = (
            metrics["predicted_out_of_dataset"] / len(results) if results else 0.0
        )
        all_metrics[dataset_name] = metrics

        metrics_file = os.path.join(dataset_out_dir, "mlp_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        metadata = {
            "dataset_name": dataset_name,
            "train_datasets": DATASET_NAMES,
            "data_dir": args.data_dir,
            "data_root": data_root,
            "model_path": args.model_path,
            "embedding_cache_dir": embedding_base_dir,
            "device": str(device),
            "num_train_fragments": len(train_sequences),
            "num_test_fragments": len(test_sequences),
            "num_global_labels": len(label_to_idx),
            "dataset_label_counts": data["dataset_label_counts"],
            "hidden_dims": parse_hidden_dims(args.hidden_dims),
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "class_weight": args.class_weight,
            "metrics": metrics,
        }

        metadata_file = os.path.join(dataset_out_dir, "mlp_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {csv_output}")
        print(f"Metrics saved to {metrics_file}")
        print(f"Total predictions: {len(results)}")
        print(f"Accuracy: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"Average confidence: {metrics['avg_confidence']:.4f}")
        print(
            "Predicted outside dataset: "
            f"{metrics['predicted_out_of_dataset']} "
            f"({metrics['predicted_out_of_dataset_rate']:.4f})"
        )

    all_metrics_file = os.path.join(mlp_base_dir, "mlp_all_metrics.json")
    with open(all_metrics_file, "w") as f:
        json.dump(
            {
                "train_datasets": DATASET_NAMES,
                "eval_datasets": eval_datasets,
                "num_train_fragments": len(train_sequences),
                "num_global_labels": len(label_to_idx),
                "dataset_label_counts": data["dataset_label_counts"],
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    print(f"All metrics saved to {all_metrics_file}")
    return csv_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLP baseline on cached PLM embeddings for VenusX fragment classification"
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

    parser.add_argument(
        "--embedding_dir",
        type=str,
        default=os.path.join(project_root, "baselines", "plm_ref_cls_results"),
        help="Directory containing or receiving plm_ref_cls.py embedding caches.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(project_root, "baselines", "plm_ref_cls_mlp_results"),
        help="Output directory for MLP predictions, metrics, and checkpoints.",
    )
    parser.add_argument("--data_dir", type=str, default="data_70", help="Dataset root directory")
    # parser.add_argument("--data_dir", type=str, default="data_30", help="Dataset root directory")
    parser.add_argument("--encode_batch_size", type=int, default=16, help="Batch size for PLM encoding")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size for MLP training")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size for MLP evaluation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=str, default="512,256")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class_weight", choices=["none", "inverse", "sqrt_inverse"], default="none")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", default=False, action="store_true", help="Force CPU usage")

    main(parser.parse_args())
