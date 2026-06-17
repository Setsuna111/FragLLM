"""
Compute retrieval recall for referring description results.

The metric maps generated descriptions into the same embedding space as the
dataset label descriptions, retrieves the nearest labels, and checks whether
the true InterPro label is in top-k.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import hashlib
import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


argParser = argparse.ArgumentParser()
argParser.add_argument("--results_path", type=str, default="./eval_results/referring_desc/0529_all_123500/ActRefDesc_results.csv", help="path to referring description results CSV")
argParser.add_argument("--embedding_model", type=str, default="/home/dataset-local/projects_dir/pretrained_model/Qwen3-Embedding-0.6B/")
argParser.add_argument("--interpro_db_path", type=str, default="/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json")
argParser.add_argument("--cache_dir", type=str, default="/home/dataset-local/projects_dir/FragLLM/eval/cache/")
argParser.add_argument("--batch_size", type=int, default=16)
argParser.add_argument("--top_k", type=int, default=5)
argParser.add_argument("--device", type=str, default="cuda")
argParser.add_argument("--verbose", action="store_true")


def sanitize_name(value: str) -> str:
    return value.strip("/").replace("/", "_").replace("\\", "_").replace(":", "_")


def load_results(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    if "dataset_idx" in df.columns:
        df = df.drop_duplicates(subset=["dataset_idx"], keep="first")

    required_cols = ["generated", "reference", "interpro_ids"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {results_path}: {missing_cols}")

    df = df[required_cols].copy()
    df["generated"] = df["generated"].fillna("").astype(str)
    df["reference"] = df["reference"].fillna("").astype(str)
    df["interpro_ids"] = df["interpro_ids"].fillna("").astype(str)
    return df


def load_interpro_label_table(interpro_db_path: str) -> Tuple[List[str], List[str]]:
    with open(interpro_db_path, "r", encoding="utf-8") as f:
        interpro_data = json.load(f)

    label_ids = []
    label_texts = []
    for interpro_id, entry in interpro_data.items():
        label_ids.append(str(interpro_id))
        label_texts.append(str(entry.get("description") or ""))

    return label_ids, label_texts


def label_hash(label_ids: List[str], label_texts: List[str]) -> str:
    payload = json.dumps(
        [{"interpro_id": label_id, "description": text} for label_id, text in zip(label_ids, label_texts)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


class QwenDescriptionEncoder:
    def __init__(self, embedding_model: str, batch_size: int, device: str):
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model)
        if self.device != "cpu":
            self.embedding_model = self.embedding_model.to(self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        print(f"Encoding {len(texts)} texts in {total_batches} batches (batch_size={self.batch_size})")

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batch_num = start // self.batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")

            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                )
            all_embeddings.append(batch_embeddings)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embeddings = np.vstack(all_embeddings)
        print(f"Successfully encoded {len(texts)} texts to embeddings shape: {embeddings.shape}")
        return embeddings


def get_label_cache_path(
    cache_dir: str,
    interpro_db_path: str,
    embedding_model: str,
    label_ids: List[str],
    label_texts: List[str],
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"desc_label_embeddings_interpro_description_for_topk_accuracy.pkl"
    return os.path.join(cache_dir, filename)


def load_or_compute_label_embeddings(
    cache_path: str,
    encoder: QwenDescriptionEncoder,
    label_ids: List[str],
    label_texts: List[str],
) -> np.ndarray:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            if cache_data.get("interpro_ids") == label_ids and cache_data.get("descriptions") == label_texts:
                print(f"Loaded description label embeddings from cache: {cache_path}")
                return cache_data["embeddings"]
            print("Description label cache exists but label set changed, recomputing...")
        except Exception as exc:
            print(f"Error loading description label cache: {exc}, recomputing...")

    print("Computing description label embeddings...")
    label_embeddings = encoder.encode(label_texts)
    cache_data = {
        "embeddings": label_embeddings,
        "interpro_ids": label_ids,
        "descriptions": label_texts,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved description label embeddings to cache: {cache_path}")
    return label_embeddings


def compute_recall(
    prediction_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    true_label_ids: List[str],
    label_ids: List[str],
    top_k: int,
) -> Dict[str, float]:
    top_k = min(top_k, len(label_ids))
    label_id_to_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    true_indices = np.array([label_id_to_index[label_id] for label_id in true_label_ids])

    pred_tensor = torch.tensor(prediction_embeddings, dtype=torch.float32)
    label_tensor = torch.tensor(label_embeddings, dtype=torch.float32)
    pred_tensor = F.normalize(pred_tensor, p=2, dim=1)
    label_tensor = F.normalize(label_tensor, p=2, dim=1)

    similarities = torch.matmul(pred_tensor, label_tensor.T)
    top_indices = torch.topk(similarities, k=top_k, dim=1).indices.cpu().numpy()

    top1_hits = top_indices[:, 0] == true_indices
    topk_hits = np.array([true_idx in row for true_idx, row in zip(true_indices, top_indices)])

    return {
        "num_samples": int(len(true_label_ids)),
        "num_labels": int(len(label_ids)),
        "recall_at_1": float(top1_hits.mean()) if len(top1_hits) else 0.0,
        f"recall_at_{top_k}": float(topk_hits.mean()) if len(topk_hits) else 0.0,
        "top1_correct": int(top1_hits.sum()),
        f"top{top_k}_correct": int(topk_hits.sum()),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    df = load_results(args.results_path)
    label_ids, label_texts = load_interpro_label_table(args.interpro_db_path)
    unseen_labels = sorted(set(df["interpro_ids"].tolist()) - set(label_ids))
    if unseen_labels:
        raise ValueError(f"Found true labels not present in InterPro database: {unseen_labels[:5]}")

    print(f"Loaded {len(df)} unique samples from {args.results_path}")
    print(f"Loaded {len(label_ids)} InterPro description labels from {args.interpro_db_path}")

    encoder = QwenDescriptionEncoder(
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    cache_path = get_label_cache_path(
        cache_dir=args.cache_dir,
        interpro_db_path=args.interpro_db_path,
        embedding_model=args.embedding_model,
        label_ids=label_ids,
        label_texts=label_texts,
    )
    label_embeddings = load_or_compute_label_embeddings(cache_path, encoder, label_ids, label_texts)
    prediction_embeddings = encoder.encode(df["generated"].tolist())

    metrics = compute_recall(
        prediction_embeddings=prediction_embeddings,
        label_embeddings=label_embeddings,
        true_label_ids=df["interpro_ids"].tolist(),
        label_ids=label_ids,
        top_k=args.top_k,
    )
    metrics["label_cache_path"] = cache_path

    save_results_path = args.results_path.replace(".csv", "_desc_recall_metrics.json")
    with open(save_results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved recall metrics to {save_results_path}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parsed_args = argParser.parse_args()
    print("####################")
    for key, value in parsed_args.__dict__.items():
        print(f"{key}: {value}")
    print("####################")
    evaluate(parsed_args)
