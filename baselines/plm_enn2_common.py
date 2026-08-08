import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baselines.grounding_alignment import (  # noqa: E402
    CROP_POLICY_VERSION,
    attach_view_identity,
    cache_dir,
    crop_sequence,
    load_cached_embedding,
    localize_spans,
    plan_center_crop,
    read_jsonl,
    validate_manifest_record,
)

DEFAULT_DATA_DIR = os.path.join(project_root, "data_70")
# DEFAULT_DATA_DIR = os.path.join(project_root, "data_frag_50")
DEFAULT_MODEL_PATH = "/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/"
DEFAULT_EMBEDDINGS_ROOT = os.path.join(project_root, "baselines", "plm_embedding_plmenn")
LEGACY_EMBEDDINGS_ROOT = os.path.join(project_root, "baselines", "plm_embeddings")
DEFAULT_RESULTS_ROOT = os.path.join(project_root, "baselines", "plm_enn2_results")


def parse_dataset_names(datasets):
    return [name.strip() for name in datasets.split(",") if name.strip()]


def get_data_dir_name(data_dir):
    return os.path.basename(os.path.normpath(data_dir))


def get_model_name(model_path):
    return model_path.rstrip("/").split("/")[-1]


def get_embedding_dir(
    data_dir,
    model_path,
    dataset_names,
    max_seq_len=1021,
    embeddings_root=DEFAULT_EMBEDDINGS_ROOT,
):
    return str(
        cache_dir(
            embeddings_root,
            data_dir,
            model_path,
            dataset_names,
            max_seq_len,
        )
    )


def get_result_dir(out_dir, data_dir, model_path, dataset_names, num_ensemble):
    return os.path.join(
        out_dir,
        get_data_dir_name(data_dir),
        get_model_name(model_path),
        "+".join(dataset_names),
        f"ensemble_{num_ensemble}",
    )


def load_venusx_dataset(dataset_name, split, data_dir=None):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    path = os.path.join(data_dir, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_combined_data(dataset_names, split, data_dir=None):
    combined = []
    for name in dataset_names:
        for raw_index, protein in enumerate(
            load_venusx_dataset(name, split, data_dir=data_dir)
        ):
            record = dict(protein)
            record["_dataset_name"] = name
            record["_raw_record_index"] = raw_index
            record["_split"] = split
            combined.append(record)
    return combined


def build_label_map(train_data):
    ids = sorted({fg["interpro_id"] for p in train_data for fg in p["fragments"]})
    return {iid: i for i, iid in enumerate(ids)}


def build_idx_to_label(label_map):
    return {idx: label for label, idx in label_map.items()}


def expand_labels(sparse_labels, seq_len, num_classes, device):
    label = torch.zeros(seq_len, num_classes, dtype=torch.bool, device=device)
    for cls_idx, s, e in sparse_labels:
        label[s:e, cls_idx] = True
    return label


def load_plm_model(model_path, device):
    print(f"Loading PLM from {model_path}...")
    if "t5" in model_path.lower() and "ankh" not in model_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_path)
        is_t5 = True
    elif "ankh" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5EncoderModel.from_pretrained(model_path)
        is_t5 = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        is_t5 = False
    model = model.to(device).eval()
    return model, tokenizer, is_t5


@torch.no_grad()
def encode_sequence(plm, tokenizer, sequence, device, is_t5):
    seq_input = " ".join(list(sequence)) if is_t5 else sequence
    inputs = tokenizer(seq_input, return_tensors="pt", truncation=True, max_length=1026)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = plm(**inputs)
    hidden = out.last_hidden_state[0]
    mask = inputs["attention_mask"][0].bool()
    valid = hidden[mask]
    if len(valid) > 2:
        valid = valid[1:-1]
    L = len(sequence)
    if len(valid) > L:
        valid = valid[:L]
    return valid


def _fragment_group_record(group, group_index, crop_start):
    global_spans = [
        (int(frag["start_position"]), int(frag["end_position"]))
        for frag in group["frags"]
    ]
    local_spans = [
        (start - crop_start, end - crop_start + 1)
        for start, end in global_spans
    ]
    return {
        "fragment_group_index": int(group_index),
        "interpro_id": str(group.get("interpro_id", "")),
        "category": str(group.get("category", "")),
        "global_spans": global_spans,
        "local_spans": local_spans,
    }


def _base_planned_record(protein, split, task_mode, sample_id, crop, sequence):
    cropped = crop_sequence(sequence, crop)
    return {
        "sample_id": sample_id,
        "dataset_name": str(protein["_dataset_name"]),
        "split": split,
        "task_mode": task_mode,
        "uid": str(protein["uid"]),
        "raw_record_index": int(protein["_raw_record_index"]),
        "original_sequence_length": len(sequence),
        "crop_start": crop.crop_start,
        "crop_end": crop.crop_end,
        "cropped_sequence_length": crop.crop_length,
        "envelope_start": crop.envelope_start,
        "envelope_end": crop.envelope_end,
        "envelope_length": crop.envelope_length,
        "sequence": cropped,
    }


def plan_group_samples(raw_data, split, max_seq_len, model_path=None):
    samples = []
    filtered = []
    for protein in raw_data:
        sequence = str(protein["sequence"])
        all_spans = [
            (int(frag["start_position"]), int(frag["end_position"]))
            for group in protein["fragments"]
            for frag in group["frags"]
        ]
        sample_id = (
            f"group:{split}:{protein['_dataset_name']}:{protein['uid']}:"
            f"{protein['_raw_record_index']}"
        )
        crop = plan_center_crop(len(sequence), all_spans, max_seq_len)
        if crop is None:
            envelope_start = min(start for start, _ in all_spans)
            envelope_end = max(end for _, end in all_spans)
            filtered.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": protein["_dataset_name"],
                    "split": split,
                    "task_mode": f"{split}_group",
                    "uid": str(protein["uid"]),
                    "raw_record_index": int(protein["_raw_record_index"]),
                    "envelope_length": envelope_end - envelope_start + 1,
                    "max_seq_len": int(max_seq_len),
                    "reason": "envelope_exceeds_max_len",
                }
            )
            continue
        record = _base_planned_record(
            protein, split, f"{split}_group", sample_id, crop, sequence
        )
        record["fragment_groups"] = [
            _fragment_group_record(group, group_index, crop.crop_start)
            for group_index, group in enumerate(protein["fragments"])
        ]
        record["global_spans"] = all_spans
        record["local_spans"] = localize_spans(all_spans, crop)
        if model_path is not None:
            record = attach_view_identity(record, model_path)
        validate_manifest_record(record, max_seq_len)
        samples.append(record)
    stats = {
        "seen": len(raw_data),
        "retained": len(samples),
        "filtered_envelope": len(filtered),
    }
    return samples, filtered, stats


def plan_single_samples(raw_data, split, max_seq_len, model_path=None):
    samples = []
    filtered = []
    for protein in raw_data:
        sequence = str(protein["sequence"])
        for group_index, group in enumerate(protein["fragments"]):
            spans = [
                (int(frag["start_position"]), int(frag["end_position"]))
                for frag in group["frags"]
            ]
            sample_id = (
                f"single:{split}:{protein['_dataset_name']}:{protein['uid']}:"
                f"{protein['_raw_record_index']}:{group.get('interpro_id', '')}:{group_index}"
            )
            crop = plan_center_crop(len(sequence), spans, max_seq_len)
            if crop is None:
                envelope_start = min(start for start, _ in spans)
                envelope_end = max(end for _, end in spans)
                filtered.append(
                    {
                        "sample_id": sample_id,
                        "dataset_name": protein["_dataset_name"],
                        "split": split,
                        "task_mode": f"{split}_single",
                        "uid": str(protein["uid"]),
                        "raw_record_index": int(protein["_raw_record_index"]),
                        "fragment_group_index": group_index,
                        "interpro_id": str(group.get("interpro_id", "")),
                        "envelope_length": envelope_end - envelope_start + 1,
                        "max_seq_len": int(max_seq_len),
                        "reason": "envelope_exceeds_max_len",
                    }
                )
                continue
            record = _base_planned_record(
                protein, split, f"{split}_single", sample_id, crop, sequence
            )
            record["fragment_group_index"] = int(group_index)
            record["interpro_id"] = str(group.get("interpro_id", ""))
            record["category"] = str(group.get("category", ""))
            record["global_spans"] = spans
            record["local_spans"] = localize_spans(spans, crop)
            if model_path is not None:
                record = attach_view_identity(record, model_path)
            validate_manifest_record(record, max_seq_len)
            samples.append(record)
    stats = {
        "seen": len(samples) + len(filtered),
        "retained": len(samples),
        "filtered_envelope": len(filtered),
    }
    return samples, filtered, stats


def load_planned_samples(embedding_dir, mode, max_seq_len):
    records = read_jsonl(Path(embedding_dir) / mode / "manifest.jsonl")
    for record in records:
        validate_manifest_record(record, max_seq_len)
    return records


def load_sample_embedding(embedding_dir, sample):
    return load_cached_embedding(embedding_dir, sample)


class ProteinDatasetPrecomputed(Dataset):
    def __init__(self, samples, label_map, num_classes, emb_dir):
        self.num_classes = num_classes
        self.emb_dir = emb_dir
        self.samples = []

        for sample in tqdm(samples, desc="Building dataset"):
            sparse_labels = []
            for fg in sample["fragment_groups"]:
                cls_idx = label_map.get(fg["interpro_id"])
                if cls_idx is None:
                    continue
                for start, end in fg["local_spans"]:
                    sparse_labels.append((cls_idx, int(start), int(end)))
            self.samples.append({**sample, "sparse_labels": sparse_labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        emb = load_cached_embedding(self.emb_dir, sample)
        label = expand_labels(
            sample["sparse_labels"],
            int(sample["cropped_sequence_length"]),
            self.num_classes,
            torch.device("cpu"),
        ).float()
        return sample["sample_id"], emb, label


def collate_fn(batch):
    uids, embs, labels = zip(*batch)
    lengths = [e.shape[0] for e in embs]
    max_len = max(lengths)
    H = embs[0].shape[1]
    C = labels[0].shape[1]
    B = len(embs)

    padded_embs = torch.zeros(B, max_len, H)
    padded_labels = torch.zeros(B, max_len, C)
    for i, (e, l) in enumerate(zip(embs, labels)):
        L = e.shape[0]
        padded_embs[i, :L] = e
        padded_labels[i, :L] = l
    return list(uids), padded_embs, padded_labels, torch.tensor(lengths, dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, dilation_rate, bottleneck_factor=0.5, dropout=0.1):
        super().__init__()
        bottleneck_channels = max(1, int(num_channels * bottleneck_factor))
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.conv1 = nn.Conv1d(
            num_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding="same",
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.conv2 = nn.Conv1d(bottleneck_channels, num_channels, kernel_size=1, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class ProtENN2StyleClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_filters=512,
        kernel_size=9,
        num_layers=5,
        dilation_rate=3,
        first_dilated_layer=2,
        bottleneck_factor=0.5,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(hidden_dim, num_filters, kernel_size=1)
        self.residual_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            shifted_idx = layer_idx - first_dilated_layer + 1
            dilation = max(1, dilation_rate ** shifted_idx) if shifted_idx > 0 else 1
            self.residual_blocks.append(
                ResidualBlock(
                    num_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    bottleneck_factor=bottleneck_factor,
                    dropout=dropout,
                )
            )
        self.output_proj = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)


def get_architecture_from_metadata(metadata):
    return metadata.get("architecture", {})


def create_classifier(hidden_dim, num_classes, architecture, device):
    return ProtENN2StyleClassifier(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_filters=architecture.get("num_filters", 512),
        kernel_size=architecture.get("kernel_size", 9),
        num_layers=architecture.get("num_layers", 5),
        dilation_rate=architecture.get("dilation_rate", 3),
        first_dilated_layer=architecture.get("first_dilated_layer", 2),
        bottleneck_factor=architecture.get("bottleneck_factor", 0.5),
        dropout=architecture.get("dropout", 0.1),
    ).to(device)


def load_metadata(result_dir):
    with open(os.path.join(result_dir, "metadata.json")) as f:
        return json.load(f)


def load_ensemble_models(result_dir, hidden_dim, num_classes, architecture, device, ensemble_size):
    models = []
    for idx in range(ensemble_size):
        model = create_classifier(hidden_dim, num_classes, architecture, device)
        model_path = os.path.join(result_dir, "models", f"model_{idx}.pt")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_logits_ensemble(models, emb, device):
    emb = emb.unsqueeze(0).to(device)
    logits_sum = None
    for model in models:
        logits = model(emb)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return (logits_sum / len(models))[0].cpu()


def residue_iou(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return None
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float(intersection / union)


def protein_fragment_groups_to_masks(sample, label_map, seq_len, max_seq_len=1021):
    masks = {}
    for fg in sample["fragment_groups"]:
        cls_idx = label_map.get(fg["interpro_id"])
        if cls_idx is None:
            continue
        mask = masks.setdefault(cls_idx, np.zeros(seq_len, dtype=bool))
        for start, end in fg["local_spans"]:
            start = int(start)
            end = int(end)
            if not (0 <= start < end <= seq_len):
                raise ValueError(
                    f"invalid local span {(start, end)} for {sample['sample_id']}"
                )
            mask[start:end] = True
    return masks
