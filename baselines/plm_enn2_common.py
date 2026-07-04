import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

DEFAULT_DATA_DIR = os.path.join(project_root, "data_70")
DEFAULT_MODEL_PATH = "/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/"
DEFAULT_EMBEDDINGS_ROOT = os.path.join(project_root, "baselines", "plm_embeddings")
DEFAULT_RESULTS_ROOT = os.path.join(project_root, "baselines", "plm_enn2_results")


def parse_dataset_names(datasets):
    return [name.strip() for name in datasets.split(",") if name.strip()]


def get_data_dir_name(data_dir):
    return os.path.basename(os.path.normpath(data_dir))


def get_model_name(model_path):
    return model_path.rstrip("/").split("/")[-1]


def get_embedding_dir(data_dir, model_path, dataset_names):
    return os.path.join(
        DEFAULT_EMBEDDINGS_ROOT,
        get_data_dir_name(data_dir),
        get_model_name(model_path),
        "+".join(dataset_names),
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
        combined.extend(load_venusx_dataset(name, split, data_dir=data_dir))
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


def precompute_embeddings(model_path, dataset_names, splits, output_dir, device, max_seq_len=1024, data_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    plm, tokenizer, is_t5 = load_plm_model(model_path, device)

    for split in splits:
        print(f"\n=== Processing {split} split ===")
        data = load_combined_data(dataset_names, split, data_dir=data_dir)
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for protein in tqdm(data, desc=f"Encoding {split}"):
            uid = protein["uid"]
            sequence = protein["sequence"][:max_seq_len]
            emb_path = os.path.join(split_dir, f"{uid}.pt")
            if os.path.exists(emb_path):
                continue
            emb = encode_sequence(plm, tokenizer, sequence, device, is_t5)
            torch.save(emb.cpu(), emb_path)


class ProteinDatasetPrecomputed(Dataset):
    def __init__(self, raw_data, label_map, num_classes, emb_dir, max_seq_len=1024):
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.emb_dir = emb_dir
        self.samples = []

        for protein in tqdm(raw_data, desc="Building dataset"):
            uid = protein["uid"]
            seq = protein["sequence"][:max_seq_len]
            L = len(seq)
            sparse_labels = []
            for fg in protein["fragments"]:
                cls_idx = label_map.get(fg["interpro_id"])
                if cls_idx is None:
                    continue
                for frag in fg["frags"]:
                    s = frag["start_position"]
                    e = min(frag["end_position"] + 1, max_seq_len)
                    if s < L:
                        sparse_labels.append((cls_idx, s, e))
            self.samples.append({"uid": uid, "sparse_labels": sparse_labels, "seq_len": L})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample["uid"]
        emb = torch.load(os.path.join(self.emb_dir, f"{uid}.pt"))
        label = expand_labels(
            sample["sparse_labels"],
            sample["seq_len"],
            self.num_classes,
            torch.device("cpu"),
        ).float()
        return uid, emb, label


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


def protein_fragment_groups_to_masks(protein, label_map, seq_len, max_seq_len=1024):
    masks = {}
    for fg in protein["fragments"]:
        cls_idx = label_map.get(fg["interpro_id"])
        if cls_idx is None:
            continue
        mask = masks.setdefault(cls_idx, np.zeros(seq_len, dtype=bool))
        for frag in fg["frags"]:
            s = frag["start_position"]
            e = min(frag["end_position"] + 1, max_seq_len, seq_len)
            if s < seq_len:
                mask[s:e] = True
    return masks

