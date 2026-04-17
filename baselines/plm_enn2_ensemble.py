import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_venusx_dataset(dataset_name, split, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")
    path = os.path.join(data_dir, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_combined_data(dataset_names, split, data_dir=None):
    """Load and concatenate multiple VenusX datasets."""
    combined = []
    for name in dataset_names:
        combined.extend(load_venusx_dataset(name, split, data_dir=data_dir))
    return combined


def build_label_map(train_data):
    """Collect all interpro_ids from training data and assign integer indices."""
    ids = sorted({fg['interpro_id'] for p in train_data for fg in p['fragments']})
    return {iid: i for i, iid in enumerate(ids)}


def expand_labels(sparse_labels, seq_len, num_classes, device):
    """Materialize sparse label list into a (seq_len, num_classes) bool tensor on device."""
    label = torch.zeros(seq_len, num_classes, dtype=torch.bool, device=device)
    for cls_idx, s, e in sparse_labels:
        label[s:e, cls_idx] = True
    return label


# ─── Precompute Embeddings ───────────────────────────────────────────────────

def load_plm_model(model_path, device):
    """Load PLM model for embedding extraction."""
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
    """Encode a single sequence; returns per-residue embeddings (L, hidden)."""
    seq_input = ' '.join(list(sequence)) if is_t5 else sequence
    inputs = tokenizer(seq_input, return_tensors="pt", truncation=True, max_length=1026)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = plm(**inputs)
    hidden = out.last_hidden_state[0]  # (T, H)
    # strip CLS/EOS special tokens
    mask = inputs['attention_mask'][0].bool()
    valid = hidden[mask]
    if len(valid) > 2:
        valid = valid[1:-1]
    # valid should now align with residues; trim/pad to match sequence length
    L = len(sequence)
    if len(valid) > L:
        valid = valid[:L]
    return valid  # (L, H)


def precompute_embeddings(model_path, dataset_names, splits, output_dir, device, max_seq_len=1024, data_dir=None):
    """
    Precompute embeddings for all proteins and save as individual .pt files.

    Args:
        model_path: Path to PLM model
        dataset_names: List of dataset names
        splits: List of splits (e.g., ['train', 'test'])
        output_dir: Directory to save embeddings
        device: torch device
        max_seq_len: Maximum sequence length
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load PLM
    plm, tokenizer, is_t5 = load_plm_model(model_path, device)

    # Process each split
    for split in splits:
        print(f"\n=== Processing {split} split ===")
        data = load_combined_data(dataset_names, split, data_dir=data_dir)

        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for protein in tqdm(data, desc=f"Encoding {split}"):
            uid = protein['uid']
            sequence = protein['sequence'][:max_seq_len]

            # Check if already computed
            emb_path = os.path.join(split_dir, f"{uid}.pt")
            if os.path.exists(emb_path):
                continue

            # Encode and save
            emb = encode_sequence(plm, tokenizer, sequence, device, is_t5)
            torch.save(emb.cpu(), emb_path)

    print(f"\nEmbeddings saved to {output_dir}")


# ─── Dataset with Precomputed Embeddings ─────────────────────────────────────

class ProteinDatasetPrecomputed(Dataset):
    """
    Dataset that loads precomputed embeddings from disk on-the-fly.
    Stores sparse labels to save memory.
    """
    def __init__(self, raw_data, label_map, num_classes, emb_dir, max_seq_len=1024):
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.emb_dir = emb_dir
        self.samples = []

        for protein in tqdm(raw_data, desc="Building dataset"):
            uid = protein['uid']
            seq = protein['sequence'][:max_seq_len]
            L = len(seq)

            # Build sparse labels
            sparse_labels = []
            for fg in protein['fragments']:
                cls_idx = label_map.get(fg['interpro_id'])
                if cls_idx is None:
                    continue
                for frag in fg['frags']:
                    s = frag['start_position']
                    e = min(frag['end_position'] + 1, max_seq_len)
                    if s < L:
                        sparse_labels.append((cls_idx, s, e))

            self.samples.append({
                'uid': uid,
                'sparse_labels': sparse_labels,
                'seq_len': L
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample['uid']

        # Load embedding from disk
        emb_path = os.path.join(self.emb_dir, f"{uid}.pt")
        emb = torch.load(emb_path)  # (L, H)

        # Expand labels
        label = expand_labels(
            sample['sparse_labels'],
            sample['seq_len'],
            self.num_classes,
            torch.device('cpu')
        ).float()  # (L, C)

        return uid, emb, label


def collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
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


# ─── ProtENN2-style Residual CNN ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolution, inspired by ProtENN2.

    Architecture:
        Input (C channels)
          ↓
        BatchNorm → ReLU
          ↓
        Conv1D (bottleneck, C*bottleneck_factor channels, dilation)
          ↓
        BatchNorm → ReLU
          ↓
        Conv1D (1x1, C channels)
          ↓
        Residual connection (+Input)
          ↓
        Output (C channels)
    """
    def __init__(self, num_channels, kernel_size, dilation_rate, bottleneck_factor=0.5, dropout=0.1):
        super().__init__()
        bottleneck_channels = max(1, int(num_channels * bottleneck_factor))

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.conv1 = nn.Conv1d(
            num_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding='same'
        )

        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            num_channels,
            kernel_size=1,
            padding='same'
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
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
    """
    ProtENN2-inspired residual CNN for per-residue classification.

    Architecture:
        Input embeddings (L, hidden_dim)
          ↓
        Transpose to (batch, hidden_dim, L)
          ↓
        Initial Conv1D (hidden_dim → num_filters)
          ↓
        5x Residual Blocks (with increasing dilation)
          ↓
        Final Conv1D (num_filters → num_classes)
          ↓
        Transpose to (L, num_classes)
    """
    def __init__(self, hidden_dim, num_classes, num_filters=512, kernel_size=9,
                 num_layers=5, dilation_rate=3, first_dilated_layer=2,
                 bottleneck_factor=0.5, dropout=0.1):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(hidden_dim, num_filters, kernel_size=1)

        # Residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            shifted_idx = layer_idx - first_dilated_layer + 1
            dilation = max(1, dilation_rate ** shifted_idx) if shifted_idx > 0 else 1

            block = ResidualBlock(
                num_channels=num_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                bottleneck_factor=bottleneck_factor,
                dropout=dropout
            )
            self.residual_blocks.append(block)

        # Output projection
        self.output_proj = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, L, hidden_dim) - batched per-residue embeddings
        Returns:
            (B, L, num_classes) - per-residue logits
        """
        # (B, L, H) → (B, H, L)
        x = x.transpose(1, 2)

        # Initial projection
        x = self.input_proj(x)  # (B, num_filters, L)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)  # (B, num_classes, L)

        # (B, num_classes, L) → (B, L, num_classes)
        x = x.transpose(1, 2)

        return x


# ─── Training ────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_model(classifier, dataset, device, epochs, lr, batch_size,
                       num_workers=4, eval_dataset=None, threshold=0.5, seed=42):
    """Train a single model with given seed."""
    set_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    ) if eval_dataset is not None else None

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    classifier.train()

    best_iou = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(loader, desc=f"[Seed {seed}] Epoch {epoch+1}/{epochs}")
        for _, embs, labels, lengths in pbar:
            embs = embs.to(device)      # (B, L_max, H)
            labels = labels.to(device)  # (B, L_max, C)
            lengths = lengths.to(device)

            logits = classifier(embs)   # (B, L_max, C)

            # Build mask to exclude padding from loss
            L_max = embs.shape[1]
            mask = torch.arange(L_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L_max)

            loss = criterion(logits[mask], labels[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{total_loss / total_tokens:.6f}")

        avg_loss = total_loss / total_tokens
        print(f"  [Seed {seed}] Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        if eval_loader is not None:
            print(f"  [Seed {seed}] Evaluating on test set...")
            _, mean_iou = evaluate(classifier, eval_loader, device, threshold)
            print(f"  [Seed {seed}] Test Mean IoU: {mean_iou:.6f}")
            if mean_iou > best_iou:
                best_iou = mean_iou
            classifier.train()

    return best_iou


def train_ensemble(hidden_dim, num_classes, train_dataset, test_dataset, device,
                   args, num_models=10):
    """
    Train multiple models with different random seeds for ensemble.

    Args:
        hidden_dim: Input embedding dimension
        num_classes: Number of output classes
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: torch device
        args: Training arguments
        num_models: Number of models in ensemble

    Returns:
        List of trained models
    """
    models = []
    best_ious = []

    for model_idx in range(num_models):
        seed = args.base_seed + model_idx
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx+1}/{num_models} (seed={seed})")
        print(f"{'='*60}")

        # Create new model
        classifier = ProtENN2StyleClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_filters=args.num_filters,
            kernel_size=args.kernel_size,
            num_layers=args.num_layers,
            dilation_rate=args.dilation_rate,
            first_dilated_layer=args.first_dilated_layer,
            bottleneck_factor=args.bottleneck_factor,
            dropout=args.dropout
        ).to(device)

        # Train
        best_iou = train_single_model(
            classifier=classifier,
            dataset=train_dataset,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_dataset=test_dataset,
            threshold=args.threshold,
            seed=seed
        )

        models.append(classifier)
        best_ious.append(best_iou)

        print(f"Model {model_idx+1} best IoU: {best_iou:.6f}")

    print(f"\n{'='*60}")
    print(f"Ensemble Training Complete")
    print(f"Mean best IoU: {np.mean(best_ious):.6f} ± {np.std(best_ious):.6f}")
    print(f"{'='*60}\n")

    return models


# ─── Evaluation ──────────────────────────────────────────────────────────────

def compute_iou(pred_bin, label_bin):
    """
    pred_bin, label_bin: (L, C) binary numpy arrays
    Returns mean IoU over classes that appear in ground truth.
    """
    ious = []
    for c in range(label_bin.shape[1]):
        p = pred_bin[:, c]
        g = label_bin[:, c]
        if g.sum() == 0 and p.sum() == 0:
            continue
        intersection = (p & g).sum()
        union = (p | g).sum()
        ious.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate(classifier, loader, device, threshold=0.5):
    """Evaluate a single model using a DataLoader."""
    classifier.eval()
    all_iou = []
    results = []

    for uids, embs, labels, lengths in tqdm(loader, desc="Evaluating"):
        embs = embs.to(device)   # (B, L_max, H)
        logits = classifier(embs).cpu()  # (B, L_max, C)

        for i, (uid, length) in enumerate(zip(uids, lengths)):
            L = length.item()
            pred_bin = (logits[i, :L].numpy() > threshold).astype(np.int32)
            label_bin = labels[i, :L].numpy().astype(np.int32)
            iou = compute_iou(pred_bin, label_bin)
            all_iou.append(iou)
            results.append({'uid': uid, 'iou': iou})

    mean_iou = float(np.mean(all_iou))
    print(f"Mean IoU: {mean_iou:.6f}")
    return results, mean_iou


@torch.no_grad()
def evaluate_ensemble(models, loader, device, threshold=0.5):
    """Evaluate ensemble by averaging logits across models."""
    for model in models:
        model.eval()

    all_iou = []
    results = []

    for uids, embs, labels, lengths in tqdm(loader, desc="Evaluating Ensemble"):
        embs = embs.to(device)  # (B, L_max, H)

        # Sum logits from all models
        logits_sum = None
        for model in models:
            logits = model(embs)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = (logits_sum / len(models)).cpu()  # (B, L_max, C)

        for i, (uid, length) in enumerate(zip(uids, lengths)):
            L = length.item()
            pred_bin = (logits_avg[i, :L].numpy() > threshold).astype(np.int32)
            label_bin = labels[i, :L].numpy().astype(np.int32)
            iou = compute_iou(pred_bin, label_bin)
            all_iou.append(iou)
            results.append({'uid': uid, 'iou': iou})

    mean_iou = float(np.mean(all_iou))
    print(f"Ensemble Mean IoU: {mean_iou:.6f}")
    return results, mean_iou


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    dataset_names = [d.strip() for d in args.datasets.split(',')]
    data_dir = args.data_dir
    model_name = args.model_path.rstrip('/').split('/')[-1]

    # Setup output directory
    out_dir = os.path.join(
        args.out_dir,
        model_name,
        '+'.join(dataset_names),
        f"ensemble_{args.num_ensemble}"
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Datasets: {dataset_names}  |  Device: {device}")
    print(f"Ensemble size: {args.num_ensemble}")

    # Setup embedding directory
    data_dir_name = os.path.basename(data_dir.rstrip('/'))
    emb_dir = os.path.join(
        project_root,
        'baselines',
        'plm_embeddings',
        data_dir_name,
        model_name,
        '+'.join(dataset_names)
    )

    # Precompute embeddings if needed
    if args.precompute_embeddings or not os.path.exists(emb_dir):
        print("\n=== Precomputing Embeddings ===")
        precompute_embeddings(
            model_path=args.model_path,
            dataset_names=dataset_names,
            splits=['train', 'test'],
            output_dir=emb_dir,
            device=device,
            max_seq_len=args.max_seq_len,
            data_dir=data_dir
        )
    else:
        print(f"\nUsing precomputed embeddings from {emb_dir}")

    # Load data for label mapping
    train_data = load_combined_data(dataset_names, 'train', data_dir=data_dir)
    test_data = load_combined_data(dataset_names, 'test', data_dir=data_dir)

    label_map = build_label_map(train_data)
    num_classes = len(label_map)
    print(f"Total classes: {num_classes}")

    # Create datasets with precomputed embeddings
    train_dataset = ProteinDatasetPrecomputed(
        train_data, label_map, num_classes,
        os.path.join(emb_dir, 'train'),
        args.max_seq_len
    )
    test_dataset = ProteinDatasetPrecomputed(
        test_data, label_map, num_classes,
        os.path.join(emb_dir, 'test'),
        args.max_seq_len
    )

    # Get hidden dimension from first sample
    _, first_emb, _ = train_dataset[0]
    hidden_dim = first_emb.shape[1]
    print(f"Embedding dimension: {hidden_dim}")

    # Train ensemble
    print(f"\n=== Training Ensemble ({args.num_ensemble} models) ===")
    models = train_ensemble(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        args=args,
        num_models=args.num_ensemble
    )

    # Save all models
    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for idx, model in enumerate(models):
        model_path = os.path.join(models_dir, f'model_{idx}.pt')
        torch.save(model.state_dict(), model_path)
    print(f"Models saved to {models_dir}")

    # Evaluate ensemble
    print("\n=== Evaluating Ensemble ===")
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    results, mean_iou = evaluate_ensemble(models, eval_loader, device, args.threshold)

    # Save results
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    # Save metadata
    meta = {
        'datasets': dataset_names,
        'model_path': args.model_path,
        'num_classes': num_classes,
        'num_train': len(train_dataset),
        'num_test': len(test_dataset),
        'ensemble_size': args.num_ensemble,
        'base_seed': args.base_seed,
        'mean_iou': mean_iou,
        'threshold': args.threshold,
        'architecture': {
            'num_filters': args.num_filters,
            'kernel_size': args.kernel_size,
            'num_layers': args.num_layers,
            'dilation_rate': args.dilation_rate,
            'first_dilated_layer': args.first_dilated_layer,
            'bottleneck_factor': args.bottleneck_factor,
            'dropout': args.dropout
        }
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"Final Ensemble Mean IoU: {mean_iou:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ProtENN2-style ensemble for VenusX with precomputed embeddings"
    )

    # Data arguments
    parser.add_argument('--datasets', type=str, default="Act,BindI,Dom,Evo,Motif",
                        help='Comma-separated dataset names')
    parser.add_argument('--data_dir', type=str, default="/home/dataset-local/projects_dir/FragLLM/data_70",
                        help='data_dir')
    parser.add_argument('--model_path', type=str,
                        default='/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/',
                        help='Path to PLM')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--precompute_embeddings', default=False,
                        help='Force recompute embeddings even if they exist -- 只有在编码一半意外终止时才True')

    # Model architecture arguments (ProtENN2-inspired)
    parser.add_argument('--num_filters', type=int, default=512,
                        help='Number of convolutional filters')
    parser.add_argument('--kernel_size', type=int, default=9,
                        help='Convolution kernel size')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of residual blocks')
    parser.add_argument('--dilation_rate', type=int, default=3,
                        help='Dilation rate base (exponential growth)')
    parser.add_argument('--first_dilated_layer', type=int, default=2,
                        help='First layer to apply dilation (0-indexed)')
    parser.add_argument('--bottleneck_factor', type=float, default=0.5,
                        help='Bottleneck compression factor')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of proteins per batch')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader worker processes')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:3')

    # Ensemble arguments
    parser.add_argument('--num_ensemble', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (each model uses base_seed + model_idx)')

    # Output arguments
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(project_root, 'baselines', 'plm_enn2_results'))
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    main(args)
