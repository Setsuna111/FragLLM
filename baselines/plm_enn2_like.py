import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_venusx_dataset(dataset_name, split):
    path = os.path.join(project_root, "data", f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_combined_data(dataset_names, split):
    """Load and concatenate multiple VenusX datasets."""
    combined = []
    for name in dataset_names:
        combined.extend(load_venusx_dataset(name, split))
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


class ProteinDataset(Dataset):
    """
    Stores each protein as a sparse label representation.
    __getitem__ expands sparse_labels into a dense (seq_len, num_classes) float tensor.
    """
    def __init__(self, raw_data, label_map, num_classes, max_seq_len=1024):
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.samples = []
        for protein in tqdm(raw_data):
            seq = protein['sequence'][:max_seq_len]
            L = len(seq)
            sparse_labels = []
            for fg in protein['fragments']:
                cls_idx = label_map.get(fg['interpro_id'])
                if cls_idx is None:
                    continue
                for frag in fg['frags']:
                    s = frag['start_position'] - 1
                    e = min(frag['end_position'], max_seq_len)
                    if s < L:
                        sparse_labels.append((cls_idx, s, e))
            self.samples.append({'uid': protein['uid'], 'sequence': seq,
                                  'sparse_labels': sparse_labels, 'seq_len': L})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = expand_labels(
            sample['sparse_labels'], sample['seq_len'], self.num_classes, torch.device('cpu')
        ).float()  # (seq_len, num_classes)
        return sample['uid'], sample['sequence'], label


# ─── PLM Encoder ─────────────────────────────────────────────────────────────

def load_plm_model(model_path, device):
    print(f"Loading PLM from {model_path}...")
    if "t5" in model_path.lower() and "ankh" not in model_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_path)
        is_t5 = True
    elif "ankh" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5EncoderModel.from_pretrained(model_path)
        is_t5 = False  # Ankh handles its own tokenization
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


# ─── MLP Head ────────────────────────────────────────────────────────────────

class ResidueClassifier(nn.Module):
    """Per-residue MLP: hidden_dim → num_classes (multi-label)."""
    def __init__(self, hidden_dim, num_classes, mlp_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # (L, num_classes), raw logits


# ─── Training ────────────────────────────────────────────────────────────────

def train(plm, tokenizer, is_t5, classifier, dataset, device, epochs, lr, batch_proteins,
          eval_dataset=None, threshold=0.5):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    classifier.train()

    indices = list(range(len(dataset)))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        optimizer.zero_grad()
        pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}")
        for idx in pbar:
            _, sequence, label = dataset[idx]
            emb = encode_sequence(plm, tokenizer, sequence, device, is_t5)  # (L, H)
            label = label.to(device)  # (L, C)

            logits = classifier(emb)  # (L, C)
            loss = criterion(logits, label) / batch_proteins
            loss.backward()
            total_loss += loss.item() * batch_proteins
            steps += 1

            pbar.set_postfix(loss=f"{total_loss / steps:.6f}")

            if steps % batch_proteins == 0:
                optimizer.step()
                optimizer.zero_grad()

        if steps % batch_proteins != 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"  Epoch {epoch+1} avg loss: {total_loss / len(dataset):.6f}")

        if eval_dataset is not None:
            print(f"  [Epoch {epoch+1}] Evaluating on test set...")
            _, mean_iou = evaluate(plm, tokenizer, is_t5, classifier, eval_dataset, device, threshold)
            print(f"  [Epoch {epoch+1}] Test Mean IoU: {mean_iou:.6f}")
            classifier.train()


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
            continue  # skip absent classes
        intersection = (p & g).sum()
        union = (p | g).sum()
        ious.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate(plm, tokenizer, is_t5, classifier, dataset, device, threshold=0.5):
    classifier.eval()
    all_iou = []
    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        uid, sequence, label = dataset[idx]
        emb = encode_sequence(plm, tokenizer, sequence, device, is_t5)
        logits = classifier(emb).cpu().numpy()          # (L, C)
        pred_bin = (logits > threshold).astype(np.int32)
        label_bin = label.numpy().astype(np.int32)      # already expanded in __getitem__

        iou = compute_iou(pred_bin, label_bin)
        all_iou.append(iou)
        results.append({'uid': uid, 'iou': iou})

    mean_iou = float(np.mean(all_iou))
    print(f"Mean IoU: {mean_iou:.6f}")
    return results, mean_iou


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    dataset_names = [d.strip() for d in args.datasets.split(',')]
    model_name = args.model_path.rstrip('/').split('/')[-1]
    out_dir = os.path.join(args.out_dir, model_name, '+'.join(dataset_names))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Datasets: {dataset_names}  |  Device: {device}")

    # Load data
    train_data = load_combined_data(dataset_names, 'train')
    test_data  = load_combined_data(dataset_names, 'test')

    label_map = build_label_map(train_data)
    num_classes = len(label_map)
    print(f"Total classes: {num_classes}")

    train_samples = ProteinDataset(train_data, label_map, num_classes, args.max_seq_len)
    test_samples  = ProteinDataset(test_data,  label_map, num_classes, args.max_seq_len)

    # Load PLM
    plm, tokenizer, is_t5 = load_plm_model(args.model_path, device)
    hidden_dim = plm.config.hidden_size

    # Build classifier
    classifier = ResidueClassifier(hidden_dim, num_classes, args.mlp_hidden).to(device)

    # Train
    train(plm, tokenizer, is_t5, classifier, train_samples, device,
          args.epochs, args.lr, args.batch_proteins,
          eval_dataset=test_samples, threshold=args.threshold)

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, 'classifier.pt')
    torch.save(classifier.state_dict(), ckpt_path)
    print(f"Classifier saved to {ckpt_path}")

    # Evaluate
    results, mean_iou = evaluate(plm, tokenizer, is_t5, classifier, test_samples, device, args.threshold)

    # Save results
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    meta = {
        'datasets': dataset_names,
        'model_path': args.model_path,
        'num_classes': num_classes,
        'num_train': len(train_samples),
        'num_test': len(test_samples),
        'mean_iou': mean_iou,
        'threshold': args.threshold,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Results saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PLM + MLP residue segmentation baseline for VenusX")
    parser.add_argument('--datasets', type=str, default="Act,BindI,Dom,Evo,Motif",
                        help='Comma-separated dataset names, e.g. "Act,BindI"')
    parser.add_argument('--model_path', type=str,
                        default='/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/',
                        help='Path to PLM')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--mlp_hidden', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_proteins', type=int, default=32,
                        help='Gradient accumulation over N proteins before optimizer step')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(project_root, 'baselines', 'plm_enn2_results'))
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
