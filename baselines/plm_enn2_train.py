import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from plm_enn2_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_ROOT,
    ProteinDatasetPrecomputed,
    build_label_map,
    collate_fn,
    create_classifier,
    get_embedding_dir,
    get_result_dir,
    load_combined_data,
    parse_dataset_names,
    precompute_embeddings,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_iou(pred_bin, label_bin):
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
    classifier.eval()
    all_iou = []
    results = []
    for uids, embs, labels, lengths in tqdm(loader, desc="Evaluating"):
        logits = classifier(embs.to(device)).cpu()
        for i, (uid, length) in enumerate(zip(uids, lengths)):
            L = length.item()
            pred_bin = (logits[i, :L].numpy() > threshold).astype(np.int32)
            label_bin = labels[i, :L].numpy().astype(np.int32)
            iou = compute_iou(pred_bin, label_bin)
            all_iou.append(iou)
            results.append({"uid": uid, "iou": iou})
    return results, float(np.mean(all_iou)) if all_iou else 0.0


@torch.no_grad()
def evaluate_ensemble(models, loader, device, threshold=0.5):
    for model in models:
        model.eval()
    all_iou = []
    results = []
    for uids, embs, labels, lengths in tqdm(loader, desc="Evaluating Ensemble"):
        embs = embs.to(device)
        logits_sum = None
        for model in models:
            logits = model(embs)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = (logits_sum / len(models)).cpu()
        for i, (uid, length) in enumerate(zip(uids, lengths)):
            L = length.item()
            pred_bin = (logits_avg[i, :L].numpy() > threshold).astype(np.int32)
            label_bin = labels[i, :L].numpy().astype(np.int32)
            iou = compute_iou(pred_bin, label_bin)
            all_iou.append(iou)
            results.append({"uid": uid, "iou": iou})
    return results, float(np.mean(all_iou)) if all_iou else 0.0


def train_single_model(classifier, train_dataset, eval_dataset, device, args, seed):
    set_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_iou = 0.0

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        total_tokens = 0
        pbar = tqdm(loader, desc=f"[Seed {seed}] Epoch {epoch + 1}/{args.epochs}")
        for _, embs, labels, lengths in pbar:
            embs = embs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = classifier(embs)
            L_max = embs.shape[1]
            mask = torch.arange(L_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(logits[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{total_loss / total_tokens:.6f}")

        _, mean_iou = evaluate(classifier, eval_loader, device, args.threshold)
        best_iou = max(best_iou, mean_iou)
        print(f"[Seed {seed}] epoch={epoch + 1} eval_mean_iou={mean_iou:.6f} best={best_iou:.6f}")
    return best_iou


def main(args):
    dataset_names = parse_dataset_names(args.datasets)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    emb_dir = get_embedding_dir(args.data_dir, args.model_path, dataset_names)
    out_dir = get_result_dir(args.out_dir, args.data_dir, args.model_path, dataset_names, args.num_ensemble)
    os.makedirs(out_dir, exist_ok=True)

    if args.precompute_embeddings or not os.path.exists(emb_dir):
        precompute_embeddings(
            args.model_path,
            dataset_names,
            ["train", "test"],
            emb_dir,
            device,
            max_seq_len=args.max_seq_len,
            data_dir=args.data_dir,
        )

    train_data = load_combined_data(dataset_names, "train", data_dir=args.data_dir)
    test_data = load_combined_data(dataset_names, "test", data_dir=args.data_dir)
    label_map = build_label_map(train_data)
    num_classes = len(label_map)
    train_dataset = ProteinDatasetPrecomputed(train_data, label_map, num_classes, os.path.join(emb_dir, "train"), args.max_seq_len)
    test_dataset = ProteinDatasetPrecomputed(test_data, label_map, num_classes, os.path.join(emb_dir, "test"), args.max_seq_len)
    _, first_emb, _ = train_dataset[0]
    hidden_dim = first_emb.shape[1]

    architecture = {
        "num_filters": args.num_filters,
        "kernel_size": args.kernel_size,
        "num_layers": args.num_layers,
        "dilation_rate": args.dilation_rate,
        "first_dilated_layer": args.first_dilated_layer,
        "bottleneck_factor": args.bottleneck_factor,
        "dropout": args.dropout,
    }
    models = []
    best_ious = []
    for model_idx in range(args.num_ensemble):
        seed = args.base_seed + model_idx
        classifier = create_classifier(hidden_dim, num_classes, architecture, device)
        best_iou = train_single_model(classifier, train_dataset, test_dataset, device, args, seed)
        models.append(classifier)
        best_ious.append(best_iou)

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for idx, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(models_dir, f"model_{idx}.pt"))

    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    results, mean_iou = evaluate_ensemble(models, eval_loader, device, args.threshold)
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "results.csv"), index=False)

    meta = {
        "datasets": dataset_names,
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "num_classes": num_classes,
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "ensemble_size": args.num_ensemble,
        "base_seed": args.base_seed,
        "mean_iou": mean_iou,
        "best_iou_mean": float(np.mean(best_ious)) if best_ious else 0.0,
        "threshold": args.threshold,
        "architecture": architecture,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PLM-ENN2 ensemble baseline")
    parser.add_argument("--datasets", type=str, default="Act,BindI,Dom,Evo,Motif")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--precompute_embeddings", action="store_true")
    parser.add_argument("--num_filters", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dilation_rate", type=int, default=3)
    parser.add_argument("--first_dilated_layer", type=int, default=2)
    parser.add_argument("--bottleneck_factor", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_RESULTS_ROOT)
    main(parser.parse_args())
