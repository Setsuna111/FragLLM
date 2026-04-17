"""
Split esm_embeddings_3B.pt into individual per-uid .pt files.
Each output file: <output_dir>/<uid>.pt  ->  tensor of shape (seq_len, 2560)
"""

import os
import torch
from tqdm import tqdm

INPUT_PATH = "esm_embeddings/esm2_t36_3B_UR50D/data/esm_embeddings_3B.pt"
OUTPUT_DIR = "esm_embeddings/esm2_t36_3B_UR50D/data/"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {INPUT_PATH} ...")
    data = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)

    seq_emb: dict = data["sequence_embeddings"]
    total = len(seq_emb)
    print(f"Total UIDs: {total}")

    skipped = 0
    for uid, tensor in tqdm(seq_emb.items(), total=total, desc="Splitting"):
        out_path = os.path.join(OUTPUT_DIR, f"{uid}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        torch.save(tensor, out_path)

    print(f"Done. Saved to {OUTPUT_DIR}  (skipped {skipped} existing files)")


if __name__ == "__main__":
    main()
