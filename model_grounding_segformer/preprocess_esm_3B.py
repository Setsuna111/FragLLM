"""
Preprocess protein sequences using ESM2 3B model to generate embeddings cache.
This script processes all unique protein sequences from datasets and pre-computes their ESM embeddings.

Usage:
    python preprocess_esm_3B.py --data_root ../data --output_path ./esm_embeddings_3B.pt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
from transformers import EsmModel, EsmTokenizer
from typing import Dict, List
from tqdm import tqdm
import argparse


def extract_sequences_from_dataset(data_path: str, max_sequence_length: int = 1021) -> Dict[str, str]:
    """
    Extract all unique protein sequences from a dataset file.

    Args:
        data_path: Path to the JSON data file
        max_sequence_length: Maximum sequence length to include

    Returns:
        Dictionary mapping uid to sequence
    """
    if not os.path.exists(data_path):
        return {}

    with open(data_path, 'r') as f:
        data = json.load(f)

    sequences = {}
    for item in data:
        uid = item.get("uid", "")
        sequence = item.get("sequence", "")
        if uid and sequence and len(sequence) <= max_sequence_length:
            sequences[uid] = sequence

    return sequences


def parse_data_names(data_name_str: str) -> List[str]:
    """Parse data_name string to extract multiple dataset names."""
    return [name.strip() for name in data_name_str.split("||")] if "||" in data_name_str else [data_name_str]


def get_all_sequences(data_root: str, data_names: List[str], max_sequence_length: int = 1021) -> Dict[str, str]:
    """
    Get all unique protein sequences from multiple datasets.

    Args:
        data_root: Root directory containing datasets
        data_names: List of dataset names
        max_sequence_length: Maximum sequence length to include

    Returns:
        Dictionary mapping uid to sequence
    """
    all_sequences = {}

    for data_name in data_names:
        for split in ["train", "valid", "test"]:
            data_path = os.path.join(data_root, data_name, f"{split}.json")
            sequences = extract_sequences_from_dataset(data_path, max_sequence_length)

            # Merge sequences (same uid should have same sequence)
            for uid, seq in sequences.items():
                if uid in all_sequences:
                    # Verify consistency
                    if all_sequences[uid] != seq:
                        print(f"Warning: Inconsistent sequence for uid {uid}")
                else:
                    all_sequences[uid] = seq

            if sequences:
                print(f"Found {len(sequences)} sequences in {data_name}/{split}")

    print(f"Total unique sequences: {len(all_sequences)}")
    return all_sequences


def encode_sequences_with_esm(
    sequences: Dict[str, str],
    esm_model_path: str,
    batch_size: int = 4,
    max_sequence_length: int = 1021,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Encode protein sequences using ESM2 3B model.

    Note: ESM tokenizer adds BOS (<cls>) and EOS (<eos>) tokens automatically.
    The returned embeddings have BOS/EOS tokens REMOVED, so the embedding length
    equals the sequence length.

    Args:
        sequences: Dictionary mapping uid to protein sequence
        esm_model_path: Path to ESM model
        batch_size: Batch size for encoding
        max_sequence_length: Maximum sequence length
        device: Device to use

    Returns:
        Dictionary mapping uid to embeddings (without BOS/EOS tokens)
    """
    print("Loading ESM model...")
    model = EsmModel.from_pretrained(esm_model_path).to(device)
    tokenizer = EsmTokenizer.from_pretrained(esm_model_path)

    model.eval()
    sequence_embeddings = {}

    # Convert to list for batch processing
    uid_list = list(sequences.keys())
    seq_list = list(sequences.values())

    print(f"Encoding {len(seq_list)} sequences...")

    for i in tqdm(range(0, len(seq_list), batch_size)):
        batch_uids = uid_list[i:i + batch_size]
        batch_sequences = seq_list[i:i + batch_size]

        # Tokenize batch
        # ESM tokenizer automatically adds BOS (<cls>) and EOS (<eos>) tokens
        tokenized = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=max_sequence_length + 2,  # +2 for BOS/EOS tokens
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Encode with ESM
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get last hidden state
            # Shape: (batch_size, seq_len+2, hidden_size) with BOS/EOS
            hidden_states = outputs.last_hidden_state

            # Remove BOS (first token) and EOS (last valid token) to match sequence length
            # The embedding should be for the actual residues only
            # BOS is at position 0, EOS is at the last valid position
            for j, uid in enumerate(batch_uids):
                seq_len = len(batch_sequences[j])
                # Extract embeddings for actual residues (positions 1 to seq_len inclusive)
                # Position 0 is BOS, position seq_len+1 is EOS (or padding)
                embedding = hidden_states[j, 1:seq_len+1, :].cpu()  # (seq_len, hidden_size)
                sequence_embeddings[uid] = embedding

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"Encoded {len(sequence_embeddings)} sequences")
    return sequence_embeddings


def main():
    parser = argparse.ArgumentParser(description="Preprocess ESM embeddings for ProteinSAM")

    parser.add_argument("--data_root", type=str, default="../data",
                       help="Root directory for datasets")
    parser.add_argument("--data_name", type=str,
                       default="VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo",
                       help="Dataset name(s), supports || separator for multiple datasets")
    parser.add_argument("--esm_model_path", type=str,
                       default="/home/dataset-local/projects_dir/pretrained_model/esm2_t36_3B_UR50D",
                       help="Path to ESM model")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for encoding (reduce if OOM)")
    parser.add_argument("--max_sequence_length", type=int, default=1021,
                       help="Maximum protein sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for encoding")
    parser.add_argument("--output_path", type=str, default="./esm_embeddings_3B.pt",
                       help="Output path for ESM embeddings")

    args = parser.parse_args()

    print("=== ESM Embedding Preprocessing for ProteinSAM ===")
    print(f"Data root: {args.data_root}")
    print(f"ESM model: {args.esm_model_path}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Extract all sequences
    print("Step 1: Extracting sequences from datasets...")

    data_names = parse_data_names(args.data_name)
    print(f"Processing datasets: {data_names}")

    sequences = get_all_sequences(args.data_root, data_names, args.max_sequence_length)

    if len(sequences) == 0:
        print("No sequences found. Exiting.")
        return

    # Show some statistics
    seq_lengths = [len(seq) for seq in sequences.values()]
    print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")
    print()

    # Step 2: Encode sequences
    print("Step 2: Encoding sequences with ESM...")
    sequence_embeddings = encode_sequences_with_esm(
        sequences=sequences,
        esm_model_path=args.esm_model_path,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        device=args.device
    )

    # Step 3: Save embeddings
    print("Step 3: Saving ESM embeddings...")

    # Get embedding dimension from first embedding
    sample_embedding = list(sequence_embeddings.values())[0]
    embedding_dim = sample_embedding.shape[-1]

    torch.save({
        "sequence_embeddings": sequence_embeddings,
        "uid_list": list(sequences.keys()),
        "esm_model_path": args.esm_model_path,
        "max_sequence_length": args.max_sequence_length,
        "embedding_dim": embedding_dim,
        "note": "Embeddings are stored WITHOUT BOS/EOS tokens. Shape is (seq_len, hidden_size)."
    }, args.output_path)

    print(f"ESM embeddings saved to: {args.output_path}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total embeddings: {len(sequence_embeddings)}")
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
