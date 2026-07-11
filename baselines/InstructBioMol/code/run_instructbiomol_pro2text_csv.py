import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from instructbiomol_reference_common import (
    DEFAULT_FOLDSEEK_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    FRAGLLM_ROOT,
    FoldseekRunner,
    generation_kwargs,
    load_instructbiomol_model,
    masked_saprot_sequence,
)


DEFAULT_INPUT_CSV = FRAGLLM_ROOT / "data_70" / "Pro2Text" / "test_frag_no_train.csv"
QUESTION_TEMPLATE = (
    "Protein name: {fullname}; Taxon: {taxon}. "
    "Please describe its function clearly and concisely in professional language."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InstructBioMol inference on a Pro2Text CSV test split."
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--llama_ckpt_path",
        default=None,
        help="Path used for Llama config/tokenizer. Defaults to --model_path.",
    )
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--save_results_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model_identifier", default="instructbiomol_instruct")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_per_device", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--datatype", default="bf16", choices=["bf16", "half", "float"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--foldseek_path", default=str(DEFAULT_FOLDSEEK_PATH))
    parser.add_argument(
        "--missing_structure",
        default="mask",
        choices=["error", "mask"],
        help="Use 'mask' to fall back to sequence-plus-# SaProt strings when a PDB is missing.",
    )
    parser.add_argument(
        "--structure_dir",
        default=None,
        help=(
            "Optional directory containing full-protein PDB files. The script tries "
            "{AlphaFoldDB}.pdb, AF-{AlphaFoldDB}-F1-model_v4.pdb, {accession}.pdb, "
            "and AF-{accession}-F1-model_v4.pdb."
        ),
    )
    return parser.parse_args()


def clean_sequence(value) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split())


def load_samples(csv_path: Path, limit: Optional[int]) -> List[Dict]:
    frame = pd.read_csv(csv_path)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    required = {"sequence", "function"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    samples: List[Dict] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        meta = row.to_dict()
        sequence = clean_sequence(meta.get("sequence", ""))
        fullname = "" if pd.isna(meta.get("Full Name")) else str(meta.get("Full Name", ""))
        taxon = "" if pd.isna(meta.get("taxon")) else str(meta.get("taxon", ""))
        reference = "" if pd.isna(meta.get("function")) else str(meta.get("function"))
        samples.append(
            {
                "sequence": sequence,
                "reference": reference,
                "dataset_idx": idx,
                "instruction": QUESTION_TEMPLATE.format(fullname=fullname, taxon=taxon),
                "metadata": meta,
            }
        )
    return samples


def structure_candidates(sample: Dict, structure_dir: Optional[Path]) -> List[Path]:
    if structure_dir is None:
        return []
    meta = sample["metadata"]
    ids = []
    for key in ("AlphaFoldDB", "accession"):
        value = meta.get(key)
        if value is not None and not pd.isna(value):
            text = str(value).strip()
            if text and text not in ids:
                ids.append(text)
    candidates: List[Path] = []
    for protein_id in ids:
        candidates.extend(
            [
                structure_dir / f"{protein_id}.pdb",
                structure_dir / f"AF-{protein_id}-F1-model_v4.pdb",
            ]
        )
    return candidates


def resolve_structure_path(sample: Dict, structure_dir: Optional[Path]) -> Optional[Path]:
    candidates = structure_candidates(sample, structure_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def batched(items: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prepare_pro2text_batch(
    samples: Sequence[Dict],
    foldseek: FoldseekRunner,
    structure_dir: Optional[Path],
    missing_structure: str,
) -> Tuple[Tuple[List[str], List[str]], Dict]:
    input_sequences: List[str] = []
    saprot_sequences: List[str] = []
    instructions: List[str] = []
    references: List[str] = []
    ids: List[str] = []
    structure_paths: List[str] = []

    for sample in samples:
        sequence = sample["sequence"]
        structure_path = resolve_structure_path(sample, structure_dir)
        if structure_path is None:
            if missing_structure == "error":
                raise FileNotFoundError(
                    "No --structure_dir was provided, so InstructBioMol cannot build "
                    "SaProt 3Di inputs. Use --missing_structure mask or provide structures."
                )
            saprot_sequence = masked_saprot_sequence(sequence)
            structure_path_text = ""
        else:
            saprot_sequence = foldseek.sequence_for_structure(structure_path, sequence)
            structure_path_text = str(structure_path)

        input_sequences.append(sequence)
        saprot_sequences.append(saprot_sequence)
        instructions.append(sample["instruction"])
        references.append(sample["reference"])
        ids.append(str(sample["dataset_idx"]))
        structure_paths.append(structure_path_text)

    inputs = {
        "input_seqs": input_sequences,
        "target_seqs": references,
        "input_enc_seqs": input_sequences,
        "input_enc_fps": [[] for _ in input_sequences],
        "input_modality": "protein",
        "target_modality": "text",
        "instructions": instructions,
        "ids": ids,
        "data_name": "pro2text_csv",
        "structure_paths": structure_paths,
    }
    return (input_sequences, saprot_sequences), inputs


def output_path(args: argparse.Namespace) -> Path:
    if args.output_csv:
        return Path(args.output_csv)
    input_stem = Path(args.input_csv).stem
    return (
        Path(args.save_results_dir)
        / "data_70"
        / "pro2text_csv"
        / args.model_identifier
        / f"{input_stem}_results.csv"
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    input_csv = Path(args.input_csv)
    samples = load_samples(input_csv, args.limit)
    structure_dir = Path(args.structure_dir) if args.structure_dir else None
    foldseek = FoldseekRunner(args.foldseek_path, args.missing_structure)
    model = load_instructbiomol_model(args)

    generated: List[str] = []
    structure_paths: List[str] = []
    print(f"Evaluating {input_csv}: {len(samples)} samples")
    for batch_samples in tqdm(
        batched(samples, args.batch_per_device),
        total=(len(samples) + args.batch_per_device - 1) // args.batch_per_device,
        desc=f"Evaluating {input_csv.stem}",
    ):
        input_batch, inputs = prepare_pro2text_batch(
            samples=batch_samples,
            foldseek=foldseek,
            structure_dir=structure_dir,
            missing_structure=args.missing_structure,
        )
        structure_paths.extend(inputs["structure_paths"])
        with torch.no_grad():
            generated.extend(model.generate(input_batch, inputs, "text", **generation_kwargs(args)))

    rows: List[Dict] = []
    for sample, text, structure_path in zip(samples, generated, structure_paths):
        rows.append(
            {
                "generated": text,
                "reference": sample["reference"],
                "dataset_idx": sample["dataset_idx"],
                "structure_path": structure_path,
                **dict(sample["metadata"]),
            }
        )

    save_path = output_path(args)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["generated", "reference", "dataset_idx"]
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(generated)} generations to {save_path}")


if __name__ == "__main__":
    main()
