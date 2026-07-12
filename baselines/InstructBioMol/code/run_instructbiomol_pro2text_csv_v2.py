import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from exp_utils import set_seed

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
DEFAULT_METRICS_PYTHON = "/home/dataset-local/anaconda3/envs/fragllm/bin/python"
DEFAULT_STRUCTURE_DIR = FRAGLLM_ROOT / "analysis" / "0712analysis_1" / "temp_pdb"
# This is the instruction used for both the official SwissProt function training
# data and protein_to_text_swissprot_test_func evaluation data.
QUESTION_TEMPLATE = "What is the function of this protein?"


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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Defaults to the official model max_length (450) when omitted.",
    )
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--datatype", default="bf16", choices=["bf16", "half", "float"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--foldseek_path", default=str(DEFAULT_FOLDSEEK_PATH))
    parser.add_argument(
        "--missing_structure",
        default="error",
        choices=["error", "mask"],
        help="Use 'error' to require prepared AlphaFold/SaProt inputs, as in official evaluation.",
    )
    parser.add_argument(
        "--structure_dir",
        default=str(DEFAULT_STRUCTURE_DIR),
        help=(
            "Directory containing sequence-validated full-protein PDB files. The script tries "
            "{AlphaFoldDB}.pdb, AF-{AlphaFoldDB}-F1-model_v4.pdb, {accession}.pdb, "
            "and AF-{accession}-F1-model_v4.pdb."
        ),
    )
    parser.add_argument("--compute_metrics", type=str2bool, default=True)
    parser.add_argument("--evaluate_exact_match", type=str2bool, default=True)
    parser.add_argument("--evaluate_bleu", type=str2bool, default=True)
    parser.add_argument("--evaluate_rouge", type=str2bool, default=True)
    parser.add_argument("--evaluate_bert_score", type=str2bool, default=True)
    parser.add_argument("--metrics_output_json", default=None)
    parser.add_argument(
        "--metrics_python",
        default=DEFAULT_METRICS_PYTHON,
        help="Python executable used as fallback when the current environment cannot import evaluate.",
    )
    return parser.parse_args()


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
        reference = "" if pd.isna(meta.get("function")) else str(meta.get("function"))
        samples.append(
            {
                "sequence": sequence,
                "reference": reference,
                "dataset_idx": idx,
                "instruction": QUESTION_TEMPLATE,
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


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    def normalize(text: str) -> str:
        text = text.lower()
        return re.sub(r"[^\w]", "", text)

    return sum(
        normalize(pred) == normalize(ref)
        for pred, ref in zip(predictions, references)
    ) / len(predictions)


def compute_bert_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, Any]]:
    import evaluate
    from transformers import BertTokenizer, RobertaTokenizer

    results: Dict[str, Dict[str, Any]] = {}
    bert = evaluate.load(str(FRAGLLM_ROOT / "eval" / "metrics" / "bertscore"))

    tokenizer = RobertaTokenizer.from_pretrained(
        "/home/dataset-local/projects_dir/pretrained_model/roberta_large"
    )
    pred_ids = tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    ref_ids = tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    roberta_results = bert.compute(
        predictions=tokenizer.batch_decode(pred_ids, skip_special_tokens=True),
        references=tokenizer.batch_decode(ref_ids, skip_special_tokens=True),
        model_type="/home/dataset-local/projects_dir/pretrained_model/roberta_large",
        num_layers=17,
    )
    results["roberta-large"] = {
        "precision": sum(roberta_results["precision"]) / len(roberta_results["precision"]),
        "recall": sum(roberta_results["recall"]) / len(roberta_results["recall"]),
        "f1": sum(roberta_results["f1"]) / len(roberta_results["f1"]),
    }

    tokenizer = BertTokenizer.from_pretrained(
        "/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1"
    )
    pred_ids = tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    ref_ids = tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    biobert_results = bert.compute(
        predictions=tokenizer.batch_decode(pred_ids, skip_special_tokens=True),
        references=tokenizer.batch_decode(ref_ids, skip_special_tokens=True),
        model_type="/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1",
        num_layers=24,
    )
    results["biobert-large"] = {
        "precision": sum(biobert_results["precision"]) / len(biobert_results["precision"]),
        "recall": sum(biobert_results["recall"]) / len(biobert_results["recall"]),
        "f1": sum(biobert_results["f1"]) / len(biobert_results["f1"]),
    }
    return results


def compute_language_metrics(results_df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    import evaluate

    res = results_df.drop_duplicates(subset=["dataset_idx"])
    predictions = res["generated"].fillna("").astype(str).tolist()
    references = res["reference"].fillna("").astype(str).tolist()
    if not predictions:
        raise ValueError("No predictions available for metric computation")

    results: Dict[str, Any] = {}
    if args.evaluate_exact_match:
        results["exact_match"] = compute_exact_match(predictions, references)
    if args.evaluate_bleu:
        bleu = evaluate.load(str(FRAGLLM_ROOT / "eval" / "metrics" / "bleu"))
        results["bleu2"] = bleu.compute(
            predictions=predictions, references=references, max_order=2
        )
        results["bleu4"] = bleu.compute(predictions=predictions, references=references)
    if args.evaluate_rouge:
        rouge = evaluate.load(str(FRAGLLM_ROOT / "eval" / "metrics" / "rouge"))
        results["rouge"] = rouge.compute(predictions=predictions, references=references)
    if args.evaluate_bert_score:
        results["bert"] = compute_bert_score(predictions, references)
    return results


PLOT_METRICS = [
    ("BLEU-2", ("bleu2", "bleu")),
    ("BLEU-4", ("bleu4", "bleu")),
    ("ROUGE-L", ("rouge", "rougeL")),
    ("RoBERTa-large BERTScore-F1", ("bert", "roberta-large", "f1")),
    ("BioBERTScore-F1", ("bert", "biobert-large", "f1")),
]


def get_nested_metric(results: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    value: Any = results
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def print_plot_metrics(results: Dict[str, Any]) -> None:
    for metric_name, metric_path in PLOT_METRICS:
        metric_value = get_nested_metric(results, metric_path)
        if metric_value is not None:
            print(f"{metric_name}: {float(metric_value):.4g}")


def metrics_output_path(args: argparse.Namespace, save_path: Path) -> Path:
    if args.metrics_output_json:
        return Path(args.metrics_output_json)
    return save_path.with_name(f"{save_path.stem}_metrics.json")


def compute_metrics_with_fallback(
    results_df: pd.DataFrame,
    args: argparse.Namespace,
    save_path: Path,
) -> Dict[str, Any]:
    metric_path = metrics_output_path(args, save_path)
    try:
        metrics = compute_language_metrics(results_df, args)
    except ModuleNotFoundError as exc:
        if exc.name != "evaluate":
            raise
        cmd = [
            args.metrics_python,
            str(FRAGLLM_ROOT / "eval" / "language_metrics_lfj.py"),
            "--results_path",
            str(save_path),
            "--output_json",
            str(metric_path),
            "--evaluate_exact_match",
            str(args.evaluate_exact_match).lower(),
            "--evaluate_bleu",
            str(args.evaluate_bleu).lower(),
            "--evaluate_rouge",
            str(args.evaluate_rouge).lower(),
            "--evaluate_bert_score",
            str(args.evaluate_bert_score).lower(),
        ]
        subprocess.run(cmd, cwd=str(FRAGLLM_ROOT), check=True)
        with metric_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {metric_path}")
    return metrics


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    set_seed(args.random_seed)
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

    if args.compute_metrics:
        results_df = pd.DataFrame(rows)
        metrics = compute_metrics_with_fallback(results_df, args, save_path)
        print_plot_metrics(metrics)


if __name__ == "__main__":
    main()
