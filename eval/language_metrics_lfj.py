# 用于ProFunc任务的指标评测，codex给两个对比方法单独写了这么一个文件
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


FRAGLLM_ROOT = Path("/home/dataset-local/projects_dir/FragLLM")


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute language metrics for generated CSV files.")
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--evaluate_exact_match", type=str2bool, default=True)
    parser.add_argument("--evaluate_bleu", type=str2bool, default=True)
    parser.add_argument("--evaluate_rouge", type=str2bool, default=True)
    parser.add_argument("--evaluate_bert_score", type=str2bool, default=True)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_path)
    metrics = compute_language_metrics(results_df, args)
    output_json = Path(args.output_json) if args.output_json else Path(args.results_path).with_name(
        f"{Path(args.results_path).stem}_metrics.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {output_json}")
    print_plot_metrics(metrics)


if __name__ == "__main__":
    main()
