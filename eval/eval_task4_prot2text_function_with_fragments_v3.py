import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List

import evaluate
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

sys.path.append(".")

import scripts.utils_argparse as utils_argparse  # noqa: E402
from dataset.dataloader_frag import FragDataCollator  # noqa: E402
from models.protein_llama_addtoken_lfj import ProteinLlamaForCausalLM  # noqa: E402


SYSTEM_MESSAGE = (
    "You are a scientific assistant specializing in protein sequence analysis. "
    "Based on protein sequence embeddings and other related information, please "
    "answer the relevant questions using professional language. "
)

FRAGMENT_DATASETS = {
    "VenusX_Act": "active site",
    "VenusX_BindI": "binding site",
    "VenusX_Dom": "domain",
    "VenusX_Evo": "evolutionary conserved site",
    "VenusX_Motif": "motif domain",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task4 step3: Pro2Text function prediction with optional fragments."
    )
    parser.add_argument(
        "--model_path",
        default="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/",
    )
    parser.add_argument("--root_dir", default="/home/dataset-local/projects_dir/FragLLM/data_70")
    parser.add_argument(
        "--input_csv",
        default="/home/dataset-local/projects_dir/FragLLM/data_70/Pro2Text/test_frag_test.csv",
    )
    parser.add_argument(
        "--fragment_mode",
        choices=["none", "predicted", "random", "truth"],
        default="predicted",
    )
    parser.add_argument(
        "--fragment_text",
        choices=["cls", "desc"],
        default="cls",
        help="Retained for CLI compatibility; function prompts now use fragment classes only.",
    )
    parser.add_argument("--predicted_regions_path", default="/home/dataset-local/projects_dir/FragLLM/eval_results/task4_region_ref/0529_all_215000/test_frag_test_grounding_results_region_ref_results.csv")
    # parser.add_argument("--truth_splits", default="test,train")
    parser.add_argument("--truth_splits", default="test")
    parser.add_argument("--save_results_dir", default="./eval_results/task4_function")
    parser.add_argument("--model_identifier", default="0529_all_215000")
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=128009,
        help="Official Llama 3.1 end-of-turn token used by the recommended setting.",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=128002,
        help="Official Llama 3.1 reserved pad token used by the recommended setting.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=1021)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_samples", type=int, default=-1)
    parser.add_argument("--evaluate_exact_match", type=utils_argparse.str2bool, default=True)
    parser.add_argument("--evaluate_bleu", type=utils_argparse.str2bool, default=True)
    parser.add_argument("--evaluate_rouge", type=utils_argparse.str2bool, default=True)
    parser.add_argument("--evaluate_bert_score", type=utils_argparse.str2bool, default=True)
    parser.add_argument("--verbose", type=utils_argparse.str2bool, default=True)
    parser.add_argument(
        "--mock_inference",
        action="store_true",
        help="Do not load the model; echo deterministic fake function predictions.",
    )
    return parser.parse_args()


def clean_generation(text: str) -> str:
    return (
        str(text)
        .replace("<|reserved_special_token_0|>", "")
        .replace("<|eot_id|>", "")
        .strip()
    )


def normalize_text(text: Any) -> str:
    text = str(text).strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pred_class(text: Any) -> str:
    """Extract the class phrase from grounding predictions."""
    text = normalize_text(text)
    patterns = [
        r"^It is the (.*?)[\.\n]*$",
        r"^The category is (.*?)[\.\n]*$",
        r"^This belongs to the (.*?)[\.\n]*$",
        r"^It is categorized as (.*?)[\.\n]*$",
        r"^This instance is identified as (.*?)[\.\n]*$",
        r"^The determined class is (.*?)[\.\n]*$",
        r"^It has been assigned to the (.*?) group[\.\n]*$",
        r"^It is grouped under the (.*?) category[\.\n]*$",
        r"^The appropriate category for this is (.*?)[\.\n]*$",
        r"^Its class designation is (.*?)[\.\n]*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    return text.strip(" .")


def deduplicate_fragment_classes(classes: List[str]) -> List[str]:
    unique_classes = []
    seen = set()
    for class_name in classes:
        class_name = normalize_text(class_name).strip(" .")
        key = class_name.casefold()
        if class_name and key not in seen:
            unique_classes.append(class_name)
            seen.add(key)
    return unique_classes


def format_class_sentence(classes: List[str]) -> str:
    """Make class boundaries explicit even when an InterPro name contains commas."""
    return (
        f"This protein may contain {len(classes)} fragment classes: "
        f"{'; '.join(classes)}. "
    )


def fragment_class(group: Dict) -> str:
    return normalize_text(
        group.get("category") or group.get("shortname") or group.get("description") or ""
    )


def load_truth_fragments(root_dir: str, splits: List[str]) -> Dict[str, List[str]]:
    uid_to_fragments: Dict[str, List[str]] = {}
    for data_name in FRAGMENT_DATASETS:
        for split in splits:
            path = os.path.join(root_dir, data_name, f"{split}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
            for item in data:
                uid = str(item.get("uid", "")).strip()
                if not uid:
                    continue
                for group in item.get("fragments", []):
                    if not group.get("frags"):
                        continue
                    class_name = fragment_class(group)
                    if class_name:
                        uid_to_fragments.setdefault(uid, []).append(class_name)
    return uid_to_fragments


def load_random_fragment_pool(root_dir: str) -> List[str]:
    pool = []
    for data_name in FRAGMENT_DATASETS:
        for split in ["train", "test"]:
            path = os.path.join(root_dir, data_name, f"{split}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
            for item in data:
                for group in item.get("fragments", []):
                    if not group.get("frags"):
                        continue
                    class_name = fragment_class(group)
                    if class_name:
                        pool.append(class_name)
    return pool


def load_predicted_fragments(path: str) -> Dict[str, List[str]]:
    if not path:
        raise ValueError("--predicted_regions_path is required for fragment_mode=predicted")
    df = pd.read_csv(path)
    uid_to_fragments: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        text = str(row.get("pred_cls", "")).strip()
        if not text or text.lower() == "nan":
            continue
        class_name = extract_pred_class(text)
        if class_name:
            uid_to_fragments.setdefault(str(row["accession"]).strip(), []).append(class_name)
    return uid_to_fragments


def build_fragment_lookup(args) -> Dict[str, List[str]]:
    if args.fragment_mode == "none":
        return {}
    if args.fragment_mode == "predicted":
        return load_predicted_fragments(args.predicted_regions_path)
    if args.fragment_mode == "truth":
        splits = [s.strip() for s in args.truth_splits.split(",") if s.strip()]
        return load_truth_fragments(args.root_dir, splits)
    if args.fragment_mode == "random":
        pool = load_random_fragment_pool(args.root_dir)
        if not pool:
            raise ValueError("No fragments found for random mode")
        return {"__pool__": pool}
    raise ValueError(f"Unsupported fragment_mode: {args.fragment_mode}")


class Pro2TextFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, args, fragment_lookup: Dict[str, List[str]]):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.fragment_lookup = fragment_lookup
        self.sequence_placeholder = "<|reserved_special_token_1|>"

    def __len__(self):
        return len(self.df)

    def _fragments_for_uid(self, uid: str) -> List[str]:
        if self.args.fragment_mode == "none":
            return []
        if self.args.fragment_mode == "random":
            pool = self.fragment_lookup["__pool__"]
            count = random.randint(3, min(5, len(pool)))
            return deduplicate_fragment_classes(random.sample(pool, count))
        return deduplicate_fragment_classes(self.fragment_lookup.get(uid, []))

    def _build_question(self, row: pd.Series, sequence: str, fragments: List[str]) -> str:
        base = (
            "Protein name: {fullname}; Taxon: {taxon}; Sequence embeddings: "
            "{full_sequence}. "
        ).format(
            fullname=row.get("Full Name", ""),
            taxon=row.get("taxon", ""),
            full_sequence=self.sequence_placeholder * (len(sequence) + 2),
        )
        if not fragments:
            return base + "Please describe its function clearly and concisely in professional language."

        return (
            base
            + format_class_sentence(fragments)
            + "Please describe its function clearly and concisely in professional language."
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = str(row["sequence"])
        if len(sequence) > self.args.max_sequence_length:
            sequence = sequence[: self.args.max_sequence_length]
        fragments = self._fragments_for_uid(str(row["accession"]).strip())
        question = self._build_question(row, sequence, fragments)
        return {
            "sequence": sequence,
            "conversation": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": question},
            ],
            "answer": str(row.get("function", "")),
            "position_ref": None,
            "position_grd": None,
            "start": 0,
            "dataset_idx": idx,
            "interpro_id": None,
            "accession": str(row["accession"]).strip(),
            "fragment_context": "\n".join(fragments),
        }


def collate_with_metadata(base_collator):
    def collate(batch):
        accessions = [item.pop("accession") for item in batch]
        fragment_contexts = [item.pop("fragment_context") for item in batch]
        out = base_collator(batch)
        out["accessions"] = accessions
        out["fragment_contexts"] = fragment_contexts
        return out

    return collate


def run_mock(dataset: Pro2TextFunctionDataset) -> pd.DataFrame:
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        rows.append(
            {
                "generated": "Mock function prediction.",
                "reference": item["answer"],
                "dataset_idx": idx,
                "accession": item["accession"],
                "fragment_context": item["fragment_context"],
            }
        )
    return pd.DataFrame(rows)


def run_inference(dataset: Pro2TextFunctionDataset, args) -> pd.DataFrame:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, pad_token="<|reserved_special_token_0|>"
    )
    model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
    model.config.eos_token_id = args.eos_token_id
    model.config.pad_token_id = args.pad_token_id
    sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)
    model.eval()
    model = model.bfloat16().to(device)

    base_collator = FragDataCollator(
        sequence_tokenizer=sequence_tokenizer,
        llm_tokenizer=tokenizer,
        mode="inference",
        max_sequence_length=args.max_sequence_length,
        max_description_length=512,
        use_max_desc_length=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_per_device,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_metadata(base_collator),
    )
    rows = []
    for inputs in tqdm(dataloader, desc=f"task4 function {args.fragment_mode}/{args.fragment_text}"):
        references = tokenizer.batch_decode(inputs["answer_input_ids"], skip_special_tokens=True)
        row_ids = [int(x) for x in inputs["dataset_idxs"]]
        accessions = inputs["accessions"]
        fragment_contexts = inputs["fragment_contexts"]
        inputs = {
            k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in inputs.items()
            if k not in {"accessions", "fragment_contexts"}
        }
        with torch.no_grad():
            generation_kwargs = {
                "inputs": None,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "protein_input_ids": inputs["protein_input_ids"],
                "protein_attention_mask": inputs["protein_attention_mask"],
                "protein_inputs_embeds": None,
                "position_refs": inputs["position_refs"],
                "num_beams": args.num_beams,
                "early_stopping": args.num_beams > 1,
                "no_repeat_ngram_size": None,
                "length_penalty": 1.0,
                "do_sample": args.temperature > 0,
                "max_new_tokens": args.max_new_tokens,
                "eos_token_id": args.eos_token_id,
                "pad_token_id": args.pad_token_id,
                "use_cache": True,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
            else:
                # The checkpoint's generation config sets sampling defaults.
                generation_kwargs["temperature"] = None
                generation_kwargs["top_p"] = None
            tok_ids = model.generate(
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
        for generated, reference, row_id, accession, context in zip(
            decoded, references, row_ids, accessions, fragment_contexts
        ):
            rows.append(
                {
                    "generated": clean_generation(generated),
                    "reference": reference,
                    "dataset_idx": row_id,
                    "accession": accession,
                    "fragment_context": context,
                }
            )
    return pd.DataFrame(rows)


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    def normalize(text: str) -> str:
        return re.sub(r"[^\w]", "", text.lower())

    return sum(normalize(p) == normalize(r) for p, r in zip(predictions, references)) / len(predictions)


def compute_bert_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    bert = evaluate.load("./eval/metrics/bertscore")

    tokenizer = RobertaTokenizer.from_pretrained("/home/dataset-local/projects_dir/pretrained_model/roberta_large")
    pred_ids = tokenizer(predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt")["input_ids"]
    ref_ids = tokenizer(references, padding="max_length", truncation=True, max_length=495, return_tensors="pt")["input_ids"]
    trunc_pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    trunc_ref = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
    score = bert.compute(
        predictions=trunc_pred,
        references=trunc_ref,
        model_type="/home/dataset-local/projects_dir/pretrained_model/roberta_large",
        num_layers=17,
    )
    results["roberta-large"] = {
        "precision": sum(score["precision"]) / len(score["precision"]),
        "recall": sum(score["recall"]) / len(score["recall"]),
        "f1": sum(score["f1"]) / len(score["f1"]),
    }

    tokenizer = BertTokenizer.from_pretrained("/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1")
    pred_ids = tokenizer(predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt")["input_ids"]
    ref_ids = tokenizer(references, padding="max_length", truncation=True, max_length=495, return_tensors="pt")["input_ids"]
    trunc_pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    trunc_ref = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
    score = bert.compute(
        predictions=trunc_pred,
        references=trunc_ref,
        model_type="/home/dataset-local/projects_dir/pretrained_model/biobert-large-cased-v1.1",
        num_layers=24,
    )
    results["biobert-large"] = {
        "precision": sum(score["precision"]) / len(score["precision"]),
        "recall": sum(score["recall"]) / len(score["recall"]),
        "f1": sum(score["f1"]) / len(score["f1"]),
    }
    return results


def compute_metrics(df: pd.DataFrame, args) -> Dict[str, Any]:
    predictions = df["generated"].fillna("").astype(str).tolist()
    references = df["reference"].fillna("").astype(str).tolist()
    results: Dict[str, Any] = {}
    if not predictions:
        return results
    if args.evaluate_exact_match:
        results["exact_match"] = compute_exact_match(predictions, references)
    if args.evaluate_bleu:
        bleu = evaluate.load("./eval/metrics/bleu")
        results["bleu2"] = bleu.compute(predictions=predictions, references=references, max_order=2)
        results["bleu4"] = bleu.compute(predictions=predictions, references=references)
    if args.evaluate_rouge:
        rouge = evaluate.load("./eval/metrics/rouge")
        results["rouge"] = rouge.compute(predictions=predictions, references=references)
    if args.evaluate_bert_score:
        results["bert"] = compute_bert_score(predictions, references)
    if args.verbose:
        print(json.dumps(results, indent=2))
    return results


def infer_save_path(args) -> str:
    input_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_dir = os.path.join(args.save_results_dir, args.model_identifier)
    os.makedirs(output_dir, exist_ok=True)
    name = f"{input_name}_{args.fragment_mode}_{args.fragment_text}_function_results.csv"
    return os.path.join(output_dir, name)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.input_csv)
    if args.limit_samples > 0:
        df = df.iloc[: args.limit_samples].copy()
    fragment_lookup = build_fragment_lookup(args)
    dataset = Pro2TextFunctionDataset(df, args, fragment_lookup)
    if args.mock_inference:
        results_df = run_mock(dataset)
    else:
        results_df = run_inference(dataset, args)

    save_path = infer_save_path(args)
    results_df.to_csv(save_path, index=False)
    print(f"Saved function results to {save_path}")
    metrics = compute_metrics(results_df, args)
    metrics_path = save_path.replace(".csv", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
