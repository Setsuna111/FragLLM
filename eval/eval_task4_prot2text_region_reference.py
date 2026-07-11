import argparse
import ast
import os
import random
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(".")

from dataset.dataloader_frag import FragDataCollator  # noqa: E402
from dataset.templates import (  # noqa: E402
    Frag_Act_Des,
    Frag_BindI_Des,
    Frag_Class,
    Frag_Dom_Des,
    Frag_Evo_Des,
    Frag_Motif_Des,
)
from models.protein_llama_addtoken_lfj import ProteinLlamaForCausalLM  # noqa: E402


SYSTEM_MESSAGE = (
    "You are a scientific assistant specializing in protein sequence analysis. "
    "Based on protein sequence embeddings and other related information, please "
    "answer the relevant questions using professional language. "
)

TASK_CONFIGS = {
    "VenusX_Act": {
        "task_name": "active site",
        "desc_template": Frag_Act_Des,
    },
    "VenusX_BindI": {
        "task_name": "binding site",
        "desc_template": Frag_BindI_Des,
    },
    "VenusX_Dom": {
        "task_name": "domain",
        "desc_template": Frag_Dom_Des,
    },
    "VenusX_Evo": {
        "task_name": "conserved site",
        "desc_template": Frag_Evo_Des,
    },
    "VenusX_Motif": {
        "task_name": "motif domain",
        "desc_template": Frag_Motif_Des,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task4 step2: classify/describe predicted Pro2Text fragments."
    )
    parser.add_argument(
        "--model_path",
        default="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/",
    )
    parser.add_argument("--grounding_results_path", default="/home/dataset-local/projects_dir/FragLLM/eval_results/task4_grounding/0529_all_215000/test_frag_test_grounding_results.csv")
    parser.add_argument("--save_results_dir", default="./eval_results/task4_region_ref")
    parser.add_argument("--model_identifier", default="0529_all_215000")
    parser.add_argument("--tasks", default="cls,desc")
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=1021)
    parser.add_argument("--perceiver_latent_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_regions", type=int, default=-1)
    parser.add_argument(
        "--mock_inference",
        action="store_true",
        help="Do not load the model; emit deterministic fake cls/desc outputs.",
    )
    return parser.parse_args()


def parse_positions(value) -> List[Tuple[int, int]]:
    if value is None or pd.isna(value):
        return []
    try:
        raw = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    positions = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if start > end:
            start, end = end, start
        if start < 0 or end < 0:
            continue
        positions.append((start, end))
    return positions


def clean_generation(text: str) -> str:
    return (
        str(text)
        .replace("<|reserved_special_token_0|>", "")
        .replace("<|eot_id|>", "")
        .strip()
    )


def choose_crop_start(sequence_len: int, start: int, end: int, max_len: int) -> int:
    if sequence_len <= max_len:
        return 0
    max_start = max(0, sequence_len - max_len)
    center = (start + end) // 2
    crop_start = max(0, min(center - max_len // 2, max_start))
    if start < crop_start:
        crop_start = start
    if end >= crop_start + max_len:
        crop_start = end - max_len + 1
    return max(0, min(crop_start, max_start))


def collect_region_rows(args) -> List[Dict]:
    grounding_df = pd.read_csv(args.grounding_results_path)
    rows = []
    for _, result_row in grounding_df.iterrows():
        sequence = str(result_row["sequence"])
        crop_start = int(result_row.get("crop_start", 0))
        for region_idx, (local_start, local_end) in enumerate(
            parse_positions(result_row.get("pred_positions", "[]"))
        ):
            abs_start = local_start + crop_start
            abs_end = local_end + crop_start - 1
            if abs_start >= len(sequence) or abs_end < 0:
                continue
            abs_start = max(0, min(abs_start, len(sequence) - 1))
            abs_end = max(abs_start, min(abs_end, len(sequence) - 1))
            ref_crop_start = choose_crop_start(
                len(sequence), abs_start, abs_end, args.max_sequence_length
            )
            rows.append(
                {
                    "row_id": len(rows),
                    "csv_idx": int(result_row["csv_idx"]),
                    "accession": result_row["accession"],
                    "task_dataset": result_row["task_dataset"],
                    "task_name": result_row["task_name"],
                    "region_idx": region_idx,
                    "start_pos": abs_start,
                    "end_pos": abs_end,
                    "position": f"({abs_start},{abs_end})",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "crop_start": ref_crop_start,
                    "grounding_generated": result_row.get("generated", ""),
                }
            )
    if args.limit_regions > 0:
        rows = rows[: args.limit_regions]
        for idx, row in enumerate(rows):
            row["row_id"] = idx
    return rows


class RegionReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict], task: str, max_len: int, latent_size: int):
        self.rows = rows
        self.task = task
        self.max_len = max_len
        self.sequence_placeholder = "<|reserved_special_token_1|>"
        self.fragment_placeholder = "<|reserved_special_token_2|>" * latent_size

    def __len__(self):
        return len(self.rows)

    def _conversation(self, row: Dict, sequence_window: str):
        config = TASK_CONFIGS[row["task_dataset"]]
        if self.task == "cls":
            question = random.choice(Frag_Class).format(
                full_sequence=self.sequence_placeholder * (len(sequence_window) + 2),
                task_name=config["task_name"],
                fragment=self.fragment_placeholder,
            )
        elif self.task == "desc":
            question = random.choice(config["desc_template"]).format(
                full_sequence=self.sequence_placeholder * (len(sequence_window) + 2),
                fragment=self.fragment_placeholder,
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ]

    def __getitem__(self, idx):
        row = self.rows[idx]
        crop_start = int(row["crop_start"])
        sequence_window = row["sequence"][crop_start : crop_start + self.max_len]
        local_start = max(0, int(row["start_pos"]) - crop_start)
        local_end = min(len(sequence_window), int(row["end_pos"]) - crop_start + 1)
        if local_end <= local_start:
            local_end = min(len(sequence_window), local_start + 1)
        return {
            "sequence": sequence_window,
            "conversation": self._conversation(row, sequence_window),
            "answer": "",
            "position_ref": [local_start, local_end],
            "position_grd": None,
            "start": crop_start,
            "dataset_idx": row["row_id"],
            "interpro_id": None,
        }


def generate_mock(rows: List[Dict], task_list: List[str]) -> List[Dict]:
    for row in rows:
        if "cls" in task_list:
            row["pred_cls"] = f"mock {row['task_name']}"
        if "desc" in task_list:
            row["pred_desc"] = f"mock description for {row['task_name']}"
    return rows


def generate_task_outputs(rows, task, model, tokenizer, sequence_tokenizer, args, device):
    dataset = RegionReferenceDataset(
        rows,
        task=task,
        max_len=args.max_sequence_length,
        latent_size=args.perceiver_latent_size,
    )
    collator = FragDataCollator(
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
        collate_fn=collator,
    )
    outputs = {}
    for inputs in tqdm(dataloader, desc=f"task4 {task}"):
        row_ids = [int(x) for x in inputs["dataset_idxs"]]
        inputs = {
            k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            tok_ids = model.generate(
                inputs=None,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                protein_input_ids=inputs["protein_input_ids"],
                protein_attention_mask=inputs["protein_attention_mask"],
                protein_inputs_embeds=None,
                position_refs=inputs["position_refs"],
                num_beams=1,
                early_stopping=False,
                no_repeat_ngram_size=None,
                length_penalty=1.0,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=512,
                use_cache=True,
            )
        decoded = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
        for row_id, text in zip(row_ids, decoded):
            outputs[row_id] = clean_generation(text)
    return outputs


def infer_save_path(args) -> str:
    input_name = os.path.splitext(os.path.basename(args.grounding_results_path))[0]
    output_dir = os.path.join(args.save_results_dir, args.model_identifier)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{input_name}_region_ref_results.csv")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in task_list:
        if task not in {"cls", "desc"}:
            raise ValueError("--tasks must contain only cls and/or desc")

    rows = collect_region_rows(args)
    print(f"Collected {len(rows)} predicted regions")
    save_path = infer_save_path(args)
    if not rows:
        pd.DataFrame().to_csv(save_path, index=False)
        print(f"No regions found. Empty result saved to {save_path}")
        return

    if args.mock_inference:
        rows = generate_mock(rows, task_list)
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, pad_token="<|reserved_special_token_0|>"
        )
        model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)
        model.eval()
        model = model.bfloat16().to(device)

        task_outputs = {}
        for task in task_list:
            task_outputs[task] = generate_task_outputs(
                rows, task, model, tokenizer, sequence_tokenizer, args, device
            )
        for row in rows:
            row["pred_cls"] = task_outputs.get("cls", {}).get(row["row_id"], "")
            row["pred_desc"] = task_outputs.get("desc", {}).get(row["row_id"], "")

    for row in rows:
        row.pop("sequence", None)
    columns = [
        "csv_idx",
        "accession",
        "task_dataset",
        "task_name",
        "region_idx",
        "start_pos",
        "end_pos",
        "position",
        "sequence_length",
        "crop_start",
        "pred_cls",
        "pred_desc",
        "grounding_generated",
    ]
    pd.DataFrame(rows)[columns].to_csv(save_path, index=False)
    print(f"Saved region reference results to {save_path}")


if __name__ == "__main__":
    main()
