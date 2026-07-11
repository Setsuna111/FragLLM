import argparse
import ast
import json
import os
import random
import re
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
from dataset.templates import Frag_Ground_Group_Detailed  # noqa: E402
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
        description="Task4 step1: run group grounding on Pro2Text proteins."
    )
    parser.add_argument(
        "--model_path",
        default="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/",
    )
    parser.add_argument("--root_dir", default="/home/dataset-local/projects_dir/FragLLM/data_70")
    parser.add_argument(
        "--input_csv",
        default="data_70/Pro2Text/test_frag_test.csv",
        help="Pro2Text csv, e.g. data_70/Pro2Text/test_frag_test.csv.",
    )
    parser.add_argument("--save_results_dir", default="./eval_results/task4_grounding")
    parser.add_argument("--model_identifier", default="0529_all_215000")
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=1021)
    parser.add_argument("--window_stride", type=int, default=768)
    parser.add_argument(
        "--candidate_splits",
        # default="test,train",
        default="test",  # 对于frag_test子集，只需要读入test就可以了，一定是在test中重复的
        help="Splits used only to decide which fragment task types to ask.",
    )
    parser.add_argument(
        "--ask_all_tasks",
        action="store_true",
        help="Ask all five task types instead of gating by same-UID fragment annotations.",
    )
    parser.add_argument("--limit_samples", type=int, default=-1)
    parser.add_argument(
        "--mock_inference",
        action="store_true",
        help="Do not load the model; emit deterministic fake predictions for pipeline tests.",
    )
    return parser.parse_args()


def clean_generation(text: str) -> str:
    return (
        str(text)
        .replace("<|reserved_special_token_0|>", "")
        .replace("<|eot_id|>", "")
        .strip()
    )


def parse_position_list(value) -> List[Tuple[int, int]]:
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


def replace_matches_sequentially(text: str, replacements: List[Tuple[int, int]]) -> str:
    iterator = iter(replacements)

    def repl(_match):
        try:
            start, end = next(iterator)
        except StopIteration:
            return "(unknown)"
        if start > end:
            start, end = end, start
        return f"({start},{end})"

    return re.sub(r"<frag_position>", repl, text)


def load_candidate_task_index(root_dir: str, splits: List[str]) -> Dict[str, List[str]]:
    uid_to_tasks: Dict[str, set] = {}
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
                fragments = item.get("fragments") or []
                has_fragment = any(group.get("frags") for group in fragments)
                if has_fragment:
                    uid_to_tasks.setdefault(uid, set()).add(data_name)
    return {uid: sorted(tasks) for uid, tasks in uid_to_tasks.items()}


def make_windows(sequence: str, max_len: int, stride: int) -> List[Tuple[int, str]]:
    if len(sequence) <= max_len:
        return [(0, sequence)]
    starts = list(range(0, len(sequence), stride))
    final_start = max(0, len(sequence) - max_len)
    starts.append(final_start)
    starts = sorted(set(start for start in starts if start <= final_start))
    return [(start, sequence[start : start + max_len]) for start in starts]


class Pro2TextGroundingDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows
        self.sequence_placeholder = "<|reserved_special_token_1|>"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sequence = row["sequence_window"]
        question = random.choice(Frag_Ground_Group_Detailed).format(
            full_sequence=self.sequence_placeholder * (len(sequence) + 2),
            N=len(sequence),
            task_name=row["task_name"],
        )
        return {
            "sequence": sequence,
            "conversation": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": question},
            ],
            "answer": "",
            "position_ref": None,
            "position_grd": [],
            "start": row["crop_start"],
            "dataset_idx": row["row_id"],
            "interpro_id": None,
        }


def build_rows(args) -> List[Dict]:
    input_csv = args.input_csv or os.path.join(args.root_dir, "Pro2Text", "test_frag_test.csv")
    df = pd.read_csv(input_csv)
    if args.limit_samples > 0:
        df = df.iloc[: args.limit_samples].copy()
    candidate_splits = [s.strip() for s in args.candidate_splits.split(",") if s.strip()]
    candidate_index = load_candidate_task_index(args.root_dir, candidate_splits)  # 找到每个uid在哪些数据集中出现过

    rows = []
    for csv_idx, item in df.reset_index(drop=True).iterrows():
        accession = str(item["accession"]).strip()
        if args.ask_all_tasks:
            task_datasets = sorted(FRAGMENT_DATASETS)
        else:
            task_datasets = candidate_index.get(accession, [])
        for data_name in task_datasets:
            for crop_start, sequence_window in make_windows(
                str(item["sequence"]), args.max_sequence_length, args.window_stride
            ):  # 把长序列切成多个固定长度窗口，让模型分别在每个窗口里预测片段位置
                rows.append(
                    {
                        "row_id": len(rows),
                        "csv_idx": int(csv_idx),
                        "accession": accession,
                        "name": item.get("name", ""),
                        "fullname": item.get("Full Name", ""),
                        "taxon": item.get("taxon", ""),
                        "sequence": str(item["sequence"]),
                        "sequence_length": len(str(item["sequence"])),
                        "function": item.get("function", ""),
                        "task_dataset": data_name,
                        "task_name": FRAGMENT_DATASETS[data_name],
                        "crop_start": crop_start,
                        "sequence_window": sequence_window,
                    }
                )
    return rows


def generate_mock(rows: List[Dict]) -> pd.DataFrame:
    output_rows = []
    for row in rows:
        pred = [(0, min(8, len(row["sequence_window"])))]
        output_rows.append(
            {
                **{k: v for k, v in row.items() if k != "sequence_window"},
                "generated": f"Mock {row['task_name']}: <p>mock_fragment:(0,8)</p>",
                "pred_positions": str(pred),
            }
        )
    return pd.DataFrame(output_rows)


def run_inference(rows: List[Dict], args) -> pd.DataFrame:
    dataset = Pro2TextGroundingDataset(rows)
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
    row_map = {row["row_id"]: row for row in rows}
    output_rows = []
    for inputs in tqdm(dataloader, desc="task4 grounding"):
        row_ids = [int(x) for x in inputs["dataset_idxs"]]
        inputs = {
            k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            tok_ids, position_preds = model.generate(
                inputs=None,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                protein_input_ids=inputs["protein_input_ids"],
                protein_attention_mask=inputs["protein_attention_mask"],
                protein_inputs_embeds=None,
                position_refs=inputs["position_refs"],
                grounding_inference=True,
                num_beams=1,
                early_stopping=False,
                no_repeat_ngram_size=None,
                length_penalty=1.0,
                eos_token_id=128009,
                pad_token_id=128002,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=512,
                use_cache=True,
            )
        decoded = tokenizer.batch_decode(tok_ids)
        for batch_idx, row_id in enumerate(row_ids):
            if position_preds is not None and batch_idx < len(position_preds):
                pos = list(
                    zip(
                        position_preds[batch_idx]["start_positions"].tolist(),
                        position_preds[batch_idx]["end_positions"].tolist(),
                    )
                )
            else:
                pos = []
            generated = replace_matches_sequentially(decoded[batch_idx], pos)
            row = row_map[row_id]
            output_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "sequence_window"},
                    "generated": clean_generation(generated),
                    "pred_positions": str(pos),
                }
            )
    return pd.DataFrame(output_rows)


def infer_save_path(args) -> str:
    input_name = os.path.splitext(os.path.basename(args.input_csv or "test_frag_test.csv"))[0]
    output_dir = os.path.join(args.save_results_dir, args.model_identifier)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{input_name}_grounding_results.csv")


def main():
    args = parse_args()
    rows = build_rows(args)
    print(f"Built {len(rows)} grounding prompts")
    save_path = infer_save_path(args)
    if not rows:
        pd.DataFrame().to_csv(save_path, index=False)
        print(f"No prompts to run. Empty result saved to {save_path}")
        return
    if args.mock_inference:
        df = generate_mock(rows)
    else:
        df = run_inference(rows, args)
    df.to_csv(save_path, index=False)
    print(f"Saved grounding results to {save_path}")


if __name__ == "__main__":
    main()
