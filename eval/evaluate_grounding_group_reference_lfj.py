import argparse
import ast
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(".")

from dataset.dataloader_frag import FragDataCollator
from dataset.dataloader_grounding import (  # noqa: E402
    ActGroundingGroup,
    BindIGroundingGroup,
    DomainGroundingGroup,
    EvoGroundingGroup,
    MotifGroundingGroup,
)
from dataset.templates import (  # noqa: E402
    Frag_Act_Des,
    Frag_BindI_Des,
    Frag_Class,
    Frag_Dom_Des,
    Frag_Evo_Des,
    Frag_Motif_Des,
)
from models.protein_llama_addtoken_lfj import ProteinLlamaForCausalLM  # noqa: E402


GROUNDING_DATASETS = {
    "ActGroundGroup": {
        "dataset_class": ActGroundingGroup,
        "ref_prefix": "Act",
        "task_name": "active site",
        "desc_template": Frag_Act_Des,
    },
    "BindIGroundGroup": {
        "dataset_class": BindIGroundingGroup,
        "ref_prefix": "BindI",
        "task_name": "binding site",
        "desc_template": Frag_BindI_Des,
    },
    "DomGroundGroup": {
        "dataset_class": DomainGroundingGroup,
        "ref_prefix": "Dom",
        "task_name": "domain",
        "desc_template": Frag_Dom_Des,
    },
    "EvoGroundGroup": {
        "dataset_class": EvoGroundingGroup,
        "ref_prefix": "Evo",
        "task_name": "conserved site",
        "desc_template": Frag_Evo_Des,
    },
    "MotifGroundGroup": {
        "dataset_class": MotifGroundingGroup,
        "ref_prefix": "Motif",
        "task_name": "motif domain",
        "desc_template": Frag_Motif_Des,
    },
}


SYSTEM_MESSAGE = (
    "You are a scientific assistant specializing in protein sequence analysis. "
    "Based on protein sequence embeddings and other related information, please "
    "answer the relevant questions using professional language. "
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run referring classification/description on group grounding regions."
    )
    parser.add_argument(
        "--model_path",
        default="/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-215000_merge/",
    )
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--root_dir", default="./data_70")
    parser.add_argument(
        "--grounding_results_dir",
        default="./eval_results/grounding_group/grounding_lora_0529_all_215000",
        help="Directory containing {Dataset}_results.csv from group grounding.",
    )
    parser.add_argument(
        "--datasets",
        default="ActGroundGroup,BindIGroundGroup,MotifGroundGroup,EvoGroundGroup,DomGroundGroup",
        help="Comma-separated group grounding datasets.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--save_results_dir", default="./eval_results/grounding_group_ref")
    parser.add_argument("--model_identifier", default="0529_all_215000")
    parser.add_argument(
        "--tasks",
        default="cls",
        # default="cls,desc",
        help="Comma-separated tasks to run: cls, desc, or cls,desc.",
    )
    parser.add_argument("--single_gpu", action="store_true", default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=1021)
    parser.add_argument("--perceiver_latent_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_positions(value) -> List[Tuple[int, int]]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        raw_positions = value
    else:
        try:
            raw_positions = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []
    positions = []
    for item in raw_positions:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if start > end:
            start, end = end, start
        if start < 0 or end < 0:
            continue
        positions.append((start, end))
    return positions


def get_truth_half_open_positions(data_item: Dict) -> List[Tuple[int, int]]:
    positions = []
    for fragment_group in data_item["fragments"]:
        group_positions = [
            (int(frag["start_position"]), int(frag["end_position"]) + 1)
            for frag in fragment_group["frags"]
        ]
        group_positions.sort(key=lambda x: x[0])
        positions.extend(group_positions)
    return positions


def infer_crop_start_from_grounding_row(data_item: Dict, result_row) -> Tuple[int, str]:
    truth_positions = get_truth_half_open_positions(data_item)
    local_gt_positions = parse_positions(result_row.get("gt_positions", "[]"))
    if not truth_positions or not local_gt_positions:
        return 0, "missing_gt_positions_for_crop_inference"
    if len(truth_positions) != len(local_gt_positions):
        return 0, "gt_position_count_mismatch_for_crop_inference"

    candidates = []
    for abs_pos, local_pos in zip(truth_positions, local_gt_positions):
        candidates.append(abs_pos[0] - local_pos[0])
        candidates.append(abs_pos[1] - local_pos[1])
    first = candidates[0]
    if all(candidate == first for candidate in candidates):
        return max(0, first), ""
    return max(0, first), "inconsistent_crop_start_inference"


def clean_generation(text: str) -> str:
    return (
        str(text)
        .replace("<|reserved_special_token_0|>", "")
        .replace("<|eot_id|>", "")
        .strip()
    )


def create_group_dataset(dataset_name: str, args):
    config = GROUNDING_DATASETS[dataset_name]
    return config["dataset_class"](
        root_dir=args.root_dir,
        split=args.split,
        max_sequence_length=args.max_sequence_length,
        use_detailed_template=True,
        pos_decoder_type="ProteinSAM",
    )


class GroundingRegionRefDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: List[Dict],
        dataset_config: Dict,
        task: str,
        max_sequence_length: int,
        perceiver_latent_size: int,
    ):
        self.rows = rows
        self.dataset_config = dataset_config
        self.task = task
        self.max_sequence_length = max_sequence_length
        self.perceiver_latent_size = perceiver_latent_size
        self.sequence_placeholder = "<|reserved_special_token_1|>"
        self.fragment_placeholder = "<|reserved_special_token_2|>" * perceiver_latent_size

    def __len__(self):
        return len(self.rows)

    def _build_conversation(self, sequence: str):
        if self.task == "cls":
            question_template = random.choice(Frag_Class)
            question = question_template.format(
                full_sequence=self.sequence_placeholder * (len(sequence) + 2),
                task_name=self.dataset_config["task_name"],
                fragment=self.fragment_placeholder,
            )
        elif self.task == "desc":
            question_template = random.choice(self.dataset_config["desc_template"])
            question = question_template.format(
                full_sequence=self.sequence_placeholder * (len(sequence) + 2),
                fragment=self.fragment_placeholder,
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ]

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        sequence = row["sequence"]
        crop_start = row["crop_start"]
        crop_end = min(crop_start + self.max_sequence_length, len(sequence))
        sequence_window = sequence[crop_start:crop_end]
        start = max(0, row["start_pos"] - crop_start)
        end = min(len(sequence_window), row["end_pos"] - crop_start + 1)
        if end <= start:
            end = min(len(sequence_window), start + 1)

        return {
            "sequence": sequence_window,
            "conversation": self._build_conversation(sequence_window),
            "answer": "",
            "position_ref": [start, end],
            "position_grd": None,
            "start": crop_start,
            "dataset_idx": row["row_id"],
            "interpro_id": None,
        }


def choose_crop_start(
    sequence_len: int,
    start_pos: int,
    end_pos: int,
    max_sequence_length: int,
) -> int:
    if sequence_len <= max_sequence_length:
        return 0
    max_start = max(0, sequence_len - max_sequence_length)
    center = (start_pos + end_pos) // 2
    crop_start = center - max_sequence_length // 2
    crop_start = max(0, min(crop_start, max_start))
    if start_pos < crop_start:
        crop_start = start_pos
    if end_pos >= crop_start + max_sequence_length:
        crop_start = end_pos - max_sequence_length + 1
    return max(0, min(crop_start, max_start))


def collect_region_rows(dataset_name: str, args) -> List[Dict]:
    dataset = create_group_dataset(dataset_name, args)
    grounding_path = os.path.join(args.grounding_results_dir, f"{dataset_name}_results.csv")
    if not os.path.exists(grounding_path):
        raise FileNotFoundError(f"Grounding results not found: {grounding_path}")

    grounding_df = pd.read_csv(grounding_path)
    grounding_df = grounding_df.drop_duplicates(subset=["dataset_idx"], keep="first")
    data_by_idx = {int(item["dataset_idx"]): item for item in dataset.data_infos}

    rows = []
    for _, result_row in grounding_df.iterrows():
        dataset_idx = int(result_row["dataset_idx"])
        data_item = data_by_idx.get(dataset_idx)
        if data_item is None:
            continue
        inferred_crop_start, crop_warning = infer_crop_start_from_grounding_row(
            data_item, result_row
        )
        sequence = data_item["sequence"]
        positions = parse_positions(result_row.get("pred_positions", "[]"))
        for region_idx, (start_pos, end_pos) in enumerate(positions):
            absolute_start = start_pos + inferred_crop_start
            # Grounding dataloaders render end positions as half-open end+1.
            absolute_end_inclusive = end_pos + inferred_crop_start - 1
            if absolute_start >= len(sequence):
                warning = "pred_start_out_of_sequence"
                start_clamped = len(sequence) - 1
            else:
                warning = crop_warning
                start_clamped = absolute_start
            end_clamped = min(absolute_end_inclusive, len(sequence) - 1)
            if end_clamped < start_clamped:
                end_clamped = start_clamped
            crop_start = inferred_crop_start
            if (
                len(sequence) > args.max_sequence_length
                and crop_start + args.max_sequence_length > len(sequence)
            ):
                crop_start = choose_crop_start(
                    len(sequence), start_clamped, end_clamped, args.max_sequence_length
                )
                if not warning:
                    warning = "crop_start_adjusted_to_sequence_bounds"
            rows.append(
                {
                    "row_id": len(rows),
                    "dataset_name": dataset_name,
                    "ref_prefix": GROUNDING_DATASETS[dataset_name]["ref_prefix"],
                    "uid": data_item["uid"],
                    "dataset_idx": dataset_idx,
                    "region_idx": region_idx,
                    "local_start_pos": start_pos,
                    "local_end_pos": end_pos,
                    "start_pos": start_clamped,
                    "end_pos": end_clamped,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "crop_start": crop_start,
                    "coordinate_warning": warning,
                    "grounding_generated": result_row.get("generated", ""),
                    "grounding_reference": result_row.get("reference", ""),
                }
            )
    return rows


def generate_task_outputs(
    rows: List[Dict],
    dataset_config: Dict,
    task: str,
    model,
    tokenizer,
    sequence_tokenizer,
    args,
    device,
) -> Dict[int, str]:
    task_dataset = GroundingRegionRefDataset(
        rows=rows,
        dataset_config=dataset_config,
        task=task,
        max_sequence_length=args.max_sequence_length,
        perceiver_latent_size=args.perceiver_latent_size,
    )
    data_collator = FragDataCollator(
        sequence_tokenizer=sequence_tokenizer,
        llm_tokenizer=tokenizer,
        mode="inference",
        max_sequence_length=args.max_sequence_length,
        max_description_length=512,
        use_max_desc_length=True,
    )
    dataloader = DataLoader(
        task_dataset,
        batch_size=args.batch_per_device,
        num_workers=0,
        shuffle=False,
        collate_fn=data_collator,
    )

    outputs: Dict[int, str] = {}
    for inputs in tqdm(dataloader, desc=f"{task} inference"):
        row_ids = inputs["dataset_idxs"]
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
        generated = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
        for row_id, text in zip(row_ids, generated):
            outputs[int(row_id)] = clean_generation(text)
    return outputs


def infer_save_path(args, dataset_name: str) -> str:
    output_dir = os.path.join(args.save_results_dir, args.model_identifier)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{dataset_name}_group_regions_ref_results.csv")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for dataset_name in dataset_list:
        if dataset_name not in GROUNDING_DATASETS:
            raise ValueError(
                f"Unknown dataset {dataset_name}. Available: {list(GROUNDING_DATASETS)}"
            )
    for task in task_list:
        if task not in {"cls", "desc"}:
            raise ValueError("--tasks must contain only cls and/or desc")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, pad_token="<|reserved_special_token_0|>"
    )
    model = ProteinLlamaForCausalLM.from_pretrained(args.model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    sequence_tokenizer = AutoTokenizer.from_pretrained(model.config.esm_path)
    model.eval()
    model = model.bfloat16().to(device)

    for dataset_name in dataset_list:
        print(f"\n=== Processing {dataset_name} ===")
        rows = collect_region_rows(dataset_name, args)
        print(f"Collected {len(rows)} predicted regions")
        if not rows:
            save_path = infer_save_path(args, dataset_name)
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"No regions found. Empty result saved to {save_path}")
            continue

        task_outputs = {}
        for task in task_list:
            task_outputs[task] = generate_task_outputs(
                rows,
                GROUNDING_DATASETS[dataset_name],
                task,
                model,
                tokenizer,
                sequence_tokenizer,
                args,
                device,
            )

        for row in rows:
            row["pred_cls"] = task_outputs.get("cls", {}).get(row["row_id"], "")
            row["pred_desc"] = task_outputs.get("desc", {}).get(row["row_id"], "")
            row["position"] = f"({row['start_pos']},{row['end_pos']})"
            row.pop("sequence", None)

        save_path = infer_save_path(args, dataset_name)
        columns = [
            "dataset_name",
            "ref_prefix",
            "uid",
            "dataset_idx",
            "region_idx",
            "local_start_pos",
            "local_end_pos",
            "start_pos",
            "end_pos",
            "position",
            "sequence_length",
            "crop_start",
            "coordinate_warning",
            "pred_cls",
            "pred_desc",
            "grounding_generated",
            "grounding_reference",
        ]
        pd.DataFrame(rows)[columns].to_csv(save_path, index=False)
        print(f"Saved {len(rows)} region-level results to {save_path}")


if __name__ == "__main__":
    main()
