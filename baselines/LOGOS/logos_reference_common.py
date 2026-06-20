import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


FRAGLLM_ROOT = Path("/home/dataset-local/projects_dir/FragLLM")

DEFAULT_DATA_ROOT = FRAGLLM_ROOT / "data_70"
# DEFAULT_DATA_ROOT = FRAGLLM_ROOT / "data_30"

DEFAULT_OUTPUT_ROOT = FRAGLLM_ROOT / "baselines" / "logos_results"
DEFAULT_MODEL_PATH = Path("/home/dataset-local/projects_dir/pretrained_model/LOGOS-8B/")

DATASET_TO_SOURCE = {
    "ActRefClass": ("VenusX_Act", "active site"),
    "ActRefDesc": ("VenusX_Act", "active site"),
    "BindIRefClass": ("VenusX_BindI", "binding site"),
    "BindIRefDesc": ("VenusX_BindI", "binding site"),
    "DomRefClass": ("VenusX_Dom", "functional domain"),
    "DomRefDesc": ("VenusX_Dom", "functional domain"),
    "EvoRefClass": ("VenusX_Evo", "evolutionary conserved site"),
    "EvoRefDesc": ("VenusX_Evo", "evolutionary conserved site"),
    "MotifRefClass": ("VenusX_Motif", "motif domain"),
    "MotifRefDesc": ("VenusX_Motif", "motif domain"),
}

PROTEIN_START = "<ProteinS>"
PROTEIN_END = "<ProteinE>"
FRAGMENT_START = "<ProteinS>"
FRAGMENT_END = "<ProteinE>"


def str_to_dtype(value: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype: {value}") from exc


def add_common_args(parser: argparse.ArgumentParser, default_datasets: str) -> None:
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_ROOT.name,
        help="Dataset directory under the FragLLM root, such as data_70 or data_30.",
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Compatibility override for the dataset root. Takes precedence over --data_dir.",
    )
    parser.add_argument("--save_results_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model_identifier", default="logos_8b")
    parser.add_argument("--datasets", default=default_datasets)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_per_device", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, default=1021)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--torch_dtype", type=str_to_dtype, default=torch.bfloat16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--device_map",
        default="none",
        help='Use "none" for a single device, or a transformers device_map such as "auto".',
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)


def parse_dataset_list(datasets: str, allowed: Iterable[str]) -> List[str]:
    allowed_set = set(allowed)
    names = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed_set]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed datasets: {sorted(allowed_set)}")
    return names


def resolve_data_root(args: argparse.Namespace) -> Path:
    data_dir = args.root_dir if getattr(args, "root_dir", None) else args.data_dir
    data_root = Path(data_dir)
    if not data_root.is_absolute():
        data_root = FRAGLLM_ROOT / data_root
    return data_root


def make_output_dir(args: argparse.Namespace, output_subdir: str) -> Path:
    data_name = resolve_data_root(args).name
    return Path(args.save_results_dir) / data_name / output_subdir / args.model_identifier


def validate_local_model_dir(model_path: Path) -> None:
    if not model_path.exists():
        return
    if not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"LOGOS model path is empty: {model_path}. Download or copy LOGOS-8B files first."
        )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as handle:
            index = json.load(handle)
        required_shards = sorted(set(index.get("weight_map", {}).values()))
        missing_shards = [name for name in required_shards if not (model_path / name).exists()]
        if missing_shards:
            raise FileNotFoundError(
                f"LOGOS model directory is incomplete: {model_path}. "
                f"Missing weight shards: {missing_shards}"
            )

    has_tokenizer_json = (model_path / "tokenizer.json").exists()
    has_qwen_vocab = (model_path / "vocab.json").exists() and (model_path / "merges.txt").exists()
    if not (has_tokenizer_json or has_qwen_vocab):
        raise FileNotFoundError(
            f"LOGOS tokenizer files are incomplete in {model_path}. "
            "Expected tokenizer.json or vocab.json plus merges.txt."
        )


def crop_around_region(
    sequence: str,
    start_pos: int,
    end_pos: int,
    max_sequence_length: int,
) -> Tuple[str, int, int]:
    if len(sequence) <= max_sequence_length:
        return sequence, start_pos, end_pos + 1

    fragment_len = end_pos - start_pos + 1
    if fragment_len >= max_sequence_length:
        cropped = sequence[start_pos : start_pos + max_sequence_length]
        return cropped, 0, min(fragment_len, max_sequence_length)

    left_context = (max_sequence_length - fragment_len) // 2
    min_window_start = max(0, end_pos - max_sequence_length + 1)
    max_window_start = min(start_pos, len(sequence) - max_sequence_length)
    window_start = start_pos - left_context
    window_start = max(min_window_start, min(max_window_start, window_start))
    cropped = sequence[window_start : window_start + max_sequence_length]
    return cropped, start_pos - window_start, end_pos - window_start + 1


class LogosReferenceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str,
        task_type: str,
        input_mode: str,
        max_sequence_length: int,
        limit: Optional[int] = None,
    ):
        if dataset_name not in DATASET_TO_SOURCE:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        source_name, task_name = DATASET_TO_SOURCE[dataset_name]
        self.dataset_name = dataset_name
        self.source_name = source_name
        self.task_name = task_name
        self.task_type = task_type
        self.input_mode = input_mode
        self.max_sequence_length = max_sequence_length
        json_path = Path(root_dir) / source_name / f"{split}.json"
        with json_path.open() as handle:
            raw_items = json.load(handle)
        self.samples = self._flatten(raw_items)
        if limit is not None:
            self.samples = self.samples[:limit]

    def _flatten(self, raw_items: Sequence[Dict]) -> List[Dict]:
        samples: List[Dict] = []
        dataset_idx = 0
        for item in raw_items:
            for fragment_group in item["fragments"]:
                for fragment in fragment_group["frags"]:
                    start_pos = int(fragment["start_position"])
                    end_pos = int(fragment["end_position"])
                    cropped_sequence, new_start, new_end = crop_around_region(
                        sequence=item["sequence"],
                        start_pos=start_pos,
                        end_pos=end_pos,
                        max_sequence_length=self.max_sequence_length,
                    )
                    samples.append(
                        {
                            "uid": item["uid"],
                            "sequence": cropped_sequence,
                            "fragment_sequence": fragment["sequence"],
                            "category": fragment_group["category"],
                            "description": fragment_group["description"],
                            "interpro_id": fragment_group["interpro_id"],
                            "dataset_idx": dataset_idx,
                            "original_start_pos": start_pos,
                            "original_end_pos": end_pos,
                            "fragment_start_pos": new_start,
                            "fragment_end_pos": new_end,
                        }
                    )
                    dataset_idx += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        if self.task_type == "class":
            sample["reference"] = f"It is the {sample['category']}."
        else:
            sample["reference"] = sample["description"]
        sample["prompt"] = build_prompt(
            sample=sample,
            task_type=self.task_type,
            task_name=self.task_name,
            input_mode=self.input_mode,
        )
        return sample


def build_prompt(sample: Dict, task_type: str, task_name: str, input_mode: str) -> str:
    fragment = sample["fragment_sequence"]
    if input_mode == "frag_only":
        protein_block = f"{PROTEIN_START}{fragment}{PROTEIN_END}"
        context = "Only the target protein fragment is provided."
    else:
        protein_block = f"{PROTEIN_START}{sample['sequence']}{PROTEIN_END}"
        context = (
            "Use the provided protein sequence as context. "
            f"The target fragment amino-acid sequence is {FRAGMENT_START}{fragment}{FRAGMENT_END}. "
            f"Residues {sample['fragment_start_pos']}-{sample['fragment_end_pos'] - 1} "
            f"are the target fragment. "
            f"Positions are 0-based and inclusive."
        )

    if task_type == "class":
        instruction = (
            f"What is the category name of this {task_name} fragment? "
            "Answer with only the category name or one short sentence."
        )
    else:
        instruction = (
            f"Describe the biological function of this {task_name} fragment. Include likely "
            "function, important residues, mechanism, binding context, or conservation context "
            "when inferable."
        )

    return (
        f"{protein_block}\n"
        f"{context}\n"
        f"Question: {instruction}\n"
        "Answer:"
    )


def collate_prompts(batch: Sequence[Dict]) -> Dict:
    return {
        "prompts": [item["prompt"] for item in batch],
        "references": [item["reference"] for item in batch],
        "dataset_idxs": [item["dataset_idx"] for item in batch],
        "interpro_ids": [item["interpro_id"] for item in batch],
        "fragment_sequences": [item["fragment_sequence"] for item in batch],
        "original_start_positions": [item["original_start_pos"] for item in batch],
        "original_end_positions": [item["original_end_pos"] for item in batch],
        "fragment_start_positions": [item["fragment_start_pos"] for item in batch],
        "fragment_end_positions": [item["fragment_end_pos"] for item in batch],
    }


def load_logos_model_and_tokenizer(args: argparse.Namespace):
    model_path = Path(args.model_path)
    validate_local_model_dir(model_path)
    print(f"Loading LOGOS model from {args.model_path}")
    device_map = None if args.device_map == "none" else args.device_map
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=args.torch_dtype,
            device_map=device_map,
        )
    except ValueError as exc:
        raise NotImplementedError(
            f"Failed to load LOGOS model with transformers. Check if the model files are correctly placed in {args.model_path} and if the transformers library is up to date. Original error: {exc}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if device_map is None:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer, device


def generate_texts(model, tokenizer, prompts: Sequence[str], device: torch.device, args) -> List[str]:
    tokenized = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": False,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    with torch.no_grad():
        output_ids = model.generate(**tokenized, **generation_kwargs)
    prompt_len = tokenized["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    texts = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
    return [text.strip() for text in texts]


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one_dataset(
    dataset_name: str,
    task_type: str,
    input_mode: str,
    output_subdir: str,
    model,
    tokenizer,
    device: torch.device,
    args,
) -> Path:
    data_root = resolve_data_root(args)
    dataset = LogosReferenceDataset(
        root_dir=str(data_root),
        dataset_name=dataset_name,
        split=args.split,
        task_type=task_type,
        input_mode=input_mode,
        max_sequence_length=args.max_sequence_length,
        limit=args.limit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_per_device,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_prompts,
    )

    save_path = make_output_dir(args, output_subdir) / f"{dataset_name}_results.csv"
    rows: List[Dict] = []

    print(f"Evaluating {dataset_name}: {len(dataset)} samples")
    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        generated = generate_texts(model, tokenizer, batch["prompts"], device, args)
        for index, text in enumerate(generated):
            row = {
                "generated": text,
                "reference": batch["references"][index],
                "dataset_idx": batch["dataset_idxs"][index],
                "interpro_ids": batch["interpro_ids"][index],
            }
            if input_mode == "frag_only":
                row.update(
                    {
                        "fragment_sequence": batch["fragment_sequences"][index],
                        "original_start_pos": batch["original_start_positions"][index],
                        "original_end_pos": batch["original_end_positions"][index],
                    }
                )
            else:
                row.update(
                    {
                        "original_start_pos": batch["original_start_positions"][index],
                        "original_end_pos": batch["original_end_positions"][index],
                        "fragment_start_pos": batch["fragment_start_positions"][index],
                        "fragment_end_pos": batch["fragment_end_positions"][index],
                    }
                )
            rows.append(row)

    if input_mode == "frag_only":
        fieldnames = [
            "generated",
            "reference",
            "dataset_idx",
            "interpro_ids",
            "fragment_sequence",
            "original_start_pos",
            "original_end_pos",
        ]
    else:
        fieldnames = [
            "generated",
            "reference",
            "dataset_idx",
            "interpro_ids",
            "original_start_pos",
            "original_end_pos",
            "fragment_start_pos",
            "fragment_end_pos",
        ]
    write_csv(save_path, rows, fieldnames)
    print(f"Saved {save_path}")
    return save_path


def run_reference_inference(
    args,
    task_type: str,
    input_mode: str,
    output_subdir: str,
    allowed_datasets: Sequence[str],
) -> List[Path]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    dataset_names = parse_dataset_list(args.datasets, allowed_datasets)
    model, tokenizer, device = load_logos_model_and_tokenizer(args)
    saved_paths = []
    for dataset_name in dataset_names:
        saved_paths.append(
            evaluate_one_dataset(
                dataset_name=dataset_name,
                task_type=task_type,
                input_mode=input_mode,
                output_subdir=output_subdir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                args=args,
            )
        )
    return saved_paths
