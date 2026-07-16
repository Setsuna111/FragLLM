#!/usr/bin/env python
"""Task4-CoT training with unambiguous, order-augmented category prompts.

This reuses the original model, loss, LoRA, optimizer, checkpoint, and
DeepSpeed implementation. It only swaps the data factory and restores model
weights directly from an unmerged DeepSpeed checkpoint with fresh Task4
training progress.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import transformers.trainer as hf_trainer
from transformers.trainer_callback import TrainerState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataloader_frag import FragDataCollator
import scripts.train_llama_addtoken as base_training


SYSTEM_MESSAGE = (
    "You are a scientific assistant specializing in protein sequence analysis. "
    "Based on protein sequence embeddings and other related information, please "
    "answer the relevant questions using professional language. "
)
SEQUENCE_PLACEHOLDER = "<|reserved_special_token_1|>"


def normalize_category(value: object) -> str:
    return " ".join(str(value or "").split()).strip(" .")


def parse_category_list(value: object) -> List[str]:
    """Read the JSON-list schema; reject legacy comma-joined values as ambiguous."""
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "fragment_classes must be a JSON list. Rebuild task4_cot_train.csv with "
            "analysis/0716analysis_2/build_task4_cot_train_data_0716_2.py."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError("fragment_classes must decode to a JSON list")
    categories = [normalize_category(item) for item in parsed]
    categories = [category for category in categories if category]
    if not categories:
        raise ValueError("fragment_classes contains no usable category")
    return categories


def format_class_sentence(categories: List[str]) -> str:
    return (
        f"This protein may contain {len(categories)} fragment classes: "
        f"{'; '.join(categories)}. "
    )


class Task4CoTFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, max_sequence_length: int):
        self.frame = pd.read_csv(csv_path).reset_index(drop=True)
        self.max_sequence_length = max_sequence_length
        required = {
            "accession", "sequence", "function", "Full Name", "taxon", "fragment_classes"
        }
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Task4 training CSV is missing columns: {sorted(missing)}")
        self.class_lists = [parse_category_list(value) for value in self.frame["fragment_classes"]]
        declared_counts = self.frame.get("fragment_class_count")
        if declared_counts is not None:
            for index, (categories, count) in enumerate(zip(self.class_lists, declared_counts)):
                if int(count) != len(categories):
                    raise ValueError(
                        f"row {index}: fragment_class_count={count} does not match "
                        f"the JSON list length {len(categories)}"
                    )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict:
        row = self.frame.iloc[idx]
        sequence = str(row["sequence"])[: self.max_sequence_length]
        categories = list(self.class_lists[idx])
        random.shuffle(categories)  # Data augmentation: a new class order on each access.
        question = (
            "Protein name: {fullname}; Taxon: {taxon}; Sequence embeddings: "
            "{sequence_embeddings}. {class_sentence}"
            "Please describe its function clearly and concisely in professional language."
        ).format(
            fullname=row["Full Name"],
            taxon=row["taxon"],
            sequence_embeddings=SEQUENCE_PLACEHOLDER * (len(sequence) + 2),
            class_sentence=format_class_sentence(categories),
        )
        return {
            "sequence": sequence,
            "conversation": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": question},
            ],
            "answer": str(row["function"]),
            "position_ref": None,
            "position_grd": None,
            "start": 0,
            "dataset_idx": idx,
            "interpro_id": None,
        }


def make_task4_data_module(data_args) -> Dict:
    csv_path = Path(os.environ["TASK4_COT_TRAIN_CSV"])
    train_dataset = Task4CoTFunctionDataset(csv_path, data_args.max_sequence_length)
    collator = FragDataCollator(
        sequence_tokenizer=data_args.sequence_tokenizer,
        llm_tokenizer=data_args.llm_tokenizer,
        mode="train",
        max_sequence_length=data_args.max_sequence_length,
    )
    print(f"Task4-CoT category-only training rows: {len(train_dataset)}")
    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": collator}


def _remove_cli_option(option: str) -> None:
    while option in sys.argv:
        index = sys.argv.index(option)
        del sys.argv[index : index + 2]


def _load_raw_model_weights_only(deepspeed_engine, checkpoint_path, load_module_strict=True):
    """Restore DeepSpeed module state while Task4 optimizer and schedule start fresh."""
    load_path, _ = deepspeed_engine.load_checkpoint(
        checkpoint_path,
        load_module_strict=load_module_strict,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise ValueError(f"[deepspeed] failed to load model state from {checkpoint_path}")


def _prepare_reset_resume_proxy(
    checkpoint: Path,
    output_dir: Path,
    train_batch_size: int,
    logging_steps: int,
    save_steps: int,
) -> Path:
    """Expose raw DeepSpeed shards to Trainer while replacing old training progress."""
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not (checkpoint / "latest").is_file() or not list(checkpoint.glob("global_step*")):
        raise ValueError(f"{checkpoint} is not a DeepSpeed checkpoint directory")

    output_dir.mkdir(parents=True, exist_ok=True)
    proxy = output_dir / "checkpoint-0"
    marker = proxy / "task4_reset_resume.json"
    if proxy.exists():
        if not marker.is_file():
            raise FileExistsError(
                f"{proxy} already exists and is not a Task4 reset-resume proxy. "
                "Use a new --output_dir or remove that directory deliberately."
            )
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if Path(metadata["source_checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"{proxy} points to {metadata['source_checkpoint']}, not {checkpoint}")
        return proxy

    proxy.mkdir()
    for source in checkpoint.iterdir():
        if source.name == "trainer_state.json" or source.name.startswith("rng_state_"):
            continue
        (proxy / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())

    state = TrainerState(
        global_step=0,
        logging_steps=logging_steps,
        save_steps=save_steps,
        train_batch_size=train_batch_size,
    )
    state.save_to_json(proxy / "trainer_state.json")
    marker.write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint.resolve()),
                "policy": "load DeepSpeed module state only; reset Task4 progress, optimizer, scheduler, and RNG",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return proxy


def configure_task4_options() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task4_cot_train_csv",
        default="/home/dataset-local/projects_dir/FragLLM/data_70/Pro2Text/task4_cot_train.csv",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        default=None,
        help=(
            "Raw four-rank DeepSpeed checkpoint whose module state initializes Task4 "
            "training without restoring its old training progress."
        ),
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    options, _ = parser.parse_known_args()
    os.environ["TASK4_COT_TRAIN_CSV"] = options.task4_cot_train_csv
    _remove_cli_option("--task4_cot_train_csv")
    _remove_cli_option("--resume_checkpoint_path")

    if not options.resume_checkpoint_path:
        return
    if not options.output_dir:
        raise ValueError("--output_dir is required with --resume_checkpoint_path")
    checkpoint = Path(options.resume_checkpoint_path).expanduser().resolve()
    output_dir = Path(options.output_dir).expanduser().resolve()
    saved_task4_checkpoints = [
        path for path in output_dir.glob("checkpoint-*") if path.name != "checkpoint-0"
    ] if output_dir.exists() else []
    if saved_task4_checkpoints:
        print("Found an existing Task4 checkpoint; resuming it with its native optimizer and scheduler state.")
        return
    proxy = _prepare_reset_resume_proxy(
        checkpoint=checkpoint,
        output_dir=output_dir,
        train_batch_size=options.per_device_train_batch_size,
        logging_steps=options.logging_steps,
        save_steps=options.save_steps,
    )
    print(f"Task4 reset-resume proxy: {proxy}")
    # Trainer sees checkpoint-0 in output_dir and resumes through this proxy.
    # Patch its DeepSpeed loader so raw model weights are restored while the
    # optimizer and scheduler constructed for the new Task4 run remain fresh.
    hf_trainer.deepspeed_load_checkpoint = _load_raw_model_weights_only


def main() -> None:
    configure_task4_options()
    base_training.make_multitask_dataset = make_task4_data_module
    base_training.train()


if __name__ == "__main__":
    main()
