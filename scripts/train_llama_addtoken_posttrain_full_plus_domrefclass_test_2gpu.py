#!/usr/bin/env python3
"""Continue 4-GPU training on 2 GPUs with full train data + DomRefClass test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scripts.train_llama_addtoken as base_training
from dataset.dataloader_frag import FragDataCollator, HybridTrainDataset
from zero2_resume_compat_4to2 import install_zero2_4to2_resize_compat


FULL_TRAIN_CONFIG = (
    "ProFunction||ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||"
    "MotifRefClass||ActRefDesc||BindIRefDesc||DomRefDesc||EvoRefDesc||"
    "MotifRefDesc||ActGroundSingle||BindIGroundSingle||MotifGroundSingle||"
    "DomGroundSingle||EvoGroundSingle||ActGroundGroup||BindIGroundGroup||"
    "DomGroundGroup||EvoGroundGroup||MotifGroundGroup"
)


class FullTrainPlusDomainRefClassTestDataset:
    """Concatenate all formal train data with DomRefClass test samples."""

    def __init__(
        self,
        data_root: str,
        max_sequence_length: int,
        perceiver_latent_size: int,
        use_detailed_template: bool,
        pos_decoder_type: str,
        filter_sequence: bool,
    ) -> None:
        common_kwargs = {
            "max_sequence_length": max_sequence_length,
            "perceiver_latent_size": perceiver_latent_size,
            "use_detailed_template": use_detailed_template,
            "pos_decoder_type": pos_decoder_type,
            "filter_sequence": filter_sequence,
        }
        self.full_train_dataset = HybridTrainDataset(
            root_dir=data_root,
            data_train=FULL_TRAIN_CONFIG,
            split="train",
            **common_kwargs,
        )
        self.domain_test_dataset = HybridTrainDataset(
            root_dir=data_root,
            data_train="DomRefClass",
            split="test",
            **common_kwargs,
        )
        self.full_train_length = len(self.full_train_dataset)
        self.domain_test_length = len(self.domain_test_dataset)
        print(
            "[posttrain-data] full_train="
            f"{self.full_train_length}; domain_refclass_test={self.domain_test_length}; "
            f"total={len(self)}"
        )

    def __len__(self) -> int:
        return self.full_train_length + self.domain_test_length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.full_train_length:
            return self.full_train_dataset[index]
        return self.domain_test_dataset[index - self.full_train_length]

    def summary(self) -> Dict[str, Any]:
        return {
            "data_root": str(Path(self.full_train_dataset.root_dir).resolve()),
            "full_train_config": FULL_TRAIN_CONFIG,
            "full_train_split": "train",
            "domain_test_config": "DomRefClass",
            "domain_test_split": "test",
            "full_train_samples": self.full_train_length,
            "domain_refclass_test_samples": self.domain_test_length,
            "total_samples_per_epoch": len(self),
            "sampling": "full_train_once_plus_domain_refclass_test_once",
        }


def make_posttrain_data_module(data_args: Any) -> Dict[str, Any]:
    data_root = os.environ["POSTTRAIN_DATA_ROOT"]
    train_dataset = FullTrainPlusDomainRefClassTestDataset(
        data_root=data_root,
        max_sequence_length=data_args.max_sequence_length,
        perceiver_latent_size=data_args.perceiver_latent_size,
        use_detailed_template=data_args.use_detailed_template,
        pos_decoder_type=data_args.pos_decoder_type,
        filter_sequence=data_args.filter_sequence,
    )

    manifest_path = os.environ.get("POSTTRAIN_DATA_MANIFEST")
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(train_dataset.summary(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    data_collator = FragDataCollator(
        sequence_tokenizer=data_args.sequence_tokenizer,
        llm_tokenizer=data_args.llm_tokenizer,
        mode="train",
        max_sequence_length=data_args.max_sequence_length,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }


def _remove_cli_option(option: str) -> None:
    while option in sys.argv:
        index = sys.argv.index(option)
        del sys.argv[index : index + 2]


def _prepare_continuation_proxy(
    checkpoint: Path, output_dir: Path, additional_steps: int
) -> Dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    trainer_state_path = checkpoint / "trainer_state.json"
    latest_path = checkpoint / "latest"
    if not checkpoint.is_dir() or not trainer_state_path.is_file() or not latest_path.is_file():
        raise ValueError(
            "The resume checkpoint must be a full Trainer/DeepSpeed checkpoint "
            f"with trainer_state.json and latest: {checkpoint}"
        )

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    source_step = int(state["global_step"])
    scheduler_max_steps = int(state["max_steps"])
    stop_step = source_step + int(additional_steps)
    tag = latest_path.read_text(encoding="utf-8").strip()
    if not (checkpoint / tag).is_dir():
        raise ValueError(f"Missing DeepSpeed tag directory {tag} in {checkpoint}")
    if additional_steps <= 0 or stop_step > scheduler_max_steps:
        raise ValueError(
            f"Invalid continuation range: source={source_step}, "
            f"additional={additional_steps}, scheduler_max_steps={scheduler_max_steps}"
        )

    metadata = {
        "source_checkpoint": str(checkpoint),
        "source_global_step": source_step,
        "additional_training_steps": int(additional_steps),
        "stop_global_step": stop_step,
        "scheduler_max_steps": scheduler_max_steps,
        "policy": "restore model, LoRA, non-LoRA, optimizer, scheduler, RNG and TrainerState",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "posttrain_continuation.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"Existing continuation metadata differs: {metadata_path}")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    proxy = output_dir / f"checkpoint-{source_step}"
    other_checkpoints = [
        path for path in output_dir.glob("checkpoint-*") if path.name != proxy.name
    ]
    if other_checkpoints and not proxy.exists():
        raise RuntimeError(
            f"{output_dir} already contains checkpoints but no source proxy; use a new output_dir"
        )
    if not proxy.exists():
        proxy.mkdir()
        for source in checkpoint.iterdir():
            destination = proxy / source.name
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        (proxy / "continuation_proxy.json").write_text(
            json.dumps(
                {"source_checkpoint": str(checkpoint), "preserves_full_state": True},
                indent=2,
            ),
            encoding="utf-8",
        )
    return metadata


def configure_continuation() -> Dict[str, Any]:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--resume-checkpoint-path", required=True)
    wrapper_parser.add_argument("--additional-training-steps", type=int, required=True)
    wrapper_parser.add_argument("--output_dir", required=True)
    options, _ = wrapper_parser.parse_known_args()
    _remove_cli_option("--resume-checkpoint-path")
    _remove_cli_option("--additional-training-steps")
    if "--max_steps" in sys.argv:
        raise ValueError("Do not pass --max_steps; it is inherited from the source checkpoint")

    metadata = _prepare_continuation_proxy(
        Path(options.resume_checkpoint_path),
        Path(options.output_dir),
        options.additional_training_steps,
    )
    sys.argv.extend(["--max_steps", str(metadata["scheduler_max_steps"])])
    os.environ["POSTTRAIN_STOP_GLOBAL_STEP"] = str(metadata["stop_global_step"])
    return metadata


class ContinuationSaveScheduleCallback(transformers.TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = int(args.save_steps)
        print(f"[posttrain-2gpu] continuation save_steps={state.save_steps}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= int(os.environ["POSTTRAIN_STOP_GLOBAL_STEP"]):
            control.should_save = True
            control.should_training_stop = True
        return control


class ContinuedFragTrainer(base_training.FragTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(ContinuationSaveScheduleCallback())

    def _save_checkpoint(self, model, trial, metrics=None):
        result = transformers.Trainer._save_checkpoint(self, model, trial, metrics)
        self._save_non_lora_trainables()
        return result

    def _save_non_lora_trainables(self):
        if not self.is_world_process_zero():
            return
        checkpoint_dir = Path(self._get_output_dir(trial=None)) / (
            f"checkpoint-{self.state.global_step}"
        )
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
        model_to_inspect = self.model
        if getattr(self, "accelerator", None) is not None:
            model_to_inspect = self.accelerator.unwrap_model(self.model)
        non_lora_state = base_training.get_peft_state_non_lora_with_protein_sam_maybe_zero_3(
            model_to_inspect.named_parameters(), require_grad_only=True
        )
        if not non_lora_state:
            raise RuntimeError(f"No trainable non-LoRA tensors found in {checkpoint_dir}")
        non_lora_path = checkpoint_dir / "non_lora_trainables.bin"
        torch.save(non_lora_state, non_lora_path)
        metadata = {
            "format_version": 1,
            "global_step": int(self.state.global_step),
            "tensor_count": len(non_lora_state),
            "numel": sum(tensor.numel() for tensor in non_lora_state.values()),
            "keys": sorted(non_lora_state),
            "source": "get_peft_state_non_lora_with_protein_sam_maybe_zero_3",
            "require_grad_only": True,
            "optimizer_state_preserved": True,
        }
        (checkpoint_dir / "non_lora_trainables_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            "[posttrain-2gpu] saved non-LoRA trainables: "
            f"{non_lora_path} ({metadata['tensor_count']} tensors, {metadata['numel']} parameters)"
        )
        del non_lora_state


def main() -> None:
    install_zero2_4to2_resize_compat()
    metadata = configure_continuation()
    print(
        "2-GPU continuation: "
        f"step {metadata['source_global_step']} -> {metadata['stop_global_step']} "
        f"(scheduler horizon={metadata['scheduler_max_steps']})"
    )
    base_training.make_multitask_dataset = make_posttrain_data_module
    base_training.FragTrainer = ContinuedFragTrainer
    base_training.train()


if __name__ == "__main__":
    main()
