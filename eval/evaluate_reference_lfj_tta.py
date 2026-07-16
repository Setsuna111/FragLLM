#!/usr/bin/env python3
"""Three-view reference-class inference with mean semantic-score aggregation.

Use the ``generate`` subcommand in the FragLLM environment and ``aggregate``
in an environment that supports Qwen3-Embedding.  The accompanying shell
script runs both stages with the repository's validated environments.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = REPO_ROOT / "checkpoints/0529_all/checkpoint-315000_merge"
DEFAULT_DATA = REPO_ROOT / "data_70"
DEFAULT_EMBEDDING_MODEL = Path(
    "/home/dataset-local/projects_dir/pretrained_model/Qwen3-Embedding-0.6B"
)
DEFAULT_ONTOLOGY = Path(
    "/home/dataset-local/projects_dir/VenusX_dataset/final_interpro_metadata.json"
)
DEFAULT_CACHE = (
    REPO_ROOT
    / "eval/cache/interpro_embeddings_home_dataset-local_projects_dir_pretrained_model_Qwen3-Embedding-0.6B.pkl"
)
SUPPORTED_DATASETS = (
    "ActRefClass",
    "BindIRefClass",
    "DomRefClass",
    "EvoRefClass",
    "MotifRefClass",
)


def comma_separated_seeds(value: str):
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from error
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("Exactly three distinct seeds are required")
    return seeds


def comma_separated_datasets(value: str):
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("At least one dataset is required")
    unsupported = [item for item in datasets if item not in SUPPORTED_DATASETS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported datasets {unsupported}; choose from {list(SUPPORTED_DATASETS)}"
        )
    if len(set(datasets)) != len(datasets):
        raise argparse.ArgumentTypeError("Dataset names must be unique")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-class three-view inference and semantic aggregation"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    generate = subparsers.add_parser("generate", help="Generate three greedy views")
    generate.add_argument("--model_path", default=str(DEFAULT_MODEL))
    generate.add_argument("--root_dir", default=str(DEFAULT_DATA))
    generate.add_argument("--split", default="test")
    generate.add_argument(
        "--datasets",
        type=comma_separated_datasets,
        default=["ActRefClass"],
        help=f"Comma-separated datasets chosen from {list(SUPPORTED_DATASETS)}",
    )
    generate.add_argument("--output_dir", required=True)
    generate.add_argument("--seeds", type=comma_separated_seeds, default=[42, 43, 44])
    generate.add_argument("--batch_per_device", type=int, default=4)
    generate.add_argument("--gpu_id", type=int, default=0)
    generate.add_argument("--perceiver_latent_size", type=int, default=4)
    generate.add_argument("--max_new_tokens", type=int, default=512)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--show_progress", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", help="Average semantic scores from three generated views"
    )
    aggregate.add_argument("--inputs", nargs=3, required=True)
    aggregate.add_argument("--output_path", required=True)
    aggregate.add_argument("--details_path", required=True)
    aggregate.add_argument("--config_path", required=True)
    aggregate.add_argument("--dataset_name", choices=SUPPORTED_DATASETS)
    aggregate.add_argument("--embedding_model", default=str(DEFAULT_EMBEDDING_MODEL))
    aggregate.add_argument("--ontology_path", default=str(DEFAULT_ONTOLOGY))
    aggregate.add_argument("--cache_path", default=str(DEFAULT_CACHE))
    aggregate.add_argument("--batch_size", type=int, default=32)
    aggregate.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_views(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from dataset.dataloader_frag import FragDataCollator
    from eval.evaluate_reference_lfj import create_dataset
    from models.protein_llama_addtoken_lfj import ProteinLlamaForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Three-view inference requires CUDA")

    started_at = time.perf_counter()
    device = torch.device(f"cuda:{args.gpu_id}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model loading does not use Python's random module, which drives the
    # dataset's prompt choice and fragment-preserving crop. Each view resets
    # all RNGs again immediately before its dataset is iterated.
    seed_everything(args.seeds[0])
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
        max_sequence_length=1021,
        max_description_length=512,
        use_max_desc_length=True,
    )

    view_paths = {}
    view_runtimes = {}
    for dataset_name in args.datasets:
        view_paths[dataset_name] = {}
        view_runtimes[dataset_name] = {}
        for seed in args.seeds:
            view_started_at = time.perf_counter()
            seed_everything(seed)
            dataset = create_dataset(
                dataset_name,
                args.root_dir,
                args.split,
                args.perceiver_latent_size,
            )
            if args.limit is not None:
                dataset.data_infos = dataset.data_infos[: args.limit]
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_per_device,
                num_workers=0,
                shuffle=False,
                collate_fn=collator,
            )

            rows = []
            for inputs in tqdm(
                dataloader,
                desc=f"{dataset_name} seed={seed}",
                disable=not args.show_progress,
            ):
                references = tokenizer.batch_decode(
                    inputs["answer_input_ids"], skip_special_tokens=True
                )
                inputs = {
                    key: value.to(device=device, non_blocking=True)
                    if hasattr(value, "to")
                    else value
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    token_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        protein_input_ids=inputs["protein_input_ids"],
                        protein_attention_mask=inputs["protein_attention_mask"],
                        protein_inputs_embeds=None,
                        position_refs=inputs["position_refs"],
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        early_stopping=False,
                        no_repeat_ngram_size=None,
                        length_penalty=1.0,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                for row_index, (prediction, reference) in enumerate(
                    zip(generated, references)
                ):
                    rows.append(
                        {
                            "generated": prediction,
                            "reference": reference,
                            "dataset_idx": inputs["dataset_idxs"][row_index],
                            "interpro_ids": inputs["interpro_ids"][row_index],
                        }
                    )

            view_dir = output_dir / "views" / f"seed{seed}"
            view_dir.mkdir(parents=True, exist_ok=True)
            view_path = view_dir / f"{dataset_name}_results.csv"
            pd.DataFrame(rows).to_csv(view_path, index=False)
            view_paths[dataset_name][str(seed)] = str(view_path)
            view_runtimes[dataset_name][str(seed)] = (
                time.perf_counter() - view_started_at
            )
            print(
                f"Saved {dataset_name} seed {seed} view "
                f"({len(rows)} samples) to {view_path}"
            )

    config = {
        "stage": "generate",
        "datasets": args.datasets,
        "model_path": str(Path(args.model_path).resolve()),
        "root_dir": str(Path(args.root_dir).resolve()),
        "split": args.split,
        "seeds": args.seeds,
        "batch_per_device": args.batch_per_device,
        "gpu_id": args.gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "perceiver_latent_size": args.perceiver_latent_size,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "view_paths": view_paths,
        "view_runtime_seconds": view_runtimes,
        "runtime_seconds": time.perf_counter() - started_at,
    }
    (output_dir / "tta_generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_view_frames(frames) -> None:
    required = {"generated", "reference", "dataset_idx", "interpro_ids"}
    for path, frame in frames:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    base = frames[0][1]
    for path, frame in frames[1:]:
        if not base["dataset_idx"].equals(frame["dataset_idx"]):
            raise ValueError(f"dataset_idx is not aligned in {path}")


def aggregate_views(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    started_at = time.perf_counter()
    frames = [(Path(path), pd.read_csv(path)) for path in args.inputs]
    validate_view_frames(frames)
    base = frames[0][1].copy()

    with open(args.cache_path, "rb") as handle:
        cache = pickle.load(handle)
    label_embeddings = np.asarray(cache["embeddings"])
    interpro_ids = list(cache["interpro_ids"])
    if label_embeddings.shape[0] != len(interpro_ids):
        raise ValueError("Cached label embeddings and InterPro IDs have different lengths")
    with open(args.ontology_path, encoding="utf-8") as handle:
        ontology = json.load(handle)

    all_texts = [
        str(text)
        for _, frame in frames
        for text in frame["generated"].fillna("").tolist()
    ]
    unique_texts = list(dict.fromkeys(all_texts))
    model = SentenceTransformer(args.embedding_model, device=args.device)
    text_embeddings = model.encode(
        unique_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    similarities = model.similarity(text_embeddings, label_embeddings).cpu().numpy()
    scores_by_text = dict(zip(unique_texts, similarities))
    view_scores = [
        np.stack(
            [scores_by_text[str(text)] for text in frame["generated"].fillna("")]
        )
        for _, frame in frames
    ]
    view_winners = [scores.argmax(axis=1) for scores in view_scores]
    mean_scores = np.stack(view_scores).mean(axis=0)
    mean_winners = mean_scores.argmax(axis=1)

    result = base.copy()
    result["generated"] = [
        f"It is the {ontology[interpro_ids[index]]['category']}."
        for index in mean_winners
    ]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    agreement = Counter()
    details = []
    for row_index, winner in enumerate(mean_winners):
        labels = [int(view[row_index]) for view in view_winners]
        label_counts = Counter(labels)
        if len(label_counts) == 1:
            agreement["unanimous"] += 1
        elif max(label_counts.values()) == 2:
            agreement["majority"] += 1
        else:
            agreement["all_different"] += 1
        detail = {
            "dataset_idx": base.iloc[row_index]["dataset_idx"],
            "target_interpro_id": base.iloc[row_index]["interpro_ids"],
            "mean_similarity_interpro_id": interpro_ids[int(winner)],
        }
        for view_index, ((path, frame), view) in enumerate(
            zip(frames, view_winners)
        ):
            detail[f"view_{view_index}_source"] = str(path)
            detail[f"view_{view_index}_generated"] = frame.iloc[row_index]["generated"]
            detail[f"view_{view_index}_interpro_id"] = interpro_ids[
                int(view[row_index])
            ]
        details.append(detail)

    details_path = Path(args.details_path)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(details_path, index=False)
    config = {
        "stage": "aggregate",
        "method": "three_view_mean_semantic_similarity",
        "dataset": args.dataset_name,
        "inputs": [str(path.resolve()) for path, _ in frames],
        "output_path": str(output_path.resolve()),
        "details_path": str(details_path.resolve()),
        "embedding_model": str(Path(args.embedding_model).resolve()),
        "ontology_path": str(Path(args.ontology_path).resolve()),
        "cache_path": str(Path(args.cache_path).resolve()),
        "batch_size": args.batch_size,
        "device": args.device,
        "num_samples": len(result),
        "unique_generated_texts": len(unique_texts),
        "agreement": dict(agreement),
        "uses_target_for_selection": False,
        "runtime_seconds": time.perf_counter() - started_at,
    }
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Agreement counts: {dict(agreement)}")
    print(f"Saved final TTA result ({len(result)} samples) to {output_path}")


def main() -> None:
    args = parse_args()
    if args.stage == "generate":
        generate_views(args)
    elif args.stage == "aggregate":
        aggregate_views(args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
