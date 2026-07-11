import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
INSTRUCTBIOMOL_ROOT = SCRIPT_DIR.parent
FRAGLLM_ROOT = Path("/home/dataset-local/projects_dir/FragLLM")
VENUSX_STRUCT_ROOT = Path("/home/dataset-local/projects_dir/VenusX_dataset")

# DEFAULT_DATA_ROOT = FRAGLLM_ROOT / "data_70"
DEFAULT_DATA_ROOT = FRAGLLM_ROOT / "data_30"

DEFAULT_OUTPUT_ROOT = FRAGLLM_ROOT / "baselines" / "instructbiomol_results"
DEFAULT_MODEL_PATH = Path("/home/dataset-local/projects_dir/pretrained_model/InstructBioMol-instruct")
DEFAULT_FOLDSEEK_PATH = INSTRUCTBIOMOL_ROOT / "utils" / "foldseek"
REQUIRED_PRETRAINED_ASSETS = (
    "pretrained_ckpt/supervised_contextpred.pth",
    "pretrained_ckpt/geoformer.ckpt",
    "pretrained_ckpt/esm2_t12_35M_UR50D",
    "pretrained_ckpt/SaProt_35M_AF2",
)

DATASET_TO_SOURCE = {
    "ActRefClass": ("VenusX_Act", "Act", "active site"),
    "ActRefDesc": ("VenusX_Act", "Act", "active site"),
    "BindIRefClass": ("VenusX_BindI", "BindI", "binding site"),
    "BindIRefDesc": ("VenusX_BindI", "BindI", "binding site"),
    "DomRefClass": ("VenusX_Dom", "Dom", "functional domain"),
    "DomRefDesc": ("VenusX_Dom", "Dom", "functional domain"),
    "EvoRefClass": ("VenusX_Evo", "Evo", "evolutionary conserved site"),
    "EvoRefDesc": ("VenusX_Evo", "Evo", "evolutionary conserved site"),
    "MotifRefClass": ("VenusX_Motif", "Motif", "motif domain"),
    "MotifRefDesc": ("VenusX_Motif", "Motif", "motif domain"),
}


def add_common_args(parser: argparse.ArgumentParser, default_datasets: str) -> None:
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--llama_ckpt_path",
        default=None,
        help="Path used for Llama config/tokenizer. Defaults to --model_path.",
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_ROOT.name,
        help="Dataset directory under FragLLM, such as data_70 or data_30. Absolute paths are accepted.",
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Compatibility override for the dataset root. If set, it takes precedence over --data_dir.",
    )
    parser.add_argument("--save_results_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model_identifier", default="instructbiomol_instruct_prompt_v2")
    parser.add_argument("--datasets", default=default_datasets)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_per_device", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--datatype", default="bf16", choices=["bf16", "half", "float"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--foldseek_path", default=str(DEFAULT_FOLDSEEK_PATH))
    parser.add_argument("--fragment_offset", type=int, default=1)
    parser.add_argument(
        "--missing_structure",
        default="error",
        choices=["error", "mask"],
        help="Use 'mask' to fall back to sequence-plus-# SaProt strings when a PDB is missing.",
    )


def parse_dataset_list(datasets: str, allowed: Iterable[str]) -> List[str]:
    allowed_set = set(allowed)
    names = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed_set]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed datasets: {sorted(allowed_set)}")
    if not names:
        raise ValueError("At least one dataset must be selected")
    return names


def resolve_data_root(args: argparse.Namespace) -> Path:
    data_dir = args.root_dir if args.root_dir else args.data_dir
    data_root = Path(data_dir)
    if not data_root.is_absolute():
        data_root = FRAGLLM_ROOT / data_root
    return data_root


def make_output_dir(args: argparse.Namespace, output_subdir: str) -> Path:
    return Path(args.save_results_dir) / resolve_data_root(args).name / output_subdir / args.model_identifier


def encode_protein_for_instruction(sequence: str) -> str:
    return "<FASTA>" + "".join(f"<p>{aa}" for aa in sequence) + "</FASTA>"


def structure_dirs(source_dataset: str) -> Tuple[Path, Path]:
    full_dir = VENUSX_STRUCT_ROOT / f"VenusX_{source_dataset}_AlphaFold2_PDB" / "alphafold2_pdb"
    fragment_dir = VENUSX_STRUCT_ROOT / f"VenusX_{source_dataset}_AlphaFold2_PDB" / "alphafold2_pdb_fragment"
    return full_dir, fragment_dir


def full_structure_path(source_dataset: str, interpro_id: str, uniprot_id: str) -> Path:
    full_dir, _ = structure_dirs(source_dataset)
    return full_dir / f"{interpro_id}_{uniprot_id}.pdb"


def fragment_structure_path(
    source_dataset: str,
    interpro_id: str,
    uniprot_id: str,
    start: int,
    end: int,
    offset: int,
) -> Path:
    _, fragment_dir = structure_dirs(source_dataset)
    return fragment_dir / f"{interpro_id}_{uniprot_id}_{start + offset}-{end + offset}.pdb"


def masked_saprot_sequence(sequence: str) -> str:
    return "".join(f"{aa}#" for aa in sequence)


class FoldseekRunner:
    def __init__(self, foldseek_path: str, missing_structure: str):
        self.foldseek_path = Path(foldseek_path)
        self.missing_structure = missing_structure
        self.cache: Dict[Path, str] = {}

    def sequence_for_structure(self, pdb_path: Path, fallback_sequence: str) -> str:
        pdb_path = pdb_path.resolve()
        if pdb_path in self.cache:
            return self.cache[pdb_path]
        if not pdb_path.exists():
            if self.missing_structure == "mask":
                value = masked_saprot_sequence(fallback_sequence)
                self.cache[pdb_path] = value
                return value
            raise FileNotFoundError(f"Structure file not found: {pdb_path}")
        if not self.foldseek_path.exists():
            raise FileNotFoundError(f"Foldseek executable not found: {self.foldseek_path}")

        value = self._run_foldseek(pdb_path)
        self.cache[pdb_path] = value
        return value

    def _run_foldseek(self, pdb_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="instructbiomol_foldseek_") as tmpdir:
            out_base = Path(tmpdir) / "structure_3di.tsv"
            cmd = [
                str(self.foldseek_path),
                "structureto3didescriptor",
                "-v",
                "0",
                "--threads",
                "1",
                "--chain-name-mode",
                "1",
                str(pdb_path),
                str(out_base),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with out_base.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        seq, struc_seq = parts[1], parts[2]
                        return "".join(a + b.lower() for a, b in zip(seq, struc_seq))
        raise RuntimeError(f"Foldseek produced no 3Di descriptor for {pdb_path}")


def load_reference_samples(
    data_root: Path,
    dataset_name: str,
    split: str,
    task_type: str,
    limit: Optional[int],
) -> List[Dict]:
    source_name, source_dataset, task_name = DATASET_TO_SOURCE[dataset_name]
    json_path = data_root / source_name / f"{split}.json"
    with json_path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    samples: List[Dict] = []
    dataset_idx = 0
    for item in raw_items:
        for fragment_group in item["fragments"]:
            for fragment in fragment_group["frags"]:
                reference = (
                    f"It is the {fragment_group['category']}."
                    if task_type == "class"
                    else fragment_group["description"]
                )
                start = int(fragment["start_position"])
                end = int(fragment["end_position"])
                samples.append(
                    {
                        "uid": item["uid"],
                        "sequence": item["sequence"],
                        "fragment_sequence": fragment["sequence"],
                        "original_start_pos": start,
                        "original_end_pos": end,
                        "category": fragment_group["category"],
                        "description": fragment_group["description"],
                        "interpro_id": fragment_group["interpro_id"],
                        "shortname": fragment_group.get("shortname", ""),
                        "reference": reference,
                        "dataset_idx": dataset_idx,
                        "dataset_name": dataset_name,
                        "source_dataset": source_dataset,
                        "task_name": task_name,
                    }
                )
                dataset_idx += 1
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def build_instruction(task_type: str, sample: Dict, input_mode: str) -> str:
    task_name = sample["task_name"]
    if input_mode == "full":
        fragment = encode_protein_for_instruction(sample["fragment_sequence"])
        context = (
            "You are given a full protein sequence. The target fragment sequence is "
            f"{fragment}. "
        )
        if task_type == "class":
            return (
                f"{context}Identify the specific database-style category name of "
                f"the {task_name} represented by this target fragment. Return only "
                "the category name, with no prefix, explanation, coordinates, or extra sentence. "
                "Do not answer with the whole-protein name or a broad protein family "
                "when the target fragment is a more specific site, motif, or domain."
            )
        return (
            f"{context}Write one concise database-style functional description of "
            f"the {task_name} represented by this target fragment. Focus on the "
            "target fragment and avoid inventing residue numbers, coordinates, or "
            "unsupported mechanisms."
        )

    context = "You are given only the target protein fragment sequence. "
    if task_type == "class":
        return (
            f"{context}Identify the specific database-style category name of this "
            f"{task_name} fragment. Return only the category name, with no prefix, "
            "explanation, coordinates, or extra sentence. Do not answer with the "
            "whole-protein name or a broad protein family when the target fragment "
            "is a more specific site, motif, or domain."
        )
    return (
        f"{context}Write one concise database-style functional description of this "
        f"{task_name} fragment. Focus on the target fragment and avoid inventing "
        "residue numbers, coordinates, or unsupported mechanisms."
    )


def load_instructbiomol_model(args: argparse.Namespace):
    os.chdir(INSTRUCTBIOMOL_ROOT)
    sys.path.insert(0, str(SCRIPT_DIR))

    from model.unimodel import UniModel

    missing_assets = [
        str(INSTRUCTBIOMOL_ROOT / rel_path)
        for rel_path in REQUIRED_PRETRAINED_ASSETS
        if not (INSTRUCTBIOMOL_ROOT / rel_path).exists()
    ]
    if missing_assets:
        missing_text = "\n".join(f"  - {path}" for path in missing_assets)
        raise FileNotFoundError(
            "InstructBioMol auxiliary pretrained assets are missing. "
            "Place the README pretrained_ckpt contents under "
            f"{INSTRUCTBIOMOL_ROOT / 'pretrained_ckpt'}.\n{missing_text}"
        )

    with (INSTRUCTBIOMOL_ROOT / "config" / "base.yaml").open("r", encoding="utf-8") as handle:
        model_args = yaml.load(handle, Loader=yaml.FullLoader)

    model_path = str(Path(args.model_path))
    model_args.update(
        {
            "debug": False,
            "local_rank": 0,
            "datatype": args.datatype,
            "llama_ckpt_path": str(Path(args.llama_ckpt_path or model_path)),
            "load_ckpt_path_list": [model_path],
            "max_length": args.max_length or model_args.get("max_length", 450),
        }
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if args.datatype == "bf16":
        model = UniModel(model_args).bfloat16().cuda()
    elif args.datatype == "half":
        model = UniModel(model_args).half().cuda()
    else:
        model = UniModel(model_args).float().cuda()

    ckpt = torch.load(Path(model_path) / "pytorch_model.bin", map_location=torch.device("cuda"))
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def prepare_model_batch(
    samples: Sequence[Dict],
    task_type: str,
    input_mode: str,
    foldseek: FoldseekRunner,
    fragment_offset: int,
) -> Tuple[Tuple[List[str], List[str]], Dict]:
    input_sequences: List[str] = []
    saprot_sequences: List[str] = []
    instructions: List[str] = []
    references: List[str] = []
    ids: List[str] = []
    structure_paths: List[str] = []

    for sample in samples:
        if input_mode == "full":
            sequence = sample["sequence"]
            structure_path = full_structure_path(
                sample["source_dataset"], sample["interpro_id"], sample["uid"]
            )
        else:
            sequence = sample["fragment_sequence"]
            structure_path = fragment_structure_path(
                sample["source_dataset"],
                sample["interpro_id"],
                sample["uid"],
                sample["original_start_pos"],
                sample["original_end_pos"],
                fragment_offset,
            )

        input_sequences.append(sequence)
        saprot_sequences.append(foldseek.sequence_for_structure(structure_path, sequence))
        instructions.append(build_instruction(task_type, sample, input_mode))
        references.append(sample["reference"])
        ids.append(str(sample["dataset_idx"]))
        structure_paths.append(str(structure_path))

    inputs = {
        "input_seqs": input_sequences,
        "target_seqs": references,
        "input_enc_seqs": input_sequences,
        "input_enc_fps": [[] for _ in input_sequences],
        "input_modality": "protein",
        "target_modality": "text",
        "instructions": instructions,
        "ids": ids,
        "data_name": "reference_fragment",
        "structure_paths": structure_paths,
    }
    return (input_sequences, saprot_sequences), inputs


def batched(items: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generation_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        kwargs["t"] = args.temperature
        kwargs["top_p"] = args.top_p
    else:
        kwargs["num_beams"] = args.num_beams or 1
    return kwargs


def evaluate_one_dataset(
    dataset_name: str,
    task_type: str,
    input_mode: str,
    output_subdir: str,
    model,
    args: argparse.Namespace,
) -> Path:
    data_root = resolve_data_root(args)
    samples = load_reference_samples(data_root, dataset_name, args.split, task_type, args.limit)
    out_dir = make_output_dir(args, output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{dataset_name}_results.csv"

    foldseek = FoldseekRunner(args.foldseek_path, args.missing_structure)
    generated: List[str] = []
    structure_paths: List[str] = []

    print(f"Evaluating {dataset_name}: {len(samples)} samples")
    for batch_samples in tqdm(
        batched(samples, args.batch_per_device),
        total=(len(samples) + args.batch_per_device - 1) // args.batch_per_device,
        desc=f"Evaluating {dataset_name}",
    ):
        input_batch, inputs = prepare_model_batch(
            batch_samples, task_type, input_mode, foldseek, args.fragment_offset
        )
        structure_paths.extend(inputs["structure_paths"])
        with torch.no_grad():
            generated.extend(model.generate(input_batch, inputs, "text", **generation_kwargs(args)))

    rows = []
    for sample, text, structure_path in zip(samples, generated, structure_paths):
        row = {
            "generated": text,
            "reference": sample["reference"],
            "dataset_idx": sample["dataset_idx"],
            "interpro_ids": sample["interpro_id"],
            "fragment_sequence": sample["fragment_sequence"],
            "structure_path": structure_path,
        }
        if input_mode == "full":
            row.update(
                {
                    "original_start_pos": sample["original_start_pos"],
                    "original_end_pos": sample["original_end_pos"],
                    "fragment_start_pos": sample["original_start_pos"],
                    "fragment_end_pos": sample["original_end_pos"] + 1,
                }
            )
        else:
            row.update(
                {
                    "original_start_pos": sample["original_start_pos"],
                    "original_end_pos": sample["original_end_pos"],
                }
            )
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["generated", "reference", "dataset_idx", "interpro_ids"]
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {save_path}")
    return save_path


def run_reference_inference(
    args: argparse.Namespace,
    task_type: str,
    input_mode: str,
    output_subdir: str,
    allowed_datasets: Sequence[str],
) -> List[Path]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dataset_names = parse_dataset_list(args.datasets, allowed_datasets)
    model = load_instructbiomol_model(args)
    saved_paths = []
    for dataset_name in dataset_names:
        saved_paths.append(
            evaluate_one_dataset(
                dataset_name=dataset_name,
                task_type=task_type,
                input_mode=input_mode,
                output_subdir=output_subdir,
                model=model,
                args=args,
            )
        )
    return saved_paths
