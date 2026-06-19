"""Check whether VenusX test-set structures exist."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENUSX_STRUCT_ROOT = Path("/home/dataset-local/projects_dir/VenusX_dataset")

DATA_DIRS = {
    "data_70": PROJECT_ROOT / "data_70",
    "data_30": PROJECT_ROOT / "data_30",
}
DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]


def parse_csv_arg(value, valid_values, arg_name):
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in valid_values]
    if invalid:
        valid_text = ",".join(valid_values)
        invalid_text = ",".join(invalid)
        raise argparse.ArgumentTypeError(
            f"invalid {arg_name}: {invalid_text}; supported values: {valid_text}"
        )
    if not items:
        raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty")
    return items


def load_test_data(data_root, dataset_name):
    test_path = data_root / f"VenusX_{dataset_name}" / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"test file not found: {test_path}")
    with test_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_structure_dirs(dataset_name):
    base_dir = VENUSX_STRUCT_ROOT / f"VenusX_{dataset_name}_AlphaFold2_PDB"
    return base_dir / "alphafold2_pdb", base_dir / "alphafold2_pdb_fragment"


def full_structure_path(full_dir, interpro_id, uniprot_id):
    return full_dir / f"{interpro_id}_{uniprot_id}.pdb"


def fragment_structure_path(fragment_dir, interpro_id, uniprot_id, start, end, offset):
    return fragment_dir / f"{interpro_id}_{uniprot_id}_{start + offset}-{end + offset}.pdb"


def check_dataset(data_name, data_root, dataset_name, fragment_offset):
    test_data = load_test_data(data_root, dataset_name)
    full_dir, fragment_dir = get_structure_dirs(dataset_name)

    if not full_dir.is_dir():
        tqdm.write(f"[WARN] full structure directory not found: {full_dir}")
    if not fragment_dir.is_dir():
        tqdm.write(f"[WARN] fragment structure directory not found: {fragment_dir}")

    stats = {
        "samples": len(test_data),
        "full_checked": 0,
        "full_missing": 0,
        "fragments_checked": 0,
        "fragments_missing": 0,
    }

    desc = f"{data_name}/{dataset_name}"
    for sample in tqdm(test_data, desc=desc, unit="sample"):
        uniprot_id = sample["uid"]
        for fragment_group in sample.get("fragments", []):
            interpro_id = fragment_group["interpro_id"]

            full_path = full_structure_path(full_dir, interpro_id, uniprot_id)
            stats["full_checked"] += 1
            if not full_path.exists():
                stats["full_missing"] += 1
                tqdm.write(
                    "[MISSING_FULL] "
                    f"data={data_name} dataset={dataset_name} "
                    f"interpro_id={interpro_id} uniprot_id={uniprot_id} "
                    f"path={full_path}"
                )

            for frag in fragment_group.get("frags", []):
                start = int(frag["start_position"])
                end = int(frag["end_position"])
                frag_path = fragment_structure_path(
                    fragment_dir, interpro_id, uniprot_id, start, end, fragment_offset
                )
                stats["fragments_checked"] += 1
                if not frag_path.exists():
                    stats["fragments_missing"] += 1
                    tqdm.write(
                        "[MISSING_FRAGMENT] "
                        f"data={data_name} dataset={dataset_name} "
                        f"interpro_id={interpro_id} uniprot_id={uniprot_id} "
                        f"start={start} end={end} "
                        f"expected_span={start + fragment_offset}-{end + fragment_offset} "
                        f"path={frag_path}"
                    )

    return stats


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check full-protein and fragment PDB files for VenusX test sets."
    )
    parser.add_argument(
        "--data",
        default="data_70,data_30",
        help="Comma-separated data roots to check. Supported: data_70,data_30",
    )
    parser.add_argument(
        "--datasets",
        default="Act,BindI,Evo,Motif",
        # default="Dom",
        help="Comma-separated VenusX sub-datasets to check. Supported: Act,BindI,Dom,Evo,Motif",
    )
    parser.add_argument(
        "--fragment-offset",
        type=int,
        default=1,
        help="Offset added to start_position/end_position for fragment PDB filenames.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_names = parse_csv_arg(args.data, list(DATA_DIRS), "--data")
    dataset_names = parse_csv_arg(args.datasets, DATASET_NAMES, "--datasets")

    total = {
        "samples": 0,
        "full_checked": 0,
        "full_missing": 0,
        "fragments_checked": 0,
        "fragments_missing": 0,
    }

    for data_name in data_names:
        data_root = DATA_DIRS[data_name]
        for dataset_name in dataset_names:
            stats = check_dataset(data_name, data_root, dataset_name, args.fragment_offset)
            tqdm.write(
                "[SUMMARY] "
                f"data={data_name} dataset={dataset_name} "
                f"samples={stats['samples']} "
                f"full_missing={stats['full_missing']}/{stats['full_checked']} "
                f"fragment_missing={stats['fragments_missing']}/{stats['fragments_checked']}"
            )
            for key, value in stats.items():
                total[key] += value

    tqdm.write(
        "[TOTAL] "
        f"samples={total['samples']} "
        f"full_missing={total['full_missing']}/{total['full_checked']} "
        f"fragment_missing={total['fragments_missing']}/{total['fragments_checked']}"
    )


if __name__ == "__main__":
    main()
