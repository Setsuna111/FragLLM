import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from instructbiomol_reference_common_v6 import add_common_args, run_reference_inference


DESC_DATASETS = "ActRefDesc,BindIRefDesc,DomRefDesc,EvoRefDesc,MotifRefDesc"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run InstructBioMol inference on reference region description datasets "
            "using only target fragment structures and sequences."
        )
    )
    add_common_args(parser, default_datasets=DESC_DATASETS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_reference_inference(
        args=args,
        task_type="desc",
        input_mode="fragment",
        output_subdir="ref_frag_only_desc_v6",
        allowed_datasets=DESC_DATASETS.split(","),
    )


if __name__ == "__main__":
    main()
