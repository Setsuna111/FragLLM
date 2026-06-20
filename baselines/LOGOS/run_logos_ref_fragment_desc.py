import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from logos_reference_common import add_common_args, run_reference_inference


DESC_DATASETS = "ActRefDesc,BindIRefDesc,DomRefDesc,EvoRefDesc,MotifRefDesc"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run LOGOS inference on reference-region description datasets using "
            "the full protein sequence and a text-specified target fragment."
        )
    )
    add_common_args(parser, default_datasets=DESC_DATASETS)
    parser.set_defaults(max_new_tokens=512)
    return parser.parse_args()


def main():
    args = parse_args()
    run_reference_inference(
        args=args,
        task_type="desc",
        input_mode="full",
        output_subdir="referring_desc_fragment_text",
        allowed_datasets=DESC_DATASETS.split(","),
    )


if __name__ == "__main__":
    main()
