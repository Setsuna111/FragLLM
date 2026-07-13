import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from instructbiomol_reference_common_v6 import add_common_args, run_reference_inference


CLASS_DATASETS = "ActRefClass,BindIRefClass,DomRefClass,EvoRefClass,MotifRefClass"
# CLASS_DATASETS = "ActRefClass"
# CLASS_DATASETS = "BindIRefClass,DomRefClass,EvoRefClass,MotifRefClass"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run InstructBioMol inference on reference region classification datasets "
            "using full protein structures and text-specified target fragments."
        )
    )
    add_common_args(parser, default_datasets=CLASS_DATASETS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_reference_inference(
        args=args,
        task_type="class",
        input_mode="full",
        output_subdir="referring_cls_fragment_struct_v6",
        allowed_datasets=CLASS_DATASETS.split(","),
    )


if __name__ == "__main__":
    main()
