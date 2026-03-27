from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from recipe_mpr_qa.slm.finetune import (
    RunConfig,
    build_arg_parser,
    build_hf_datasets,
    main,
    namespace_to_run_config,
    parse_args,
    run_finetune,
    run_training_from_arg_list,
    seed_everything,
    str2bool,
)

__all__ = [
    "RunConfig",
    "build_arg_parser",
    "build_hf_datasets",
    "main",
    "namespace_to_run_config",
    "parse_args",
    "run_finetune",
    "run_training_from_arg_list",
    "seed_everything",
    "str2bool",
]


if __name__ == "__main__":
    main()
