from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from recipe_mpr_qa.evaluation.mc_eval import (
    _model_display_name,
    build_arg_parser,
    build_result_row,
    main,
    run_evaluation,
    run_evaluation_from_arg_list,
)

__all__ = [
    "_model_display_name",
    "build_arg_parser",
    "build_result_row",
    "main",
    "run_evaluation",
    "run_evaluation_from_arg_list",
]


if __name__ == "__main__":
    main()
