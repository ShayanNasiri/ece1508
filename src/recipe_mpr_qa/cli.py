from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.constants import (
    DEFAULT_PROCESSED_DATASET_PATH,
    DEFAULT_RAW_DATASET_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_SPLIT_SEED,
)
from recipe_mpr_qa.data.loaders import get_split_examples
from recipe_mpr_qa.data.models import DatasetValidationError
from recipe_mpr_qa.data.preparation import (
    generate_primary_split,
    prepare_dataset,
    read_prepared_dataset,
    read_split_manifest,
    write_prepared_dataset,
    write_split_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recipe-mpr-qa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATASET_PATH)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    prepare_parser.add_argument(
        "--split-output", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATASET_PATH)
    validate_parser.add_argument(
        "--kind",
        choices=("raw", "prepared"),
        default="raw",
    )

    stats_parser = subparsers.add_parser("dataset-stats")
    stats_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATASET_PATH)
    stats_parser.add_argument(
        "--kind",
        choices=("raw", "prepared"),
        default="raw",
    )

    export_parser = subparsers.add_parser("export-split")
    export_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    export_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    export_parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    return parser


def _dataset_for_kind(input_path: Path, kind: str):
    if kind == "raw":
        return prepare_dataset(input_path)
    return read_prepared_dataset(input_path)


def _command_prepare_data(args: argparse.Namespace) -> int:
    dataset = prepare_dataset(args.input)
    split_manifest = generate_primary_split(dataset.examples, seed=args.seed)
    write_prepared_dataset(dataset, args.output)
    write_split_manifest(split_manifest, args.split_output)
    print(
        f"Prepared {len(dataset.examples)} examples to {args.output.as_posix()} and "
        f"wrote split manifest to {args.split_output.as_posix()}."
    )
    return 0


def _command_validate_data(args: argparse.Namespace) -> int:
    dataset = _dataset_for_kind(args.input, args.kind)
    print(f"Validated {len(dataset.examples)} {args.kind} examples from {args.input.as_posix()}.")
    return 0


def _command_dataset_stats(args: argparse.Namespace) -> int:
    dataset = _dataset_for_kind(args.input, args.kind)
    print(json.dumps(dataset.metadata, ensure_ascii=True, indent=2))
    return 0


def _command_export_split(args: argparse.Namespace) -> int:
    dataset = read_prepared_dataset(args.dataset)
    split_manifest = read_split_manifest(args.split_manifest)
    split_examples = get_split_examples(dataset, split_manifest, args.split)
    split_rows = [json.dumps(example.to_dict(), ensure_ascii=True, separators=(",", ":")) for example in split_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    print(
        f"Exported {len(split_examples)} {args.split} examples to {args.output.as_posix()}."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            return _command_prepare_data(args)
        if args.command == "validate-data":
            return _command_validate_data(args)
        if args.command == "dataset-stats":
            return _command_dataset_stats(args)
        if args.command == "export-split":
            return _command_export_split(args)
    except (DatasetValidationError, FileNotFoundError, json.JSONDecodeError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
