from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.augmentation import (
    augment_training_examples,
    count_augmentation_strategies,
)
from recipe_mpr_qa.data.constants import (
    DEFAULT_PROCESSED_DATASET_PATH,
    DEFAULT_RAW_DATASET_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_SPLIT_SEED,
)
from recipe_mpr_qa.data.loaders import get_split_examples
from recipe_mpr_qa.data.models import DatasetValidationError, PreparedDataset
from recipe_mpr_qa.data.preparation import (
    generate_primary_split,
    prepare_dataset,
    read_prepared_dataset,
    read_split_manifest,
    write_prepared_dataset,
    write_split_manifest,
)
from recipe_mpr_qa.synthetic import (
    DEFAULT_FULL_GENERATION_MODEL,
    DEFAULT_FULL_REVIEW_MODEL,
    DEFAULT_QUERY_GENERATION_MODEL,
    DEFAULT_QUERY_REVIEW_MODEL,
    OpenAIResponsesClient,
    approve_synthetic_full_candidates,
    approve_synthetic_query_candidates,
    build_synthetic_training_artifact,
    generate_synthetic_full_candidates,
    generate_synthetic_query_candidates,
    review_synthetic_full_candidates,
    review_synthetic_query_candidates,
)
from recipe_mpr_qa.tracking import (
    DEFAULT_MLOPS_ROOT,
    build_run_comparison,
    format_run_comparison_table,
    format_run_table,
    list_registered_runs,
    promote_run,
    run_tracked_eval,
    run_tracked_train,
    write_comparison_report,
)
from recipe_mpr_qa.tracking.models import REGISTRY_STAGES, RUN_STATUSES, RUN_TYPES


def _build_openai_client() -> OpenAIResponsesClient:
    return OpenAIResponsesClient()


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

    augment_parser = subparsers.add_parser("augment-train")
    augment_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    augment_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    augment_parser.add_argument("--output", type=Path, required=True)
    augment_parser.add_argument("--max-variants", type=int, default=2)

    synthetic_query_generate_parser = subparsers.add_parser("generate-synthetic-query")
    synthetic_query_generate_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_query_generate_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    synthetic_query_generate_parser.add_argument("--output", type=Path, required=True)
    synthetic_query_generate_parser.add_argument(
        "--model", type=str, default=DEFAULT_QUERY_GENERATION_MODEL
    )
    synthetic_query_generate_parser.add_argument("--limit", type=int, default=75)
    synthetic_query_generate_parser.add_argument("--max-candidates-per-parent", type=int, default=3)
    synthetic_query_generate_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    synthetic_query_review_parser = subparsers.add_parser("review-synthetic-query")
    synthetic_query_review_parser.add_argument("--input", type=Path, required=True)
    synthetic_query_review_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_query_review_parser.add_argument("--output", type=Path, required=True)
    synthetic_query_review_parser.add_argument("--model", type=str, default=DEFAULT_QUERY_REVIEW_MODEL)

    synthetic_query_approve_parser = subparsers.add_parser("approve-synthetic-query")
    synthetic_query_approve_parser.add_argument("--input", type=Path, required=True)
    synthetic_query_approve_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_query_approve_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    synthetic_query_approve_parser.add_argument("--output", type=Path, required=True)
    synthetic_query_approve_parser.add_argument("--approval-batch-id", required=True)
    synthetic_query_approve_parser.add_argument("--max-examples", type=int, default=None)
    synthetic_query_approve_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    synthetic_full_generate_parser = subparsers.add_parser("generate-synthetic-full")
    synthetic_full_generate_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_full_generate_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    synthetic_full_generate_parser.add_argument("--output", type=Path, required=True)
    synthetic_full_generate_parser.add_argument(
        "--model", type=str, default=DEFAULT_FULL_GENERATION_MODEL
    )
    synthetic_full_generate_parser.add_argument("--limit", type=int, default=40)
    synthetic_full_generate_parser.add_argument("--max-candidates-per-seed", type=int, default=1)
    synthetic_full_generate_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    synthetic_full_review_parser = subparsers.add_parser("review-synthetic-full")
    synthetic_full_review_parser.add_argument("--input", type=Path, required=True)
    synthetic_full_review_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_full_review_parser.add_argument("--output", type=Path, required=True)
    synthetic_full_review_parser.add_argument("--model", type=str, default=DEFAULT_FULL_REVIEW_MODEL)

    synthetic_full_approve_parser = subparsers.add_parser("approve-synthetic-full")
    synthetic_full_approve_parser.add_argument("--input", type=Path, required=True)
    synthetic_full_approve_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_full_approve_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    synthetic_full_approve_parser.add_argument("--output", type=Path, required=True)
    synthetic_full_approve_parser.add_argument("--approval-batch-id", required=True)
    synthetic_full_approve_parser.add_argument("--max-examples", type=int, default=None)
    synthetic_full_approve_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    synthetic_build_parser = subparsers.add_parser("build-synthetic-train")
    synthetic_build_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH
    )
    synthetic_build_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    synthetic_build_parser.add_argument("--query-approved-path", type=Path, default=None)
    synthetic_build_parser.add_argument("--full-approved-path", type=Path, default=None)
    synthetic_build_parser.add_argument("--output", type=Path, required=True)
    synthetic_build_parser.add_argument("--max-query-examples", type=int, default=None)
    synthetic_build_parser.add_argument("--max-full-examples", type=int, default=None)
    synthetic_build_parser.add_argument("--target-ratio", type=float, default=None)
    synthetic_build_parser.add_argument("--full-share", type=float, default=0.0)
    synthetic_build_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)

    run_train_parser = subparsers.add_parser("run-train")
    run_train_parser.add_argument("--stage", choices=REGISTRY_STAGES, default="candidate")
    run_train_parser.add_argument("--mlops-root", type=Path, default=DEFAULT_MLOPS_ROOT)
    run_train_parser.add_argument("--enable-mlflow", action="store_true")
    run_train_parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    run_train_parser.add_argument("--mlflow-experiment", type=str, default="recipe-mpr-qa")

    run_eval_parser = subparsers.add_parser("run-eval")
    run_eval_parser.add_argument("--stage", choices=REGISTRY_STAGES, default="baseline")
    run_eval_parser.add_argument("--parent-run-id", type=str, default=None)
    run_eval_parser.add_argument("--mlops-root", type=Path, default=DEFAULT_MLOPS_ROOT)
    run_eval_parser.add_argument("--enable-mlflow", action="store_true")
    run_eval_parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    run_eval_parser.add_argument("--mlflow-experiment", type=str, default="recipe-mpr-qa")

    list_runs_parser = subparsers.add_parser("list-runs")
    list_runs_parser.add_argument("--mlops-root", type=Path, default=DEFAULT_MLOPS_ROOT)
    list_runs_parser.add_argument("--run-type", choices=RUN_TYPES, default=None)
    list_runs_parser.add_argument("--status", choices=RUN_STATUSES, default=None)
    list_runs_parser.add_argument("--stage", choices=REGISTRY_STAGES, default=None)
    list_runs_parser.add_argument("--format", choices=("table", "json"), default="table")

    compare_runs_parser = subparsers.add_parser("compare-runs")
    compare_runs_parser.add_argument("--mlops-root", type=Path, default=DEFAULT_MLOPS_ROOT)
    compare_runs_parser.add_argument("--run-id", action="append", required=True)
    compare_runs_parser.add_argument("--format", choices=("table", "json"), default="table")
    compare_runs_parser.add_argument("--output", type=Path, default=None)

    promote_run_parser = subparsers.add_parser("promote-run")
    promote_run_parser.add_argument("--mlops-root", type=Path, default=DEFAULT_MLOPS_ROOT)
    promote_run_parser.add_argument("--run-id", required=True)
    promote_run_parser.add_argument("--stage", choices=REGISTRY_STAGES, required=True)

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


def _command_augment_train(args: argparse.Namespace) -> int:
    dataset = read_prepared_dataset(args.dataset)
    split_manifest = read_split_manifest(args.split_manifest)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    augmented_examples = augment_training_examples(
        train_examples,
        max_variants=args.max_variants,
    )
    write_prepared_dataset(
        dataset=PreparedDataset(examples=augmented_examples, metadata={}),
        output_path=args.output,
    )
    print(
        f"Augmented {len(train_examples)} parent train examples into {len(augmented_examples)} "
        f"synthetic examples at {args.output.as_posix()}."
    )
    print(
        json.dumps(
            {
                "parent_examples": len(train_examples),
                "augmented_examples": len(augmented_examples),
                "strategy_counts": count_augmentation_strategies(augmented_examples),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


def _command_generate_synthetic_query(args: argparse.Namespace) -> int:
    summary = generate_synthetic_query_candidates(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        client=_build_openai_client(),
        model=args.model,
        limit=args.limit,
        max_candidates_per_parent=args.max_candidates_per_parent,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_review_synthetic_query(args: argparse.Namespace) -> int:
    summary = review_synthetic_query_candidates(
        input_path=args.input,
        dataset_path=args.dataset,
        output_path=args.output,
        client=_build_openai_client(),
        model=args.model,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_approve_synthetic_query(args: argparse.Namespace) -> int:
    summary = approve_synthetic_query_candidates(
        input_path=args.input,
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        approval_batch_id=args.approval_batch_id,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_generate_synthetic_full(args: argparse.Namespace) -> int:
    summary = generate_synthetic_full_candidates(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        client=_build_openai_client(),
        model=args.model,
        limit=args.limit,
        max_candidates_per_seed=args.max_candidates_per_seed,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_review_synthetic_full(args: argparse.Namespace) -> int:
    summary = review_synthetic_full_candidates(
        input_path=args.input,
        dataset_path=args.dataset,
        output_path=args.output,
        client=_build_openai_client(),
        model=args.model,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_approve_synthetic_full(args: argparse.Namespace) -> int:
    summary = approve_synthetic_full_candidates(
        input_path=args.input,
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        approval_batch_id=args.approval_batch_id,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_build_synthetic_train(args: argparse.Namespace) -> int:
    summary = build_synthetic_training_artifact(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        query_approved_path=args.query_approved_path,
        full_approved_path=args.full_approved_path,
        max_query_examples=args.max_query_examples,
        max_full_examples=args.max_full_examples,
        target_ratio=args.target_ratio,
        full_share=args.full_share,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_run_train(args: argparse.Namespace, script_args: Sequence[str]) -> int:
    manifest = run_tracked_train(
        script_args=script_args,
        stage=args.stage,
        mlops_root=args.mlops_root,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "run_type": manifest.run_type,
                "status": manifest.status,
                "stage": args.stage,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


def _command_run_eval(args: argparse.Namespace, script_args: Sequence[str]) -> int:
    manifest = run_tracked_eval(
        script_args=script_args,
        stage=args.stage,
        parent_run_id=args.parent_run_id,
        mlops_root=args.mlops_root,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "run_type": manifest.run_type,
                "status": manifest.status,
                "stage": args.stage,
                "parent_run_id": manifest.parent_run_id,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


def _command_list_runs(args: argparse.Namespace) -> int:
    entries = list_registered_runs(
        mlops_root=args.mlops_root,
        run_type=args.run_type,
        status=args.status,
        stage=args.stage,
    )
    if args.format == "json":
        print(json.dumps([entry.to_dict() for entry in entries], ensure_ascii=True, indent=2))
    else:
        print(format_run_table(entries) if entries else "No tracked runs found.")
    return 0


def _command_compare_runs(args: argparse.Namespace) -> int:
    comparison_rows = build_run_comparison(args.run_id, mlops_root=args.mlops_root)
    if args.output is not None:
        write_comparison_report(comparison_rows, args.output)
    if args.format == "json":
        print(json.dumps(comparison_rows, ensure_ascii=True, indent=2))
    else:
        print(format_run_comparison_table(comparison_rows))
    return 0


def _command_promote_run(args: argparse.Namespace) -> int:
    entry = promote_run(args.run_id, args.stage, mlops_root=args.mlops_root)
    print(json.dumps(entry.to_dict(), ensure_ascii=True, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    script_args: Sequence[str] = ()
    if args.command in {"run-train", "run-eval"}:
        script_args = tuple(unknown)
    elif unknown:
        parser.error(f"Unrecognized arguments: {' '.join(unknown)}")
    try:
        if args.command == "prepare-data":
            return _command_prepare_data(args)
        if args.command == "validate-data":
            return _command_validate_data(args)
        if args.command == "dataset-stats":
            return _command_dataset_stats(args)
        if args.command == "export-split":
            return _command_export_split(args)
        if args.command == "augment-train":
            return _command_augment_train(args)
        if args.command == "generate-synthetic-query":
            return _command_generate_synthetic_query(args)
        if args.command == "review-synthetic-query":
            return _command_review_synthetic_query(args)
        if args.command == "approve-synthetic-query":
            return _command_approve_synthetic_query(args)
        if args.command == "generate-synthetic-full":
            return _command_generate_synthetic_full(args)
        if args.command == "review-synthetic-full":
            return _command_review_synthetic_full(args)
        if args.command == "approve-synthetic-full":
            return _command_approve_synthetic_full(args)
        if args.command == "build-synthetic-train":
            return _command_build_synthetic_train(args)
        if args.command == "run-train":
            return _command_run_train(args, script_args)
        if args.command == "run-eval":
            return _command_run_eval(args, script_args)
        if args.command == "list-runs":
            return _command_list_runs(args)
        if args.command == "compare-runs":
            return _command_compare_runs(args)
        if args.command == "promote-run":
            return _command_promote_run(args)
    except (
        DatasetValidationError,
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
