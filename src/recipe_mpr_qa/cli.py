from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.artifacts import ensure_run_layout, write_json
from recipe_mpr_qa.benchmark import (
    SlurmJobSpec,
    build_benchmark_contract,
    build_benchmark_table_rows,
    build_run_manifest,
    collect_benchmark_manifests,
    read_benchmark_registry,
    register_benchmark_run,
    render_sbatch_script,
    summarize_registry_statuses,
    write_benchmark_table,
    write_run_manifest,
)
from recipe_mpr_qa.augmentation import (
    build_augmented_examples,
    read_augmented_dataset,
    write_augmented_dataset,
)
from recipe_mpr_qa.config import (
    ConfigError,
    config_to_dict,
    load_augmentation_run_config,
    load_judge_experiment_config,
    load_llm_experiment_config,
    load_slm_experiment_config,
)
from recipe_mpr_qa.data.constants import (
    DEFAULT_PROCESSED_DATASET_PATH,
    DEFAULT_RAW_DATASET_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_SPLIT_SEED,
)
from recipe_mpr_qa.data.loaders import combine_examples, get_split_examples
from recipe_mpr_qa.data.models import DatasetValidationError
from recipe_mpr_qa.data.preparation import (
    generate_primary_split,
    prepare_dataset,
    read_prepared_dataset,
    read_split_manifest,
    write_prepared_dataset,
    write_split_manifest,
)
from recipe_mpr_qa.evaluation import (
    build_run_summary,
    read_judgment_records,
    read_prediction_records,
    summarize_judgment_metrics,
    summarize_prediction_metrics,
    write_run_summary,
)
from recipe_mpr_qa.llm import HFClient, OllamaClient, judge_predictions, run_llm_predictions
from recipe_mpr_qa.llm.prompts import DEFAULT_PARSER_VERSION, OPTION_ORDER_SHUFFLE_SEED
from recipe_mpr_qa.slm import (
    evaluate_causal_slm,
    evaluate_finetuned_model,
    evaluate_finetuned_causal_model,
    evaluate_vanilla_slm,
    train_causal_slm,
    train_finetuned_model,
)
from recipe_mpr_qa.synthetic import (
    approve_synthetic_query_candidates,
    build_train_ready_dataset,
    generate_synthetic_query_candidates,
    review_synthetic_query_candidates,
)
from recipe_mpr_qa.tracking.mlflow import ExperimentContext, MLflowLogger, build_mlflow_tags


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
    validate_parser.add_argument("--kind", choices=("raw", "prepared"), default="raw")

    stats_parser = subparsers.add_parser("dataset-stats")
    stats_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATASET_PATH)
    stats_parser.add_argument("--kind", choices=("raw", "prepared"), default="raw")

    export_parser = subparsers.add_parser("export-split")
    export_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    export_parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    export_parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    augmentation_parser = subparsers.add_parser("generate-augmentation")
    augmentation_parser.add_argument("--config", type=Path, required=True)
    augmentation_parser.add_argument("--output-dir", type=Path)
    augmentation_parser.add_argument("--resume", action="store_true")

    train_parser = subparsers.add_parser("train-slm")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path)

    eval_parser = subparsers.add_parser("evaluate-slm")
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path)
    eval_parser.add_argument("--split", choices=("train", "validation", "test"))

    llm_parser = subparsers.add_parser("run-llm")
    llm_parser.add_argument("--config", type=Path, required=True)
    llm_parser.add_argument("--output-dir", type=Path)
    llm_parser.add_argument("--split", choices=("train", "validation", "test"))
    llm_parser.add_argument("--resume", action="store_true")

    judge_parser = subparsers.add_parser("judge-predictions")
    judge_parser.add_argument("--config", type=Path, required=True)
    judge_parser.add_argument("--predictions", type=Path)
    judge_parser.add_argument("--output-dir", type=Path)
    judge_parser.add_argument("--resume", action="store_true")

    summary_parser = subparsers.add_parser("summarize-run")
    summary_parser.add_argument("--config", type=Path, required=True)
    summary_parser.add_argument("--component", choices=("augmentation", "slm", "llm", "judge"), required=True)
    summary_parser.add_argument("--predictions", type=Path)
    summary_parser.add_argument("--judgments", type=Path)
    summary_parser.add_argument("--output-dir", type=Path)

    synthetic_generate_parser = subparsers.add_parser("generate-synthetic")
    synthetic_generate_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    synthetic_generate_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    synthetic_generate_parser.add_argument("--output", type=Path, required=True)
    synthetic_generate_parser.add_argument("--provider", choices=("ollama", "huggingface"), default="ollama")
    synthetic_generate_parser.add_argument("--model-name", required=True)
    synthetic_generate_parser.add_argument("--parent-limit", type=int, default=75)
    synthetic_generate_parser.add_argument("--variants-per-example", type=int, default=2)
    synthetic_generate_parser.add_argument("--temperature", type=float, default=0.2)

    synthetic_review_parser = subparsers.add_parser("review-synthetic")
    synthetic_review_parser.add_argument("--input", type=Path, required=True)
    synthetic_review_parser.add_argument("--reviewed-output", type=Path, required=True)
    synthetic_review_parser.add_argument("--review-predictions", type=Path, required=True)
    synthetic_review_parser.add_argument("--provider", choices=("ollama", "huggingface"), default="ollama")
    synthetic_review_parser.add_argument("--model-name", required=True)
    synthetic_review_parser.add_argument("--temperature", type=float, default=0.0)
    synthetic_review_parser.add_argument("--resume", action="store_true")

    synthetic_approve_parser = subparsers.add_parser("approve-synthetic")
    synthetic_approve_parser.add_argument("--input", type=Path, required=True)
    synthetic_approve_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    synthetic_approve_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    synthetic_approve_parser.add_argument("--output", type=Path, required=True)
    synthetic_approve_parser.add_argument("--approval-batch-id", required=True)
    synthetic_approve_parser.add_argument("--max-examples", type=int)
    synthetic_approve_parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)

    train_ready_parser = subparsers.add_parser("build-train-ready")
    train_ready_parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    train_ready_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    train_ready_parser.add_argument("--approved-input", type=Path, required=True)
    train_ready_parser.add_argument("--output", type=Path, required=True)
    train_ready_parser.add_argument("--manifest-output", type=Path, required=True)
    train_ready_parser.add_argument("--synthetic-ratio", type=float, required=True)

    benchmark_table_parser = subparsers.add_parser("benchmark-table")
    benchmark_table_parser.add_argument("--root", type=Path, default=Path("artifacts/runs"))
    benchmark_table_parser.add_argument("--output", type=Path, required=True)

    benchmark_status_parser = subparsers.add_parser("benchmark-status")
    benchmark_status_parser.add_argument("--registry", type=Path, default=Path("artifacts/runs/registry.jsonl"))

    slurm_parser = subparsers.add_parser("render-slurm")
    slurm_parser.add_argument("--job-name", required=True)
    slurm_parser.add_argument("--command", dest="slurm_command", nargs="+", required=True)
    slurm_parser.add_argument("--output", type=Path, required=True)
    slurm_parser.add_argument("--partition", default="gpu")
    slurm_parser.add_argument("--time-limit", default="04:00:00")
    slurm_parser.add_argument("--gpus", type=int, default=1)
    slurm_parser.add_argument("--cpus-per-task", type=int, default=4)
    slurm_parser.add_argument("--mem-gb", type=int, default=32)
    slurm_parser.add_argument("--workdir", type=Path)

    return parser


def _dataset_for_kind(input_path: Path, kind: str):
    if kind == "raw":
        return prepare_dataset(input_path)
    return read_prepared_dataset(input_path)


def _resolve_root_dir(default_root: Path, override: Path | None) -> Path:
    return override or default_root


def _write_resolved_config(config: Any, layout, file_name: str) -> Path:
    output_path = layout.configs_dir / file_name
    write_json(config_to_dict(config), output_path)
    return output_path


def _load_split_dataset(dataset_path: Path, split_manifest_path: Path, split: str):
    dataset = read_prepared_dataset(dataset_path)
    split_manifest = read_split_manifest(split_manifest_path)
    examples = get_split_examples(dataset, split_manifest, split)
    return dataset, split_manifest, examples


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(value, prefix=full_key))
        elif isinstance(value, (int, float)):
            flattened[full_key] = float(value)
    return flattened


def _build_inference_client(provider: str, *, max_retries: int = 3):
    if provider == "ollama":
        return OllamaClient(max_retries=max_retries)
    if provider == "huggingface":
        return HFClient()
    raise ValueError(f"Unsupported provider: {provider}")


def _slurm_metadata_from_env() -> dict[str, Any]:
    return {
        key.lower(): value
        for key, value in {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }.items()
        if value
    }


def _write_benchmark_manifest(
    *,
    run_layout,
    run_id: str,
    component: str,
    config_payload: Mapping[str, Any],
    dataset_path: Path,
    split_manifest_path: Path,
    split: str | None,
    prompt_version: str,
    parser_version: str | None,
    option_shuffle_strategy: str,
    option_shuffle_seed: int | None,
    model: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    notes: tuple[str, ...] = (),
) -> None:
    contract = build_benchmark_contract(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        prompt_version=prompt_version,
        parser_version=parser_version,
        option_shuffle_seed=option_shuffle_seed,
        option_shuffle_strategy=option_shuffle_strategy,
        split_name=split,
    )
    manifest = build_run_manifest(
        run_id=run_id,
        component=component,
        status="completed",
        contract=contract,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        config_payload=config_payload,
        model=model,
        artifact_paths={**artifact_paths, "summary": run_layout.summary_path().as_posix()},
        metrics=summary.get("prediction_metrics", {}),
        slurm=_slurm_metadata_from_env(),
        notes=notes,
    )
    write_run_manifest(manifest, run_layout.benchmark_manifest_path())
    register_benchmark_run(manifest, registry_path=run_layout.benchmark_registry_path())


def _maybe_log_mlflow(*, tracking_config, context: ExperimentContext, summary: Mapping[str, Any]) -> None:
    if not tracking_config.enabled:
        return
    logger = MLflowLogger(tracking_uri=tracking_config.tracking_uri)
    logger.log_run(
        context=context,
        metrics=_flatten_metrics(
            {
                "prediction_metrics": summary.get("prediction_metrics", {}),
                "judgment_metrics": summary.get("judgment_metrics", {}),
            }
        ),
        artifact_paths={key: value for key, value in summary["artifact_paths"].items()},
    )


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
    split_rows = [
        json.dumps(example.to_dict(), ensure_ascii=True, separators=(",", ":"))
        for example in split_examples
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    print(f"Exported {len(split_examples)} {args.split} examples to {args.output.as_posix()}.")
    return 0


def _command_generate_augmentation(args: argparse.Namespace) -> int:
    config = load_augmentation_run_config(args.config)
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    config_path = _write_resolved_config(config, run_layout, "augmentation_config.json")
    dataset, _split_manifest, split_examples = _load_split_dataset(
        config.data.dataset_path,
        config.data.split_manifest_path,
        config.data.split,
    )
    existing_examples = ()
    output_path = run_layout.augmentation_dir / "augmented_examples.jsonl"
    resume = args.resume or config.augmentation.resume
    if resume and output_path.exists():
        existing_examples = read_augmented_dataset(output_path)
    client = OllamaClient(max_retries=config.augmentation.max_retries)
    synthetic_examples = build_augmented_examples(
        examples=split_examples,
        client=client,
        teacher_model_name=config.augmentation.teacher_model_name,
        variants_per_example=config.augmentation.variants_per_example,
        existing_examples=existing_examples,
        temperature=config.augmentation.temperature,
    )
    write_augmented_dataset(synthetic_examples, output_path)
    metrics = {
        "source_example_count": len(split_examples),
        "synthetic_example_count": len(synthetic_examples),
    }
    metrics_path = run_layout.augmentation_dir / "metrics.json"
    write_json(metrics, metrics_path)
    summary = build_run_summary(
        run_id=config.output.run_id,
        component="augmentation",
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths={
            "resolved_config": config_path.as_posix(),
            "augmented_examples": output_path.as_posix(),
            "metrics": metrics_path.as_posix(),
        },
        extra_metadata=metrics,
    )
    write_run_summary(summary, run_layout.summary_path())
    _maybe_log_mlflow(
        tracking_config=config.tracking,
        context=ExperimentContext(
            experiment_name=config.tracking.experiment_name,
            run_name=config.output.run_id,
            tags=build_mlflow_tags(
                phase="phase2",
                provider="ollama",
                model_name=config.augmentation.teacher_model_name,
                split=config.data.split,
                prompt_version=config.augmentation.prompt_version,
            ),
            params={
                "variants_per_example": config.augmentation.variants_per_example,
            },
        ),
        summary=summary,
    )
    print(f"Wrote {len(synthetic_examples)} augmented examples to {output_path.as_posix()}.")
    return 0


def _command_train_slm(args: argparse.Namespace) -> int:
    config = load_slm_experiment_config(args.config)
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    config_path = _write_resolved_config(config, run_layout, "slm_config.json")
    dataset = read_prepared_dataset(config.data.dataset_path)
    split_manifest = read_split_manifest(config.data.split_manifest_path)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    validation_examples = get_split_examples(dataset, split_manifest, "validation")
    test_examples = get_split_examples(dataset, split_manifest, "test")
    use_augmentation = False
    if config.mode == "finetune" and config.finetune is not None:
        use_augmentation = config.finetune.use_augmentation
    elif config.mode == "causal_finetune" and config.causal_finetune is not None:
        use_augmentation = config.causal_finetune.use_augmentation
    else:
        raise ConfigError(
            "train-slm requires a config with slm.mode='finetune' or slm.mode='causal_finetune'"
        )
    if use_augmentation:
        if config.data.augmentation_dataset_path is None:
            raise ConfigError("augmentation_dataset_path is required when use_augmentation=true")
        augmented_examples = read_augmented_dataset(config.data.augmentation_dataset_path)
        train_examples = combine_examples(train_examples, augmented_examples)
    if config.mode == "finetune" and config.finetune is not None:
        result = train_finetuned_model(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            run_id=config.output.run_id,
            output_dir=run_layout.checkpoints_dir,
            model_name=config.finetune.model_name,
            max_length=config.finetune.max_length,
            learning_rate=config.finetune.learning_rate,
            train_batch_size=config.finetune.train_batch_size,
            eval_batch_size=config.finetune.eval_batch_size,
            num_train_epochs=config.finetune.num_train_epochs,
        )
        model_name = config.finetune.model_name
        prompt_version = "distilbert-cross-encoder-v1"
        decoding_mode = "pairwise_logits"
        model_interface = "classifier"
        params = {
            "learning_rate": config.finetune.learning_rate,
            "num_train_epochs": config.finetune.num_train_epochs,
            "use_augmentation": config.finetune.use_augmentation,
            "mode": config.mode,
        }
    elif config.mode == "causal_finetune" and config.causal_finetune is not None:
        result = train_causal_slm(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            run_id=config.output.run_id,
            output_dir=run_layout.checkpoints_dir,
            model_name=config.causal_finetune.model_name,
            prompt_version=config.causal_finetune.prompt_version,
            max_length=config.causal_finetune.max_length,
            max_new_tokens=config.causal_finetune.max_new_tokens,
            temperature=config.causal_finetune.temperature,
            top_p=config.causal_finetune.top_p,
            learning_rate=config.causal_finetune.learning_rate,
            train_batch_size=config.causal_finetune.train_batch_size,
            eval_batch_size=config.causal_finetune.eval_batch_size,
            num_train_epochs=config.causal_finetune.num_train_epochs,
            use_lora=config.causal_finetune.use_lora,
            lora_r=config.causal_finetune.lora_r,
            lora_alpha=config.causal_finetune.lora_alpha,
            lora_dropout=config.causal_finetune.lora_dropout,
            decoding_mode=config.causal_finetune.decoding_mode,
        )
        model_name = config.causal_finetune.model_name
        prompt_version = config.causal_finetune.prompt_version
        decoding_mode = config.causal_finetune.decoding_mode
        model_interface = "generative"
        params = {
            "learning_rate": config.causal_finetune.learning_rate,
            "num_train_epochs": config.causal_finetune.num_train_epochs,
            "use_augmentation": config.causal_finetune.use_augmentation,
            "use_lora": config.causal_finetune.use_lora,
            "mode": config.mode,
        }
    else:
        raise ConfigError("Invalid SLM config")
    validation_metrics = summarize_prediction_metrics(result["validation_records"], dataset)
    test_metrics = summarize_prediction_metrics(result["test_records"], dataset)
    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "train_example_count": len(train_examples),
    }
    metrics_path = run_layout.component_metrics_path("slm")
    write_json(metrics, metrics_path)
    summary = build_run_summary(
        run_id=config.output.run_id,
        component="slm",
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths={
            "resolved_config": config_path.as_posix(),
            "validation_predictions": (run_layout.slm_dir / "validation_predictions.jsonl").as_posix(),
            "test_predictions": (run_layout.slm_dir / "test_predictions.jsonl").as_posix(),
            "checkpoints": run_layout.checkpoints_dir.as_posix(),
            "best_checkpoint": result.get("best_checkpoint_dir", run_layout.checkpoints_dir).as_posix(),
            "checkpoint_manifest": (run_layout.slm_dir / "checkpoint_manifest.json").as_posix(),
            "metrics": metrics_path.as_posix(),
        },
        prediction_records=result["test_records"],
        extra_metadata={"validation_metrics": validation_metrics},
    )
    write_run_summary(summary, run_layout.summary_path())
    _write_benchmark_manifest(
        run_layout=run_layout,
        run_id=config.output.run_id,
        component="slm",
        config_payload=config_to_dict(config),
        dataset_path=config.data.dataset_path,
        split_manifest_path=config.data.split_manifest_path,
        split="test",
        prompt_version=prompt_version,
        parser_version=DEFAULT_PARSER_VERSION if model_interface == "generative" else None,
        option_shuffle_strategy=(
            "deterministic_per_example" if model_interface == "generative" else "not_applicable"
        ),
        option_shuffle_seed=OPTION_ORDER_SHUFFLE_SEED if model_interface == "generative" else None,
        model={
            "name": model_name,
            "provider": "huggingface",
            "interface": model_interface,
            "decoding_mode": decoding_mode,
            "train_data": "original+synthetic" if use_augmentation else "original_only",
            "synthetic_policy": "query_only_or_legacy_augmentation" if use_augmentation else "none",
        },
        summary=summary,
        artifact_paths=summary["artifact_paths"],
    )
    _maybe_log_mlflow(
        tracking_config=config.tracking,
        context=ExperimentContext(
            experiment_name=config.tracking.experiment_name,
            run_name=config.output.run_id,
            tags=build_mlflow_tags(
                phase="phase2",
                provider="huggingface",
                model_name=model_name,
                split="test",
                prompt_version=prompt_version,
            ),
            params=params,
        ),
        summary=summary,
    )
    print(f"Finished fine-tuning run in {run_layout.slm_dir.as_posix()}.")
    return 0


def _command_evaluate_slm(args: argparse.Namespace) -> int:
    config = load_slm_experiment_config(args.config)
    split = args.split or config.data.split
    dataset, _split_manifest, split_examples = _load_split_dataset(
        config.data.dataset_path,
        config.data.split_manifest_path,
        split,
    )
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    config_path = _write_resolved_config(config, run_layout, "slm_config.json")
    output_path = run_layout.component_predictions_path("slm", split)
    if config.mode == "vanilla" and config.vanilla is not None:
        prediction_records = evaluate_vanilla_slm(
            examples=split_examples,
            run_id=config.output.run_id,
            split=split,
            output_path=output_path,
            model_name=config.vanilla.model_name,
            prompt_version=config.vanilla.prompt_version,
            batch_size=config.vanilla.batch_size,
            max_length=config.vanilla.max_length,
        )
        model_name = config.vanilla.model_name
        prompt_version = config.vanilla.prompt_version
        decoding_mode = "embedding_similarity"
        model_interface = "classifier"
    elif config.mode == "finetune" and config.finetune is not None:
        checkpoint_dir = config.finetune.checkpoint_dir or run_layout.checkpoints_dir
        prediction_records = evaluate_finetuned_model(
            examples=split_examples,
            run_id=config.output.run_id,
            split=split,
            checkpoint_dir=checkpoint_dir,
            output_path=output_path,
            model_name=config.finetune.model_name,
            max_length=config.finetune.max_length,
        )
        model_name = config.finetune.model_name
        prompt_version = "distilbert-cross-encoder-v1"
        decoding_mode = "pairwise_logits"
        model_interface = "classifier"
    elif config.mode == "causal_baseline" and config.causal_baseline is not None:
        prediction_records = evaluate_causal_slm(
            examples=split_examples,
            run_id=config.output.run_id,
            split=split,
            output_path=output_path,
            model_name=config.causal_baseline.model_name,
            prompt_version=config.causal_baseline.prompt_version,
            max_length=config.causal_baseline.max_length,
            max_new_tokens=config.causal_baseline.max_new_tokens,
            temperature=config.causal_baseline.temperature,
            top_p=config.causal_baseline.top_p,
            decoding_mode=config.causal_baseline.decoding_mode,
        )
        model_name = config.causal_baseline.model_name
        prompt_version = config.causal_baseline.prompt_version
        decoding_mode = config.causal_baseline.decoding_mode
        model_interface = "generative"
    elif config.mode == "causal_finetune" and config.causal_finetune is not None:
        checkpoint_dir = config.causal_finetune.checkpoint_dir or run_layout.checkpoints_dir
        prediction_records = evaluate_finetuned_causal_model(
            examples=split_examples,
            run_id=config.output.run_id,
            split=split,
            checkpoint_dir=checkpoint_dir,
            output_path=output_path,
            model_name=config.causal_finetune.model_name,
            prompt_version=config.causal_finetune.prompt_version,
            max_length=config.causal_finetune.max_length,
            max_new_tokens=config.causal_finetune.max_new_tokens,
            temperature=config.causal_finetune.temperature,
            top_p=config.causal_finetune.top_p,
            decoding_mode=config.causal_finetune.decoding_mode,
        )
        model_name = config.causal_finetune.model_name
        prompt_version = config.causal_finetune.prompt_version
        decoding_mode = config.causal_finetune.decoding_mode
        model_interface = "generative"
    else:
        raise ConfigError("Invalid SLM config")
    metrics = summarize_prediction_metrics(prediction_records, dataset)
    metrics_path = run_layout.slm_dir / f"{split}_metrics.json"
    write_json(metrics, metrics_path)
    summary = build_run_summary(
        run_id=config.output.run_id,
        component="slm",
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths={
            "resolved_config": config_path.as_posix(),
            "predictions": output_path.as_posix(),
            "metrics": metrics_path.as_posix(),
        },
        prediction_records=prediction_records,
    )
    write_run_summary(summary, run_layout.summary_path())
    _write_benchmark_manifest(
        run_layout=run_layout,
        run_id=config.output.run_id,
        component="slm",
        config_payload=config_to_dict(config),
        dataset_path=config.data.dataset_path,
        split_manifest_path=config.data.split_manifest_path,
        split=split,
        prompt_version=prompt_version,
        parser_version=DEFAULT_PARSER_VERSION if model_interface == "generative" else None,
        option_shuffle_strategy=(
            "deterministic_per_example" if model_interface == "generative" else "not_applicable"
        ),
        option_shuffle_seed=OPTION_ORDER_SHUFFLE_SEED if model_interface == "generative" else None,
        model={
            "name": model_name,
            "provider": "huggingface",
            "interface": model_interface,
            "decoding_mode": decoding_mode,
            "train_data": "n/a",
            "synthetic_policy": "n/a",
        },
        summary=summary,
        artifact_paths=summary["artifact_paths"],
    )
    _maybe_log_mlflow(
        tracking_config=config.tracking,
        context=ExperimentContext(
            experiment_name=config.tracking.experiment_name,
            run_name=config.output.run_id,
            tags=build_mlflow_tags(
                phase="phase2",
                provider="huggingface",
                model_name=model_name,
                split=split,
                prompt_version=prompt_version,
            ),
            params={"mode": config.mode},
        ),
        summary=summary,
    )
    print(f"Wrote {len(prediction_records)} {split} SLM predictions to {output_path.as_posix()}.")
    return 0


def _command_run_llm(args: argparse.Namespace) -> int:
    config = load_llm_experiment_config(args.config)
    split = args.split or config.data.split
    dataset, _split_manifest, split_examples = _load_split_dataset(
        config.data.dataset_path,
        config.data.split_manifest_path,
        split,
    )
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    config_path = _write_resolved_config(config, run_layout, "llm_config.json")
    output_path = run_layout.component_predictions_path("llm", split)
    client = _build_inference_client(config.llm.provider, max_retries=config.llm.max_retries)
    prediction_records = run_llm_predictions(
        examples=split_examples,
        client=client,
        run_id=config.output.run_id,
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        split=split,
        output_path=output_path,
        prompt_version=config.llm.prompt_version,
        temperature=config.llm.temperature,
        resume=args.resume or config.llm.resume,
    )
    metrics = summarize_prediction_metrics(prediction_records, dataset)
    metrics_path = run_layout.llm_dir / f"{split}_metrics.json"
    write_json(metrics, metrics_path)
    summary = build_run_summary(
        run_id=config.output.run_id,
        component="llm",
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths={
            "resolved_config": config_path.as_posix(),
            "predictions": output_path.as_posix(),
            "metrics": metrics_path.as_posix(),
        },
        prediction_records=prediction_records,
    )
    write_run_summary(summary, run_layout.summary_path())
    _write_benchmark_manifest(
        run_layout=run_layout,
        run_id=config.output.run_id,
        component="llm",
        config_payload=config_to_dict(config),
        dataset_path=config.data.dataset_path,
        split_manifest_path=config.data.split_manifest_path,
        split=split,
        prompt_version=config.llm.prompt_version,
        parser_version=DEFAULT_PARSER_VERSION,
        option_shuffle_strategy="deterministic_per_example",
        option_shuffle_seed=OPTION_ORDER_SHUFFLE_SEED,
        model={
            "name": config.llm.model_name,
            "provider": config.llm.provider,
            "interface": "generative",
            "decoding_mode": "generate",
            "train_data": "n/a",
            "synthetic_policy": "n/a",
        },
        summary=summary,
        artifact_paths=summary["artifact_paths"],
    )
    _maybe_log_mlflow(
        tracking_config=config.tracking,
        context=ExperimentContext(
            experiment_name=config.tracking.experiment_name,
            run_name=config.output.run_id,
            tags=build_mlflow_tags(
                phase="phase3",
                provider=config.llm.provider,
                model_name=config.llm.model_name,
                split=split,
                prompt_version=config.llm.prompt_version,
            ),
            params={"temperature": config.llm.temperature},
        ),
        summary=summary,
    )
    print(f"Wrote {len(prediction_records)} {split} LLM predictions to {output_path.as_posix()}.")
    return 0


def _command_judge_predictions(args: argparse.Namespace) -> int:
    config = load_judge_experiment_config(args.config)
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    config_path = _write_resolved_config(config, run_layout, "judge_config.json")
    dataset = read_prepared_dataset(config.data.dataset_path)
    predictions_path = args.predictions or config.predictions_path
    if predictions_path is None:
        predictions_path = run_layout.component_predictions_path("llm", config.data.split)
    prediction_records = read_prediction_records(predictions_path)
    output_path = run_layout.judge_dir / "judgments.jsonl"
    client = _build_inference_client(config.judge.provider, max_retries=config.judge.max_retries)
    judgment_records = judge_predictions(
        dataset=dataset,
        prediction_records=prediction_records,
        client=client,
        run_id=config.output.run_id,
        model_name=config.judge.model_name,
        output_path=output_path,
        provider=config.judge.provider,
        temperature=config.judge.temperature,
        resume=args.resume or config.judge.resume,
    )
    metrics = summarize_judgment_metrics(judgment_records)
    metrics_path = run_layout.judge_dir / "metrics.json"
    write_json(metrics, metrics_path)
    summary = build_run_summary(
        run_id=config.output.run_id,
        component="judge",
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths={
            "resolved_config": config_path.as_posix(),
            "predictions": Path(predictions_path).as_posix(),
            "judgments": output_path.as_posix(),
            "metrics": metrics_path.as_posix(),
        },
        prediction_records=prediction_records,
        judgment_records=judgment_records,
    )
    write_run_summary(summary, run_layout.summary_path())
    _write_benchmark_manifest(
        run_layout=run_layout,
        run_id=config.output.run_id,
        component="judge",
        config_payload=config_to_dict(config),
        dataset_path=config.data.dataset_path,
        split_manifest_path=config.data.split_manifest_path,
        split=config.data.split,
        prompt_version=config.judge.prompt_version,
        parser_version=None,
        option_shuffle_strategy="not_applicable",
        option_shuffle_seed=None,
        model={
            "name": config.judge.model_name,
            "provider": config.judge.provider,
            "interface": "judge",
            "decoding_mode": "json",
            "train_data": "n/a",
            "synthetic_policy": "n/a",
        },
        summary=summary,
        artifact_paths=summary["artifact_paths"],
    )
    _maybe_log_mlflow(
        tracking_config=config.tracking,
        context=ExperimentContext(
            experiment_name=config.tracking.experiment_name,
            run_name=config.output.run_id,
            tags=build_mlflow_tags(
                phase="phase4",
                provider=config.judge.provider,
                model_name=config.judge.model_name,
                split=config.data.split,
                prompt_version=config.judge.prompt_version,
            ),
            params={"temperature": config.judge.temperature},
        ),
        summary=summary,
    )
    print(f"Wrote {len(judgment_records)} judgments to {output_path.as_posix()}.")
    return 0


def _command_summarize_run(args: argparse.Namespace) -> int:
    if args.component == "augmentation":
        config = load_augmentation_run_config(args.config)
    elif args.component == "slm":
        config = load_slm_experiment_config(args.config)
    elif args.component == "llm":
        config = load_llm_experiment_config(args.config)
    else:
        config = load_judge_experiment_config(args.config)
    run_layout = ensure_run_layout(
        config.output.run_id,
        _resolve_root_dir(config.output.artifacts_root, args.output_dir),
    )
    dataset = read_prepared_dataset(config.data.dataset_path)
    prediction_records = ()
    judgment_records = ()
    artifact_paths = {"run_dir": run_layout.run_dir.as_posix()}
    if args.predictions is not None:
        prediction_records = read_prediction_records(args.predictions)
        artifact_paths["predictions"] = args.predictions.as_posix()
    if args.judgments is not None:
        judgment_records = read_judgment_records(args.judgments)
        artifact_paths["judgments"] = args.judgments.as_posix()
    summary = build_run_summary(
        run_id=config.output.run_id,
        component=args.component,
        dataset=dataset,
        config=config_to_dict(config),
        artifact_paths=artifact_paths,
        prediction_records=prediction_records,
        judgment_records=judgment_records,
    )
    write_run_summary(summary, run_layout.summary_path())
    print(f"Wrote run summary to {run_layout.summary_path().as_posix()}.")
    return 0


def _command_generate_synthetic(args: argparse.Namespace) -> int:
    client = _build_inference_client(args.provider)
    summary = generate_synthetic_query_candidates(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        client=client,
        model_name=args.model_name,
        parent_limit=args.parent_limit,
        variants_per_example=args.variants_per_example,
        temperature=args.temperature,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_review_synthetic(args: argparse.Namespace) -> int:
    client = _build_inference_client(args.provider)
    summary = review_synthetic_query_candidates(
        input_path=args.input,
        reviewed_output_path=args.reviewed_output,
        review_predictions_path=args.review_predictions,
        reviewer_client=client,
        reviewer_provider=args.provider,
        reviewer_model_name=args.model_name,
        temperature=args.temperature,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_approve_synthetic(args: argparse.Namespace) -> int:
    summary = approve_synthetic_query_candidates(
        input_path=args.input,
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        approval_batch_id=args.approval_batch_id,
        max_examples=args.max_examples,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_build_train_ready(args: argparse.Namespace) -> int:
    summary = build_train_ready_dataset(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        approved_input_path=args.approved_input,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        synthetic_ratio=args.synthetic_ratio,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _command_benchmark_table(args: argparse.Namespace) -> int:
    manifests = collect_benchmark_manifests(args.root)
    rows = build_benchmark_table_rows(manifests)
    write_benchmark_table(rows, output_path=args.output)
    print(f"Wrote {len(rows)} benchmark rows to {args.output.as_posix()}.")
    return 0


def _command_benchmark_status(args: argparse.Namespace) -> int:
    manifests = read_benchmark_registry(args.registry)
    print(
        json.dumps(
            {
                "registry_path": args.registry.as_posix(),
                "run_count": len(manifests),
                "status_counts": summarize_registry_statuses(manifests),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


def _command_render_slurm(args: argparse.Namespace) -> int:
    spec = SlurmJobSpec(
        job_name=args.job_name,
        command=tuple(args.slurm_command),
        partition=args.partition,
        time_limit=args.time_limit,
        gpus=args.gpus,
        cpus_per_task=args.cpus_per_task,
        mem_gb=args.mem_gb,
        workdir=args.workdir.as_posix() if args.workdir is not None else None,
        output_path=(args.output.parent / f"{args.job_name}.out").as_posix(),
        error_path=(args.output.parent / f"{args.job_name}.err").as_posix(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_sbatch_script(spec), encoding="utf-8")
    print(f"Wrote Slurm script to {args.output.as_posix()}.")
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
        if args.command == "generate-augmentation":
            return _command_generate_augmentation(args)
        if args.command == "train-slm":
            return _command_train_slm(args)
        if args.command == "evaluate-slm":
            return _command_evaluate_slm(args)
        if args.command == "run-llm":
            return _command_run_llm(args)
        if args.command == "judge-predictions":
            return _command_judge_predictions(args)
        if args.command == "summarize-run":
            return _command_summarize_run(args)
        if args.command == "generate-synthetic":
            return _command_generate_synthetic(args)
        if args.command == "review-synthetic":
            return _command_review_synthetic(args)
        if args.command == "approve-synthetic":
            return _command_approve_synthetic(args)
        if args.command == "build-train-ready":
            return _command_build_train_ready(args)
        if args.command == "benchmark-table":
            return _command_benchmark_table(args)
        if args.command == "benchmark-status":
            return _command_benchmark_status(args)
        if args.command == "render-slurm":
            return _command_render_slurm(args)
    except (
        ConfigError,
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
