"""Tracked wrappers around the direct training and evaluation entrypoints.

These helpers do not implement separate modeling logic. They run the existing
train/eval scripts, capture their inputs and outputs as artifact references,
and persist run manifests plus optional MLflow mirrors.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.tracking.artifacts import (
    build_artifact_ref,
    build_environment_summary,
    collect_git_metadata,
    generate_run_id,
    utc_now_iso,
)
from recipe_mpr_qa.tracking.mlflow import mirror_run_to_mlflow
from recipe_mpr_qa.tracking.models import RUN_STATUSES, SCHEMA_VERSION, RunManifest
from recipe_mpr_qa.tracking.registry import DEFAULT_MLOPS_ROOT, read_run_manifest, register_run

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_tracked_train(
    *,
    script_args: Sequence[str],
    stage: str,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "recipe-mpr-qa",
) -> RunManifest:
    """Run fine-tuning and persist a tracked manifest for the resulting run."""
    from recipe_mpr_qa.slm.finetune import build_arg_parser, namespace_to_run_config, run_training_from_arg_list

    run_id = generate_run_id("train")
    downstream_args = _normalize_script_args(script_args)
    if not _arg_present(downstream_args, "--output-dir"):
        downstream_args.extend(["--output-dir", f"outputs/tracked/{run_id}"])

    parsed_args = build_arg_parser().parse_args(downstream_args)
    config = namespace_to_run_config(parsed_args)
    manifest = _build_train_manifest(
        run_id=run_id,
        downstream_args=downstream_args,
        config=config,
        status="running",
        finished_at=None,
        output_artifacts=(),
        metrics={},
        metadata={
            "stage": stage,
            "tracking_backend": "filesystem",
            "mlflow_enabled": enable_mlflow,
        },
    )
    register_run(manifest, stage=stage, mlops_root=mlops_root)

    try:
        result = run_training_from_arg_list(downstream_args)
    except Exception as exc:
        failed_manifest = replace(
            manifest,
            status="failed",
            finished_at=utc_now_iso(),
            output_artifacts=_train_output_artifacts(
                {
                    "output_dir": config.output_dir,
                    "final_dir": Path(config.output_dir) / "final",
                    "run_config_path": Path(config.output_dir) / "run_config.json",
                    "log_history_path": Path(config.output_dir) / "log_history.json",
                    "trainer_state_path": Path(config.output_dir) / "trainer_state_summary.json",
                }
            ),
            metadata={
                **dict(manifest.metadata),
                "error": str(exc),
            },
        )
        register_run(failed_manifest, stage=stage, mlops_root=mlops_root)
        raise

    completed_manifest = replace(
        manifest,
        status="completed",
        finished_at=utc_now_iso(),
        output_artifacts=_train_output_artifacts(result),
        metrics=_train_metrics(result),
        metadata={
            **dict(manifest.metadata),
            "dataset_sizes": result.get("dataset_sizes", {}),
        },
    )
    register_run(completed_manifest, stage=stage, mlops_root=mlops_root)
    _maybe_mirror_to_mlflow(
        completed_manifest,
        enable_mlflow=enable_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return completed_manifest


def run_tracked_eval(
    *,
    script_args: Sequence[str],
    stage: str,
    parent_run_id: str | None = None,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "recipe-mpr-qa",
) -> RunManifest:
    """Run evaluation and persist a tracked manifest for the resulting run."""
    from recipe_mpr_qa.evaluation.mc_eval import build_arg_parser, run_evaluation_from_arg_list

    run_id = generate_run_id("eval")
    downstream_args = _normalize_script_args(script_args)

    if parent_run_id is not None and not _arg_present(downstream_args, "--model"):
        downstream_args.extend(["--model", _resolve_parent_model_path(parent_run_id, mlops_root=mlops_root)])

    if not _arg_present(downstream_args, "--output"):
        downstream_args.extend(["--output", f"llm_evaluation/results/tracked/{run_id}.json"])

    parsed_args = build_arg_parser().parse_args(downstream_args)
    manifest = _build_eval_manifest(
        run_id=run_id,
        downstream_args=downstream_args,
        args=parsed_args,
        parent_run_id=parent_run_id,
        status="running",
        finished_at=None,
        output_artifacts=(),
        metrics={},
        metadata={
            "stage": stage,
            "tracking_backend": "filesystem",
            "mlflow_enabled": enable_mlflow,
        },
    )
    register_run(manifest, stage=stage, mlops_root=mlops_root)

    try:
        result = run_evaluation_from_arg_list(downstream_args)
    except Exception as exc:
        failed_manifest = replace(
            manifest,
            status="failed",
            finished_at=utc_now_iso(),
            output_artifacts=_eval_output_artifacts(
                {"output_path": parsed_args.output}
            ),
            metadata={
                **dict(manifest.metadata),
                "error": str(exc),
            },
        )
        register_run(failed_manifest, stage=stage, mlops_root=mlops_root)
        raise

    completed_manifest = replace(
        manifest,
        status="completed",
        finished_at=utc_now_iso(),
        output_artifacts=_eval_output_artifacts(result),
        metrics=dict(result.get("metrics", {})),
        prompt=dict(result.get("prompt", {})),
        metadata={
            **dict(manifest.metadata),
            "temperature": result.get("temperature"),
            "example_count": result.get("example_count"),
        },
    )
    register_run(completed_manifest, stage=stage, mlops_root=mlops_root)
    _maybe_mirror_to_mlflow(
        completed_manifest,
        enable_mlflow=enable_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return completed_manifest


def _build_train_manifest(
    *,
    run_id: str,
    downstream_args: Sequence[str],
    config,
    status: str,
    finished_at: str | None,
    output_artifacts,
    metrics,
    metadata,
) -> RunManifest:
    git_metadata = collect_git_metadata(REPO_ROOT)
    input_artifacts = [
        build_artifact_ref(name="dataset", path=config.data_path, repo_root=REPO_ROOT),
        build_artifact_ref(
            name="split_manifest",
            path=config.split_manifest_path,
            repo_root=REPO_ROOT,
        ),
    ]
    if config.augmented_train_path:
        input_artifacts.append(
            build_artifact_ref(
                name="augmented_train",
                path=config.augmented_train_path,
                repo_root=REPO_ROOT,
            )
        )
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_type="train",
        status=status,
        created_at=utc_now_iso(),
        finished_at=finished_at,
        entrypoint="recipe_mpr_qa.cli:run-train",
        command=tuple(["recipe-mpr-qa", "run-train", *downstream_args]),
        git_commit=git_metadata["git_commit"],
        git_dirty=git_metadata["git_dirty"],
        environment=build_environment_summary(),
        input_artifacts=tuple(input_artifacts),
        output_artifacts=tuple(output_artifacts),
        model={
            "name": config.model_name,
            "backend": "huggingface",
            "use_lora": config.use_lora,
            "output_dir": str(config.output_dir),
        },
        prompt=_default_prompt_summary(),
        metrics=dict(metrics),
        metadata=dict(metadata),
    )


def _build_eval_manifest(
    *,
    run_id: str,
    downstream_args: Sequence[str],
    args,
    parent_run_id: str | None,
    status: str,
    finished_at: str | None,
    output_artifacts,
    metrics,
    metadata,
) -> RunManifest:
    git_metadata = collect_git_metadata(REPO_ROOT)
    input_artifacts = [
        build_artifact_ref(name="dataset", path=args.data, repo_root=REPO_ROOT),
        build_artifact_ref(
            name="split_manifest",
            path=args.split_manifest,
            repo_root=REPO_ROOT,
        ),
        build_artifact_ref(name="config", path=args.config, repo_root=REPO_ROOT),
    ]
    model_path = Path(args.model)
    if model_path.exists():
        input_artifacts.append(
            build_artifact_ref(
                name="model_input",
                path=model_path,
                repo_root=REPO_ROOT,
                include_hash=False,
            )
        )
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_type="eval",
        status=status,
        created_at=utc_now_iso(),
        finished_at=finished_at,
        entrypoint="recipe_mpr_qa.cli:run-eval",
        command=tuple(["recipe-mpr-qa", "run-eval", *downstream_args]),
        git_commit=git_metadata["git_commit"],
        git_dirty=git_metadata["git_dirty"],
        environment=build_environment_summary(),
        input_artifacts=tuple(input_artifacts),
        output_artifacts=tuple(output_artifacts),
        model={
            "name": args.model,
            "backend": args.backend,
        },
        prompt=_default_prompt_summary(),
        metrics=dict(metrics),
        parent_run_id=parent_run_id,
        metadata=dict(metadata),
    )


def _train_metrics(result: dict[str, object]) -> dict[str, object]:
    trainer_state = dict(result.get("trainer_state", {}))
    dataset_sizes = dict(result.get("dataset_sizes", {}))
    return {
        "epoch": trainer_state.get("epoch"),
        "global_step": trainer_state.get("global_step"),
        "best_metric": trainer_state.get("best_metric"),
        "train_examples": dataset_sizes.get("train"),
        "validation_examples": dataset_sizes.get("validation"),
        "test_examples": dataset_sizes.get("test"),
    }


def _train_output_artifacts(result: dict[str, object]):
    return tuple(
        artifact
        for artifact in (
            build_artifact_ref(
                name="output_dir",
                path=result["output_dir"],
                repo_root=REPO_ROOT,
                include_hash=False,
            )
            if "output_dir" in result
            else None,
            build_artifact_ref(
                name="final_model_dir",
                path=result["final_dir"],
                repo_root=REPO_ROOT,
                include_hash=False,
            )
            if "final_dir" in result
            else None,
            build_artifact_ref(
                name="run_config_json",
                path=result["run_config_path"],
                repo_root=REPO_ROOT,
            )
            if "run_config_path" in result
            else None,
            build_artifact_ref(
                name="log_history_json",
                path=result["log_history_path"],
                repo_root=REPO_ROOT,
            )
            if "log_history_path" in result
            else None,
            build_artifact_ref(
                name="trainer_state_summary_json",
                path=result["trainer_state_path"],
                repo_root=REPO_ROOT,
            )
            if "trainer_state_path" in result
            else None,
        )
        if artifact is not None
    )


def _eval_output_artifacts(result: dict[str, object]):
    if "output_path" not in result:
        return tuple()
    return (
        build_artifact_ref(
            name="result_json",
            path=result["output_path"],
            repo_root=REPO_ROOT,
        ),
    )


def _default_prompt_summary() -> dict[str, object]:
    from recipe_mpr_qa.formats import DEFAULT_PROMPT_SPEC, OPTION_ORDER_SHUFFLE_SEED

    return {
        "version": DEFAULT_PROMPT_SPEC.version,
        "option_order": "deterministic_per_example_shuffle",
        "shuffle_seed": OPTION_ORDER_SHUFFLE_SEED,
    }


def _normalize_script_args(script_args: Sequence[str]) -> list[str]:
    args = list(script_args)
    if args and args[0] == "--":
        return args[1:]
    return args


def _arg_present(args: Sequence[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in args)


def _resolve_parent_model_path(run_id: str, *, mlops_root: Path | str) -> str:
    manifest = read_run_manifest(run_id, mlops_root=mlops_root)
    for artifact in manifest.output_artifacts:
        if artifact.name == "final_model_dir":
            return _resolve_repo_path(artifact.path)
    raise FileNotFoundError(f"Run {run_id} does not expose a final_model_dir artifact")


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _maybe_mirror_to_mlflow(
    manifest: RunManifest,
    *,
    enable_mlflow: bool,
    tracking_uri: str | None,
    experiment_name: str,
) -> None:
    if not enable_mlflow:
        return
    try:
        mirror_run_to_mlflow(
            manifest,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
    except Exception as exc:
        print(f"Warning: failed to mirror run {manifest.run_id} to MLflow: {exc}", file=sys.stderr)
