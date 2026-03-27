from __future__ import annotations

from collections.abc import Mapping

from recipe_mpr_qa.tracking.models import RunManifest


def mirror_run_to_mlflow(
    manifest: RunManifest,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "recipe-mpr-qa",
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is not installed. Install with `pip install -e \".[mlops]\"`.") from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=manifest.run_id):
        mlflow.set_tags(
            {
                "run_id": manifest.run_id,
                "run_type": manifest.run_type,
                "status": manifest.status,
                "git_commit": manifest.git_commit or "",
                "entrypoint": manifest.entrypoint,
                "model_name": str(manifest.model.get("name", "")),
            }
        )

        params = _flatten_scalars(
            {
                "model": manifest.model,
                "prompt": manifest.prompt,
                "metadata": manifest.metadata,
            }
        )
        if params:
            mlflow.log_params(params)

        metrics = _flatten_numeric_scalars(manifest.metrics)
        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.log_dict(manifest.to_dict(), "run_manifest.json")


def _flatten_scalars(payload: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        joined_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_scalars(value, joined_key))
        elif value is not None and isinstance(value, (str, int, float, bool)):
            flattened[joined_key] = str(value)
    return flattened


def _flatten_numeric_scalars(payload: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        joined_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_numeric_scalars(value, joined_key))
        elif isinstance(value, bool):
            flattened[joined_key] = float(value)
        elif isinstance(value, (int, float)):
            flattened[joined_key] = float(value)
    return flattened
