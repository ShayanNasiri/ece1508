from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentContext:
    experiment_name: str
    run_name: str
    tags: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": dict(self.tags),
            "params": dict(self.params),
        }


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[tracking] to use MLflow integration") from exc
    return mlflow


def build_mlflow_tags(
    *,
    phase: str,
    provider: str,
    model_name: str,
    split: str,
    prompt_version: str | None = None,
) -> dict[str, str]:
    tags = {
        "phase": phase,
        "provider": provider,
        "model_name": model_name,
        "split": split,
    }
    if prompt_version is not None:
        tags["prompt_version"] = prompt_version
    return tags


class MLflowLogger:
    def __init__(self, *, tracking_uri: str | None = None) -> None:
        self._mlflow = _require_mlflow()
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)

    def log_run(
        self,
        *,
        context: ExperimentContext,
        metrics: Mapping[str, float] | None = None,
        artifact_paths: Mapping[str, Path | str] | None = None,
    ) -> None:
        self._mlflow.set_experiment(context.experiment_name)
        with self._mlflow.start_run(run_name=context.run_name):
            for key, value in context.tags.items():
                self._mlflow.set_tag(key, value)
            for key, value in context.params.items():
                self._mlflow.log_param(key, value)
            for key, value in (metrics or {}).items():
                self._mlflow.log_metric(key, value)
            for key, value in (artifact_paths or {}).items():
                path = Path(value)
                if path.is_dir():
                    self._mlflow.log_artifacts(path.as_posix(), artifact_path=key)
                else:
                    self._mlflow.log_artifact(path.as_posix(), artifact_path=key)
