from __future__ import annotations

import sys
from pathlib import Path

from recipe_mpr_qa.tracking.mlflow import ExperimentContext, MLflowLogger


class FakeMLflowModule:
    def __init__(self):
        self.calls = []

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    def start_run(self, run_name: str):
        self.calls.append(("start_run", run_name))
        return self._Run()

    def set_tag(self, key: str, value: str) -> None:
        self.calls.append(("set_tag", key, value))

    def log_param(self, key: str, value) -> None:
        self.calls.append(("log_param", key, value))

    def log_metric(self, key: str, value: float) -> None:
        self.calls.append(("log_metric", key, value))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifact", path, artifact_path))

    def log_artifacts(self, path: str, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifacts", path, artifact_path))


def test_mlflow_logger_logs_metrics_and_artifacts(monkeypatch, tmp_path: Path) -> None:
    fake_mlflow = FakeMLflowModule()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact_file = tmp_path / "metrics.json"
    artifact_file.write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "checkpoints"
    artifact_dir.mkdir()

    logger = MLflowLogger(tracking_uri="file:///tmp/mlruns")
    logger.log_run(
        context=ExperimentContext(
            experiment_name="recipe-mpr-qa",
            run_name="unit-test",
            tags={"phase": "phase2"},
            params={"epochs": 3},
        ),
        metrics={"accuracy": 0.8},
        artifact_paths={"metrics": artifact_file, "checkpoints": artifact_dir},
    )

    assert ("set_experiment", "recipe-mpr-qa") in fake_mlflow.calls
    assert ("log_metric", "accuracy", 0.8) in fake_mlflow.calls
    assert any(call[0] == "log_artifact" for call in fake_mlflow.calls)
    assert any(call[0] == "log_artifacts" for call in fake_mlflow.calls)
