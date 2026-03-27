from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from recipe_mpr_qa.tracking.mlflow import mirror_run_to_mlflow
from recipe_mpr_qa.tracking.models import RunManifest


def test_mirror_run_to_mlflow_smoke(tmp_path) -> None:
    manifest = RunManifest(
        schema_version="mlops-run-v1",
        run_id="train-mlflow",
        run_type="train",
        status="completed",
        created_at="2026-03-26T00:00:00+00:00",
        finished_at="2026-03-26T00:10:00+00:00",
        entrypoint="recipe_mpr_qa.cli:run-train",
        command=("recipe-mpr-qa", "run-train"),
        git_commit="abc123",
        git_dirty=False,
        environment={"python_version": "3.13"},
        model={"name": "demo-model"},
        metrics={"best_metric": 0.9},
    )

    mirror_run_to_mlflow(
        manifest,
        tracking_uri=tmp_path.as_uri(),
        experiment_name="recipe-mpr-qa-test",
    )
