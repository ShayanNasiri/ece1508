from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.tracking.artifacts import build_artifact_ref
from recipe_mpr_qa.tracking.models import ArtifactRef, RunManifest
from recipe_mpr_qa.tracking.registry import (
    load_registry_entries,
    promote_run,
    read_run_manifest,
    register_run,
)


def test_build_artifact_ref_hash_is_deterministic(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"ok": true}\n', encoding="utf-8")

    first = build_artifact_ref(name="payload", path=payload_path, repo_root=tmp_path)
    second = build_artifact_ref(name="payload", path=payload_path, repo_root=tmp_path)

    assert first.sha256 is not None
    assert first.sha256 == second.sha256
    assert first.path == "payload.json"


def test_register_run_writes_manifest_and_registries(tmp_path: Path) -> None:
    final_dir = tmp_path / "outputs" / "demo" / "final"
    final_dir.mkdir(parents=True)
    manifest = RunManifest(
        schema_version="mlops-run-v1",
        run_id="train-001",
        run_type="train",
        status="completed",
        created_at="2026-03-26T00:00:00+00:00",
        finished_at="2026-03-26T00:10:00+00:00",
        entrypoint="recipe_mpr_qa.cli:run-train",
        command=("recipe-mpr-qa", "run-train"),
        git_commit="abc123",
        git_dirty=False,
        environment={"python_version": "3.13"},
        input_artifacts=(),
        output_artifacts=(
            ArtifactRef(
                name="final_model_dir",
                path=(final_dir.relative_to(tmp_path)).as_posix(),
                artifact_type="dir",
                exists=True,
            ),
        ),
        model={"name": "demo-model"},
        metrics={"best_metric": 0.9},
    )

    register_run(manifest, stage="candidate", mlops_root=tmp_path / "mlops")

    restored = read_run_manifest("train-001", mlops_root=tmp_path / "mlops")
    run_entries = load_registry_entries("runs", mlops_root=tmp_path / "mlops")
    model_entries = load_registry_entries("models", mlops_root=tmp_path / "mlops")
    summary_payload = json.loads(
        (tmp_path / "mlops" / "runs" / "train-001" / "summary.json").read_text(encoding="utf-8")
    )

    assert restored == manifest
    assert len(run_entries) == 1
    assert len(model_entries) == 1
    assert run_entries[0].stage == "candidate"
    assert model_entries[0].artifact_path == "outputs/demo/final"
    assert summary_payload["metrics"]["best_metric"] == 0.9


def test_promote_run_updates_registry_stage(tmp_path: Path) -> None:
    manifest = RunManifest(
        schema_version="mlops-run-v1",
        run_id="eval-001",
        run_type="eval",
        status="completed",
        created_at="2026-03-26T00:00:00+00:00",
        finished_at="2026-03-26T00:10:00+00:00",
        entrypoint="recipe_mpr_qa.cli:run-eval",
        command=("recipe-mpr-qa", "run-eval"),
        git_commit="abc123",
        git_dirty=False,
        environment={"python_version": "3.13"},
        input_artifacts=(),
        output_artifacts=(),
        model={"name": "demo-model"},
        metrics={"overall": 0.8},
    )
    register_run(manifest, stage="baseline", mlops_root=tmp_path / "mlops")

    updated = promote_run("eval-001", "validated", mlops_root=tmp_path / "mlops")
    run_entries = load_registry_entries("runs", mlops_root=tmp_path / "mlops")
    summary_payload = json.loads(
        (tmp_path / "mlops" / "runs" / "eval-001" / "summary.json").read_text(encoding="utf-8")
    )

    assert updated.stage == "validated"
    assert run_entries[0].stage == "validated"
    assert summary_payload["stage"] == "validated"
