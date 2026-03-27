from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.tracking.registry import load_registry_entries, register_run
from recipe_mpr_qa.tracking.runner import run_tracked_eval, run_tracked_train
from recipe_mpr_qa.tracking.models import ArtifactRef, RunManifest


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


def test_run_tracked_train_writes_manifest_and_registry(tmp_path: Path, monkeypatch) -> None:
    def fake_run_training_from_arg_list(argv):
        output_dir = tmp_path / "outputs" / "tracked-train"
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True)
        run_config_path = output_dir / "run_config.json"
        log_history_path = output_dir / "log_history.json"
        trainer_state_path = output_dir / "trainer_state_summary.json"
        run_config_path.write_text("{}", encoding="utf-8")
        log_history_path.write_text("[]", encoding="utf-8")
        trainer_state_path.write_text(
            json.dumps({"epoch": 1.0, "global_step": 5, "best_metric": 0.75}),
            encoding="utf-8",
        )
        return {
            "output_dir": str(output_dir),
            "final_dir": str(final_dir),
            "run_config_path": str(run_config_path),
            "log_history_path": str(log_history_path),
            "trainer_state_path": str(trainer_state_path),
            "trainer_state": {"epoch": 1.0, "global_step": 5, "best_metric": 0.75},
            "dataset_sizes": {"train": 350, "validation": 75, "test": 75},
            "model": {"name": "demo-model"},
        }

    monkeypatch.setattr(
        "recipe_mpr_qa.slm.finetune.run_training_from_arg_list",
        fake_run_training_from_arg_list,
    )

    manifest = run_tracked_train(
        script_args=(
            "--data-path",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest-path",
            str(SPLIT_MANIFEST_PATH),
            "--output-dir",
            str(tmp_path / "outputs" / "tracked-train"),
        ),
        stage="candidate",
        mlops_root=tmp_path / "mlops",
    )

    run_entries = load_registry_entries("runs", mlops_root=tmp_path / "mlops")
    model_entries = load_registry_entries("models", mlops_root=tmp_path / "mlops")

    assert manifest.status == "completed"
    assert manifest.metrics["best_metric"] == 0.75
    assert any(artifact.name == "trainer_state_summary_json" for artifact in manifest.output_artifacts)
    assert len(run_entries) == 1
    assert len(model_entries) == 1
    assert model_entries[0].stage == "candidate"


def test_run_tracked_eval_links_parent_run_and_writes_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent_final_dir = tmp_path / "outputs" / "parent" / "final"
    parent_final_dir.mkdir(parents=True)
    parent_manifest = RunManifest(
        schema_version="mlops-run-v1",
        run_id="train-parent",
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
                path=str(parent_final_dir.resolve()),
                artifact_type="dir",
                exists=True,
            ),
        ),
        model={"name": "demo-parent"},
    )
    register_run(parent_manifest, stage="candidate", mlops_root=tmp_path / "mlops")

    captured = {}

    def fake_run_evaluation_from_arg_list(argv):
        captured["argv"] = list(argv)
        output_path = tmp_path / "llm_results" / "eval.json"
        output_path.parent.mkdir(parents=True)
        output_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {
            "output_path": str(output_path),
            "metrics": {"overall": 0.8, "parse_failures": 1, "total": 10},
            "prompt": {"version": "recipe-mpr-mc-v1"},
            "temperature": 0,
            "example_count": 10,
        }

    monkeypatch.setattr(
        "recipe_mpr_qa.evaluation.mc_eval.run_evaluation_from_arg_list",
        fake_run_evaluation_from_arg_list,
    )

    manifest = run_tracked_eval(
        script_args=(
            "--backend",
            "huggingface",
            "--data",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--config",
            str(ROOT / "llm_evaluation" / "config.json"),
        ),
        stage="baseline",
        parent_run_id="train-parent",
        mlops_root=tmp_path / "mlops",
    )

    run_entries = load_registry_entries("runs", mlops_root=tmp_path / "mlops")

    assert manifest.status == "completed"
    assert manifest.parent_run_id == "train-parent"
    assert manifest.metrics["overall"] == 0.8
    assert "--model" in captured["argv"]
    assert any(entry.run_id == manifest.run_id for entry in run_entries)
