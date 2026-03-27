from __future__ import annotations

import json

from recipe_mpr_qa.cli import main
from recipe_mpr_qa.tracking.models import RegistryEntry, RunManifest


def _make_manifest(run_id: str, run_type: str = "train", parent_run_id: str | None = None) -> RunManifest:
    return RunManifest(
        schema_version="mlops-run-v1",
        run_id=run_id,
        run_type=run_type,
        status="completed",
        created_at="2026-03-26T00:00:00+00:00",
        finished_at="2026-03-26T00:10:00+00:00",
        entrypoint="recipe_mpr_qa.cli:test",
        command=("recipe-mpr-qa",),
        git_commit="abc123",
        git_dirty=False,
        environment={"python_version": "3.13"},
        model={"name": "demo-model"},
        parent_run_id=parent_run_id,
    )


def test_cli_run_train_passes_unknown_args_to_tracked_runner(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run_tracked_train(**kwargs):
        captured.update(kwargs)
        return _make_manifest("train-123")

    monkeypatch.setattr("recipe_mpr_qa.cli.run_tracked_train", fake_run_tracked_train)

    exit_code = main(["run-train", "--stage", "candidate", "--model-name", "demo/model"])

    assert exit_code == 0
    assert tuple(captured["script_args"]) == ("--model-name", "demo/model")
    assert json.loads(capsys.readouterr().out)["run_id"] == "train-123"


def test_cli_run_eval_passes_parent_run_id(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run_tracked_eval(**kwargs):
        captured.update(kwargs)
        return _make_manifest("eval-123", run_type="eval", parent_run_id=kwargs["parent_run_id"])

    monkeypatch.setattr("recipe_mpr_qa.cli.run_tracked_eval", fake_run_tracked_eval)

    exit_code = main(
        [
            "run-eval",
            "--parent-run-id",
            "train-001",
            "--backend",
            "huggingface",
            "--model",
            "demo/model",
        ]
    )

    assert exit_code == 0
    assert captured["parent_run_id"] == "train-001"
    assert tuple(captured["script_args"]) == ("--backend", "huggingface", "--model", "demo/model")
    assert json.loads(capsys.readouterr().out)["parent_run_id"] == "train-001"


def test_cli_list_runs_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "recipe_mpr_qa.cli.list_registered_runs",
        lambda **kwargs: [
            RegistryEntry(
                entry_id="train-001",
                run_id="train-001",
                run_type="train",
                stage="candidate",
                status="completed",
                model_name="demo-model",
                created_at="2026-03-26T00:00:00+00:00",
                updated_at="2026-03-26T00:10:00+00:00",
            )
        ],
    )

    exit_code = main(["list-runs", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["run_id"] == "train-001"


def test_cli_compare_runs_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "recipe_mpr_qa.cli.build_run_comparison",
        lambda run_ids, **kwargs: [{"run_id": run_ids[0], "overall": 0.9}],
    )

    exit_code = main(["compare-runs", "--run-id", "eval-001", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["run_id"] == "eval-001"


def test_cli_promote_run_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "recipe_mpr_qa.cli.promote_run",
        lambda run_id, stage, **kwargs: RegistryEntry(
            entry_id=run_id,
            run_id=run_id,
            run_type="train",
            stage=stage,
            status="completed",
            model_name="demo-model",
            created_at="2026-03-26T00:00:00+00:00",
            updated_at="2026-03-26T00:20:00+00:00",
        ),
    )

    exit_code = main(["promote-run", "--run-id", "train-001", "--stage", "validated"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "validated"
