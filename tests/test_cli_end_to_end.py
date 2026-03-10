from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.cli import main
from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.evaluation.records import JudgmentRecord, PredictionRecord, write_prediction_records


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_PATH = ROOT / "data" / "processed" / "primary_split.json"


def _write_config(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_cli_end_to_end_dry_run(monkeypatch, tmp_path: Path) -> None:
    dataset = read_prepared_dataset(DATASET_PATH)

    def fake_build_augmented_examples(**kwargs):
        examples = kwargs["examples"]
        return (
            type(examples[0])(
                example_id=f"{examples[0].example_id}-aug-001",
                query="synthetic query",
                options=examples[0].options,
                answer_option_id=examples[0].answer_option_id,
                query_type_flags=examples[0].query_type_flags,
                correctness_explanation=examples[0].correctness_explanation,
                source_metadata={"synthetic": True, "source_example_id": examples[0].example_id},
            ),
        )

    def fake_evaluate_vanilla_slm(**kwargs):
        output_path = kwargs["output_path"]
        example = kwargs["examples"][0]
        records = (
            PredictionRecord(
                run_id=kwargs["run_id"],
                phase="phase2",
                provider="slm",
                model_name=kwargs["model_name"],
                split=kwargs["split"],
                example_id=example.example_id,
                prompt_version=kwargs["prompt_version"],
                raw_response="[]",
                parsed_choice=None,
                predicted_option_id=example.answer_option_id,
                gold_option_id=example.answer_option_id,
                is_correct=True,
                latency_ms=1.0,
            ),
        )
        write_prediction_records(records, output_path)
        return records

    def fake_run_llm_predictions(**kwargs):
        output_path = kwargs["output_path"]
        example = kwargs["examples"][0]
        records = (
            PredictionRecord(
                run_id=kwargs["run_id"],
                phase="phase3",
                provider="ollama",
                model_name=kwargs["model_name"],
                split=kwargs["split"],
                example_id=example.example_id,
                prompt_version=kwargs["prompt_version"],
                raw_response="A",
                parsed_choice="A",
                predicted_option_id=example.answer_option_id,
                gold_option_id=example.answer_option_id,
                is_correct=True,
                latency_ms=1.0,
            ),
        )
        write_prediction_records(records, output_path)
        return records

    def fake_judge_predictions(**kwargs):
        output_path = kwargs["output_path"]
        prediction = kwargs["prediction_records"][0]
        records = (
            JudgmentRecord(
                run_id=kwargs["run_id"],
                phase="phase4",
                provider="ollama",
                model_name=prediction.model_name,
                split=prediction.split,
                example_id=prediction.example_id,
                prediction_run_id=prediction.run_id,
                predicted_option_id=prediction.predicted_option_id,
                gold_option_id=prediction.gold_option_id,
                judge_model_name=kwargs["model_name"],
                ingredient_alignment=5,
                constraint_satisfaction=5,
                reasoning_quality=5,
                overall_verdict="correct",
                rationale="All good.",
                metadata={},
            ),
        )
        from recipe_mpr_qa.evaluation.records import write_judgment_records

        write_judgment_records(records, output_path)
        return records

    monkeypatch.setattr("recipe_mpr_qa.cli.build_augmented_examples", fake_build_augmented_examples)
    monkeypatch.setattr("recipe_mpr_qa.cli.evaluate_vanilla_slm", fake_evaluate_vanilla_slm)
    monkeypatch.setattr("recipe_mpr_qa.cli.run_llm_predictions", fake_run_llm_predictions)
    monkeypatch.setattr("recipe_mpr_qa.cli.judge_predictions", fake_judge_predictions)

    augmentation_config = tmp_path / "augmentation.toml"
    slm_config = tmp_path / "slm.toml"
    llm_config = tmp_path / "llm.toml"
    judge_config = tmp_path / "judge.toml"
    run_root = (tmp_path / "runs").as_posix()

    _write_config(
        augmentation_config,
        f"""
[output]
run_id = "aug-run"
artifacts_root = "{run_root}"

[data]
dataset_path = "{DATASET_PATH.as_posix()}"
split_manifest_path = "{SPLIT_PATH.as_posix()}"
split = "train"

[augmentation]
teacher_model_name = "llama3.1:8b"
""",
    )
    _write_config(
        slm_config,
        f"""
[output]
run_id = "slm-run"
artifacts_root = "{run_root}"

[data]
dataset_path = "{DATASET_PATH.as_posix()}"
split_manifest_path = "{SPLIT_PATH.as_posix()}"
split = "test"

[slm]
mode = "vanilla"
""",
    )
    _write_config(
        llm_config,
        f"""
[output]
run_id = "llm-run"
artifacts_root = "{run_root}"

[data]
dataset_path = "{DATASET_PATH.as_posix()}"
split_manifest_path = "{SPLIT_PATH.as_posix()}"
split = "test"

[llm]
model_name = "llama3.1:8b"
""",
    )
    _write_config(
        judge_config,
        f"""
[output]
run_id = "judge-run"
artifacts_root = "{run_root}"

[data]
dataset_path = "{DATASET_PATH.as_posix()}"
split_manifest_path = "{SPLIT_PATH.as_posix()}"
split = "test"

[judge]
model_name = "llama3.1:8b"
""",
    )

    assert main(["generate-augmentation", "--config", str(augmentation_config)]) == 0
    assert main(["evaluate-slm", "--config", str(slm_config)]) == 0
    assert main(["run-llm", "--config", str(llm_config)]) == 0
    assert (
        main(
            [
                "judge-predictions",
                "--config",
                str(judge_config),
                "--predictions",
                str(tmp_path / "runs" / "llm-run" / "llm" / "test_predictions.jsonl"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "summarize-run",
                "--config",
                str(llm_config),
                "--component",
                "llm",
                "--predictions",
                str(tmp_path / "runs" / "llm-run" / "llm" / "test_predictions.jsonl"),
            ]
        )
        == 0
    )

    summary_path = tmp_path / "runs" / "llm-run" / "manifests" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["prediction_metrics"]["accuracy"] == 1.0
