from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.evaluation.records import (
    JudgmentRecord,
    PredictionRecord,
    read_judgment_records,
    read_prediction_records,
    write_judgment_records,
    write_prediction_records,
)


def test_prediction_record_jsonl_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    records = (
        PredictionRecord(
            run_id="run-001",
            phase="phase3",
            provider="ollama",
            model_name="deepseek-r1:7b",
            split="test",
            example_id="rmpr-0001",
            prompt_version="recipe-mpr-mc-v1",
            raw_response="A",
            parsed_choice="A",
            predicted_option_id="08cb462fdf",
            gold_option_id="08cb462fdf",
            is_correct=True,
            latency_ms=142.5,
            metadata={"temperature": 0},
        ),
    )

    write_prediction_records(records, output_path)
    restored_records = read_prediction_records(output_path)

    assert restored_records == records


def test_judgment_record_jsonl_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "judgments.jsonl"
    records = (
        JudgmentRecord(
            run_id="judge-001",
            phase="phase4",
            provider="ollama",
            model_name="deepseek-r1:7b",
            split="test",
            example_id="rmpr-0001",
            prediction_run_id="run-001",
            predicted_option_id="08cb462fdf",
            gold_option_id="08cb462fdf",
            judge_model_name="llama3.1:8b",
            ingredient_alignment=5,
            constraint_satisfaction=4,
            reasoning_quality=4,
            overall_verdict="correct",
            rationale="Matches the ingredients.",
            metadata={"temperature": 0},
        ),
    )

    write_judgment_records(records, output_path)
    restored_records = read_judgment_records(output_path)

    assert restored_records == records
