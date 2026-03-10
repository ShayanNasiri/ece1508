from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.evaluation.records import PredictionRecord, read_judgment_records, read_prediction_records
from recipe_mpr_qa.llm.inference import run_llm_predictions
from recipe_mpr_qa.llm.judge import judge_predictions


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, *, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        del model_name, prompt, temperature
        return self.responses.pop(0)


def test_run_llm_predictions_supports_resume(tmp_path: Path) -> None:
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    output_path = tmp_path / "predictions.jsonl"
    client = FakeClient(["Answer: A", "B"])

    first_run = run_llm_predictions(
        examples=dataset.examples[:1],
        client=client,
        run_id="llm-run",
        provider="ollama",
        model_name="llama3.1:8b",
        split="test",
        output_path=output_path,
    )
    second_run = run_llm_predictions(
        examples=dataset.examples[:2],
        client=client,
        run_id="llm-run",
        provider="ollama",
        model_name="llama3.1:8b",
        split="test",
        output_path=output_path,
        resume=True,
    )

    assert len(first_run) == 1
    assert len(second_run) == 2
    assert len(read_prediction_records(output_path)) == 2


def test_judge_predictions_writes_jsonl(tmp_path: Path) -> None:
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    prediction = PredictionRecord(
        run_id="llm-run",
        phase="phase3",
        provider="ollama",
        model_name="llama3.1:8b",
        split="test",
        example_id=dataset.examples[0].example_id,
        prompt_version="recipe-mpr-mc-v1",
        raw_response="A",
        parsed_choice="A",
        predicted_option_id=dataset.examples[0].answer_option_id,
        gold_option_id=dataset.examples[0].answer_option_id,
        is_correct=True,
        latency_ms=12.5,
    )
    output_path = tmp_path / "judgments.jsonl"
    client = FakeClient(
        [
            '{"ingredient_alignment": 5, "constraint_satisfaction": 5, '
            '"reasoning_quality": 4, "overall_verdict": "correct", "rationale": "Matches the request."}'
        ]
    )

    judgments = judge_predictions(
        dataset=dataset,
        prediction_records=(prediction,),
        client=client,
        run_id="judge-run",
        model_name="llama3.1:8b",
        output_path=output_path,
    )

    assert len(judgments) == 1
    assert judgments[0].overall_verdict == "correct"
    assert read_judgment_records(output_path) == judgments
