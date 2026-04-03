from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.llm.prompts import DEFAULT_PROMPT_SPEC, benchmark_prompt_metadata, build_multiple_choice_prompt
from recipe_mpr_qa.synthetic.pipeline import (
    approve_synthetic_query_candidates,
    build_train_ready_dataset,
    generate_synthetic_query_candidates,
    review_synthetic_query_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_PATH = ROOT / "data" / "processed" / "primary_split.json"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, *, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        del model_name, prompt, temperature
        return self.responses.pop(0)


def test_synthetic_query_pipeline_builds_train_ready_dataset(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.jsonl"
    reviewed_path = tmp_path / "reviewed.jsonl"
    review_predictions_path = tmp_path / "review_predictions.jsonl"
    approved_path = tmp_path / "approved.jsonl"
    train_ready_path = tmp_path / "train_ready.jsonl"
    manifest_path = tmp_path / "train_ready_manifest.json"

    generate_summary = generate_synthetic_query_candidates(
        dataset_path=DATASET_PATH,
        split_manifest_path=SPLIT_PATH,
        output_path=candidate_path,
        client=FakeClient(['{"rewrites": ["synthetic benchmark query"]}']),
        model_name="generator",
        parent_limit=1,
        variants_per_example=1,
    )
    candidate_dataset = read_prepared_dataset(candidate_path)
    candidate_example = candidate_dataset.examples[0]
    metadata = benchmark_prompt_metadata(
        example_id=candidate_example.example_id,
        prompt_version=DEFAULT_PROMPT_SPEC.version,
    )
    _prompt, letter_to_option_id = build_multiple_choice_prompt(
        query=candidate_example.query,
        options=candidate_example.options,
        prompt_spec=DEFAULT_PROMPT_SPEC,
        shuffle_key=metadata["shuffle_key"],
        shuffle_seed=metadata["shuffle_seed"],
    )
    correct_letter = next(
        letter for letter, option_id in letter_to_option_id.items() if option_id == candidate_example.answer_option_id
    )

    review_summary = review_synthetic_query_candidates(
        input_path=candidate_path,
        reviewed_output_path=reviewed_path,
        review_predictions_path=review_predictions_path,
        reviewer_client=FakeClient([correct_letter]),
        reviewer_provider="ollama",
        reviewer_model_name="reviewer",
    )
    approve_summary = approve_synthetic_query_candidates(
        input_path=reviewed_path,
        dataset_path=DATASET_PATH,
        split_manifest_path=SPLIT_PATH,
        output_path=approved_path,
        approval_batch_id="batch-001",
    )
    train_ready_summary = build_train_ready_dataset(
        dataset_path=DATASET_PATH,
        split_manifest_path=SPLIT_PATH,
        approved_input_path=approved_path,
        output_path=train_ready_path,
        manifest_output_path=manifest_path,
        synthetic_ratio=0.1,
    )

    assert generate_summary["candidate_count"] == 1
    assert review_summary["review_status_counts"]["approved"] == 1
    assert approve_summary["approved_count"] == 1
    assert train_ready_summary["synthetic_selected_count"] == 1
