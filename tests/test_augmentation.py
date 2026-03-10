from __future__ import annotations

from pathlib import Path

import pytest

from recipe_mpr_qa.augmentation import (
    build_augmented_examples,
    read_augmented_dataset,
    write_augmented_dataset,
)
from recipe_mpr_qa.data.preparation import read_prepared_dataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, *, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        del model_name, prompt, temperature
        return self.responses.pop(0)


def test_build_augmented_examples_creates_stable_ids_and_metadata() -> None:
    examples = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[:1]
    client = FakeClient(['{"rewrites": ["rewrite one", "rewrite two"]}'])

    augmented = build_augmented_examples(
        examples=examples,
        client=client,
        teacher_model_name="llama3.1:8b",
        variants_per_example=2,
    )

    assert [example.example_id for example in augmented] == [
        "rmpr-0001-aug-001",
        "rmpr-0001-aug-002",
    ]
    assert all(example.answer_option_id == examples[0].answer_option_id for example in augmented)
    assert augmented[0].source_metadata["source_example_id"] == "rmpr-0001"
    assert augmented[0].source_metadata["synthetic"] is True


def test_build_augmented_examples_rejects_malformed_json() -> None:
    examples = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[:1]
    client = FakeClient(["not json"])

    with pytest.raises(ValueError):
        build_augmented_examples(
            examples=examples,
            client=client,
            teacher_model_name="llama3.1:8b",
            variants_per_example=1,
        )


def test_write_and_read_augmented_dataset_round_trip(tmp_path: Path) -> None:
    examples = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[:1]
    client = FakeClient(['{"rewrites": ["rewrite one"]}'])
    augmented = build_augmented_examples(
        examples=examples,
        client=client,
        teacher_model_name="llama3.1:8b",
        variants_per_example=1,
    )
    output_path = tmp_path / "augmented.jsonl"

    write_augmented_dataset(augmented, output_path)
    restored = read_augmented_dataset(output_path)

    assert restored == augmented


def test_build_augmented_examples_resumes_partial_source() -> None:
    examples = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[:1]
    existing = build_augmented_examples(
        examples=examples,
        client=FakeClient(['{"rewrites": ["rewrite one"]}']),
        teacher_model_name="llama3.1:8b",
        variants_per_example=1,
    )
    resumed = build_augmented_examples(
        examples=examples,
        client=FakeClient(['{"rewrites": ["rewrite two"]}']),
        teacher_model_name="llama3.1:8b",
        variants_per_example=2,
        existing_examples=existing,
    )

    assert [example.example_id for example in resumed] == [
        "rmpr-0001-aug-001",
        "rmpr-0001-aug-002",
    ]
