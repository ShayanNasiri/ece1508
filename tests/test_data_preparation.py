from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.models import DatasetValidationError, RecipeExample
from recipe_mpr_qa.data.preparation import (
    generate_primary_split,
    prepare_dataset,
    prepare_examples,
    read_prepared_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT / "data" / "500QA.json"
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


def _load_raw_records() -> list[dict]:
    return json.loads(RAW_DATASET_PATH.read_text(encoding="utf-8"))


def test_prepare_dataset_assigns_stable_ids_and_metadata() -> None:
    dataset = prepare_dataset(RAW_DATASET_PATH)

    assert len(dataset.examples) == 500
    assert dataset.examples[0].example_id == "rmpr-0001"
    assert dataset.examples[-1].example_id == "rmpr-0500"
    assert dataset.metadata["example_count"] == 500
    assert set(dataset.metadata["query_type_counts"]) == set(QUERY_TYPE_NAMES)


def test_prepare_dataset_normalizes_query_whitespace() -> None:
    dataset = prepare_dataset(RAW_DATASET_PATH)
    normalized_example = dataset.examples[3]

    assert normalized_example.query == "I would like a shrimp recipe and I'm trying to eat a balanced diet"
    assert normalized_example.source_metadata["normalization"] == ["strip_query_outer_whitespace"]


def test_prepare_examples_rejects_missing_keys() -> None:
    raw_records = _load_raw_records()
    raw_records[0] = {"query": "missing fields"}

    with pytest.raises(DatasetValidationError):
        prepare_examples(raw_records, RAW_DATASET_PATH)


def test_prepare_examples_rejects_invalid_answer() -> None:
    raw_records = _load_raw_records()
    raw_records[0]["answer"] = "not-an-option"

    with pytest.raises(DatasetValidationError):
        prepare_examples(raw_records, RAW_DATASET_PATH)


def test_prepare_examples_rejects_wrong_option_count() -> None:
    raw_records = _load_raw_records()
    raw_records[0]["options"] = {"a": "one"}

    with pytest.raises(DatasetValidationError):
        prepare_examples(raw_records, RAW_DATASET_PATH)


def test_prepared_example_rejects_duplicate_option_ids() -> None:
    example = prepare_dataset(RAW_DATASET_PATH).examples[0].to_dict()
    example["options"] = [
        {"option_id": "dup", "text": "First"},
        {"option_id": "dup", "text": "Second"},
        {"option_id": "c", "text": "Third"},
        {"option_id": "d", "text": "Fourth"},
        {"option_id": "e", "text": "Fifth"},
    ]
    example["answer_option_id"] = "dup"

    with pytest.raises(DatasetValidationError):
        RecipeExample.from_dict(example)


def test_generate_primary_split_has_expected_sizes_and_is_disjoint() -> None:
    dataset = prepare_dataset(RAW_DATASET_PATH)
    split_manifest = generate_primary_split(dataset.examples)

    assert len(split_manifest.splits["train"]) == 350
    assert len(split_manifest.splits["validation"]) == 75
    assert len(split_manifest.splits["test"]) == 75
    all_ids = set().union(*map(set, split_manifest.splits.values()))
    assert len(all_ids) == 500
    assert set(split_manifest.splits["train"]).isdisjoint(split_manifest.splits["validation"])
    assert set(split_manifest.splits["train"]).isdisjoint(split_manifest.splits["test"])
    assert set(split_manifest.splits["validation"]).isdisjoint(split_manifest.splits["test"])


def test_committed_processed_dataset_matches_generator() -> None:
    generated_dataset = prepare_dataset(RAW_DATASET_PATH)
    committed_dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)

    assert [example.to_dict() for example in committed_dataset.examples] == [
        example.to_dict() for example in generated_dataset.examples
    ]


def test_committed_split_manifest_matches_generator() -> None:
    dataset = prepare_dataset(RAW_DATASET_PATH)
    generated_split = generate_primary_split(dataset.examples).to_dict()
    committed_split = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert committed_split == generated_split
