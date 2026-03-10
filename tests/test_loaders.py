from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.data.loaders import (
    build_option_scoring_examples,
    get_split_examples,
    load_dataset,
    load_option_scoring_split,
    load_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


def test_get_split_examples_uses_manifest_order() -> None:
    dataset = load_dataset(PROCESSED_DATASET_PATH)
    split_manifest = load_split_manifest(SPLIT_MANIFEST_PATH)

    train_examples = get_split_examples(dataset, split_manifest, "train")

    assert len(train_examples) == 350
    assert tuple(example.example_id for example in train_examples[:5]) == split_manifest.splits["train"][:5]


def test_build_option_scoring_examples_expands_each_question() -> None:
    dataset = load_dataset(PROCESSED_DATASET_PATH)
    examples = dataset.examples[:2]

    scoring_examples = build_option_scoring_examples(examples)

    assert len(scoring_examples) == 10
    for example in examples:
        group = [item for item in scoring_examples if item.example_id == example.example_id]
        assert len(group) == 5
        assert sum(item.label for item in group) == 1


def test_build_option_scoring_examples_supports_tokenizer_passthrough() -> None:
    dataset = load_dataset(PROCESSED_DATASET_PATH)
    example = dataset.examples[:1]

    def tokenizer(query: str, option_text: str, **kwargs):
        return {
            "input_ids": [len(query), len(option_text)],
            "attention_mask": [1, 1],
            "truncation": kwargs["truncation"],
        }

    scoring_examples = build_option_scoring_examples(
        example,
        tokenizer=tokenizer,
        tokenizer_kwargs={"truncation": True},
    )

    assert len(scoring_examples) == 5
    assert scoring_examples[0].tokenized_inputs == {
        "input_ids": [len(example[0].query), len(example[0].options[0].text)],
        "attention_mask": [1, 1],
        "truncation": True,
    }
    assert scoring_examples[0].to_model_input()["group_size"] == 5


def test_load_option_scoring_split_returns_five_rows_per_example() -> None:
    scoring_examples = load_option_scoring_split(
        split_name="validation",
        dataset_path=PROCESSED_DATASET_PATH,
        split_manifest_path=SPLIT_MANIFEST_PATH,
    )

    assert len(scoring_examples) == 75 * 5
