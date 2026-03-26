from __future__ import annotations

from recipe_mpr_qa.data.augmentation import (
    MAX_AUGMENTED_VARIANTS,
    augment_example,
    augment_training_examples,
)
from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.models import DatasetValidationError, RecipeExample, RecipeOption


def _make_example(query: str, example_id: str = "rmpr-0001") -> RecipeExample:
    return RecipeExample(
        example_id=example_id,
        query=query,
        options=tuple(
            RecipeOption(option_id=f"opt-{idx}", text=f"Recipe option {idx}")
            for idx in range(1, 6)
        ),
        answer_option_id="opt-1",
        query_type_flags={name: (name == "Commonsense") for name in QUERY_TYPE_NAMES},
        correctness_explanation={"reason": "matches the request"},
        source_metadata={"raw_index": 0},
    )


def test_augment_example_preserves_core_fields_and_adds_metadata() -> None:
    example = _make_example("Can I have a fish recipe but not salmon?")

    augmented_examples = augment_example(example)

    assert len(augmented_examples) == 2
    first_augmented = augmented_examples[0]
    assert first_augmented.options == example.options
    assert first_augmented.answer_option_id == example.answer_option_id
    assert dict(first_augmented.query_type_flags) == dict(example.query_type_flags)
    assert dict(first_augmented.correctness_explanation) == dict(example.correctness_explanation)
    assert first_augmented.query != example.query
    assert first_augmented.example_id == "rmpr-0001-aug-01"
    assert first_augmented.source_metadata["parent_example_id"] == example.example_id
    assert first_augmented.source_metadata["augmentation_strategy"] == "lead_in_rewrite"
    assert first_augmented.source_metadata["variant_index"] == 1


def test_augment_example_deduplicates_and_caps_variants() -> None:
    example = _make_example("Show me a pasta recipe")

    augmented_examples = augment_example(example, max_variants=1)

    assert len(augmented_examples) == 1
    assert augmented_examples[0].query == "Looking for a pasta recipe"


def test_augment_training_examples_skips_queries_without_safe_rewrites() -> None:
    examples = (
        _make_example("Need brunch ideas", example_id="rmpr-0001"),
        _make_example("What are recipes for fish?", example_id="rmpr-0002"),
    )

    augmented_examples = augment_training_examples(examples)

    assert augmented_examples == ()


def test_augment_example_rejects_invalid_max_variants() -> None:
    example = _make_example("Can I have a fish recipe but not salmon?")

    try:
        augment_example(example, max_variants=MAX_AUGMENTED_VARIANTS + 1)
    except DatasetValidationError:
        pass
    else:
        raise AssertionError("Expected DatasetValidationError for invalid max_variants")
