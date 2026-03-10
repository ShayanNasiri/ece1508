from __future__ import annotations

from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.data.preparation import read_prepared_dataset, write_prepared_dataset
from recipe_mpr_qa.llm.prompts import (
    AUGMENTATION_PROMPT_SPEC,
    build_augmentation_prompt,
    parse_augmentation_response,
)


def build_augmented_examples(
    *,
    examples: Sequence[RecipeExample],
    client,
    teacher_model_name: str,
    variants_per_example: int,
    existing_examples: Sequence[RecipeExample] = (),
    temperature: float = 0.2,
) -> tuple[RecipeExample, ...]:
    augmented_examples: list[RecipeExample] = list(existing_examples)
    seen_ids: set[str] = {example.example_id for example in existing_examples}
    synthetic_counts: dict[str, int] = {}
    for existing in existing_examples:
        if not existing.source_metadata.get("synthetic"):
            continue
        source_example_id = existing.source_metadata.get("source_example_id")
        if isinstance(source_example_id, str):
            synthetic_counts[source_example_id] = synthetic_counts.get(source_example_id, 0) + 1
    for example in examples:
        if synthetic_counts.get(example.example_id, 0) >= variants_per_example:
            continue
        prompt = build_augmentation_prompt(
            example,
            requested_count=variants_per_example,
            prompt_spec=AUGMENTATION_PROMPT_SPEC,
        )
        response_text = client.generate(
            model_name=teacher_model_name,
            prompt=prompt,
            temperature=temperature,
        )
        rewrites = parse_augmentation_response(response_text)
        existing_count = synthetic_counts.get(example.example_id, 0)
        remaining = variants_per_example - existing_count
        if len(rewrites) < remaining:
            raise ValueError(
                f"Expected at least {remaining} rewrites for {example.example_id}, got {len(rewrites)}"
            )
        for index, query in enumerate(rewrites[:remaining], start=existing_count + 1):
            synthetic_id = f"{example.example_id}-aug-{index:03d}"
            if synthetic_id in seen_ids:
                raise ValueError(f"Duplicate synthetic example id: {synthetic_id}")
            seen_ids.add(synthetic_id)
            synthetic_counts[example.example_id] = synthetic_counts.get(example.example_id, 0) + 1
            augmented_examples.append(
                RecipeExample(
                    example_id=synthetic_id,
                    query=query,
                    options=example.options,
                    answer_option_id=example.answer_option_id,
                    query_type_flags=example.query_type_flags,
                    correctness_explanation=example.correctness_explanation,
                    source_metadata={
                        **dict(example.source_metadata),
                        "source_example_id": example.example_id,
                        "synthetic": True,
                        "teacher_model_name": teacher_model_name,
                        "prompt_version": AUGMENTATION_PROMPT_SPEC.version,
                    },
                )
            )
    return tuple(augmented_examples)


def write_augmented_dataset(
    examples: Sequence[RecipeExample],
    output_path: Path | str,
) -> None:
    from recipe_mpr_qa.data.models import PreparedDataset

    metadata = {
        "example_count": len(examples),
        "synthetic_count": sum(
            1 for example in examples if example.source_metadata.get("synthetic")
        ),
    }
    write_prepared_dataset(PreparedDataset(examples=tuple(examples), metadata=metadata), output_path)


def read_augmented_dataset(dataset_path: Path | str) -> tuple[RecipeExample, ...]:
    return read_prepared_dataset(dataset_path).examples
