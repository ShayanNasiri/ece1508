from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from recipe_mpr_qa.data.constants import (
    DEFAULT_PROCESSED_DATASET_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
)
from recipe_mpr_qa.data.models import OptionScoringExample, PreparedDataset, RecipeExample, SplitManifest
from recipe_mpr_qa.data.preparation import read_prepared_dataset, read_split_manifest


Tokenizer = Callable[..., Mapping[str, Any]]


def load_dataset(dataset_path: Path | str = DEFAULT_PROCESSED_DATASET_PATH) -> PreparedDataset:
    return read_prepared_dataset(dataset_path)


def load_split_manifest(
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
) -> SplitManifest:
    return read_split_manifest(split_manifest_path)


def get_split_examples(
    dataset: PreparedDataset, split_manifest: SplitManifest, split_name: str
) -> tuple[RecipeExample, ...]:
    if split_name not in split_manifest.splits:
        raise KeyError(split_name)
    dataset_by_id = {example.example_id: example for example in dataset.examples}
    return tuple(dataset_by_id[example_id] for example_id in split_manifest.splits[split_name])


def build_option_scoring_examples(
    examples: Sequence[RecipeExample],
    *,
    tokenizer: Tokenizer | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[OptionScoringExample, ...]:
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    scoring_examples: list[OptionScoringExample] = []
    for example in examples:
        for option_index, option in enumerate(example.options):
            tokenized_inputs = None
            if tokenizer is not None:
                tokenized_payload = tokenizer(example.query, option.text, **tokenizer_kwargs)
                tokenized_inputs = dict(tokenized_payload)
            scoring_examples.append(
                OptionScoringExample(
                    example_id=example.example_id,
                    option_id=option.option_id,
                    option_index=option_index,
                    group_size=len(example.options),
                    query=example.query,
                    option_text=option.text,
                    label=int(option.option_id == example.answer_option_id),
                    tokenized_inputs=tokenized_inputs,
                )
            )
    return tuple(scoring_examples)


def load_option_scoring_split(
    *,
    split_name: str,
    dataset_path: Path | str = DEFAULT_PROCESSED_DATASET_PATH,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
    tokenizer: Tokenizer | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[OptionScoringExample, ...]:
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    examples = get_split_examples(dataset, split_manifest, split_name)
    return build_option_scoring_examples(
        examples,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )
