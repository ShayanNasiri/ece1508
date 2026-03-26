from __future__ import annotations

import re
from typing import Sequence

from recipe_mpr_qa.data.models import DatasetValidationError, RecipeExample

MAX_AUGMENTED_VARIANTS = 2


def _normalize_query_key(query: str) -> str:
    collapsed = re.sub(r"\s+", " ", query).strip().casefold()
    return collapsed.rstrip("?.!")


def _split_terminal_punctuation(query: str) -> tuple[str, str]:
    stripped = query.strip()
    if stripped.endswith(("?", ".", "!")):
        return stripped[:-1].rstrip(), stripped[-1]
    return stripped, ""


def _rewrite_with_prefix(query: str, prefix: str, replacement: str) -> str | None:
    body, punctuation = _split_terminal_punctuation(query)
    lowered = body.casefold()
    if not lowered.startswith(prefix):
        return None
    remainder = body[len(prefix):].strip()
    if not remainder:
        return None
    if prefix.endswith(" ") and remainder.casefold().startswith("to ") and replacement.casefold() == "looking for ":
        return None
    return f"{replacement}{remainder}{punctuation}"


def _apply_lead_in_rewrite(query: str) -> str | None:
    prefix_rules = (
        ("i want to make ", "Looking for "),
        ("i want to prepare ", "Looking for "),
        ("i want to cook ", "Looking for "),
        ("can i have ", "Looking for "),
        ("could i have ", "Looking for "),
        ("show me ", "Looking for "),
        ("give me ", "Need "),
        ("any ideas for ", "Looking for "),
        ("how do i make ", "Looking for "),
        ("how to make ", "Looking for "),
    )
    for prefix, replacement in prefix_rules:
        rewritten = _rewrite_with_prefix(query, prefix, replacement)
        if rewritten is not None:
            return rewritten

    body, punctuation = _split_terminal_punctuation(query)
    lowered = body.casefold()
    if lowered.startswith("i want ") and not lowered.startswith("i want to "):
        remainder = body[len("i want ") :].strip()
        if remainder:
            return f"Looking for {remainder}{punctuation}"
    if lowered.startswith("i need ") and not lowered.startswith("i need to "):
        remainder = body[len("i need ") :].strip()
        if remainder:
            return f"Looking for {remainder}{punctuation}"
    return None


def _apply_constraint_rewrite(query: str) -> str | None:
    body, punctuation = _split_terminal_punctuation(query)
    rewritten = body
    substitution_count = 0
    pattern_replacements = (
        (r"\bbut not\s+(.+)$", r"without \1"),
        (r"\bbut I don't like\s+(.+)$", r"without \1"),
        (r"\bbut I do not like\s+(.+)$", r"without \1"),
        (r"\bthat don't contain\s+(.+)$", r"without \1"),
        (r"\bthat do not contain\s+(.+)$", r"without \1"),
        (r"\bthat doesn't contain\s+(.+)$", r"without \1"),
        (r"\bthat does not contain\s+(.+)$", r"without \1"),
    )
    for pattern, replacement in pattern_replacements:
        rewritten, replacements = re.subn(pattern, replacement, rewritten, flags=re.IGNORECASE)
        substitution_count += replacements
    if substitution_count == 0:
        return None
    return f"{rewritten}{punctuation}"


def _build_augmented_example(
    example: RecipeExample,
    *,
    query: str,
    strategy_name: str,
    variant_index: int,
) -> RecipeExample:
    return RecipeExample(
        example_id=f"{example.example_id}-aug-{variant_index:02d}",
        query=query,
        options=example.options,
        answer_option_id=example.answer_option_id,
        query_type_flags=dict(example.query_type_flags),
        correctness_explanation=dict(example.correctness_explanation),
        source_metadata={
            **dict(example.source_metadata),
            "parent_example_id": example.example_id,
            "augmentation_strategy": strategy_name,
            "variant_index": variant_index,
        },
    )


def augment_example(example: RecipeExample, *, max_variants: int = MAX_AUGMENTED_VARIANTS) -> tuple[RecipeExample, ...]:
    if max_variants < 1 or max_variants > MAX_AUGMENTED_VARIANTS:
        raise DatasetValidationError(
            f"max_variants must be between 1 and {MAX_AUGMENTED_VARIANTS}"
        )

    candidate_queries: list[tuple[str, str]] = []
    lead_in_query = _apply_lead_in_rewrite(example.query)
    if lead_in_query is not None:
        candidate_queries.append(("lead_in_rewrite", lead_in_query))

    constraint_query = _apply_constraint_rewrite(example.query)
    if constraint_query is not None:
        candidate_queries.append(("constraint_rewrite", constraint_query))

    if lead_in_query is not None:
        combined_query = _apply_constraint_rewrite(lead_in_query)
        if combined_query is not None:
            candidate_queries.append(("lead_in_plus_constraint_rewrite", combined_query))

    seen_queries = {_normalize_query_key(example.query)}
    augmented_examples: list[RecipeExample] = []
    for strategy_name, candidate_query in candidate_queries:
        normalized_candidate = _normalize_query_key(candidate_query)
        if normalized_candidate in seen_queries:
            continue
        seen_queries.add(normalized_candidate)
        augmented_examples.append(
            _build_augmented_example(
                example,
                query=candidate_query,
                strategy_name=strategy_name,
                variant_index=len(augmented_examples) + 1,
            )
        )
        if len(augmented_examples) >= max_variants:
            break

    return tuple(augmented_examples)


def augment_training_examples(
    examples: Sequence[RecipeExample],
    *,
    max_variants: int = MAX_AUGMENTED_VARIANTS,
) -> tuple[RecipeExample, ...]:
    augmented_examples: list[RecipeExample] = []
    for example in examples:
        augmented_examples.extend(augment_example(example, max_variants=max_variants))
    return tuple(augmented_examples)


def count_augmentation_strategies(examples: Sequence[RecipeExample]) -> dict[str, int]:
    strategy_counts: dict[str, int] = {}
    for example in examples:
        strategy_name = str(example.source_metadata.get("augmentation_strategy", "unknown"))
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    return dict(sorted(strategy_counts.items()))
