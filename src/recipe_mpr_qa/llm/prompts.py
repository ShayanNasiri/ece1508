from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from recipe_mpr_qa.data.models import RecipeOption

LETTER_MAP = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class PromptSpec:
    version: str
    template: str

    def render(
        self, *, query: str, options: Sequence[RecipeOption] | Mapping[str, str]
    ) -> tuple[str, dict[str, str]]:
        return build_multiple_choice_prompt(query=query, options=options, prompt_spec=self)


DEFAULT_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-mc-v1",
    template=(
        "Given the following recipe preference query, select the best matching recipe.\n\n"
        "Query: {query}\n\n"
        "Options:\n"
        "A) {option_a}\n"
        "B) {option_b}\n"
        "C) {option_c}\n"
        "D) {option_d}\n"
        "E) {option_e}\n\n"
        "Respond with only the letter (A-E) of the best option."
    ),
)


def build_multiple_choice_prompt(
    *,
    query: str,
    options: Sequence[RecipeOption] | Mapping[str, str],
    prompt_spec: PromptSpec = DEFAULT_PROMPT_SPEC,
) -> tuple[str, dict[str, str]]:
    if isinstance(options, Mapping):
        option_items = list(options.items())
    else:
        option_items = [(option.option_id, option.text) for option in options]
    if len(option_items) != 5:
        raise ValueError(f"Expected exactly 5 options, got {len(option_items)}")
    letter_to_option_id = {
        LETTER_MAP[index]: option_items[index][0] for index in range(len(LETTER_MAP))
    }
    prompt = prompt_spec.template.format(
        query=query,
        option_a=option_items[0][1],
        option_b=option_items[1][1],
        option_c=option_items[2][1],
        option_d=option_items[3][1],
        option_e=option_items[4][1],
    )
    return prompt, letter_to_option_id


def parse_multiple_choice_response(response_text: str) -> str | None:
    text = response_text.strip().upper()
    if not text:
        return None
    if text in LETTER_MAP:
        return text
    patterns = (
        r"^\s*([A-E])[\)\.\:\-]?\s*$",
        r"\(([A-E])\)",
        r"ANSWER\s*(?:IS|:)?\s*([A-E])\b",
        r"OPTION\s*([A-E])\b",
        r"\b([A-E])\)",
        r"\b([A-E])\.",
        r"\b([A-E])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None
