from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.benchmark.provenance import BENCHMARK_CONTRACT_VERSION
from recipe_mpr_qa.data.models import RecipeExample, RecipeOption

LETTER_MAP = ("A", "B", "C", "D", "E")
OPTION_ORDER_SHUFFLE_SEED = 1508
DEFAULT_PARSER_VERSION = "recipe-mpr-mc-parser-v2"


@dataclass(frozen=True)
class PromptSpec:
    version: str
    template: str

    def render(
        self,
        *,
        query: str,
        options: Sequence[RecipeOption] | Mapping[str, str],
        shuffle_key: str | None = None,
        shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
    ) -> tuple[str, dict[str, str]]:
        return build_multiple_choice_prompt(
            query=query,
            options=options,
            prompt_spec=self,
            shuffle_key=shuffle_key,
            shuffle_seed=shuffle_seed,
        )


DEFAULT_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-mc-v2",
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

AUGMENTATION_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-augmentation-v1",
    template=(
        "You are generating paraphrased recipe preference queries for data augmentation.\n"
        "Return valid JSON only with the shape "
        '{{"rewrites":["query one","query two"]}}.\n'
        "Keep the meaning, dietary constraints, ingredient preferences, and answer choice unchanged.\n"
        "Do not mention option ids or change the number of requested rewrites.\n\n"
        "Original query: {query}\n"
        "Requested rewrites: {requested_count}\n"
        "Correct option text: {gold_option}\n"
        "Distractor options:\n"
        "{distractors}\n"
    ),
)

JUDGE_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-judge-v1",
    template=(
        "You are an expert evaluator for recipe recommendation quality.\n"
        "Return valid JSON only with keys "
        '"ingredient_alignment", "constraint_satisfaction", "reasoning_quality", '
        '"overall_verdict", and "rationale".\n'
        "Each score must be an integer from 1 to 5. "
        'overall_verdict must be one of "correct", "partially_correct", or "incorrect".\n\n'
        "Query: {query}\n"
        "Predicted option: {predicted_option}\n"
        "Gold option: {gold_option}\n"
        "Gold evidence: {gold_evidence}\n"
        "Model rationale: {model_rationale}\n"
    ),
)

CAUSAL_SLM_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-chat-mc-v2",
    template=(
        "User request: {query}\n\n"
        "Options:\n"
        "A. {option_a}\n"
        "B. {option_b}\n"
        "C. {option_c}\n"
        "D. {option_d}\n"
        "E. {option_e}\n\n"
        "Reply with only one letter: A, B, C, D, or E."
    ),
)


@dataclass(frozen=True)
class ParseResult:
    parsed_choice: str | None
    status: str
    parser_version: str = DEFAULT_PARSER_VERSION
    candidate_letters: tuple[str, ...] = ()


def build_multiple_choice_prompt(
    *,
    query: str,
    options: Sequence[RecipeOption] | Mapping[str, str],
    prompt_spec: PromptSpec = DEFAULT_PROMPT_SPEC,
    shuffle_key: str | None = None,
    shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
) -> tuple[str, dict[str, str]]:
    if isinstance(options, Mapping):
        option_items = list(options.items())
    else:
        option_items = [(option.option_id, option.text) for option in options]
    if len(option_items) != 5:
        raise ValueError(f"Expected exactly 5 options, got {len(option_items)}")
    if shuffle_key is not None:
        derived_seed = int(
            hashlib.sha256(f"{shuffle_seed}:{shuffle_key}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        shuffled = list(option_items)
        random.Random(derived_seed).shuffle(shuffled)
        option_items = shuffled
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


def build_causal_multiple_choice_prompt(
    *,
    query: str,
    options: Sequence[RecipeOption] | Mapping[str, str],
    prompt_spec: PromptSpec = CAUSAL_SLM_PROMPT_SPEC,
    shuffle_key: str | None = None,
    shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
) -> tuple[str, dict[str, str]]:
    return build_multiple_choice_prompt(
        query=query,
        options=options,
        prompt_spec=prompt_spec,
        shuffle_key=shuffle_key,
        shuffle_seed=shuffle_seed,
    )


def parse_multiple_choice_response(response_text: str) -> str | None:
    return parse_multiple_choice_response_detail(response_text).parsed_choice


def parse_multiple_choice_response_detail(response_text: str) -> ParseResult:
    text = response_text.strip().upper()
    if not text:
        return ParseResult(parsed_choice=None, status="empty")

    exact_match = re.fullmatch(r"\(?([A-E])\)?[\)\.\:\-]?", text)
    if exact_match:
        return ParseResult(parsed_choice=exact_match.group(1), status="parsed")

    disjunctive_letters = tuple(
        sorted({match.group(1) for match in re.finditer(r"\b([A-E])\b", text)})
    )
    if len(disjunctive_letters) > 1 and (re.search(r"\b(?:OR|AND)\b", text) or "/" in text):
        return ParseResult(parsed_choice=None, status="ambiguous", candidate_letters=disjunctive_letters)

    candidates: list[str] = []
    explicit_patterns = (
        r"\b(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER|BEST\s+OPTION|BEST\s+MATCH|PREDICTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-E])\)?",
        r"\b(?:I\s+CHOOSE|I\s+PICK|I\s+SELECT|CHOOSE|PICK|SELECT)\s+(?:OPTION\s*)?\(?([A-E])\)?",
        r"\bOPTION\s+([A-E])\b",
        r"\(([A-E])\)",
        r"\b([A-E])\)",
        r"\b([A-E])\.",
        r"^\s*([A-E])[\)\.\:\-]?\s+\S",
    )
    for pattern in explicit_patterns:
        candidates.extend(match.group(1) for match in re.finditer(pattern, text))
    distinct_candidates = tuple(sorted(set(candidates)))
    if len(distinct_candidates) == 1:
        return ParseResult(parsed_choice=distinct_candidates[0], status="parsed", candidate_letters=distinct_candidates)
    if len(distinct_candidates) > 1:
        return ParseResult(parsed_choice=None, status="ambiguous", candidate_letters=distinct_candidates)

    standalone_letters = tuple(
        sorted(
            {
                match.group(1)
                for match in re.finditer(r"(?:^|[\s>])\(?([A-E])\)?(?:[\)\.\:\-]|\s|$)", text)
            }
        )
    )
    if len(standalone_letters) == 1 and text.count(standalone_letters[0]) == 1:
        return ParseResult(parsed_choice=standalone_letters[0], status="parsed", candidate_letters=standalone_letters)
    if len(standalone_letters) > 1:
        return ParseResult(parsed_choice=None, status="ambiguous", candidate_letters=standalone_letters)
    return ParseResult(parsed_choice=None, status="invalid")


def benchmark_prompt_metadata(
    *,
    example_id: str,
    prompt_version: str,
    shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
) -> dict[str, Any]:
    return {
        "contract_version": BENCHMARK_CONTRACT_VERSION,
        "parser_version": DEFAULT_PARSER_VERSION,
        "shuffle_key": f"{prompt_version}:{example_id}",
        "shuffle_seed": shuffle_seed,
    }


def build_augmentation_prompt(
    example: RecipeExample,
    *,
    requested_count: int,
    prompt_spec: PromptSpec = AUGMENTATION_PROMPT_SPEC,
) -> str:
    gold_option = next(
        option.text for option in example.options if option.option_id == example.answer_option_id
    )
    distractors = "\n".join(
        f"- {option.text}"
        for option in example.options
        if option.option_id != example.answer_option_id
    )
    return prompt_spec.template.format(
        query=example.query,
        requested_count=requested_count,
        gold_option=gold_option,
        distractors=distractors,
    )


def parse_augmentation_response(response_text: str) -> tuple[str, ...]:
    payload = _extract_json_payload(response_text)
    rewrites = payload.get("rewrites") or payload.get("queries")
    if not isinstance(rewrites, list) or not rewrites:
        raise ValueError("Augmentation response must contain a non-empty rewrites list")
    parsed = []
    for item in rewrites:
        if isinstance(item, str) and item.strip():
            parsed.append(item.strip())
        elif isinstance(item, Mapping):
            query = item.get("query")
            if isinstance(query, str) and query.strip():
                parsed.append(query.strip())
            else:
                raise ValueError("Augmentation rewrite objects must contain a non-empty query")
        else:
            raise ValueError("Augmentation rewrites must be strings or objects with query")
    return tuple(parsed)


def build_judge_prompt(
    *,
    example: RecipeExample,
    predicted_option_text: str,
    model_rationale: str | None,
    prompt_spec: PromptSpec = JUDGE_PROMPT_SPEC,
) -> str:
    gold_option = next(
        option.text for option in example.options if option.option_id == example.answer_option_id
    )
    return prompt_spec.template.format(
        query=example.query,
        predicted_option=predicted_option_text,
        gold_option=gold_option,
        gold_evidence=json.dumps(example.correctness_explanation, ensure_ascii=True),
        model_rationale=model_rationale or "",
    )


def parse_judge_response(response_text: str) -> dict[str, Any]:
    payload = _extract_json_payload(response_text)
    required = {
        "ingredient_alignment",
        "constraint_satisfaction",
        "reasoning_quality",
        "overall_verdict",
        "rationale",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Judge response missing keys: {sorted(missing)}")
    verdict = payload["overall_verdict"]
    if verdict not in {"correct", "partially_correct", "incorrect"}:
        raise ValueError(f"Invalid overall_verdict: {verdict!r}")
    parsed: dict[str, Any] = {"overall_verdict": verdict}
    for key in ("ingredient_alignment", "constraint_satisfaction", "reasoning_quality"):
        value = payload[key]
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"{key} must be an integer in [1, 5]")
        parsed[key] = value
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    parsed["rationale"] = rationale.strip()
    return parsed


def _extract_json_payload(response_text: str) -> Mapping[str, Any]:
    text = response_text.strip()
    if not text:
        raise ValueError("Response text is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Response did not contain JSON") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, Mapping):
        raise ValueError("Expected JSON object response")
    return payload
