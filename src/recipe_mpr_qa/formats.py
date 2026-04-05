from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.models import RecipeOption

LETTER_MAP = ("A", "B", "C", "D", "E")
OPTION_ORDER_SHUFFLE_SEED = 1508


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
        shuffled_option_items = list(option_items)
        random.Random(derived_seed).shuffle(shuffled_option_items)
        option_items = shuffled_option_items
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


_EXPLICIT_PATTERNS = (
    r"\\BOXED\{([A-E])\}",
    r"\b(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER|BEST\s+OPTION|BEST\s+MATCHING\s+RECIPE|PREDICTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-E])\)?[\)\.\:\-]?(?=\s|$)",
    r"\b(?:I\s+CHOOSE|I\s+PICK|I\s+SELECT|CHOOSE|PICK|SELECT)\s+(?:OPTION\s*)?\(?([A-E])\)?(?=\b)",
    r"\b(?:MY\s+ANSWER|MY\s+CHOICE)\s*(?:IS|:)\s*\(?([A-E])\)?[\)\.\:\-]?(?=\s|$)",
)
_MIN_WORD_OVERLAP = 0.5


def _try_exact_match(text: str) -> str | None:
    """Single uppercase letter, optionally wrapped in parens or followed by punctuation."""
    m = re.fullmatch(r"\(?([A-E])\)?[\)\.\:\-]?", text)
    return m.group(1) if m else None


def _try_option_prefix(text: str) -> str | None:
    """'Option A' or 'OPTION A)' style response."""
    m = re.fullmatch(r"OPTION\s*([A-E])[\)\.\:\-]?", text)
    return m.group(1) if m else None


def _try_explicit_patterns(text: str) -> str | None:
    """Explicit declarations: 'The answer is A', '\\boxed{A}', 'I choose B', etc."""
    latest: tuple[int, str] | None = None
    for pattern in _EXPLICIT_PATTERNS:
        for match in re.finditer(pattern, text):
            if latest is None or match.end() > latest[0]:
                latest = (match.end(), match.group(1))
    return latest[1] if latest is not None else None


def _try_tail_match(text: str) -> str | None:
    """Letter at the end of the response, not preceded by 'Option/Choice'."""
    m = re.search(r"(?:^|[\s>])\(?([A-E])\)?[\)\.\:\-]?\s*$", text)
    if m and not re.search(r"(?:OPTION|CHOICE)\s*$", text[:m.start(1)]):
        return m.group(1)
    return None


def _try_leading_match(text: str) -> str | None:
    """Letter at the start of the response, only when few other letters are mentioned."""
    mentioned = {
        match.group(1) or match.group(2)
        for match in re.finditer(r"OPTION\s+([A-E])\b|(?:^|[\s])([A-E])\)", text)
    }
    if len(mentioned) >= 3:
        return None
    m = re.match(r"\s*\(?([A-E])\)?[\)\.\:\-]\s+\S", text)
    return m.group(1) if m else None


def _try_word_overlap(text: str, options: dict[str, str]) -> str | None:
    """Fuzzy match 'best option is <description>' against option text by word overlap."""
    desc_match = None
    for match in re.finditer(r"BEST OPTION IS\s+(.+?)(?:\.|##|\Z)", text):
        desc_match = match
    if desc_match is None:
        return None
    candidate = desc_match.group(1).strip()
    best_letter: str | None = None
    best_score = 0.0
    for letter, option_text in options.items():
        words = [w for w in option_text.upper().split() if len(w) > 3]
        if not words:
            continue
        score = sum(1 for w in words if w in candidate) / len(words)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter if best_score >= _MIN_WORD_OVERLAP else None


def parse_multiple_choice_response(
    response_text: str,
    options: dict[str, str] | None = None,
) -> str | None:
    text = response_text.strip().upper()
    if not text:
        return None
    return (
        _try_exact_match(text)
        or _try_option_prefix(text)
        or _try_explicit_patterns(text)
        or _try_tail_match(text)
        or _try_leading_match(text)
        or (options and _try_word_overlap(text, options))
        or None
    )


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    phase: str
    provider: str
    model_name: str
    split: str
    example_id: str
    prompt_version: str
    raw_response: str
    parsed_choice: str | None
    predicted_option_id: str | None
    gold_option_id: str
    is_correct: bool
    latency_ms: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "provider": self.provider,
            "model_name": self.model_name,
            "split": self.split,
            "example_id": self.example_id,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "parsed_choice": self.parsed_choice,
            "predicted_option_id": self.predicted_option_id,
            "gold_option_id": self.gold_option_id,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionRecord":
        return cls(
            run_id=payload["run_id"],
            phase=payload["phase"],
            provider=payload["provider"],
            model_name=payload["model_name"],
            split=payload["split"],
            example_id=payload["example_id"],
            prompt_version=payload["prompt_version"],
            raw_response=payload["raw_response"],
            parsed_choice=payload.get("parsed_choice"),
            predicted_option_id=payload.get("predicted_option_id"),
            gold_option_id=payload["gold_option_id"],
            is_correct=bool(payload["is_correct"]),
            latency_ms=payload.get("latency_ms"),
            metadata=payload.get("metadata", {}),
        )


def write_prediction_records(
    records: list[PredictionRecord] | tuple[PredictionRecord, ...], output_path: Path | str
) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":")) for record in records]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prediction_records(input_path: Path | str) -> tuple[PredictionRecord, ...]:
    resolved_path = Path(input_path)
    lines = [line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(PredictionRecord.from_dict(json.loads(line)) for line in lines)
