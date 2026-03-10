from recipe_mpr_qa.llm.ollama import OllamaClient
from recipe_mpr_qa.llm.prompts import (
    DEFAULT_PROMPT_SPEC,
    LETTER_MAP,
    PromptSpec,
    build_multiple_choice_prompt,
    parse_multiple_choice_response,
)

__all__ = [
    "DEFAULT_PROMPT_SPEC",
    "LETTER_MAP",
    "OllamaClient",
    "PromptSpec",
    "build_multiple_choice_prompt",
    "parse_multiple_choice_response",
]
from recipe_mpr_qa.llm.inference import run_llm_predictions
from recipe_mpr_qa.llm.judge import judge_predictions
from recipe_mpr_qa.llm.ollama import OllamaClient
from recipe_mpr_qa.llm.prompts import (
    AUGMENTATION_PROMPT_SPEC,
    CAUSAL_SLM_PROMPT_SPEC,
    DEFAULT_PROMPT_SPEC,
    JUDGE_PROMPT_SPEC,
    PromptSpec,
    build_augmentation_prompt,
    build_causal_multiple_choice_prompt,
    build_judge_prompt,
    build_multiple_choice_prompt,
    parse_augmentation_response,
    parse_judge_response,
    parse_multiple_choice_response,
)

__all__ = [
    "AUGMENTATION_PROMPT_SPEC",
    "CAUSAL_SLM_PROMPT_SPEC",
    "DEFAULT_PROMPT_SPEC",
    "JUDGE_PROMPT_SPEC",
    "OllamaClient",
    "PromptSpec",
    "build_augmentation_prompt",
    "build_causal_multiple_choice_prompt",
    "build_judge_prompt",
    "build_multiple_choice_prompt",
    "judge_predictions",
    "parse_augmentation_response",
    "parse_judge_response",
    "parse_multiple_choice_response",
    "run_llm_predictions",
]
