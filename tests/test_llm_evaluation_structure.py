from __future__ import annotations

from llm_evaluation.ollama_client import OllamaClient as LegacyOllamaClient
from llm_evaluation.prompts import build_mc_prompt as legacy_build_mc_prompt
from llm_evaluation.prompts import parse_mc_response as legacy_parse_mc_response
from recipe_mpr_qa.llm_evaluation.ollama_client import OllamaClient as PackagedOllamaClient
from recipe_mpr_qa.llm_evaluation.prompts import (
    build_mc_prompt as packaged_build_mc_prompt,
)
from recipe_mpr_qa.llm_evaluation.prompts import (
    parse_mc_response as packaged_parse_mc_response,
)


def test_packaged_llm_evaluation_helpers_match_legacy_wrappers() -> None:
    options = {
        "id-a": "Option A",
        "id-b": "Option B",
        "id-c": "Option C",
        "id-d": "Option D",
        "id-e": "Option E",
    }

    legacy_prompt, legacy_mapping = legacy_build_mc_prompt("Need a roasted fish recipe", options)
    packaged_prompt, packaged_mapping = packaged_build_mc_prompt(
        "Need a roasted fish recipe", options
    )

    assert legacy_prompt == packaged_prompt
    assert legacy_mapping == packaged_mapping
    assert legacy_parse_mc_response("Answer: C") == packaged_parse_mc_response("Answer: C") == "C"
    assert LegacyOllamaClient is PackagedOllamaClient
