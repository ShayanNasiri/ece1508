from recipe_mpr_qa.formats import (
    DEFAULT_PROMPT_SPEC,
    LETTER_MAP as _LETTER_MAP,
    build_multiple_choice_prompt,
    parse_multiple_choice_response,
)

MC_TEMPLATE = DEFAULT_PROMPT_SPEC.template
LETTER_MAP = list(_LETTER_MAP)


def build_mc_prompt(query, options_dict):
    """Build a multiple-choice prompt from a query and options dict."""
    return build_multiple_choice_prompt(
        query=query,
        options=options_dict,
        prompt_spec=DEFAULT_PROMPT_SPEC,
    )


def parse_mc_response(response_text):
    """Extract a letter A-E from model output."""
    return parse_multiple_choice_response(response_text)
