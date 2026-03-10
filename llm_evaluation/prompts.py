import re

MC_TEMPLATE = """Given the following recipe preference query, select the best matching recipe.

Query: {query}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Respond with only the letter (A-E) of the best option."""

LETTER_MAP = ["A", "B", "C", "D", "E"]


def build_mc_prompt(query, options_dict):
    """Build a multiple-choice prompt from a query and options dict.

    Args:
        query: The recipe preference query string.
        options_dict: dict mapping recipe_id -> description (5 entries).

    Returns:
        (prompt_str, letter_to_id) where letter_to_id maps 'A'-'E' to recipe IDs.
    """
    ids = list(options_dict.keys())
    letter_to_id = {LETTER_MAP[i]: ids[i] for i in range(len(ids))}

    prompt = MC_TEMPLATE.format(
        query=query,
        option_a=options_dict[ids[0]],
        option_b=options_dict[ids[1]],
        option_c=options_dict[ids[2]],
        option_d=options_dict[ids[3]],
        option_e=options_dict[ids[4]],
    )
    return prompt, letter_to_id


def parse_mc_response(response_text):
    """Extract a letter A-E from model output.

    Returns the letter (uppercase) or None if no valid letter found.
    """
    text = response_text.strip().upper()

    if text in LETTER_MAP:
        return text

    patterns = [
        r"\b([A-E])\)",
        r"\b([A-E])\.",
        r"\(([A-E])\)",
        r"answer\s*[:is]*\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None
