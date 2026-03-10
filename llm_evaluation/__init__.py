from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_evaluation.ollama_client import OllamaClient
from llm_evaluation.prompts import LETTER_MAP, MC_TEMPLATE, build_mc_prompt, parse_mc_response

__all__ = [
    "LETTER_MAP",
    "MC_TEMPLATE",
    "OllamaClient",
    "build_mc_prompt",
    "parse_mc_response",
]
