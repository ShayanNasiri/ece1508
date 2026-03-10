from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recipe_mpr_qa.llm_evaluation.ollama_client import OllamaClient

__all__ = ["OllamaClient"]
