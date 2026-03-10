from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434/api/generate"
    max_retries: int = 3
    retry_delay: float = 2.0

    def query(self, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install recipe-mpr-qa[llm] to use OllamaClient") from exc

        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.base_url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["response"]
            except (requests.RequestException, KeyError) as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Ollama query failed after {self.max_retries} attempts: {exc}"
                    ) from exc
        raise RuntimeError("Ollama query failed unexpectedly")

    def generate(self, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        return self.query(model_name=model_name, prompt=prompt, temperature=temperature)
