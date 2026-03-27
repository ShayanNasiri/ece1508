from __future__ import annotations

import time

import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/api/generate",
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def query(self, model_name: str, prompt: str, temperature: float = 0) -> str:
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
                    print(
                        f"  Attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Ollama query failed after {self.max_retries} attempts: {exc}"
                    ) from exc
