import logging
import requests
import time

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api/generate", max_retries=3, retry_delay=2, timeout=120):
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def query(self, model_name, prompt, temperature=0):
        """Send a prompt to an Ollama model and return the response text."""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %ss...",
                        attempt + 1, self.max_retries, e, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(
                    f"Ollama request failed after {self.max_retries} attempts: {e}"
                ) from e

            data = resp.json()
            if "response" not in data:
                raise RuntimeError(
                    f"Ollama response missing 'response' key. Got keys: {sorted(data.keys())}"
                )
            return data["response"]
