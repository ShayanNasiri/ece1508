import time


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api/generate", max_retries=3, retry_delay=2):
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def query(self, model_name, prompt, temperature=0):
        """Send a prompt to an Ollama model and return the response text."""
        import requests

        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.base_url, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()["response"]
            except (requests.RequestException, KeyError) as e:
                if attempt < self.max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Ollama query failed after {self.max_retries} attempts: {e}"
                    )
