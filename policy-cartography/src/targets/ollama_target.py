import os
import time
import requests
from typing import Optional


class OllamaTarget:
    """
    Minimal, robust Ollama HTTP client for Promptfoo / agent loops.

    Goals:
    - Never crash the whole run on a slow model
    - Be easy to debug
    - Keep behavior obvious
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_s: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")

        # Allow override via env, default = 600s (10 min)
        if timeout_s is None:
            timeout_s = int(os.environ.get("OLLAMA_TIMEOUT_S", "600"))

        self.timeout_s = timeout_s

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Single Ollama generation call.

        - Retries once on timeout
        - Raises only on *non-timeout* fatal errors
        """

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        last_error = None

        for attempt in range(2):  # retry once
            try:
                r = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout_s,
                )
                r.raise_for_status()
                data = r.json()

                # Ollama returns {"response": "..."}
                return data.get("response", "")

            except requests.exceptions.ReadTimeout as e:
                last_error = e
                if attempt == 0:
                    # brief pause before retry
                    time.sleep(2.0)
                    continue
                break

            except Exception:
                # Non-timeout error → bubble up
                raise

        # If we get here: timeout twice
        return (
            "[ERROR] Ollama request timed out twice. "
            "Model did not return a response."
        )
