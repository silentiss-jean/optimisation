import json
import requests
from typing import Generator, Optional
from providers.llm_provider_interface import LLMProvider


class LMStudioProvider(LLMProvider):
    """
    Implementation of LLMProvider using an OpenAI-compatible API (LM Studio).
    Assumes local availability at localhost:1234/v1.
    """
    def __init__(self, model: str = "local-model", api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url="http://localhost:1234/v1")
        self.model = model

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "Bearer lm-studio"
        }

    def generate_response(self, prompt: str, history: list) -> str:
        messages = list(history) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "temperature": 0.7}
        try:
            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"[ERROR] Connexion LM Studio impossible : {e}"

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        messages = list(history) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "stream": True}
        print("--- Démarrage du streaming LM Studio ---")
        try:
            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=self._headers(), json=payload, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    raw = line.decode('utf-8')
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)['choices'][0]['delta'].get('content')
                        if chunk:
                            yield chunk
                    except Exception as e:
                        print(f"Avertissement chunk LM Studio : {e}")
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Connexion LM Studio impossible ({e})."
        finally:
            print("--- Fin du streaming LM Studio ---")
