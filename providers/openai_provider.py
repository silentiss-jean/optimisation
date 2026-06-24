import json
import requests
from typing import Generator, Optional
from providers.llm_provider_interface import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    Implementation of LLMProvider using the OpenAI Chat Completion API.
    Compatible avec OpenAI et OpenRouter (via base_url).
    """
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key=api_key, base_url=base_url)
        self.model = model

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
            if "401" in str(e):
                return "[ERROR] Clé API invalide ou manquante."
            return f"[ERROR] Connexion OpenAI impossible : {e}"

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        messages = list(history) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "stream": True}
        print("--- Démarrage du streaming OpenAI ---")
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
                        print(f"Avertissement chunk OpenAI : {e}")
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Connexion OpenAI impossible ({e})."
        finally:
            print("--- Fin du streaming OpenAI ---")
