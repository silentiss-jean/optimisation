import requests
from typing import Generator, Optional
from providers.llm_provider_interface import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Implementation of LLMProvider using the Ollama REST API.
    Assumes local availability at localhost:11434.
    """
    def __init__(self, model: str = "qwen3:8b", api_key: Optional[str] = None):
        super().__init__(api_key=None, base_url="http://localhost:11434/api")
        self.model = model

    def generate_response(self, prompt: str, history: list) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            }
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.RequestException as e:
            return f"[ERROR] Impossible de se connecter à Ollama : {e}"

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        print("--- Démarrage du streaming Ollama ---")
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True
            }
            response = requests.post(f"{self.base_url}/generate", json=payload, stream=True)
            response.raise_for_status()

            import json
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if isinstance(data, dict) and data.get("response"):
                            yield data["response"]
                    except Exception as e:
                        print(f"Avertissement chunk Ollama : {e}")

        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Connexion Ollama impossible ({e})."
        finally:
            print("--- Fin du streaming Ollama ---")
