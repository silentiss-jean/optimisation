import json
import requests
from typing import Generator, Optional
from providers.llm_provider_interface import LLMProvider

# Documentation : https://docs.perplexity.ai/docs/sonar/quickstart
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# Modèles Sonar disponibles (recherche web intégrée)
AVAILABLE_MODELS = [
    "sonar",                  # Modèle léger avec recherche web
    "sonar-pro",              # Modèle pro avec recherche web avancée
    "sonar-reasoning",        # Reasoning + recherche web
    "sonar-reasoning-pro",    # Reasoning pro + recherche web
    "sonar-deep-research",    # Recherche approfondie multi-étapes
]


class PerplexityProvider(LLMProvider):
    """
    Provider pour l'API Perplexity (Sonar API).
    Compatible OpenAI Chat Completions — recherche web intégrée dans chaque réponse.
    Endpoint : https://api.perplexity.ai/chat/completions
    Clé API   : https://console.perplexity.ai → API Keys
    """
    def __init__(self, model: str = "sonar", api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=PERPLEXITY_BASE_URL)
        self.model = model

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_response(self, prompt: str, history: list) -> str:
        if not self.api_key:
            return "[ERROR] Clé API Perplexity manquante. Obtenez-la sur https://console.perplexity.ai"

        messages = list(history) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Ajouter les citations si présentes
            citations = data.get("citations", [])
            if citations:
                content += "\n\n📚 Sources :"
                for i, url in enumerate(citations[:5], 1):
                    content += f"\n  [{i}] {url}"

            return content

        except requests.exceptions.HTTPError as e:
            if "401" in str(e):
                return "[ERROR] Clé API Perplexity invalide. Vérifiez sur https://console.perplexity.ai"
            return f"[ERROR] API Perplexity : {e}"
        except requests.exceptions.RequestException as e:
            return f"[ERROR] Connexion Perplexity impossible : {e}"

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        if not self.api_key:
            yield "[ERROR] Clé API Perplexity manquante. Obtenez-la sur https://console.perplexity.ai"
            return

        messages = list(history) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }

        print(f"--- Démarrage streaming Perplexity ({self.model}) ---")
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        chunk = data["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk
                    except Exception as e:
                        print(f"Avertissement chunk Perplexity : {e}")

        except requests.exceptions.HTTPError as e:
            yield f"[ERROR] API Perplexity : {e}"
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Connexion Perplexity impossible : {e}"
        finally:
            print("--- Fin streaming Perplexity ---")
