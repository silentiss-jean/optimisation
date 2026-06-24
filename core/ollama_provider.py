import requests
import json
from typing import Generator, List, Dict
from .base_provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Génère un stream de tokens depuis /api/generate.
        Utilise requests en mode stream=True et iter_lines() pour un vrai streaming token par token.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        # FIX 1 : stream=True dans requests.post + FIX 2 : timeout=120
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        # FIX 1 : iter_lines() lit ligne par ligne au fil de l'eau (vrai streaming)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")
                if data.get("done", False):
                    break

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse complète (non streaming) depuis /api/generate.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        # FIX 2 : timeout=120
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> str:
        """
        FIX 3 : Méthode chat() utilisant l'endpoint /api/chat.
        Adapté aux conversations multi-tours via une liste de messages.

        Args:
            messages : liste de dicts au format [{"role": "user"|"assistant"|"system", "content": "..."}]
            stream   : si True, retourne la réponse concaténée token par token (bloquant)
            **kwargs : paramètres supplémentaires (temperature, top_p, etc.)

        Returns:
            La réponse complète de l'assistant sous forme de chaîne.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        # FIX 2 : timeout=120
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=120
        )
        response.raise_for_status()

        if stream:
            # Lire le flux ligne par ligne et concaténer le contenu
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    full_response += content
                    if data.get("done", False):
                        break
            return full_response
        else:
            return response.json().get("message", {}).get("content", "")
