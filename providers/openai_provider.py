import requests
from typing import Generator, Optional, List, Dict
# Assurez-vous que l'interface est accessible
from .llm_provider_interface import LLMProvider

class OpenAIProvider(LLMProvider):
    """
    Implementation of LLMProvider using the official OpenAI API (Remote access).
    Requires an API Key set for authorization. This handles services like GPT-4, Claude via Anthropic/etc.,
    si le modèle supporte l'API Chat Completion standard.
    """
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        # Pour les API externes, nous utilisons généralement le base_url de l'API cible (ex: OpenAI)
        super().__init__(api_key=api_key, base_url="https://api.openai.com/v1")
        self.model = model

    def generate_response(self, prompt: str, history: list) -> str:
        """
        Generates a complete response from the OpenAI Chat Completion API using the standardized chat format.
        """
        messages = []
        # Le dernier élément est le prompt de l'utilisateur qui déclenche l'action
        messages.append({"role": "user", "content": prompt})

        # L'histoire passée doit suivre le format OpenAI: list of {"role": str, "content": str}
        if history:
             for msg in history:
                 messages.append(msg)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

        try:
            # Appel POST à l'endpoint 'chat/completions'
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
            data = response.json()
            return data['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            # Tentative de donner un message plus précis à l'utilisateur sur l'échec réseau/API
            if "401" in str(e) or "Authentication" in str(e):
                 return f"[ERROR] Échec d'authentification OpenAI. Vérifiez votre clé API (Authorization header)."
            return f"[ERROR] Impossible de se connecter à l'API OpenAI ou de générer une réponse : {e}"

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        """
        Streams the response from the official OpenAI API endpoint.
        """
        messages = []
        messages.append({"role": "user", "content": prompt})

        if history:
             for msg in history:
                 messages.append(msg)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        print("--- Démarrage du streaming de réponse depuis l'API OpenAI ---")
        try:
            # Appel POST à l'endpoint 'chat/completions' avec stream=True
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, stream=True)
            response.raise_for_status()

            full_text = ""
            # Parcourir chaque chunk de données reçu du flux
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        content_chunk = data['choices'][0]['delta']['content']
                        if content_chunk is not None:
                            yield content_chunk # Rend immédiatement le morceau de texte traité
                            full_text += content_chunk

                    except Exception as e:
                        # Ceci attrape les lignes mal formatées ou les erreurs JSON
                        print(f"Avertissement lors du traitement d'un chunk OpenAI : {e}")

        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Impossible de se connecter à l'API OpenAI en mode streaming ({e}). Vérifiez votre clé API et la connectivité."
        finally:
             print("--- Fin du streaming OpenAI ---")