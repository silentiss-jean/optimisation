import requests
from typing import Generator, Optional, List, Dict
# Assurez-vous que l'interface est accessible
from .llm_provider_interface import LLMProvider

class LMStudioProvider(LLMProvider):
    """
    Implementation of LLMProvider using an OpenAI-compatible API (e.g., running via LM Studio).
    Assumes local availability at localhost:1234/v1.
    """
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        # Utilisation du même pattern que OllamaProvider mais adapté à l'endpoint OpenAI /v1
        super().__init__(api_key=api_key, base_url="http://localhost:1234/v1")
        self.model = model

    def generate_response(self, prompt: str, history: list) -> str:
        """
        Generates a complete response from OpenAI-compatible API using the 'chat' endpoint.
        The conversation history is modeled as per OpenAI's message format.
        """
        # Structuration de l'historique pour le format chat/openai
        messages = []
        # Le premier prompt est souvent traité comme la dernière interaction utilisateur
        messages.append({"role": "user", "content": prompt})
        # Ici, on doit retransmettre toute l'histoire du contexte (si elle existe)
        # Pour un modèle simple, nous allons traiter 'history' comme une liste de messages au format OpenAI
        # Note: Si history est déjà dans le bon format JSONList[Dict], il suffit de les ajouter.
        if history:
             for msg in history:
                 # Assurez-vous que l'historique respecte un format (role, content)
                 messages.append({"role": "user", "content": msg})

        headers = {
            "Content-Type": "application/json",
            # L'API key est nécessaire même si elle est locale et ne fait pas d'appel externe réel
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

        try:
            # Appel POST à l'endpoint 'chat/completions'
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion à LM Studio/OpenAI lors de generate_response: {e}")
            return f"[ERROR] Impossible de se connecter ou d'utiliser le modèle '{self.model}'. Vérifiez qu'LM Studio est en cours d'exécution et que le model est chargé."

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        """
        Streams the response from OpenAI-compatible API using the streaming endpoint.
        """
        messages = []
        messages.append({"role": "user", "content": prompt})
        if history:
             for msg in history:
                 messages.append({"role": "user", "content": msg})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True # Indispensable pour le streaming
        }

        print("--- Démarrage du streaming de réponse depuis LM Studio/OpenAI ---")
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
                        print(f"Avertissement lors du traitement d'un chunk LMStudio/OpenAI : {e}")

        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Impossible de se connecter à LM Studio/OpenAI en mode streaming ({e}). Vérifiez l'état du service."
        finally:
             print("--- Fin du streaming LM Studio/OpenAI ---")

