import requests
from typing import Generator, Optional, List, Dict
# Assurez-vous que l'interface est accessible
from .llm_provider_interface import LLMProvider

class OllamaProvider(LLMProvider):
    """
    Implementation of LLMProvider using the Ollama REST API.
    Assumes local availability at localhost:11434.
    """
    def __init__(self, model: str = "llama2", api_key: Optional[str] = None):
        # Pour Ollama, le base_url est fixe et ne nécessite pas d'API Key dans ce contexte simple
        super().__init__(api_key=None, base_url="http://localhost:11434/api")
        self.model = model

    def generate_response(self, prompt: str, history: list) -> str:
        """
        Generates a complete response from Ollama API (non-streaming).
        Note: The current implementation only uses the latest prompt and ignores history for simplicity,
        but ideally, it should structure the conversation payload correctly.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt + "\n\n", # Ajouter un saut de ligne pour la cohérence du chat
                "options": {"temperature": 0.7}
            }
            # Appel POST à l'endpoint 'generate'
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion à Ollama lors de generate_response: {e}")
            return f"[ERROR] Impossible de se connecter ou d'utiliser le modèle '{self.model}'. Vérifiez qu'Ollama est en cours d'exécution et que le model est téléchargé."

    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        """
        Streams the response from Ollama using a streaming request.
        This is crucial for non-blocking UI updates (PyQt6).
        """
        print("--- Démarrage du streaming de réponse depuis Ollama ---")
        try:
            # Appel POST à l'endpoint /api/generate avec 'stream': true
            payload = {
                "model": self.model,
                "prompt": prompt + "\n\n",
                "stream": True # Indispensable pour le streaming
            }
            response = requests.post(f"{self.base_url}/generate", json=payload, stream=True)
            response.raise_for_status()

            full_text = ""
            # Parcourir chaque chunk de données reçu du flux
            for line in response.iter_lines():
                if line:
                    try:
                        data = eval(line.decode('utf-8')) # Ollama renvoie parfois des lignes non JSON pures, mais en général c'est une structure dict/json

                        # Dans le cas d'une réponse structurée de stream (comme celui attendu ici)
                        if isinstance(data, dict) and data.get("response"):
                            chunk = data["response"]
                            yield chunk # Rend immédiatement le morceau de texte traité
                            full_text += chunk
                    except Exception as e:
                        # Ceci attrape les lignes non-JSON ou mal formatées
                        print(f"Avertissement lors du traitement d'un chunk Ollama : {e}")

        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Impossible de se connecter à Ollama en mode streaming ({e}). Vérifiez l'état du service."
        finally:
             # On assure que le générateur sait quand il est terminé
             print("--- Fin du streaming Ollama ---")

