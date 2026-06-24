from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    """
    Classe de base pour tous les outils de l'agent.
    Chaque outil déclare son nom, sa description et ses paramètres dans __init__.
    ToolDispatcher les enregistre et les dispatche automatiquement.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name        = name
        self.description = description
        self.parameters  = parameters

    @property
    def definition(self) -> Dict[str, Any]:
        """Définition au format function-calling (OpenAI compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Exécute l'outil et retourne une chaîne lisible par le LLM."""
        pass
