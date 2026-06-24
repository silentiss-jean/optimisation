from typing import Dict, Any, List
from abc import ABC, abstractmethod

class Tool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @property
    def definition(self) -> Dict[str, Any]:
        """Returns the tool definition in a format compatible with most LLM APIs."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool logic and return a string result."""
        pass
