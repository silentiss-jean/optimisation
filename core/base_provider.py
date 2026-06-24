from abc import ABC, abstractmethod
from typing import Generator, Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generates a stream of tokens from the LLM.
        
        Args:
            prompt: The user input or system instructions + user input.
            **kwargs: Additional parameters like temperature, max_tokens, model name, etc.
        Returns:
            A generator yielding strings (chunks of text).
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a full response from the LLM (non-streaming).
        
        Args:
            prompt: The user input or system instructions + user input.
            **kwargs: Additional parameters.
        Returns:
            The complete response as a string.
        """
        pass
