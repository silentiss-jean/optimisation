import abc
from typing import Generator, Optional

class LLMProvider(abc.ABC):
    """
    Abstract Base Class for all Large Language Model providers.
    Defines the common contract/interface required by the application's core logic.
    All concrete implementations (Ollama, OpenAI, etc.) MUST inherit from this class.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "http://localhost"):
        """
        Initializes the provider with necessary credentials and base URL.
        """
        self.api_key = api_key
        self.base_url = base_url

    @abc.abstractmethod
    def generate_response(self, prompt: str, history: list) -> str:
        """
        Generates a complete response from the model based on a single prompt and conversation history.

        :param prompt: The user's latest message/prompt.
        :param history: List of previous messages (e.g., [{"role": "user", "content": "..."}]).
        :return: The full generated text response from the model.
        """
        pass

    @abc.abstractmethod
    def stream_response(self, prompt: str, history: list) -> Generator[str, None, None]:
        """
        Streams the model's response in chunks for real-time UI feedback (critical for PyQt6).

        This generator yields each chunk of text as it is received from the API.

        :param prompt: The user's latest message/prompt.
        :param history: List of previous messages.
        :return: A generator that yields strings representing chunks of text.
        """
        pass

    # Future methods could include:
    # @abc.abstractmethod
    # def get_model_list(self) -> list[str]: ...
