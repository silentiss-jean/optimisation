from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class Tool(ABC):
    """
    Abstract Base Class for all system tools accessible to the LLM Agent.
    Every tool must inherit from this class to enforce a standard execution contract.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initializes the tool with necessary configuration (e.g., API keys, directories)."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """
        Executes the core logic of the tool and returns a human-readable string result
        which is then passed back to the LLM for context.

        :param args: Positional arguments for the function call (e.g., file_path).
        :param kwargs: Keyword arguments (e.g., regex_pattern, output_mode).
        :return: The string result of the execution.
        """
        pass


class FileSystemTool(Tool):
    """
    Tool for managing files and directories within the project scope.
    Expose functions like read_file, write_file, find_files, etc.
    """
    def __init__(self, root_dir: str = "G:\\optimisation"):
        super().__init__()
        self.root_dir = root_dir

    @abstractmethod
    def read_file(self, file_path: str) -> str:
        """Reads the content of a specified file path."""
        pass

    # ... autres méthodes abstraites à compléter (write_file, find_files, etc.)
