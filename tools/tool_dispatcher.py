from tools.filesystem_tool import FileSystemTool
from tools.command_line_tool import CommandLineTool


class ToolDispatcher:
    """
    Routes incoming tool calls from the LLM Agent to the correct implementation.
    """
    def __init__(self):
        self.fs_tool = FileSystemTool()
        self.cmd_tool = CommandLineTool()  # whitelist gérée en interne par CommandLineTool

    def get_tools_description(self) -> str:
        return """Outils disponibles :
- read_file(file_path: str) : Lire le contenu d'un fichier
- write_file(file_path: str, content: str) : Écrire dans un fichier
- find_files(pattern: str, directory: str) : Chercher des fichiers par pattern
- command_line_execute(command: str) : Exécuter une commande OS (ls, dir, echo, pip, python, git, mkdir)"""

    def dispatch_call(self, tool_name: str, **kwargs) -> str:
        print(f"--- Dispatcher : appel '{tool_name}' avec {kwargs} ---")
        try:
            if tool_name == "read_file":
                return self.fs_tool.read_file(**kwargs)
            elif tool_name == "find_files":
                return self.fs_tool.find_files(**kwargs)
            elif tool_name == "write_file":
                return self.fs_tool.write_file(**kwargs)
            elif tool_name == "command_line_execute":
                if 'command' not in kwargs:
                    return "[ERROR] command_line_execute requiert 'command'."
                return self.cmd_tool.execute(command=kwargs['command'])
            else:
                return f"[DISPATCH ERROR] Outil '{tool_name}' non reconnu."
        except Exception as e:
            return f"[DISPATCH FAILURE] {e}"
