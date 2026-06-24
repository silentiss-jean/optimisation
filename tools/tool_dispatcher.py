from typing import Dict, Any, List
# Importez tous les outils disponibles
from .filesystem_tool import FileSystemTool
from .command_line_tool import CommandLineTool

class ToolDispatcher:
    """
    Manages and routes incoming tool calls from the LLM Agent.
    It acts as the central switchboard, translating abstract intent into concrete tool execution.
    This class is the core of the ReAct loop observation phase.
    """
    def __init__(self):
        # Initialisation des instances de tous les outils disponibles
        self.fs_tool = FileSystemTool()
        self.cmd_tool = CommandLineTool(allowed_commands=["echo", "Get-Date"])

    def dispatch_call(self, tool_name: str, **kwargs) -> str:
        """
        Receives the name of the intended tool and its arguments from the LLM (via JSON/JSON structure).
        Executes the correct underlying method and returns the string result.
        """
        print(f"--- Dispatcher : Tentative d'appel à l'outil '{tool_name}' ---")

        try:
            if tool_name == "read_file":
                # Assure-toi que les arguments sont passés correctement (le chemin)
                return self.fs_tool.read_file(**kwargs)
            elif tool_name == "find_files":
                return self.fs_tool.find_files(**kwargs)
            elif tool_name == "write_file":
                return self.fs_tool.write_file(**kwargs)
            # Ajouter ici les appels pour CommandLineTool et WebBrowserTool, etc.

            elif tool_name == "command_line_execute":
                 if 'command' not in kwargs or 'args' not in kwargs:
                     return "[ERROR] Le tool CommandLineTool nécessite 'command' et 'args'."
                 # Ici, nous devons adapter les kwargs au format de la méthode réelle du tool
                 return self.cmd_tool.execute(command=kwargs['command'], args=kwargs['args'])

            else:
                return f"[DISPATCH ERROR] Outil '{tool_name}' non reconnu ou non implémenté."

        except Exception as e:
            return f"[CRITICAL DISPATCH FAILURE] Une erreur inattendue est survenue lors de l'appel du tool : {e}"

# Exemple d'utilisation pour le test (à retirer en production)
if __name__ == '__main__':
    dispatcher = ToolDispatcher()
    print(dispatcher.dispatch_call("read_file", file_path="test/dummy.txt"))