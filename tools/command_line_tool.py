import subprocess
from .base_tool import Tool


class CommandLineTool(Tool):
    def __init__(self):
        super().__init__(
            name="command_line_execute",
            description="Exécuter une commande OS (dir, ls, echo, pip, python, git, mkdir...).",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Commande complète à exécuter."}
                },
                "required": ["command"]
            }
        )

    def execute(self, command: str, **kwargs) -> str:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip() or "Commande exécutée (pas de sortie)."
            return f"[ERREUR] {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "[ERREUR] Timeout : commande trop longue (> 60s)."
        except Exception as e:
            return f"[ERREUR command_line_execute] {e}"
