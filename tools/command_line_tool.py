import subprocess
import os
from typing import Dict, Any
from .base_tool import Tool

class CommandLineTool(Tool):
    def __init__(self):
        super().__init__(
            name="command_line_execution",
            description="Execute a command in the system shell. Use this only when necessary and for specific system-level operations.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The full command to execute (e.g., 'pip install requests')."
                    }
                },
                "required": ["command"]
            }
        )

    def _is_whitelisted(self, command: str) -> bool:
        allowed = ["ls", "dir", "echo", "pip", "python", "git", "mkdir", "type", "copy", "move"]
        base_cmd = command.split()[0]
        return base_cmd in allowed

    def execute(self, command: str) -> str:
        if not self._is_whitelisted(command):
            return f"Warning: Command '{command}' is not on the standard whitelist and might require extra caution."

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return result.stdout if result.stdout else "Command executed successfully (no output)."
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Execution failed with error: {str(e)}"
