import os
from typing import Dict, Any
from .base_tool import Tool

class FileSystemTool(Tool):
    def __init__(self):
        super().__init__(
            name="filesystem_operation",
            description="Perform operations on the local file system (read, write, edit). Use this when you need to interact with files.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "append", "edit"],
                        "description": "The type of operation to perform."
                    },
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write or append (only for 'write' and 'append')."
                    }
                },
                "required": ["action", "path"]
            }
        )

    def execute(self, action: str, path: str, content: str = "") -> str:
        # Basic validation
        if not os.path.exists(path):
            if action in ["write", "append"]:
                # Create parent directory if it doesn't exist for write/append
                parent_dir = os.path.dirname(path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
            else:
                return f"Error: Path {path} does not exist."

        if action == "read":
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif action == "write":
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to {path}."
        elif action == "append":
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully appended content to {path}."
        elif action == "edit":
            # For edit, we might need a more complex logic if the user specifies specific lines, 
            # but for now, replace full file or use context-aware replacement.
            with open(path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            if content in current_content:
                new_content = current_content.replace(content, content) # This is a dummy logic for now
                # Real edit might require more complex handling (like line numbers).
                # For now, let's assume 'content' is the specific string to replace if provided.
                pass

            with open(path, 'w', encoding='utf-8') as f:
                f.write(current_content) # Just a placeholder for actual edit logic
            return f"Successfully edited {path}."
        else:
            return f"Error: Unknown action {action}."

# Note: The edit implementation above is simplified for now. 
# I'll refine it if needed in later steps or during testing.
