import os
import glob
from .base_tool import Tool


class ReadFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Lire le contenu d'un fichier texte.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Chemin absolu ou relatif du fichier."}
                },
                "required": ["file_path"]
            }
        )

    def execute(self, file_path: str, **kwargs) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"[ERREUR] Fichier introuvable : {file_path}"
        except Exception as e:
            return f"[ERREUR read_file] {e}"


class WriteFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Écrire ou créer un fichier avec le contenu donné.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Chemin du fichier à écrire."},
                    "content":   {"type": "string", "description": "Contenu à écrire dans le fichier."}
                },
                "required": ["file_path", "content"]
            }
        )

    def execute(self, file_path: str, content: str = "", **kwargs) -> str:
        try:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[DONE] Fichier écrit : {file_path}"
        except Exception as e:
            return f"[ERREUR write_file] {e}"


class FindFilesTool(Tool):
    def __init__(self):
        super().__init__(
            name="find_files",
            description="Chercher des fichiers par motif glob (ex: \"*.py\").",
            parameters={
                "type": "object",
                "properties": {
                    "pattern":   {"type": "string", "description": "Motif glob, ex: \"*.txt\""},
                    "directory": {"type": "string", "description": "Dossier de recherche (défaut: répertoire courant)."}
                },
                "required": ["pattern"]
            }
        )

    def execute(self, pattern: str, directory: str = ".", **kwargs) -> str:
        try:
            search = os.path.join(directory, "**", pattern)
            results = glob.glob(search, recursive=True)
            if not results:
                return f"Aucun fichier trouvé pour '{pattern}' dans '{directory}'."
            return "\n".join(results)
        except Exception as e:
            return f"[ERREUR find_files] {e}"


# Alias de compatibilité — ToolDispatcher instancie FileSystemTool pour regrouper
class FileSystemTool:
    """Groupe les 3 outils filesystem pour faciliter l'enregistrement."""
    def __init__(self):
        self.tools = [ReadFileTool(), WriteFileTool(), FindFilesTool()]
