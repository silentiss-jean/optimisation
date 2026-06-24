from tools.filesystem_tool import FileSystemTool
from tools.command_line_tool import CommandLineTool
from tools.open_url_tool import OpenUrlTool
from tools.web_scraper_tool import WebScraperTool


class ToolDispatcher:
    """
    Routes les appels d'outils du LLM vers les implémentations correspondantes.
    Outils disponibles :
      - Filesystem  : read_file, write_file, find_files
      - CommandLine : command_line_execute
      - Browser     : open_url
      - Web         : web_scrape
    """

    def __init__(self):
        self.fs_tool      = FileSystemTool()
        self.cmd_tool     = CommandLineTool()
        self.browser_tool = OpenUrlTool()
        self.scraper_tool = WebScraperTool()

    def get_tools_description(self) -> str:
        return """Outils disponibles :

[Fichiers]
- read_file(file_path: str)
    Lire le contenu d'un fichier texte.
- write_file(file_path: str, content: str)
    Écrire ou créer un fichier avec le contenu donné.
- find_files(pattern: str, directory: str)
    Chercher des fichiers par motif glob (ex: "*.py").

[Système]
- command_line_execute(command: str)
    Exécuter une commande OS (dir, ls, echo, pip, python, git, mkdir...).

[Navigateur]
- open_url(url: str)
    Ouvrir une URL dans le navigateur par défaut du système.
    Exemple : open_url(url="https://google.com")

[Web]
- web_scrape(url: str)
    Télécharger et retourner le texte brut d'une page web.
    Exemple : web_scrape(url="https://example.com")
"""

    def dispatch_call(self, tool_name: str, **kwargs) -> str:
        try:
            if tool_name == "read_file":
                return self.fs_tool.read_file(**kwargs)

            elif tool_name == "write_file":
                return self.fs_tool.write_file(**kwargs)

            elif tool_name == "find_files":
                return self.fs_tool.find_files(**kwargs)

            elif tool_name == "command_line_execute":
                if 'command' not in kwargs:
                    return "[ERREUR] command_line_execute requiert 'command'."
                return self.cmd_tool.execute(command=kwargs['command'])

            elif tool_name == "open_url":
                if 'url' not in kwargs:
                    return "[ERREUR] open_url requiert 'url'."
                return self.browser_tool.open_url(url=kwargs['url'])

            elif tool_name == "web_scrape":
                if 'url' not in kwargs:
                    return "[ERREUR] web_scrape requiert 'url'."
                return self.scraper_tool.scrape(url=kwargs['url'])

            else:
                return (
                    f"[DISPATCH ERROR] Outil '{tool_name}' non reconnu. "
                    f"Outils valides : read_file, write_file, find_files, "
                    f"command_line_execute, open_url, web_scrape."
                )
        except Exception as e:
            return f"[DISPATCH FAILURE] {tool_name} : {e}"
