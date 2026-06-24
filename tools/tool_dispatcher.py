from typing import Dict
from tools.base_tool import Tool
from tools.filesystem_tool import FileSystemTool
from tools.command_line_tool import CommandLineTool
from tools.open_url_tool import OpenUrlTool
from tools.web_scraper_tool import WebScraperTool


class ToolDispatcher:
    """
    Enregistre les outils et dispatche les appels du LLM.
    Pour ajouter un outil : créer la classe, l'instancier ici avec self._register().
    Aucun if/elif nécessaire.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

        # Outils filesystem (3 outils groupés)
        for tool in FileSystemTool().tools:
            self._register(tool)

        # Outils individuels
        self._register(CommandLineTool())
        self._register(OpenUrlTool())
        self._register(WebScraperTool())

    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_tools_description(self) -> str:
        """Génère automatiquement la description des outils depuis leurs métadonnées."""
        sections = {
            "Fichiers":    ["read_file", "write_file", "find_files"],
            "Système":    ["command_line_execute"],
            "Navigateur": ["open_url"],
            "Web":        ["web_scrape"],
        }
        lines = ["Outils disponibles :\n"]
        for section, names in sections.items():
            lines.append(f"[{section}]")
            for name in names:
                tool = self._tools.get(name)
                if tool:
                    # Extraire les paramètres requis pour l'affichage
                    props = tool.parameters.get("properties", {})
                    params_str = ", ".join(
                        f"{k}: {v.get('type', 'str')}" for k, v in props.items()
                    )
                    lines.append(f"- {name}({params_str})")
                    lines.append(f"    {tool.description}")
            lines.append("")
        return "\n".join(lines)

    def dispatch_call(self, tool_name: str, **kwargs) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            return (
                f"[DISPATCH ERROR] Outil '{tool_name}' non reconnu. "
                f"Outils disponibles : {available}."
            )
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"[DISPATCH FAILURE] {tool_name} : {e}"
