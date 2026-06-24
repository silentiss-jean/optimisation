import webbrowser
from .base_tool import Tool


class OpenUrlTool(Tool):
    def __init__(self):
        super().__init__(
            name="open_url",
            description="Ouvrir une URL dans le navigateur par défaut du système.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL complète à ouvrir (https://...)."}
                },
                "required": ["url"]
            }
        )

    def execute(self, url: str, **kwargs) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            webbrowser.open(url)
            return (
                f"[DONE] Le navigateur a été ouvert sur : {url}\n"
                "La tâche est accomplie. Utilise maintenant 'final_answer'."
            )
        except Exception as e:
            return f"[ERREUR open_url] {e}"
