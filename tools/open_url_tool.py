import webbrowser


class OpenUrlTool:
    """
    Ouvre une URL dans le navigateur par défaut du système.
    Retourne un message éxplicite DONE pour que le LLM sache que la tâche est terminée.
    """

    def open_url(self, url: str) -> str:
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
