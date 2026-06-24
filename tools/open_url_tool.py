import webbrowser


class OpenUrlTool:
    """
    Ouvre une URL dans le navigateur par défaut du système.
    Utilise uniquement la bibliothèque standard Python (webbrowser).
    """

    def open_url(self, url: str) -> str:
        """
        Ouvre l'URL donnée dans le navigateur par défaut.
        Retourne un message de confirmation.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            webbrowser.open(url)
            return f"[OK] Navigateur ouvert sur : {url}"
        except Exception as e:
            return f"[ERREUR open_url] {e}"
