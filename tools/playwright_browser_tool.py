import os
from typing import Optional
from .base_tool import Tool

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class _PlaywrightSession:
    """
    Session Playwright persistante entre les appels du LLM.
    Auto-reconnexion si le browser sous-jacent a planté (thread mort).
    """
    _instance: Optional["_PlaywrightSession"] = None

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    @classmethod
    def get(cls) -> "_PlaywrightSession":
        if cls._instance is None:
            cls._instance = _PlaywrightSession()
        return cls._instance

    def _is_alive(self) -> bool:
        """Vérifie si la session Playwright est réellement utilisable."""
        if self._browser is None or self._page is None:
            return False
        try:
            # Opération légère qui échoue si le thread/process Playwright est mort
            _ = self._page.url
            return True
        except Exception:
            return False

    def _teardown(self):
        """Libère les ressources sans lever d'exception."""
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._page = None
        self._playwright = None

    def start(self) -> bool:
        if not PLAYWRIGHT_AVAILABLE:
            return False

        # Si la session existe mais est morte, on la recrée
        if self._browser is not None and not self._is_alive():
            self._teardown()

        if self._browser is not None:
            return True

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self._page = self._browser.new_page()
            return True
        except Exception:
            self._teardown()
            return False

    @property
    def page(self) -> Optional[Page]:
        return self._page

    def close(self):
        self._teardown()
        _PlaywrightSession._instance = None


# ─── Actions ─────────────────────────────────────────────────────────────────

class BrowserNavigateTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_navigate",
            description="Naviguer vers une URL dans le navigateur contrôlé (Playwright). Lance ou restaure le navigateur si besoin.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL cible (https://...)"}
                },
                "required": ["url"]
            }
        )

    def execute(self, url: str, **kwargs) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        session = _PlaywrightSession.get()
        if not session.start():
            import webbrowser
            webbrowser.open(url)
            return f"[FALLBACK] Playwright indisponible. Ouvert via webbrowser : {url}"

        try:
            session.page.goto(url, wait_until="domcontentloaded", timeout=8000)
            return f"[DONE] Navigé vers : {session.page.url}"
        except Exception as e:
            err = str(e).split("\n")[0]
            return f"[ERREUR browser_navigate] {err}. Essaie une URL alternative."


class BrowserClickTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_click",
            description="Cliquer sur un élément de la page via un sélecteur CSS.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Sélecteur CSS (ex: \"button#submit\", \"a.nav-link\")"}
                },
                "required": ["selector"]
            }
        )

    def execute(self, selector: str, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or session.page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            session.page.click(selector, timeout=8000)
            return f"[DONE] Cliqué sur '{selector}'."
        except Exception as e:
            return f"[ERREUR browser_click] {str(e).split(chr(10))[0]}"


class BrowserFillTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_fill",
            description="Remplir un champ de saisie avec du texte via un sélecteur CSS.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Sélecteur CSS du champ input."},
                    "text":     {"type": "string", "description": "Texte à saisir."}
                },
                "required": ["selector", "text"]
            }
        )

    def execute(self, selector: str, text: str, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or session.page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            session.page.fill(selector, text, timeout=8000)
            return f"[DONE] Champ '{selector}' rempli."
        except Exception as e:
            return f"[ERREUR browser_fill] {str(e).split(chr(10))[0]}"


class BrowserScreenshotTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_screenshot",
            description="Capturer une capture d'écran de la page courante et la sauvegarder.",
            parameters={
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Chemin de sauvegarde (ex: \"screenshot.png\"). Défaut: screenshot.png"
                    }
                },
                "required": []
            }
        )

    def execute(self, save_path: str = "screenshot.png", **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or session.page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            abs_path = os.path.abspath(save_path)
            session.page.screenshot(path=abs_path, full_page=True)
            return f"[DONE] Screenshot sauvegardé : {abs_path}"
        except Exception as e:
            return f"[ERREUR browser_screenshot] {str(e).split(chr(10))[0]}"


class BrowserGetTextTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_get_text",
            description="Récupérer le texte visible de la page courante ou d'un élément spécifique.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Sélecteur CSS optionnel. Si vide, retourne le texte complet de la page."
                    }
                },
                "required": []
            }
        )

    def execute(self, selector: str = "", **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or session.page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            if selector:
                text = session.page.locator(selector).first.inner_text(timeout=8000)
            else:
                text = session.page.locator("body").inner_text()
            return text[:3000]
        except Exception as e:
            return f"[ERREUR browser_get_text] {str(e).split(chr(10))[0]}"


class PlaywrightBrowserTool:
    """Groupe les 5 actions Playwright pour l'enregistrement dans ToolDispatcher."""
    def __init__(self):
        self.tools = [
            BrowserNavigateTool(),
            BrowserClickTool(),
            BrowserFillTool(),
            BrowserScreenshotTool(),
            BrowserGetTextTool(),
        ]
