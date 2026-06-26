import os
from typing import Optional
from .base_tool import Tool

# Tentative d'import Playwright — fallback webbrowser si absent
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class _PlaywrightSession:
    """
    Session Playwright persistante entre les appels du LLM.
    Une seule instance partagée par toutes les actions du navigateur.
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

    def start(self) -> bool:
        if not PLAYWRIGHT_AVAILABLE:
            return False
        if self._browser is not None:
            return True
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self._page = self._browser.new_page()
            return True
        except Exception:
            return False

    @property
    def page(self) -> Optional[Page]:
        return self._page

    def close(self):
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        finally:
            self._browser = None
            self._page = None
            self._playwright = None
            _PlaywrightSession._instance = None


# ─── Actions ─────────────────────────────────────────────────────────────────

class BrowserNavigateTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_navigate",
            description="Naviguer vers une URL dans le navigateur controllé (Playwright). Lance le navigateur si besoin.",
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
            # Fallback webbrowser
            import webbrowser
            webbrowser.open(url)
            return f"[FALLBACK] Playwright indisponible. Ouvert via webbrowser : {url}"

        try:
            session.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return f"[DONE] Navigué vers : {session.page.url}"
        except Exception as e:
            return f"[ERREUR browser_navigate] {e}"


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
        page = _PlaywrightSession.get().page
        if page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            page.click(selector, timeout=8000)
            return f"[DONE] Cliqué sur '{selector}'."
        except Exception as e:
            return f"[ERREUR browser_click] {e}"


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
        page = _PlaywrightSession.get().page
        if page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            page.fill(selector, text, timeout=8000)
            return f"[DONE] Champ '{selector}' rempli avec le texte fourni."
        except Exception as e:
            return f"[ERREUR browser_fill] {e}"


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
        page = _PlaywrightSession.get().page
        if page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            abs_path = os.path.abspath(save_path)
            page.screenshot(path=abs_path, full_page=True)
            return f"[DONE] Screenshot sauvegardé : {abs_path}"
        except Exception as e:
            return f"[ERREUR browser_screenshot] {e}"


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
        page = _PlaywrightSession.get().page
        if page is None:
            return "[ERREUR] Aucune session navigateur active. Utilise browser_navigate d'abord."
        try:
            if selector:
                text = page.locator(selector).first.inner_text(timeout=8000)
            else:
                text = page.locator("body").inner_text()
            return text[:3000]
        except Exception as e:
            return f"[ERREUR browser_get_text] {e}"


# Groupe pour ToolDispatcher
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
