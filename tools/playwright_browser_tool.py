import os
import threading
from typing import Optional
from .base_tool import Tool

try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class _PlaywrightSession:
    _instance: Optional["_PlaywrightSession"] = None
    _owner_thread: Optional[int] = None

    def __init__(self):
        self._playwright  = None
        self._browser: Optional[Browser]        = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page]              = None

    @classmethod
    def get(cls) -> "_PlaywrightSession":
        current = threading.get_ident()
        if cls._instance is None or cls._owner_thread != current:
            if cls._instance is not None:
                cls._instance._teardown()
            cls._instance     = _PlaywrightSession()
            cls._owner_thread = current
        return cls._instance

    def _teardown(self):
        for obj in (self._context, self._browser, self._playwright):
            try:
                if obj: obj.close() if hasattr(obj, 'close') else obj.stop()
            except Exception:
                pass
        self._page = self._context = self._browser = self._playwright = None

    def _is_alive(self) -> bool:
        if not self._page:
            return False
        try:
            _ = self._page.url
            return True
        except Exception:
            return False

    def start(self) -> bool:
        if not PLAYWRIGHT_AVAILABLE:
            return False
        if self._browser is not None and not self._is_alive():
            self._teardown()
        if self._browser is not None:
            return True
        try:
            self._playwright = sync_playwright().start()
            self._browser    = self._playwright.chromium.launch(headless=False)
            # ignore_https_errors=True : accepte les certs auto-signés et
            # les erreurs SSL sur les sites locaux (http:// compris)
            self._context    = self._browser.new_context(
                ignore_https_errors=True
            )
            self._page       = self._context.new_page()
            return True
        except Exception:
            self._teardown()
            return False

    @property
    def page(self) -> Optional[Page]:
        return self._page

    def new_tab(self, url: str) -> str:
        if not self._context:
            return "[ERREUR] Aucun contexte navigateur actif."
        try:
            tab = self._context.new_page()
            tab.goto(url, wait_until="load", timeout=15000)
            self._page = tab
            return f"[DONE] Nouvel onglet ouvert : {tab.url}"
        except Exception as e:
            return f"[ERREUR browser_new_tab] {str(e).split(chr(10))[0]}"

    def close(self):
        self._teardown()
        _PlaywrightSession._instance    = None
        _PlaywrightSession._owner_thread = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Préfixe https:// uniquement si aucun schéma n'est présent."""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _goto(page: Page, url: str, timeout: int = 15000) -> str:
    """
    Navigue vers url avec fallback progressif sur wait_until :
      load → domcontentloaded → commit
    Retourne l'URL finale ou lève l'exception du dernier essai.
    """
    for wait in ("load", "domcontentloaded", "commit"):
        try:
            page.goto(url, wait_until=wait, timeout=timeout)
            return page.url
        except Exception as e:
            last_exc = e
    raise last_exc


# ─── Actions ─────────────────────────────────────────────────────────────────

class BrowserNavigateTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_navigate",
            description="Naviguer vers une URL dans l'onglet actif du navigateur. Fonctionne aussi pour les sites locaux (http://) et les certificats auto-signés.",
            parameters={"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}
        )

    def execute(self, url: str, **kwargs) -> str:
        url = _normalize_url(url)
        session = _PlaywrightSession.get()
        if not session.start():
            import webbrowser; webbrowser.open(url)
            return f"[FALLBACK] Playwright indisponible. Ouvert via webbrowser : {url}"
        try:
            final_url = _goto(session.page, url)
            return f"[DONE] Navigé vers : {final_url}"
        except Exception as e:
            return f"[ERREUR browser_navigate] {str(e).split(chr(10))[0]}. Essaie une URL alternative."


class BrowserNewTabTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_new_tab",
            description="Ouvrir une URL dans un NOUVEL ONGLET de la fenêtre existante (sans ouvrir une nouvelle fenêtre). Fonctionne aussi pour les sites locaux (http://).",
            parameters={"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}
        )

    def execute(self, url: str, **kwargs) -> str:
        url = _normalize_url(url)
        session = _PlaywrightSession.get()
        if not session.start():
            import webbrowser; webbrowser.open(url)
            return f"[FALLBACK] Playwright indisponible. Ouvert via webbrowser : {url}"
        return session.new_tab(url)


class BrowserClickTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_click",
            description="Cliquer sur un élément de la page via un sélecteur CSS.",
            parameters={"type":"object","properties":{"selector":{"type":"string"}},"required":["selector"]}
        )
    def execute(self, selector: str, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
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
            parameters={"type":"object","properties":{"selector":{"type":"string"},"text":{"type":"string"}},"required":["selector","text"]}
        )
    def execute(self, selector: str, text: str, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
        try:
            session.page.fill(selector, text, timeout=8000)
            return f"[DONE] Champ '{selector}' rempli."
        except Exception as e:
            return f"[ERREUR browser_fill] {str(e).split(chr(10))[0]}"


class BrowserScreenshotTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_screenshot",
            description="Capturer une capture d'écran de la page courante.",
            parameters={"type":"object","properties":{"save_path":{"type":"string","description":"Chemin de sauvegarde. Défaut: screenshot.png"}},"required":[]}
        )
    def execute(self, save_path: str = "screenshot.png", **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
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
            description="Récupérer le texte visible de la page courante ou d'un élément CSS.",
            parameters={"type":"object","properties":{"selector":{"type":"string","description":"Sélecteur CSS optionnel. Vide = toute la page."}},"required":[]}
        )
    def execute(self, selector: str = "", **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
        try:
            target = session.page.locator(selector).first if selector else session.page.locator("body")
            return target.inner_text(timeout=8000)[:3000]
        except Exception as e:
            return f"[ERREUR browser_get_text] {str(e).split(chr(10))[0]}"


class BrowserScrollTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_scroll",
            description="Scroller la page vers le bas ou le haut pour charger du contenu lazy.",
            parameters={
                "type":"object",
                "properties":{
                    "direction":{"type":"string","description":"'down' ou 'up'. Défaut: down."},
                    "amount":{"type":"number","description":"Pixels à scroller. Défaut: 600."}
                },
                "required":[]
            }
        )
    def execute(self, direction: str = "down", amount: int = 600, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
        try:
            delta = amount if direction == "down" else -amount
            session.page.evaluate(f"window.scrollBy(0, {delta})")
            return f"[DONE] Scrollé {direction} de {abs(delta)}px."
        except Exception as e:
            return f"[ERREUR browser_scroll] {str(e).split(chr(10))[0]}"


class BrowserWaitForTool(Tool):
    def __init__(self):
        super().__init__(
            name="browser_wait_for",
            description="Attendre qu'un sélecteur CSS soit visible avant de continuer (utile après un clic ou navigation lente).",
            parameters={
                "type":"object",
                "properties":{
                    "selector":{"type":"string","description":"Sélecteur CSS à attendre."},
                    "timeout":{"type":"number","description":"Timeout en ms. Défaut: 5000."}
                },
                "required":["selector"]
            }
        )
    def execute(self, selector: str, timeout: int = 5000, **kwargs) -> str:
        session = _PlaywrightSession.get()
        if not session.start() or not session.page:
            return "[ERREUR] Aucune session active. Utilise browser_navigate d'abord."
        try:
            session.page.wait_for_selector(selector, timeout=timeout)
            return f"[DONE] Élément '{selector}' visible."
        except Exception as e:
            return f"[ERREUR browser_wait_for] {str(e).split(chr(10))[0]}"


class PlaywrightBrowserTool:
    """Groupe les 8 actions Playwright pour ToolDispatcher."""
    def __init__(self):
        self.tools = [
            BrowserNavigateTool(),
            BrowserNewTabTool(),
            BrowserClickTool(),
            BrowserFillTool(),
            BrowserScreenshotTool(),
            BrowserGetTextTool(),
            BrowserScrollTool(),
            BrowserWaitForTool(),
        ]
