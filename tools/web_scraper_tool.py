import requests
from .base_tool import Tool


class WebScraperTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_scrape",
            description="Télécharger et retourner le texte brut d'une page web.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL de la page à scraper."}
                },
                "required": ["url"]
            }
        )

    def execute(self, url: str, **kwargs) -> str:
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(separator="\n", strip=True)[:3000]
            except ImportError:
                return response.text[:3000]
        except requests.Timeout:
            return f"[ERREUR] Timeout (8s) lors du scraping de {url}. Essaie une URL alternative."
        except requests.ConnectionError:
            return f"[ERREUR] Impossible de joindre {url} (connexion refusée ou DNS). Essaie une URL alternative."
        except Exception as e:
            return f"[ERREUR web_scrape] {e}"
