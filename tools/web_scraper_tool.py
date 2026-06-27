import requests
from .base_tool import Tool

# Limite de caractères pour les réponses HTML (pages web normales)
HTML_MAX = 3000
# Limite pour les API JSON — assez large pour un tableau de repos GitHub complet
JSON_MAX = 50_000


class WebScraperTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_scrape",
            description="Télécharger et retourner le texte brut d'une page web ou le contenu JSON d'une API.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL de la page ou de l'API à scraper."}
                },
                "required": ["url"]
            }
        )

    def execute(self, url: str, **kwargs) -> str:
        try:
            headers = {"Accept": "application/json, text/html;q=0.9, */*;q=0.8"}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            is_json = "application/json" in content_type or response.text.strip()[:1] in ("[", "{")

            if is_json:
                # API JSON : retourner le contenu brut sans transformation
                return response.text[:JSON_MAX]

            # Page HTML : extraire le texte avec BeautifulSoup si disponible
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(separator="\n", strip=True)[:HTML_MAX]
            except ImportError:
                return response.text[:HTML_MAX]

        except requests.Timeout:
            return f"[ERREUR] Timeout (10s) lors du scraping de {url}. Essaie une URL alternative."
        except requests.ConnectionError:
            return f"[ERREUR] Impossible de joindre {url} (connexion refusée ou DNS). Essaie une URL alternative."
        except Exception as e:
            return f"[ERREUR web_scrape] {e}"
