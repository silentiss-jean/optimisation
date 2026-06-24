import requests
from typing import Dict, Any
from .base_tool import Tool

class WebScraperTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Extract specific information from a web page using simple scraping. Useful for structured data extraction.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape."
                    },
                    "query_selector": {
                        "type": "string",
                        "description": "CSS selector or path for the specific content you want to extract."
                    }
                },
                "required": ["url"]
            }
        )

    def execute(self, url: str, query_selector: str = "") -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Simplified scraper logic (using BeautifulSoup if available or basic text extraction)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if query_selector:
                content = soup.select_one(query_selector)
                return content.get_text() if content else "Content not found for selector."
            else:
                # Just return the text from the body if no specific selector is provided
                return soup.body.get_text() if soup.body else response.text[:1000]
        except Exception as e:
            return f"Scraping failed for {url}: {str(e)}"
