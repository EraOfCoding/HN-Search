import re
from typing import Dict, Any
from urllib.parse import urlparse
import httpx


class ContentScraper:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def fetch_url_text(self, story: Dict[str, Any]) -> str:
        title = story.get("title", "")

        # Stories without URLs (Ask HN, etc.)
        if "url" not in story or not story["url"]:
            text = story.get("text", "")
            return f"{title}. {text}".strip() if title or text else "No content"

        url = story["url"]

        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return title or "No title"
        except:
            return title or "No title"

        # Fetch content
        try:
            response = await self.client.get(url)

            if response is None or response.status_code != 200:
                return title or "No title"

            content_type = response.headers.get("content-type", "").lower()

            if "text" in content_type or not content_type:
                text = response.text[:5000]
                text = re.sub(r"]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    return f"{title}. {text}".strip()

            return title or "No title"

        except Exception as e:
            print(f"Error fetching {url}: {type(e).__name__}")
            return title or "No title"
