from typing import List, Dict, Any, Tuple
import numpy as np
from app.hn_client import HNClient
from app.scraper import ContentScraper
from app.embeddings import EmbeddingService


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(dot_product / (norm_v1 * norm_v2))


class SearchService:
    def __init__(
        self, hn_client: HNClient, scraper: ContentScraper, embedder: EmbeddingService
    ):
        self.hn_client = hn_client
        self.scraper = scraper
        self.embedder = embedder

    async def fetch_and_embed_stories(
        self, stories: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[List[float]]]:
        print(f"Fetching content for {len(stories)} stories...")
        texts = []
        for i, story in enumerate(stories):
            text = await self.scraper.fetch_url_text(story)
            texts.append(text)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(stories)} stories")

        print("Generating embeddings...")
        embeddings = await self.embedder.embed_batch(texts)

        return texts, embeddings

    async def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        # Fetch stories
        print("Fetching top stories...")
        stories = await self.hn_client.get_top_stories(limit=200)
        print(f"Fetched {len(stories)} stories")

        # Get content and embeddings
        story_texts, story_embeddings = await self.fetch_and_embed_stories(stories)

        # Embed query
        print(f"Embedding search query: '{query}'")
        query_embedding = await self.embedder.embed_text(query)

        # Calculate similarities
        print("Calculating similarities...")
        similarities = []
        for i, story_embedding in enumerate(story_embeddings):
            similarity = cosine_similarity(query_embedding, story_embedding)
            similarities.append(
                {
                    "story": stories[i],
                    "similarity": similarity,
                    "text_preview": story_texts[i][:200],
                }
            )

        # Sort and get top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:limit]

        return {
            "query": query,
            "total_stories_searched": len(stories),
            "results": [
                {
                    "title": r["story"].get("title"),
                    "url": r["story"].get("url"),
                    "hn_url": f"https://news.ycombinator.com/item?id={r['story'].get('id')}",
                    "score": r["story"].get("score"),
                    "similarity": r["similarity"],
                    "text_preview": r["text_preview"],
                }
                for r in top_results
            ],
        }
