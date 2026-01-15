import asyncio
from typing import List
from openai import AsyncOpenAI
from app.config import settings


class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    async def embed_text(self, text: str) -> List[float]:
        text_to_embed = text[:50000]
        resp = await self.client.embeddings.create(
            model=self.model,
            input=text_to_embed,
        )
        return resp.data[0].embedding

    async def embed_batch(
        self, texts: List[str], batch_size: int = 50
    ) -> List[List[float]]:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self.embed_text(text) for text in batch]
            )
            embeddings.extend(batch_embeddings)
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} stories")

        return embeddings
