from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import os


class PineconeStore:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "hn-stories"

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

    def upsert_stories(self, stories: List[Dict], embeddings: List[List[float]]):
        """Store stories with embeddings."""
        vectors = [
            {
                "id": str(story["id"]),
                "values": embedding,
                "metadata": {
                    "title": story.get("title", ""),
                    "url": story.get("url", ""),
                    "score": story.get("score", 0),
                    "author": story.get("by", ""),
                    "time": story.get("time", 0),
                },
            }
            for story, embedding in zip(stories, embeddings)
        ]

        for i in range(0, len(vectors), 100):
            batch = vectors[i : i + 100]
            self.index.upsert(vectors=batch)

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search similar stories."""
        results = self.index.query(
            vector=query_embedding, top_k=limit, include_metadata=True
        )

        return [
            {
                "id": int(match["id"]),
                "title": match["metadata"]["title"],
                "url": match["metadata"]["url"],
                "score": match["metadata"]["score"],
                "similarity": match["score"],
                "hn_url": f"https://news.ycombinator.com/item?id={match['id']}",
            }
            for match in results["matches"]
        ]

    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {"total_vectors": stats.total_vector_count, "dimension": stats.dimension}
