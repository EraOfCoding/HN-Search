import asyncio
from typing import List, Dict, Any, Optional
import httpx


class HNClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def fetch_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        try:
            response = await self.client.get(f"/item/{item_id}.json")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Failed to fetch item {item_id}: {e}")
            return None

    async def get_top_stories(self, limit: int = 500) -> List[Dict[str, Any]]:
        response = await self.client.get("/topstories.json")
        ids = response.json()[:limit]

        tasks = [self.fetch_item(id) for id in ids]
        stories = await asyncio.gather(*tasks)

        return [story for story in stories if story]
