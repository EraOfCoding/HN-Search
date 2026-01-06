import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Query
import httpx
from pydantic import BaseModel

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
http_client: httpx.AsyncClient = None
external_http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create the client
    global http_client
    http_client = httpx.AsyncClient(base_url=HN_BASE_URL)
    yield
    # Shutdown: Clean up the client
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def fetch_item(item_id: int) -> Dict[str, Any]:
    """Helper function to fetch a single item by ID."""
    try:
        response = await http_client.get(f"/item/{item_id}.json")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Failed to fetch item {item_id}: {e}")
        return None


@app.get("/top-stories")
async def get_top_stories(limit: int = Query(500, le=500)):  # ?
    """
    Fetches the top stories from Hacker News.
    """
    try:
        response = await http_client.get("/topstories.json")
        ids = response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Could not reach Hacker News")

    target_ids = ids[:limit]

    tasks = [fetch_item(id) for id in target_ids]
    stories = await asyncio.gather(*tasks)  # ?

    return [story for story in stories if story]


async def fetch_url_text(story: Dict[str, Any]) -> str:
    """
    Fetch the text content of url.
    Returns the extracted text or falls back to title if URL fetch fails.
    """
    # If story doesn't have a URL (Ask HN, Show HN with text), use title and text
    if "url" not in story or not story["url"]:
        title = story.get("title", "")
        text = story.get("text", "")
        return f"{title}. {text}".strip()

    url = story["url"]
    title = story.get("title", "")

    # Check if URL seems valid
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            print(f"Invalid URL format: {url}")
            return title
    except Exception as e:
        print(f"Failed to parse URL {url}: {e}")
        return title

    # Try to fetch the URL content
    try:
        response = await external_http_client.get(url)
        response.raise_for_status()

        # Simple text extraction - get first 5000 chars of content
        # Use BeatifulSoup or similar for better extraction
        if hasattr(response, "headers") and response.headers:
            content_type = response.headers.get("content-type", "").lower()

        if "text/html" in content_type or "text/plain" in content_type:
            text_content = response.text[:5000]
            # Basic cleanup: remove HTML tags (simple regex approach)
            import re

            text_content = re.sub(r"<[^>]+>", " ", text_content)
            text_content = re.sub(r"\s+", " ", text_content).strip()

            return f"{title}. {text_content}".strip()
        else:
            print(f"Non-text content type for {url}: {content_type}")
            return title

    except httpx.TimeoutException:
        print(f"Timeout fetching {url}")
        return title
    except httpx.HTTPError as e:
        print(f"HTTP error fetching {url}: {e}")
        return title
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
        return title


async def embed_text(text: str) -> List[float]:
    """
    Embed the given text into a vector.
    """
    client = AsyncOpenAI()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Truncate
    text_to_embed = text[:50_000]

    resp = await client.embeddings.create(
        model=model,
        input=text_to_embed,
    )
    return resp.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


class SearchRequest(BaseModel):
    prompt: str
    limit: int = 10


@app.post("/search-stories")
async def search_stories(request: SearchRequest):
    """
    Search top 500 HN stories using cosine similarity.
    Returns top 10 (or specified limit) most relevant stories.
    """
    # Step 1: Fetch top 500 stories
    print("Fetching top 500 stories...")
    stories = await get_top_stories(limit=200)  # change to 500 when in production
    print(f"Fetched {len(stories)} stories")

    # Step 2: Fetch text content for each story
    print("Fetching text content from URLs...")
    story_texts = []
    for i, story in enumerate(stories):
        text = await fetch_url_text(story)
        story_texts.append(text)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(stories)} stories")

    # Step 3: Embed all story texts
    print("Generating embeddings for stories...")
    story_embeddings = []
    # Process in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(story_texts), batch_size):
        batch = story_texts[i : i + batch_size]
        batch_embeddings = await asyncio.gather(*[embed_text(text) for text in batch])
        story_embeddings.extend(batch_embeddings)
        print(
            f"Embedded {min(i + batch_size, len(story_texts))}/{len(story_texts)} stories"
        )

    # Step 4: Embed the search prompt
    print(f"Embedding search prompt: '{request.prompt}'")
    prompt_embedding = await embed_text(request.prompt)

    # Step 5: Calculate cosine similarity for each story
    print("Calculating cosine similarities...")
    similarities = []
    for i, story_embedding in enumerate(story_embeddings):
        similarity = cosine_similarity(prompt_embedding, story_embedding)
        similarities.append(
            {
                "story": stories[i],
                "similarity": float(similarity),
                "text_preview": story_texts[i][:200],  # First 200 chars as preview
            }
        )

    # Step 6: Sort by similarity and return top results
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[: request.limit]

    return {
        "query": request.prompt,
        "total_stories_searched": len(stories),
        "results": [
            {
                "title": result["story"].get("title"),
                "url": result["story"].get("url"),
                "hn_url": f"https://news.ycombinator.com/item?id={result['story'].get('id')}",
                "score": result["story"].get("score"),
                "similarity": result["similarity"],
                "text_preview": result["text_preview"],
            }
            for result in top_results
        ],
    }


@app.get("/hi")
async def read_root():
    return {"message": "Waleikum Hi!"}
