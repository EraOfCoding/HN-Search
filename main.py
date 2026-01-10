import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import numpy as np
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel


load_dotenv()

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
http_client: Optional[httpx.AsyncClient] = None
external_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, external_http_client
    print("Starting up - creating HTTP clients...")
    http_client = httpx.AsyncClient(base_url=HN_BASE_URL, timeout=30.0)
    external_http_client = httpx.AsyncClient(
        timeout=10.0,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
    print("HTTP clients created successfully")
    yield
    print("Shutting down - closing HTTP clients...")
    if http_client:
        await http_client.aclose()
    if external_http_client:
        await external_http_client.aclose()
    print("HTTP clients closed")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_http_client():
    if http_client is None:
        raise HTTPException(
            status_code=503,
            detail="HTTP client not initialized. Server is starting up.",
        )
    return http_client


def get_external_http_client():
    if external_http_client is None:
        raise HTTPException(
            status_code=503,
            detail="External HTTP client not initialized. Server is starting up.",
        )
    return external_http_client


async def fetch_item(item_id: int) -> Dict[str, Any]:
    """Helper function to fetch a single item by ID."""
    try:
        client = get_http_client()
        response = await client.get(f"/item/{item_id}.json")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Failed to fetch item {item_id}: {e}")
        return None


@app.get("/top-stories")
async def get_top_stories(limit: int = Query(500, le=500)):
    """
    Fetches the top stories from Hacker News.
    """
    try:
        client = get_http_client()
        response = await client.get("/topstories.json")
        ids = response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Could not reach Hacker News")

    target_ids = ids[:limit]

    tasks = [fetch_item(id) for id in target_ids]
    stories = await asyncio.gather(*tasks)

    return [story for story in stories if story]


async def fetch_url_text(story: Dict[str, Any]) -> str:
    """
    Fetch the text content of url.
    Returns the extracted text or falls back to title if URL fetch fails.
    """
    title = story.get("title", "")

    if "url" not in story or not story["url"]:
        text = story.get("text", "")
        return f"{title}. {text}".strip() if title or text else "No content"

    url = story["url"]

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return title or "No title"
    except:
        return title or "No title"

    try:
        client = get_external_http_client()
        response = await client.get(url)

        if response is None or response.status_code != 200:
            return title or "No title"

        content_type = ""
        try:
            if hasattr(response, "headers") and response.headers:
                content_type = response.headers.get("content-type", "").lower()
        except:
            pass

        if "text" in content_type or not content_type:
            try:
                import re

                text_content = response.text[:5000]
                text_content = re.sub(r"<[^>]+>", " ", text_content)
                text_content = re.sub(r"\s+", " ", text_content).strip()

                if text_content:
                    return f"{title}. {text_content}".strip()
            except:
                pass

        return title or "No title"

    except Exception as e:
        print(f"Error fetching {url}: {type(e).__name__}")
        return title or "No title"


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


async def embed_stories(
    stories: List[Dict[str, Any]], batch_size: int = 50
) -> Tuple[List[str], List[List[float]]]:
    """
    Fetch text content and generate embeddings for a list of stories.

    Args:
        stories: List of HN story dictionaries
        batch_size: Number of stories to embed in each batch

    Returns:
        Tuple of (story_texts, story_embeddings)
    """
    print(f"Fetching text content from URLs for {len(stories)} stories...")
    story_texts = []
    for i, story in enumerate(stories):
        text = await fetch_url_text(story)
        story_texts.append(text)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(stories)} stories")

    print("Generating embeddings for stories...")
    story_embeddings = []

    for i in range(0, len(story_texts), batch_size):
        batch = story_texts[i : i + batch_size]
        batch_embeddings = await asyncio.gather(*[embed_text(text) for text in batch])
        story_embeddings.extend(batch_embeddings)
        print(
            f"Embedded {min(i + batch_size, len(story_texts))}/{len(story_texts)} stories"
        )

    return story_texts, story_embeddings


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

    print("Fetching top 500 stories...")
    stories = await get_top_stories(limit=200)  # change to 500 when in production
    print(f"Fetched {len(stories)} stories")

    story_texts, story_embeddings = await embed_stories(stories)

    print(f"Embedding search prompt: '{request.prompt}'")
    prompt_embedding = await embed_text(request.prompt)

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


@app.get("/")
async def read_root():
    return {"message": "Hey there!"}
