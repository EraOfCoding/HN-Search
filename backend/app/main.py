from contextlib import asynccontextmanager
from typing import Optional
import httpx
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.hn_client import HNClient
from app.scraper import ContentScraper
from app.embeddings import EmbeddingService
from app.vector_store import PineconeStore

hn_http_client: Optional[httpx.AsyncClient] = None
external_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hn_http_client, external_http_client

    print("Starting up...")
    hn_http_client = httpx.AsyncClient(base_url=settings.hn_base_url, timeout=30.0)
    external_http_client = httpx.AsyncClient(
        timeout=10.0,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
    print("HTTP clients ready")

    yield

    print("Shutting down...")
    if hn_http_client:
        await hn_http_client.aclose()
    if external_http_client:
        await external_http_client.aclose()
    print("Closed")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    prompt: str
    limit: int = 50


class UpdateDatabaseRequest(BaseModel):
    limit: int = 500


async def ingest_stories_to_pinecone(limit: int):
    """Fetch stories, embed them, and store in Pinecone."""
    hn_client = HNClient(hn_http_client)
    scraper = ContentScraper(external_http_client)
    embedder = EmbeddingService()
    vector_store = PineconeStore()

    print(f"Fetching top {limit} stories from Hacker News...")
    stories = await hn_client.get_top_stories(limit)
    print(f"Fetched {len(stories)} stories")

    print("Scraping content from URLs...")
    contents = []
    for i, story in enumerate(stories):
        content = await scraper.fetch_url_text(story)
        contents.append(content)
        if (i + 1) % 50 == 0:
            print(f"Scraped {i + 1}/{len(stories)} stories")

    print("Generating embeddings...")
    embeddings = await embedder.embed_batch(contents)

    print("Storing stories in Pinecone...")
    vector_store.upsert_stories(stories, embeddings)
    print(f"Successfully stored {len(stories)} stories in Pinecone!")

    return {
        "status": "success",
        "stories_processed": len(stories),
        "message": f"Successfully ingested {len(stories)} stories into Pinecone",
    }


@app.get("/")
async def root():
    return {
        "message": "HN Semantic Search API",
        "endpoints": {
            "/top-stories": "Get latest HN stories",
            "/update-database": "Ingest stories into Pinecone",
            "/search-stories": "Search stories using semantic search",
            "/stats": "Get database statistics",
        },
    }


@app.get("/top-stories")
async def get_top_stories(limit: int = Query(500, le=500)):
    """Fetch top stories directly from Hacker News API."""
    hn_client = HNClient(hn_http_client)
    stories = await hn_client.get_top_stories(limit)
    return {"count": len(stories), "stories": stories}


@app.post("/update-database")
async def update_database(
    request: UpdateDatabaseRequest, background_tasks: BackgroundTasks
):
    """
    Fetch top HN stories, embed them, and store in Pinecone.
    Runs in background to avoid timeout on large ingestions.
    """
    # For small updates, run synchronously
    if request.limit <= 100:
        result = await ingest_stories_to_pinecone(request.limit)
        return result

    # For large updates, run in background
    background_tasks.add_task(ingest_stories_to_pinecone, request.limit)

    return {
        "status": "processing",
        "message": f"Started ingesting {request.limit} stories in background",
        "note": "This may take several minutes. Check /stats to see progress.",
    }


@app.post("/search-stories")
async def search_stories(request: SearchRequest):
    """
    Search stories using semantic similarity.
    Queries Pinecone vector database for relevant stories.
    """
    embedder = EmbeddingService()
    vector_store = PineconeStore()

    print(f"Embedding search query: '{request.prompt}'")
    query_embedding = await embedder.embed_text(request.prompt)

    print("Searching Pinecone for similar stories...")
    results = vector_store.search(query_embedding, request.limit)

    return {"query": request.prompt, "results_count": len(results), "results": results}


@app.get("/stats")
async def get_stats():
    """Get statistics about the Pinecone database."""
    vector_store = PineconeStore()
    stats = vector_store.get_stats()

    return {
        "database": "Pinecone",
        "total_stories": stats.get("total_vectors", 0),
        "embedding_dimension": stats.get("dimension", 1536),
        "status": "operational",
    }


@app.delete("/clear-database")
async def clear_database():
    """
    Clear all data from Pinecone index.
    USE WITH CAUTION - This deletes all stored stories!
    """
    vector_store = PineconeStore()

    # Delete all vectors from the index
    vector_store.index.delete(delete_all=True)

    return {
        "status": "success",
        "message": "All stories deleted from Pinecone",
        "warning": "You'll need to run /update-database to repopulate",
    }
