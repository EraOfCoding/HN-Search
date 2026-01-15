from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    hn_base_url: str = "https://hacker-news.firebaseio.com/v0"

    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "hn-stories"

    class Config:
        env_file = ".env"


settings = Settings()
