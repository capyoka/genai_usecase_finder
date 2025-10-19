from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "生成AI事例検索"
    data_dir: Path = Path("data")
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    fallback_embedding_dim: int = 1024
    openai_api_key: Optional[str] = None
    case_search_top_k: int = 50
    chat_search_top_k: int = 6
    default_page_size: int = 20
    max_page_size: int = 100

    class Config:
        env_prefix = "APP_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
