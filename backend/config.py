"""
config.py - Central configuration management
"""
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class Settings(BaseSettings):
    # --- Groq API (LLM) ---
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    llm_model: str = Field(default="openai/gpt-oss-120b", env="LLM_MODEL")

    # --- MongoDB Atlas ---
    mongodb_url: str = Field(default="", env="MONGODB_URL")
    mongodb_db_name: str = Field(default="research_impact_db", env="MONGODB_DB_NAME")

    # --- JWT ---
    jwt_secret_key: str = Field(default="changeme_secret_key_32chars_min!!", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=60, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # --- File Storage ---
    papers_storage_path: str = Field(default="./data/papers", env="PAPERS_STORAGE_PATH")
    chroma_persist_path: str = Field(default="./data/chromadb", env="CHROMA_PERSIST_PATH")

    # --- Embedding (local, no API key) ---
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # --- App ---
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=True, env="DEBUG")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        """
        Prefer the project-local .env for local development so machine-wide
        environment variables do not silently override app secrets.
        """
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        """Accept common deployment strings like 'release' or 'development'."""
        if isinstance(value, bool) or value is None:
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "y", "on", "debug", "dev", "development"}
            falsy = {"0", "false", "no", "n", "off", "release", "prod", "production"}
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False

        return value

    def groq_key_configured(self) -> bool:
        """Return True when a non-placeholder Groq key is configured."""
        key = (self.groq_api_key or "").strip()
        return bool(key) and "your_groq_api_key_here" not in key.lower()

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.papers_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_path).mkdir(parents=True, exist_ok=True)
        Path("./data/logs").mkdir(parents=True, exist_ok=True)


# NOTE: ensure_directories() is NOT called here at import time.
# It is called explicitly in main.py startup and scripts.
settings = Settings()
