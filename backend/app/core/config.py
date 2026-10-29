from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the exact path to the backend/.env file regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    # Application metadata
    PROJECT_NAME: str = "AI Knowledge-Base RAG Assistant"
    ENVIRONMENT: str = "development"
    
    # Use 127.0.0.1:5433 explicitly to hit the Docker container and bypass Windows IPv6 localhost
    DATABASE_URL: str = "postgresql+asyncpg://rag_user:rag_password@127.0.0.1:55432/rag_saas"    
    # Redis URL for caching, rate limiting, and ARQ background worker
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    
    # JWT Security configuration
    JWT_SECRET_KEY: str = "super-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # AI & RAG Pipeline configuration
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

    # File storage and per-user rate limiting
    UPLOAD_DIR: str = "./uploads"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 20

    # Load variables from the exact backend/.env path
    model_config = SettingsConfigDict(env_file=str(ENV_FILE_PATH), extra="ignore")

settings = Settings()