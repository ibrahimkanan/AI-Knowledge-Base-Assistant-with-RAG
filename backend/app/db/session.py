from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.core.config import settings
from app.db.models import Base

# Disable SSL negotiation for local Docker development to prevent asyncpg connection_lost() on Windows
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"ssl": False}
)

# Factory for creating new AsyncSession instances per request
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db() -> None:
    """Initializes the pgvector extension and creates all database tables."""
    print(f"--> Connecting to Database at: {settings.DATABASE_URL.split('@')[-1]}")
    async with engine.begin() as conn:
        # Enable the pgvector extension inside PostgreSQL if not already enabled
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables defined in SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """FastAPI dependency that provides an async database session per request."""
    async with AsyncSessionLocal() as session:
        yield session