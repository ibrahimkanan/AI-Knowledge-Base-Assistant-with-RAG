from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.db.session import init_db, get_db, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Initialize pgvector extension and database tables
    await init_db()
    yield
    # Shutdown logic: Cleanly dispose of the database connection pool
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint to verify FastAPI and PostgreSQL + pgvector connectivity."""
    # Execute a simple raw SQL query to ensure the database and pgvector are active
    result = await db.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    vector_ext = result.scalar_one_or_none()
    
    return {
        "status": "ok",
        "pgvector_enabled": vector_ext == "vector"
    }