from contextlib import asynccontextmanager
from arq import create_pool
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.rate_limit import redis_client
from app.db.session import init_db, get_db, engine
from app.worker.tasks import parse_redis_url
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Initialize pgvector database tables and ARQ Redis queue pool
    await init_db()
    app.state.arq_pool = await create_pool(parse_redis_url(settings.REDIS_URL))
    yield
    # Shutdown logic: Cleanly close ARQ pool, Redis client, and DB engine
    await app.state.arq_pool.close()
    await redis_client.aclose()
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Register API routers
app.include_router(auth_router)
app.include_router(documents_router)


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint to verify FastAPI and PostgreSQL + pgvector connectivity."""
    result = await db.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    vector_ext = result.scalar_one_or_none()
    
    return {
        "status": "ok",
        "pgvector_enabled": vector_ext == "vector"
    }