import asyncio
from urllib.parse import urlparse
from arq.connections import RedisSettings
from sqlalchemy import select, delete

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.models import Document, DocumentChunk
from app.rag.parser import parse_document
from app.rag.chunker import chunk_paragraphs
from app.rag.embeddings import embed_texts, get_embedding_model


def parse_redis_url(url: str) -> RedisSettings:
    """Converts a standard REDIS_URL string into an ARQ RedisSettings object."""
    parsed = urlparse(url)
    database = int(parsed.path.lstrip("/") or 0)
    return RedisSettings(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 6379,
        database=database,
    )


async def process_document_task(ctx: dict, document_id: str) -> dict:
    """
    Background job executed by ARQ worker:
    1. Marks document status as 'processing'.
    2. Parses Arabic/English PDF or text file into paragraphs.
    3. Splits paragraphs into citation-preserving chunks.
    4. Generates 1024-d BGE-M3 embeddings in batches.
    5. Stores chunks and vectors in PostgreSQL (pgvector) and marks 'completed'.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return {"status": "error", "reason": "Document not found"}

        try:
            # Step 1: Transition state to processing
            doc.status = "processing"
            await db.commit()

            # Step 2 & 3: Offload CPU-bound parsing and chunking to a thread
            total_pages, paragraphs = await asyncio.to_thread(parse_document, doc.file_path)
            chunks = await asyncio.to_thread(chunk_paragraphs, paragraphs)

            if not chunks:
                doc.status = "failed"
                await db.commit()
                return {"status": "failed", "reason": "No extractable text found in document"}

            # Step 4: Generate 1024-dim vector embeddings in a background thread
            texts_to_embed = [c.content for c in chunks]
            vectors = await asyncio.to_thread(embed_texts, texts_to_embed)

            # Remove any old chunks if this document is being re-processed
            await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))

            # Step 5: Persist chunks + embeddings into pgvector
            db_chunks = [
                DocumentChunk(
                    document_id=doc.id,
                    user_id=doc.user_id,
                    page_number=chunk.page_number,
                    paragraph_index=chunk.paragraph_index,
                    content=chunk.content,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            db.add_all(db_chunks)

            doc.total_pages = total_pages
            doc.status = "completed"
            await db.commit()

            return {
                "status": "completed",
                "document_id": str(doc.id),
                "total_pages": total_pages,
                "chunks_created": len(db_chunks),
            }

        except Exception as exc:
            await db.rollback()
            doc.status = "failed"
            await db.commit()
            print(f"[Worker Error] Failed processing document {document_id}: {exc}")
            raise


async def startup(ctx: dict) -> None:
    """Pre-loads the multilingual embedding model into RAM when the ARQ worker starts."""
    await asyncio.to_thread(get_embedding_model)


class WorkerSettings:
    """ARQ Worker configuration class invoked via CLI: `arq app.worker.tasks.WorkerSettings`."""
    functions = [process_document_task]
    on_startup = startup
    redis_settings = parse_redis_url(settings.REDIS_URL)