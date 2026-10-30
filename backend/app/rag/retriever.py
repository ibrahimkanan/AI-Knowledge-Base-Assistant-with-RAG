import asyncio
import uuid
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.db.models import DocumentChunk, Document
from app.rag.embeddings import embed_query

# Global singleton instance for the Cross-Encoder re-ranking model
_reranker_model: CrossEncoder | None = None


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk along with its source document citation and relevance score."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    paragraph_index: int
    content: str
    vector_similarity: float
    rerank_score: float | None = None


def get_reranker_model() -> CrossEncoder:
    """Lazy-loads and returns the multilingual Cross-Encoder re-ranker model."""
    global _reranker_model
    if _reranker_model is None:
        print(f"--> Loading Multilingual Re-ranker Model: {settings.RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
    return _reranker_model


def _score_pairs(query: str, passages: list[str]) -> list[float]:
    """Runs synchronous Cross-Encoder scoring over (query, passage) pairs."""
    reranker = get_reranker_model()
    pairs = [[query, passage] for passage in passages]
    scores = reranker.predict(pairs, show_progress_bar=False)
    return [float(s) for s in scores]


async def retrieve_relevant_chunks(
    db: AsyncSession,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    initial_k: int = 15,
    use_reranker: bool = True,
) -> list[RetrievedChunk]:
    """
    Executes Two-Stage Multi-Tenant Retrieval:
    Stage 1: Bi-Encoder semantic search in PostgreSQL using pgvector cosine distance.
    Stage 2: Cross-Encoder re-ranking (BAAI/bge-reranker-v2-m3) to boost precision and hit-rate.
    """
    # 1. Embed the user query into a 1024-d normalized vector
    query_vector = await asyncio.to_thread(embed_query, query)

    # 2. Stage 1: Fetch top candidate chunks from pgvector scoped strictly to user_id
    distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    candidate_limit = initial_k if use_reranker else top_k

    stmt = (
        select(DocumentChunk, Document.filename, distance_col)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.user_id == user_id,
            Document.status == "completed",
        )
        .order_by(distance_col.asc())
        .limit(candidate_limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    candidates: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            filename=filename,
            page_number=chunk.page_number,
            paragraph_index=chunk.paragraph_index,
            content=chunk.content,
            vector_similarity=round(1.0 - float(distance), 4),
        )
        for chunk, filename, distance in rows
    ]

    # If re-ranking is disabled (e.g., baseline evaluation) or only 1 chunk exists, return Stage 1 top_k
    if not use_reranker or len(candidates) <= 1:
        return candidates[:top_k]

    # 3. Stage 2: Cross-Encoder Re-Ranking in a worker thread
    passages = [c.content for c in candidates]
    scores = await asyncio.to_thread(_score_pairs, query, passages)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = round(score, 4)

    # Sort descending by Cross-Encoder relevance score and return top_k
    candidates.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -999.0, reverse=True)
    return candidates[:top_k]