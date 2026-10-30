import json
from typing import AsyncGenerator
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.rate_limit import check_rate_limit
from app.db.session import get_db, AsyncSessionLocal
from app.db.models import User, UsageLog
from app.rag.retriever import retrieve_relevant_chunks, RetrievedChunk

router = APIRouter(prefix="/api/chat", tags=["RAG Chat & Streaming"])

SYSTEM_INSTRUCTION = """You are a bilingual (Arabic + English) AI Knowledge-Base Assistant.
Answer the user's question strictly using ONLY the provided context sources.
Rules:
1. Respond in the EXACT same language as the user's question (Arabic or English).
2. Cite your sources inline after every claim using the format: [Filename, p.X, para.Y].
3. If the answer cannot be found in the provided context, state clearly that the uploaded documents do not contain this information. Never invent facts."""


# ==========================================
# Pydantic Schemas
# ==========================================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["What are the best video ideas mentioned in the document?"])
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranker: bool = Field(default=True, description="Enable Cross-Encoder re-ranking")


class CitationSchema(BaseModel):
    filename: str
    page_number: int
    paragraph_index: int
    snippet: str
    vector_similarity: float
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]


# ==========================================
# Helper Functions
# ==========================================
def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks with explicit citation headers for the LLM."""
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        header = f"[Source {idx}: {c.filename}, p.{c.page_number}, para.{c.paragraph_index}]"
        context_blocks.append(f"{header}\n{c.content}")

    joined_context = "\n\n".join(context_blocks)
    return f"Context Sources:\n{joined_context}\n\nUser Question: {question}"


def format_citations(chunks: list[RetrievedChunk]) -> list[CitationSchema]:
    """Converts internal RetrievedChunk objects into clean API citation responses."""
    return [
        CitationSchema(
            filename=c.filename,
            page_number=c.page_number,
            paragraph_index=c.paragraph_index,
            snippet=c.content[:240] + ("..." if len(c.content) > 240 else ""),
            vector_similarity=c.vector_similarity,
            rerank_score=c.rerank_score,
        )
        for c in chunks
    ]


# ==========================================
# Endpoints
# ==========================================
@router.post("/query", response_model=ChatResponse)
async def query_knowledge_base(
    payload: ChatRequest,
    current_user: User = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Standard JSON RAG endpoint (great for testing in Swagger /docs):
    Retrieves & re-ranks tenant chunks, generates grounded answer with citations, and logs token usage.
    """
    chunks = await retrieve_relevant_chunks(
        db=db,
        user_id=current_user.id,
        query=payload.question,
        top_k=payload.top_k,
        use_reranker=payload.use_reranker,
    )

    if not chunks:
        return ChatResponse(
            answer="No processed documents found in your workspace. Please upload a document first.",
            citations=[],
        )

    citations = format_citations(chunks)
    prompt = build_rag_prompt(payload.question, chunks)

    # Check if Gemini API Key is configured
    if not settings.GEMINI_API_KEY or "replace" in settings.GEMINI_API_KEY or "paste" in settings.GEMINI_API_KEY:
        fallback_answer = (
            "[Retrieval Successful - Configure GEMINI_API_KEY in .env for LLM generation]\n\n"
            f"Top matched passage from [{chunks[0].filename}, p.{chunks[0].page_number}, para.{chunks[0].paragraph_index}]:\n"
            f"{chunks[0].content}"
        )
        return ChatResponse(answer=fallback_answer, citations=citations)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )

    # Extract token usage metadata and save to UsageLog
    in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
    out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    db.add(
        UsageLog(
            user_id=current_user.id,
            action="rag_query",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
    )
    await db.commit()

    return ChatResponse(answer=response.text or "", citations=citations)


@router.post("/stream")
async def stream_knowledge_base(
    payload: ChatRequest,
    current_user: User = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Server-Sent Events (SSE) Streaming endpoint for the React UI:
    1. Emits 'citations' event immediately after pgvector + Cross-Encoder retrieval.
    2. Emits 'token' events as the LLM generates the grounded answer in real time.
    3. Logs token usage and emits 'done' event.
    """
    chunks = await retrieve_relevant_chunks(
        db=db,
        user_id=current_user.id,
        query=payload.question,
        top_k=payload.top_k,
        use_reranker=payload.use_reranker,
    )
    citations = [c.model_dump() for c in format_citations(chunks)]
    prompt = build_rag_prompt(payload.question, chunks) if chunks else ""
    user_id = current_user.id

    async def event_generator() -> AsyncGenerator[str, None]:
        # Event 1: Send retrieved source citations immediately to the UI
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations}, ensure_ascii=False)}\n\n"

        if not chunks:
            msg = "No processed documents found in your workspace."
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if not settings.GEMINI_API_KEY or "replace" in settings.GEMINI_API_KEY or "paste" in settings.GEMINI_API_KEY:
            fallback = (
                f"Matched [{chunks[0].filename}, p.{chunks[0].page_number}, para.{chunks[0].paragraph_index}]: "
                f"{chunks[0].content}"
            )
            yield f"data: {json.dumps({'type': 'token', 'content': fallback}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        in_tokens = 0
        out_tokens = 0

        stream = await client.aio.models.generate_content_stream(
            model=settings.GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )

        async for chunk in stream:
            if chunk.text:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.text}, ensure_ascii=False)}\n\n"
            if chunk.usage_metadata:
                in_tokens = getattr(chunk.usage_metadata, "prompt_token_count", in_tokens) or in_tokens
                out_tokens = getattr(chunk.usage_metadata, "candidates_token_count", out_tokens) or out_tokens

        # Record usage asynchronously in a fresh session after streaming completes
        async with AsyncSessionLocal() as log_db:
            log_db.add(
                UsageLog(
                    user_id=user_id,
                    action="rag_stream_query",
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            )
            await log_db.commit()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")