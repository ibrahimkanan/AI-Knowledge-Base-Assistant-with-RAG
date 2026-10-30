import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.core.rate_limit import check_rate_limit
from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models import User, Document, DocumentChunk, UsageLog

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    total_pages: int | None
    chunk_count: int = 0
    created_at: datetime


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Uploads a document (.pdf, .txt, .md) to the tenant's workspace, records usage,
    and enqueues an asynchronous ARQ background job for parsing, chunking, and embedding.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Create tenant-specific upload directory: ./uploads/<user_id>/
    tenant_dir = Path(settings.UPLOAD_DIR) / str(current_user.id)
    tenant_dir.mkdir(parents=True, exist_ok=True)

    # Save file with a unique UUID prefix to prevent filename collisions
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = tenant_dir / unique_filename

    content_bytes = await file.read()
    saved_path.write_bytes(content_bytes)

    # Create Document record in 'queued' state
    new_doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(saved_path),
        status="queued",
    )
    db.add(new_doc)

    # Log tenant usage action
    usage = UsageLog(user_id=current_user.id, action="document_upload")
    db.add(usage)
    await db.commit()
    await db.refresh(new_doc)

    # Enqueue background processing job in ARQ Redis pool
    arq_pool = request.app.state.arq_pool
    await arq_pool.enqueue_job("process_document_task", str(new_doc.id))

    return DocumentResponse(
        id=str(new_doc.id),
        filename=new_doc.filename,
        status=new_doc.status,
        total_pages=new_doc.total_pages,
        chunk_count=0,
        created_at=new_doc.created_at,
    )


@router.get("/", response_model=list[DocumentResponse])
async def list_my_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns all documents belonging strictly to the authenticated user (Multi-tenant isolation)."""
    stmt = (
        select(
            Document,
            func.count(DocumentChunk.id).label("chunk_count")
        )
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .where(Document.user_id == current_user.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        DocumentResponse(
            id=str(doc.id),
            filename=doc.filename,
            status=doc.status,
            total_pages=doc.total_pages,
            chunk_count=chunk_count,
            created_at=doc.created_at,
        )
        for doc, chunk_count in rows
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a document and its pgvector chunks (enforcing tenant ownership)."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Delete physical file from disk if it exists
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    await db.delete(doc)
    await db.commit()