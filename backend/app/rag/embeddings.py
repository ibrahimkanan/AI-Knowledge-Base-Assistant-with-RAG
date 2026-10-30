from sentence_transformers import SentenceTransformer
from app.core.config import settings

# Global singleton instance so the model is loaded into memory/GPU only once
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-loads and returns the multilingual SentenceTransformer model (1024 dimensions)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"--> Loading Multilingual Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """
    Generates normalized 1024-dimensional embeddings for a list of Arabic/English texts.
    L2 normalization ensures cosine similarity is equivalent to inner product in pgvector.
    """
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Generates a normalized 1024-dimensional embedding for a single user search query."""
    model = get_embedding_model()
    vector = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()