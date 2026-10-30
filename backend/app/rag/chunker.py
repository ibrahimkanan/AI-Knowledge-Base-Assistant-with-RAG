from dataclasses import dataclass
from app.rag.parser import ParsedParagraph


@dataclass
class TextChunk:
    """Represents a chunk ready for embedding, retaining its source page and paragraph citation."""
    page_number: int
    paragraph_index: int
    content: str


def chunk_paragraphs(
    paragraphs: list[ParsedParagraph],
    chunk_size: int = 120,
    chunk_overlap: int = 25,
) -> list[TextChunk]:
    """
    Splits parsed paragraphs into word-based sliding window chunks while preserving
    page_number and paragraph_index metadata for accurate citations.

    Args:
        paragraphs: List of ParsedParagraph objects from the parser.
        chunk_size: Maximum number of words per chunk (default 120 words ~ 600-800 chars).
        chunk_overlap: Number of overlapping words between consecutive chunks of a long paragraph.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap

    for para in paragraphs:
        words = para.text.split()

        # If the paragraph fits inside a single chunk, keep it intact
        if len(words) <= chunk_size:
            chunks.append(
                TextChunk(
                    page_number=para.page_number,
                    paragraph_index=para.paragraph_index,
                    content=para.text,
                )
            )
            continue

        # Otherwise, slide a window with overlap across the long paragraph
        for start_idx in range(0, len(words), step):
            window_words = words[start_idx : start_idx + chunk_size]
            if not window_words:
                break

            chunk_text = " ".join(window_words)
            chunks.append(
                TextChunk(
                    page_number=para.page_number,
                    paragraph_index=para.paragraph_index,
                    content=chunk_text,
                )
            )

            # Stop if this window already reached the end of the paragraph
            if start_idx + chunk_size >= len(words):
                break

    return chunks