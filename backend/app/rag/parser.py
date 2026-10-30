import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import pymupdf as fitz

@dataclass
class ParsedParagraph:
    """Represents an extracted paragraph with exact source citation coordinates."""
    page_number: int
    paragraph_index: int
    text: str


def clean_bilingual_text(raw_text: str) -> str:
    """
    Normalizes Arabic and English text extracted from PDFs:
    1. Applies Unicode NFKC normalization to convert Arabic presentation forms/ligatures
       into standard searchable Unicode characters.
    2. Removes zero-width and control characters.
    3. Collapses multiple spaces/newlines into clean single spaces.
    """
    # Normalize Unicode characters (critical for Arabic ligatures like 'لا')
    normalized = unicodedata.normalize("NFKC", raw_text)

    # Remove invisible control characters and zero-width joiners
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]", "", normalized)

    # Replace multiple whitespace/newlines inside a paragraph block with a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_document(file_path: str | Path) -> tuple[int, list[ParsedParagraph]]:
    """
    Parses a PDF or plain-text document into structured paragraphs with page and paragraph numbers.
    Returns (total_pages, list_of_parsed_paragraphs).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {path}")

    suffix = path.suffix.lower()
    paragraphs: list[ParsedParagraph] = []

    # Case 1: Plain text or Markdown files (.txt, .md)
    if suffix in {".txt", ".md"}:
        raw_content = path.read_text(encoding="utf-8", errors="ignore")
        raw_blocks = [b for b in re.split(r"\n\s*\n", raw_content) if b.strip()]
        for idx, block in enumerate(raw_blocks, start=1):
            cleaned = clean_bilingual_text(block)
            if len(cleaned) >= 15:  # Ignore tiny noise fragments
                paragraphs.append(
                    ParsedParagraph(page_number=1, paragraph_index=idx, text=cleaned)
                )
        return 1, paragraphs

    # Case 2: PDF documents (.pdf) using PyMuPDF block extraction
    if suffix == ".pdf":
        doc = fitz.open(str(path))
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            # Extract layout blocks: each tuple is (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            # Sort blocks top-to-bottom (y0) to preserve reading order
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

            para_counter = 0
            for block in blocks:
                block_text, block_type = block[4], block[6]
                # block_type == 0 indicates a text block (ignores image blocks)
                if block_type != 0:
                    continue

                cleaned = clean_bilingual_text(block_text)
                if len(cleaned) >= 15:  # Skip page headers/numbers shorter than 15 chars
                    para_counter += 1
                    paragraphs.append(
                        ParsedParagraph(
                            page_number=page_num,
                            paragraph_index=para_counter,
                            text=cleaned,
                        )
                    )

        doc.close()
        return total_pages, paragraphs

    raise ValueError(f"Unsupported file format: {suffix}. Only .pdf, .txt, and .md are supported.")