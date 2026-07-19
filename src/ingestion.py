"""
ingestion.py

Handles loading PDF documents and splitting them into overlapping
text chunks suitable for embedding and retrieval.

Why chunking matters for RAG:
- Embedding models have a limited context window, and embedding a
  whole document as one vector loses precision (the vector becomes
  a blurry average of everything in the document).
- Smaller chunks give more precise retrieval: when a user asks a
  specific question, we want to retrieve the exact paragraph that
  answers it, not the entire document.
- Overlap between chunks prevents losing context at chunk boundaries
  (e.g. a sentence that gets cut in half between two chunks).
"""

from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Chunk:
    """A single chunk of text with metadata tracing it back to its source."""
    text: str
    source: str       # filename the chunk came from
    chunk_id: int      # position of this chunk within the document
    page: int          # page number the chunk starts on


def load_pdf_text(pdf_path: str) -> list[tuple[str, int]]:
    """
    Extract text from a PDF, page by page.

    Returns a list of (page_text, page_number) tuples.
    We keep page numbers because they're useful later for citing sources
    precisely (e.g. "see page 3").
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:  # skip empty pages (scanned images, blank pages, etc.)
            pages.append((text, i + 1))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size`
    characters, with `overlap` characters shared between consecutive
    chunks.

    This is a simple character-based splitter. A more advanced version
    could split on sentence boundaries instead of raw character counts
    (worth exploring later if retrieval quality isn't good enough).
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # move forward, keeping overlap

    return chunks


def process_document(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Full pipeline for one document: load PDF -> extract text per page
    -> chunk each page -> return a list of Chunk objects with metadata.
    """
    filename = Path(pdf_path).name
    pages = load_pdf_text(pdf_path)

    all_chunks = []
    chunk_counter = 0

    for page_text, page_number in pages:
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk_str in page_chunks:
            all_chunks.append(
                Chunk(
                    text=chunk_str,
                    source=filename,
                    chunk_id=chunk_counter,
                    page=page_number,
                )
            )
            chunk_counter += 1

    return all_chunks


def process_directory(
    raw_dir: str = "data/raw",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Process every PDF in a directory and return one combined list of
    Chunk objects across all documents.
    """
    raw_path = Path(raw_dir)
    all_chunks = []

    pdf_files = sorted(raw_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return all_chunks

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        chunks = process_document(str(pdf_file), chunk_size, overlap)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    # Quick manual test: run `python src/ingestion.py` from the project
    # root after adding at least one PDF to data/raw/
    chunks = process_directory()
    print(f"\nTotal chunks across all documents: {len(chunks)}")
    if chunks:
        print("\nExample chunk:")
        print(chunks[0])
