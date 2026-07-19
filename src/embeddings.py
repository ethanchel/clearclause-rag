"""
embeddings.py

Generates vector embeddings for document chunks and builds a FAISS
index for fast similarity search.

Why FAISS: comparing a query embedding to every chunk embedding with
a naive loop works, but doesn't scale. FAISS is a library built for
efficient nearest-neighbor search over sets of vectors — it's what
keeps retrieval fast as the number of documents grows.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion import Chunk, process_directory

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "data/processed"


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence-transformers embedding model.
    all-MiniLM-L6-v2 is small (~80MB), fast, runs on CPU, and produces
    384-dimensional embeddings — a solid default for a project that
    needs to run without a GPU and deploy easily on HF Spaces.
    """
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of Chunk objects into a matrix of embeddings.
    Shape: (num_chunks, embedding_dim)
    """
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")  # FAISS requires float32


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from a matrix of embeddings.

    IndexFlatL2 does an exhaustive (brute-force) search using L2
    (Euclidean) distance. It's not the fastest option for millions of
    vectors, but for a project-scale corpus (a handful of documents,
    a few thousand chunks at most) it's simple, exact, and fast enough.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk], output_dir: str = INDEX_DIR) -> None:
    """
    Persist the FAISS index and the chunk metadata to disk.

    FAISS only stores vectors, not the original text or metadata — it
    just returns positions (integer ids) when searched. So we save the
    list of chunks separately, in the same order they were added to
    the index. Position i in the index always corresponds to position
    i in this metadata list.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_path / "index.faiss"))

    metadata = [
        {
            "text": c.text,
            "source": c.source,
            "chunk_id": c.chunk_id,
            "page": c.page,
        }
        for c in chunks
    ]
    with open(output_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved index ({index.ntotal} vectors) and metadata to {output_dir}")


def load_index(input_dir: str = INDEX_DIR) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously saved FAISS index and its associated metadata.
    """
    input_path = Path(input_dir)
    index = faiss.read_index(str(input_path / "index.faiss"))

    with open(input_path / "chunks.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """
    Embed a query and retrieve the top_k most similar chunks.

    Returns matching chunk metadata (text, source, page) plus a
    distance score (lower = more similar, since we use L2 distance).
    """
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
            continue
        chunk_data = metadata[idx].copy()
        chunk_data["distance"] = float(distances[0][rank])
        results.append(chunk_data)

    return results


if __name__ == "__main__":
    # Build the index from scratch: run `python src/embeddings.py`
    # from the project root after adding PDFs to data/raw/
    model = get_embedding_model()
    chunks = process_directory()

    if not chunks:
        print("No chunks to index. Add PDFs to data/raw/ first.")
    else:
        embeddings = embed_chunks(chunks, model)
        index = build_faiss_index(embeddings)
        save_index(index, chunks)

        # quick sanity check
        test_query = "cancellation policy"
        metadata_preview = [
            {"text": c.text, "source": c.source, "chunk_id": c.chunk_id, "page": c.page}
            for c in chunks
        ]
        results = search(test_query, index, metadata_preview, model)
        print(f"\nTest query: '{test_query}'")
        for r in results:
            print(f"  [{r['source']} p.{r['page']}] (dist={r['distance']:.3f}) {r['text'][:100]}...")
