"""
embeddings.py

Generates vector embeddings for document chunks and builds a FAISS
index for fast similarity search.

Why FAISS: comparing a query embedding to every chunk embedding with
a naive loop works, but doesn't scale. FAISS is a library built for
efficient nearest-neighbor search over sets of vectors — it's what
keeps retrieval fast as the number of documents grows.

Similarity metric: all-MiniLM-L6-v2 is trained for cosine similarity.
Embeddings are L2-normalized at encoding time, which makes the inner
product (IndexFlatIP) equal to the cosine similarity. Scores therefore
range from -1 to 1, and higher = more similar — which also makes it
easy to apply a minimum relevance threshold at search time.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

# sentence-transformers (and its torch dependency) is optional: memory-
# constrained deployments (e.g. Render's free tier) skip it and embed
# queries through the Hugging Face Inference API instead — same model,
# same vectors, a fraction of the memory footprint.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.ingestion import Chunk, process_directory

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Fully-qualified name required by the Inference API.
API_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "data/processed"


class ApiEmbedder:
    """
    Drop-in replacement for SentenceTransformer.encode() that computes
    embeddings through the Hugging Face Inference API. Used when the
    EMBEDDINGS_VIA_API env var is set, or when sentence-transformers is
    not installed. Same model as the local path, so the vectors are
    interchangeable with the prebuilt FAISS index.
    """

    def __init__(self, model_name: str = API_EMBEDDING_MODEL_NAME):
        from huggingface_hub import InferenceClient

        self.model_name = model_name
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        vectors = np.asarray(
            [self.client.feature_extraction(t, model=self.model_name) for t in texts],
            dtype="float32",
        )
        if vectors.ndim == 3:  # some providers return token-level embeddings
            vectors = vectors.mean(axis=1)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer | ApiEmbedder:
    """
    Load the sentence-transformers embedding model (cached: loading the
    model takes a few seconds, so it must happen once per process, not
    once per query).

    all-MiniLM-L6-v2 is small (~80MB), fast, runs on CPU, and produces
    384-dimensional embeddings — a solid default for a project that
    needs to run without a GPU and deploy easily on HF Spaces.

    When EMBEDDINGS_VIA_API is set (or sentence-transformers is not
    installed), queries are embedded through the Inference API instead.
    """
    if os.getenv("EMBEDDINGS_VIA_API") or SentenceTransformer is None:
        print("Embedding queries via the Hugging Face Inference API.")
        return ApiEmbedder()
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of Chunk objects into a matrix of embeddings.
    Shape: (num_chunks, embedding_dim). Vectors are L2-normalized so
    that inner-product search is equivalent to cosine similarity.
    """
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")  # FAISS requires float32


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from a matrix of normalized embeddings.

    IndexFlatIP does an exhaustive (brute-force) inner-product search,
    which equals cosine similarity on normalized vectors. It's not the
    fastest option for millions of vectors, but for a project-scale
    corpus (a handful of documents, a few thousand chunks at most)
    it's simple, exact, and fast enough.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
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
    index_file = input_path / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No index found in {input_dir}. Build it first with: python -m src.embeddings"
        )
    index = faiss.read_index(str(index_file))

    with open(input_path / "chunks.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = 3,
    min_score: float = 0.0,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Embed a query and retrieve the top_k most similar chunks.

    Returns matching chunk metadata (text, source, page) plus a cosine
    similarity score (higher = more similar, max 1.0).

    `min_score` filters out weak matches: FAISS always returns the
    *nearest* neighbors, even for a question that has nothing to do
    with the documents. A threshold turns "nearest" into "near enough",
    so the pipeline can honestly answer "I don't know" instead of
    passing irrelevant chunks to the LLM.

    `source_filter` restricts results to a single document (by
    filename). FAISS itself has no notion of metadata, so this is done
    by over-fetching neighbors and keeping only those from the wanted
    source, until top_k results are collected.
    """
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # When filtering by source, fetch enough neighbors that top_k of
    # them can plausibly come from the wanted document.
    fetch_k = index.ntotal if source_filter else top_k
    scores, indices = index.search(query_vector, fetch_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 if fewer than fetch_k results exist
            continue
        score = float(scores[0][rank])
        if score < min_score:
            continue
        chunk_data = metadata[idx]
        if source_filter and chunk_data["source"] != source_filter:
            continue
        chunk_data = chunk_data.copy()
        chunk_data["score"] = score
        results.append(chunk_data)
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    # Build the index from scratch: run `python -m src.embeddings`
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
            print(f"  [{r['source']} p.{r['page']}] (score={r['score']:.3f}) {r['text'][:100]}...")
