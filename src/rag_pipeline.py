"""
rag_pipeline.py

Ties retrieval (FAISS + embeddings) together with generation (an LLM
via the Hugging Face Inference API) to answer user questions about
the ingested documents, grounded in the retrieved source passages.

Performance note: the embedding model and the FAISS index are loaded
once per process (lazily, on the first question) and reused for every
subsequent question. Loading them per query would add several seconds
of latency to each answer for no benefit — only the search and the
LLM call actually depend on the question.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.embeddings import (
    build_faiss_index,
    embed_chunks,
    get_embedding_model,
    load_index,
    save_index,
    search,
)
from src.ingestion import process_directory

load_dotenv()

GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Smaller sibling of the API model, used as an automatic fallback when
# the Inference API is unavailable (no token, or free credits used up).
# Runs locally through transformers — slower and a bit less fluent than
# the 7B, but keeps the whole pipeline working at zero cost.
LOCAL_GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 3
# Minimum cosine similarity for a chunk to count as relevant. Below
# this, the question is likely off-topic for the indexed documents and
# it's more honest to say "nothing found" than to let the LLM improvise
# on unrelated passages.
MIN_SCORE = 0.25
# How many previous conversation turns to pass back to the LLM so it
# can resolve follow-up questions ("and what about the deposit?").
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are ClearClause, an assistant that explains legal and \
administrative documents in plain, simple language. You must ONLY use the \
provided source passages to answer — never rely on outside knowledge. If the \
passages don't contain the answer, say so clearly instead of guessing. \
Always mention which source and page the answer comes from."""


@lru_cache(maxsize=1)
def _get_retriever():
    """
    Load the embedding model and the FAISS index exactly once per
    process. lru_cache turns this into a lazy singleton: the first call
    pays the loading cost, every later call returns the cached objects.

    If no index exists yet (e.g. first launch on a fresh Hugging Face
    Space), it is built automatically from the PDFs in data/raw/.
    """
    model = get_embedding_model()
    try:
        index, metadata = load_index()
    except FileNotFoundError:
        print("No index found — building it from data/raw/ ...")
        chunks = process_directory()
        if not chunks:
            raise
        index = build_faiss_index(embed_chunks(chunks, model))
        save_index(index, chunks)
        index, metadata = load_index()
    return model, index, metadata


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient | None:
    """
    Create the Hugging Face Inference API client once per process.
    Returns None when no token is configured — generation then runs on
    the local fallback model instead of failing.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set — generation will use the local model.")
        return None
    return InferenceClient(token=token)


@lru_cache(maxsize=1)
def _get_local_generator():
    """
    Load the local fallback generation model once per process. Uses
    Apple Silicon GPU (MPS) when available, CPU otherwise.
    """
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading local generation model {LOCAL_GENERATION_MODEL} on {device}...")
    return pipeline(
        "text-generation",
        model=LOCAL_GENERATION_MODEL,
        device=device,
        dtype="auto",
    )


def warm_up() -> None:
    """
    Preload the embedding model and index so the first user question
    doesn't pay the loading cost. Called at app startup.
    """
    _get_retriever()


def list_documents() -> list[str]:
    """Return the filenames of all indexed documents (for the UI selector)."""
    _, _, metadata = _get_retriever()
    return sorted({m["source"] for m in metadata})


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Assemble the retrieved chunks and the user's question into a single
    prompt for the LLM. This is the "augmented" part of Retrieval-
    Augmented Generation: instead of asking the model to answer from
    its own training data, we hand it the exact source text it should
    base its answer on.
    """
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Source: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""Context passages from the document:

{context}

Question: {question}

Answer in plain, simple language based only on the context above. \
Cite the source and page number for your answer."""
    return prompt


# Set to True after the first "out of credits" (402) response so later
# calls skip the doomed API round-trip and go straight to the local model.
_api_disabled = False


def _local_chat_completion(messages: list[dict]) -> str:
    """Generate an answer with the local fallback model."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "The generation API is unavailable and the local fallback model "
            "is not installed on this deployment. Please try again later."
        ) from None
    generator = _get_local_generator()
    output = generator(
        messages,
        max_new_tokens=400,
        do_sample=False,  # deterministic: grounded answers, not creative ones
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def _chat_completion(messages: list[dict], client: InferenceClient | None) -> str:
    """
    Send a chat completion request and unwrap the answer text.

    The Inference API is tried first (best quality). If no token is
    configured or the account has no credits left (HTTP 402), the local
    fallback model takes over so the app keeps working at zero cost.
    """
    global _api_disabled

    if client is not None and not _api_disabled:
        try:
            response = client.chat_completion(
                messages=messages,
                model=GENERATION_MODEL,
                max_tokens=400,
                temperature=0.2,  # low: we want grounded, consistent answers, not creative ones
            )
            return response.choices[0].message.content
        except Exception as exc:
            if "402" not in str(exc):
                raise RuntimeError(
                    f"The generation request to the Hugging Face API failed: {exc}"
                ) from exc
            _api_disabled = True
            print("Inference API credits exhausted — switching to the local model.")

    return _local_chat_completion(messages)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    client: InferenceClient,
    history: list[dict] | None = None,
) -> str:
    """
    Call the LLM with the system prompt, the recent conversation
    history (so follow-up questions make sense) and the augmented
    user prompt, and return the generated answer.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": build_prompt(question, retrieved_chunks)})
    return _chat_completion(messages, client)


def answer_question(
    question: str,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
    source_filter: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline for a single question:
    1. Retrieve the top_k most relevant chunks (cached model + index),
       optionally restricted to one document
    2. Drop chunks below the relevance threshold
    3. Generate an answer grounded in the remaining chunks, aware of
       the recent conversation history
    4. Return the answer along with the sources used (for citation display)

    `history` is a list of {"role": "user"|"assistant", "content": str}
    messages from earlier turns of the conversation.
    """
    model, index, metadata = _get_retriever()

    retrieved_chunks = search(
        question, index, metadata, model,
        top_k=top_k, min_score=min_score, source_filter=source_filter,
    )

    if not retrieved_chunks:
        scope = f"in {source_filter}" if source_filter else "in the indexed documents"
        return {
            "answer": (
                f"I couldn't find anything relevant to that question {scope}. "
                "Try rephrasing, or check that the question is about the "
                "selected document."
            ),
            "sources": [],
        }

    answer = generate_answer(question, retrieved_chunks, _get_client(), history=history)

    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }


def answer_without_rag(question: str) -> str:
    """
    Generate an answer WITHOUT any retrieved context — the LLM answers
    from its own general knowledge only. This exists specifically for
    the evaluation step: comparing RAG vs. LLM-only answers is what
    proves the RAG pipeline actually adds value over just asking the
    model directly.
    """
    return _chat_completion(
        [
            {"role": "system", "content": "Answer the question as best you can."},
            {"role": "user", "content": question},
        ],
        _get_client(),
    )


if __name__ == "__main__":
    # Quick manual test: run `python -m src.rag_pipeline` after building
    # the index with `python -m src.embeddings`
    test_question = "Can I cancel before the end of the contract?"
    result = answer_question(test_question)

    print(f"Question: {test_question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources used:")
    for src in result["sources"]:
        print(f"  - {src['source']} (page {src['page']}, score={src['score']:.3f})")
