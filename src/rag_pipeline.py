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

GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
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
def _get_client() -> InferenceClient:
    """Create the Hugging Face Inference API client once per process."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Copy .env.example to .env and add your "
            "Hugging Face token (on Hugging Face Spaces, add HF_TOKEN as a "
            "Space secret)."
        )
    return InferenceClient(token=token)


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


def _chat_completion(messages: list[dict], client: InferenceClient) -> str:
    """Send a chat completion request and unwrap the answer text."""
    try:
        response = client.chat_completion(
            messages=messages,
            model=GENERATION_MODEL,
            max_tokens=400,
            temperature=0.2,  # low: we want grounded, consistent answers, not creative ones
        )
    except Exception as exc:
        raise RuntimeError(
            f"The generation request to the Hugging Face API failed: {exc}"
        ) from exc
    return response.choices[0].message.content


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
