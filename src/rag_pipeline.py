"""
rag_pipeline.py

Ties retrieval (FAISS + embeddings) together with generation (an LLM
via the Hugging Face Inference API) to answer user questions about
the ingested documents, grounded in the retrieved source passages.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.embeddings import get_embedding_model, load_index, search

load_dotenv()

GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TOP_K = 3

SYSTEM_PROMPT = """You are ClearClause, an assistant that explains legal and \
administrative documents in plain, simple language. You must ONLY use the \
provided source passages to answer — never rely on outside knowledge. If the \
passages don't contain the answer, say so clearly instead of guessing. \
Always mention which source and page the answer comes from."""


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


def generate_answer(question: str, retrieved_chunks: list[dict], client: InferenceClient) -> str:
    """
    Call the LLM with the system prompt + the augmented user prompt,
    and return the generated answer.
    """
    user_prompt = build_prompt(question, retrieved_chunks)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=GENERATION_MODEL,
        max_tokens=400,
        temperature=0.2,  # low: we want grounded, consistent answers, not creative ones
    )
    return response.choices[0].message.content


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline for a single question:
    1. Load the FAISS index + embedding model
    2. Retrieve the top_k most relevant chunks
    3. Generate an answer grounded in those chunks
    4. Return the answer along with the sources used (for citation display)
    """
    model = get_embedding_model()
    index, metadata = load_index()

    retrieved_chunks = search(question, index, metadata, model, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information in the indexed documents.",
            "sources": [],
        }

    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    answer = generate_answer(question, retrieved_chunks, client)

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
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "Answer the question as best you can."},
            {"role": "user", "content": question},
        ],
        model=GENERATION_MODEL,
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test: run `python src/rag_pipeline.py` after building
    # the index with `python src/embeddings.py`
    test_question = "Can I cancel before the end of the contract?"
    result = answer_question(test_question)

    print(f"Question: {test_question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources used:")
    for src in result["sources"]:
        print(f"  - {src['source']} (page {src['page']}, distance={src['distance']:.3f})")
