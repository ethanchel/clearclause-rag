"""
app.py

Gradio interface for ClearClause, at the repository root so Hugging
Face Spaces picks it up as the app entry point.

Three tabs:
- Chat: conversational interface with history, document selector and
  source citations for every answer.
- RAG vs LLM-only: side-by-side comparison of a grounded answer and a
  general-knowledge answer to the same question.
- Evaluation: dashboard displaying the metrics produced by
  `python -m evaluation.evaluate`.

Run locally from the project root with: python app.py
"""

# On Hugging Face Spaces (ZeroGPU hosting), the `spaces` package must be
# imported before torch-touching imports, and the app must define at least
# one @spaces.GPU function — even though ClearClause does its real work on
# CPU (FAISS + embeddings) and the Inference API. Locally the package is
# absent and everything below is skipped.
try:
    import spaces
except ImportError:
    spaces = None

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

from src.rag_pipeline import (
    answer_question,
    answer_without_rag,
    list_documents,
    warm_up,
)

if spaces is not None:

    @spaces.GPU(duration=10)
    def _zerogpu_placeholder():
        """Required by ZeroGPU hosting; never called by the app."""
        return "ok"


ALL_DOCUMENTS = "All documents"

# Friendly labels for the document selector. Unknown files fall back
# to their filename, so adding a new PDF requires no code change.
DOC_LABELS = {
    "hud_model_lease_subsidized_programs.pdf": "Lease — HUD model lease (subsidized programs)",
    "website_terms_of_service.pdf": "Terms of Service — sample website ToS",
    "nfip_flood_insurance_summary.pdf": "Insurance — NFIP flood insurance summary",
}

RESULTS_PATH = Path("evaluation/results.json")


def document_choices() -> list[str]:
    try:
        files = list_documents()
    except FileNotFoundError:
        files = []
    return [ALL_DOCUMENTS] + [DOC_LABELS.get(f, f) for f in files]


def label_to_file(label: str) -> str | None:
    if label == ALL_DOCUMENTS:
        return None
    for filename, doc_label in DOC_LABELS.items():
        if doc_label == label:
            return filename
    return label  # label is already a raw filename


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    blocks = []
    for s in sources:
        blocks.append(
            f"**{s['source']}**, page {s['page']} "
            f"(relevance: {s['score']:.0%})\n> {s['text'][:300]}..."
        )
    return "\n\n".join(blocks)


def clean_history(history: list[dict]) -> list[dict]:
    """
    Keep only what the LLM API expects from the Gradio chat history:
    role + plain text content, for user/assistant messages.
    """
    cleaned = []
    for msg in history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            cleaned.append({"role": role, "content": content})
    return cleaned


def chat(message: str, history: list[dict], doc_label: str):
    history = list(history or [])
    if not message.strip():
        return history, "", ""

    try:
        result = answer_question(
            message,
            source_filter=label_to_file(doc_label),
            history=clean_history(history),
        )
        answer = result["answer"]
        sources_md = format_sources(result["sources"])
    except (FileNotFoundError, RuntimeError) as exc:
        answer = f"⚠️ {exc}"
        sources_md = ""

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, "", sources_md


def compare(question: str):
    if not question.strip():
        return "Please enter a question.", ""
    try:
        rag_result = answer_question(question)
        rag_md = rag_result["answer"]
        if rag_result["sources"]:
            rag_md += "\n\n---\n**Sources**\n\n" + format_sources(rag_result["sources"])
        baseline_md = answer_without_rag(question)
    except (FileNotFoundError, RuntimeError) as exc:
        return f"⚠️ {exc}", ""
    return rag_md, baseline_md


def load_dashboard():
    if not RESULTS_PATH.exists():
        return (
            "No evaluation results found. Run `python -m evaluation.evaluate` "
            "from the project root first, then reload this tab.",
            pd.DataFrame(),
        )

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        return "evaluation/results.json is empty.", pd.DataFrame()

    n = len(results)
    avg_rag = sum(r["rag_rouge1_f1"] for r in results) / n
    avg_baseline = sum(r["baseline_rouge1_f1"] for r in results) / n
    hits = [r["retrieval_hit"] for r in results if r.get("retrieval_hit") is not None]
    p_hits = [r["page_hit"] for r in results if r.get("page_hit") is not None]

    summary = [
        f"### Evaluation summary — {n} questions",
        f"- **Average ROUGE-1 F1 (RAG):** {avg_rag:.3f}",
        f"- **Average ROUGE-1 F1 (LLM-only baseline):** {avg_baseline:.3f}",
    ]
    if hits:
        summary.append(f"- **Retrieval hit rate (document):** {sum(hits) / len(hits):.1%}")
    if p_hits:
        summary.append(f"- **Retrieval hit rate (exact page):** {sum(p_hits) / len(p_hits):.1%}")

    table = pd.DataFrame(
        [
            {
                "Question": r["question"],
                "ROUGE-1 (RAG)": round(r["rag_rouge1_f1"], 3),
                "ROUGE-1 (baseline)": round(r["baseline_rouge1_f1"], 3),
                "Doc hit": r.get("retrieval_hit"),
                "Page hit": r.get("page_hit"),
            }
            for r in results
        ]
    )
    return "\n".join(summary), table


with gr.Blocks(title="ClearClause") as demo:
    gr.Markdown("# ClearClause")
    gr.Markdown(
        "Ask questions about legal and administrative documents. "
        "Answers are grounded in the source text, with citations."
    )

    with gr.Tab("Chat"):
        doc_selector = gr.Dropdown(
            choices=document_choices(),
            value=ALL_DOCUMENTS,
            label="Document",
            info="Restrict answers to a single document, or search across all of them.",
        )
        chatbot = gr.Chatbot(label="Conversation", height=420)
        with gr.Row():
            chat_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. What happens if I pay my rent late?",
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Accordion("Sources for the last answer", open=False):
            chat_sources = gr.Markdown()
        gr.ClearButton([chatbot, chat_input, chat_sources], value="Clear conversation")
        gr.Examples(
            examples=[
                "What happens if I pay my rent late?",
                "Can I have a pet in the unit?",
                "How old do I have to be to create an account?",
                "Are my belongings in the basement covered against flooding?",
            ],
            inputs=chat_input,
        )

        chat_args = {
            "fn": chat,
            "inputs": [chat_input, chatbot, doc_selector],
            "outputs": [chatbot, chat_input, chat_sources],
        }
        send_btn.click(**chat_args)
        chat_input.submit(**chat_args)

    with gr.Tab("RAG vs LLM-only"):
        gr.Markdown(
            "Ask the same question with and without retrieval. The RAG answer "
            "is grounded in the documents; the baseline answers from the "
            "model's general knowledge only."
        )
        compare_input = gr.Textbox(
            label="Question",
            placeholder="e.g. How long is the waiting period before flood coverage begins?",
        )
        compare_btn = gr.Button("Compare", variant="primary")
        with gr.Row():
            rag_output = gr.Markdown(label="With RAG", container=True)
            baseline_output = gr.Markdown(label="LLM only", container=True)

        compare_btn.click(fn=compare, inputs=compare_input, outputs=[rag_output, baseline_output])
        compare_input.submit(fn=compare, inputs=compare_input, outputs=[rag_output, baseline_output])

    with gr.Tab("Evaluation"):
        refresh_btn = gr.Button("Load evaluation results")
        dashboard_summary = gr.Markdown()
        dashboard_table = gr.Dataframe(interactive=False)

        refresh_btn.click(fn=load_dashboard, outputs=[dashboard_summary, dashboard_table])
        demo.load(fn=load_dashboard, outputs=[dashboard_summary, dashboard_table])


if __name__ == "__main__":
    # Preload (and if needed build) the embedding model + index so the
    # first question is fast. On a fresh deployment this is where the
    # index gets built from the PDFs in data/raw/.
    try:
        warm_up()
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")

    if os.getenv("RENDER"):
        # Render routes traffic to the port in $PORT and requires
        # binding on all interfaces.
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
    else:
        demo.launch()
