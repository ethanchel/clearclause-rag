"""
app.py

Gradio interface for ClearClause. Lets a user type a question about
the ingested documents and see the plain-language answer along with
the source passages it was grounded in.
"""

import gradio as gr

from src.rag_pipeline import answer_question


def ask(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    result = answer_question(question)

    sources_text = "\n\n".join(
        f"**{s['source']}**, page {s['page']} (distance: {s['distance']:.3f})\n> {s['text'][:300]}..."
        for s in result["sources"]
    )

    return result["answer"], sources_text


with gr.Blocks(title="ClearClause") as demo:
    gr.Markdown("# ClearClause")
    gr.Markdown(
        "Ask a question about your legal or administrative document. "
        "Answers are grounded in the source text, with citations."
    )

    question_input = gr.Textbox(
        label="Your question",
        placeholder="e.g. Can I cancel before the end of the contract?",
    )
    submit_btn = gr.Button("Ask")

    answer_output = gr.Markdown(label="Answer")
    sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=ask,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )
    question_input.submit(
        fn=ask,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )

if __name__ == "__main__":
    demo.launch()
