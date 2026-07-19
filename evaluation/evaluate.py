"""
evaluate.py

Evaluates the RAG pipeline against a ground-truth Q&A dataset, and
compares it to a no-RAG (LLM-only) baseline.

Two things are measured:
1. Retrieval accuracy: did the pipeline retrieve a chunk from the
   expected source document? (proves the retrieval step works)
2. Answer quality: how closely does the generated answer match the
   expected answer, using ROUGE (same metric family used in the
   Week 16 summarization evaluation).
"""

import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.rag_pipeline import answer_question, answer_without_rag

QA_DATASET_PATH = "evaluation/qa_dataset.json"


def load_qa_dataset(path: str = QA_DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(generated: str, expected: str) -> dict:
    """
    ROUGE compares n-gram overlap between the generated answer and the
    expected answer. It's not a perfect measure of correctness (two
    correct answers can be phrased very differently), but it's a
    consistent, reproducible signal for tracking answer quality across
    iterations of the pipeline.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(expected, generated)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def retrieval_hit(sources: list[dict], expected_source: str) -> bool:
    """Did the retrieved chunks include the expected source document?"""
    return any(s["source"] == expected_source for s in sources)


def page_hit(sources: list[dict], expected_source: str, expected_page: int) -> bool:
    """
    Stricter version of retrieval_hit: did the retrieved chunks include
    a chunk from the expected page of the expected document? This is
    the metric that validates the "cite the exact page" feature.
    """
    return any(
        s["source"] == expected_source and s["page"] == expected_page
        for s in sources
    )


def run_evaluation(qa_dataset: list[dict]) -> list[dict]:
    results = []

    for item in qa_dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_source = item.get("expected_source")
        expected_page = item.get("expected_page")

        # RAG answer
        rag_result = answer_question(question)
        rag_answer = rag_result["answer"]
        rag_rouge = compute_rouge(rag_answer, expected_answer)
        hit = retrieval_hit(rag_result["sources"], expected_source) if expected_source else None
        p_hit = (
            page_hit(rag_result["sources"], expected_source, expected_page)
            if expected_source and expected_page
            else None
        )

        # Baseline: LLM without any retrieved context
        baseline_answer = answer_without_rag(question)
        baseline_rouge = compute_rouge(baseline_answer, expected_answer)

        results.append({
            "question": question,
            "rag_answer": rag_answer,
            "rag_rouge1_f1": rag_rouge["rouge1_f1"],
            "retrieval_hit": hit,
            "page_hit": p_hit,
            "baseline_answer": baseline_answer,
            "baseline_rouge1_f1": baseline_rouge["rouge1_f1"],
        })

    return results


def summarize(results: list[dict]) -> None:
    n = len(results)
    avg_rag = sum(r["rag_rouge1_f1"] for r in results) / n
    avg_baseline = sum(r["baseline_rouge1_f1"] for r in results) / n
    hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    hit_rate = sum(hits) / len(hits) if hits else None
    p_hits = [r["page_hit"] for r in results if r["page_hit"] is not None]
    page_hit_rate = sum(p_hits) / len(p_hits) if p_hits else None

    print(f"\n=== Evaluation summary ({n} questions) ===")
    print(f"Average ROUGE-1 F1 — RAG:      {avg_rag:.3f}")
    print(f"Average ROUGE-1 F1 — Baseline: {avg_baseline:.3f}")
    if hit_rate is not None:
        print(f"Retrieval hit rate (document): {hit_rate:.1%}")
    if page_hit_rate is not None:
        print(f"Retrieval hit rate (exact page): {page_hit_rate:.1%}")


if __name__ == "__main__":
    dataset = load_qa_dataset()
    if not dataset:
        print("qa_dataset.json is empty — add ground-truth Q&A pairs first.")
    else:
        results = run_evaluation(dataset)
        summarize(results)

        Path("evaluation").mkdir(exist_ok=True)
        with open("evaluation/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nDetailed results saved to evaluation/results.json")
