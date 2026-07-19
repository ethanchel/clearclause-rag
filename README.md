---
title: ClearClause
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# ClearClause

**An AI assistant that explains your legal and administrative documents in plain language — with sources.**

ClearClause is a Retrieval-Augmented Generation (RAG) application that lets users ask questions about legal or administrative documents (terms of service, lease agreement, insurance contract...) in natural language. The assistant retrieves the relevant clauses and generates a plain-language answer, always citing the exact source passage and page.

Final project — GenAI & Machine Learning Bootcamp, Developer Institute.

## Problem

Most people accept or sign legal documents without truly understanding them, due to dense legal language and lack of time. ClearClause makes these documents queryable and understandable, while staying grounded in the source text (no hallucinated legal advice).

## Features

**MVP**
- PDF ingestion of legal/administrative documents (sentence-aware chunking)
- Semantic search over document chunks (embeddings + FAISS, cosine similarity with a relevance threshold)
- Plain-language answer generation with cited source passages and pages
- Chat interface with conversation history (Gradio)

**Additional / Bonus**
- RAG vs. LLM-only side-by-side comparison tab on the same questions
- Evaluation dashboard tab with accuracy metrics against a ground-truth Q&A set (ROUGE, document and page retrieval hit rates)
- Multi-document support with document-type selection (lease, terms of service, insurance)
- LoRA fine-tuning experiment on legal Q&A data (`finetuning/`)

## Documents Used

All source documents are public-domain or public-information based — no confidential or personal data:

- `hud_model_lease_subsidized_programs.pdf` — model lease adapted from HUD form 90105a (U.S. Government work, public domain)
- `website_terms_of_service.pdf` — generic sample terms of service adapted from public-domain U.S. Government website policies
- `nfip_flood_insurance_summary.pdf` — adapted from the FEMA National Flood Insurance Program Summary of Coverage (U.S. Government work, public domain)

## Tech Stack

- **Language:** Python
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS (`IndexFlatIP` over normalized embeddings = cosine similarity)
- **Generation:** Mistral-7B-Instruct via the Hugging Face Inference API
- **Interface:** Gradio (chat + comparison + evaluation dashboard)
- **Evaluation:** ROUGE + retrieval hit rates, RAG vs. LLM-only baseline
- **Fine-tuning (bonus):** LoRA via PEFT on a lightweight open-source model
- **Deployment:** Hugging Face Spaces

## Project Structure

```
clearclause-rag/
├── app.py              # Gradio interface (Spaces entry point)
├── data/
│   ├── raw/            # source PDFs
│   └── processed/      # FAISS index + chunk metadata (auto-built)
├── src/
│   ├── ingestion.py    # PDF loading + sentence-aware chunking
│   ├── embeddings.py   # embedding generation + FAISS index (cosine similarity)
│   └── rag_pipeline.py # retrieval + generation, relevance threshold, chat history
├── evaluation/
│   ├── qa_dataset.json # ground-truth Q&A set (20 pairs across 3 documents)
│   └── evaluate.py     # RAG vs. baseline evaluation (ROUGE + retrieval hit rates)
├── finetuning/
│   ├── lora_finetune.py        # bonus: LoRA fine-tuning experiment
│   └── requirements-lora.txt   # extra deps for the bonus only
├── notebooks/          # exploration notebooks
├── requirements.txt
└── .env.example
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Hugging Face token:
   ```bash
   cp .env.example .env
   ```
4. (Optional) Add more source PDF documents to `data/raw/`.

## Usage

The `run.sh` script is the single entry point (it creates the venv and installs dependencies automatically if needed):

```bash
./run.sh          # launch the app (FAISS index auto-built on first launch)
./run.sh index    # rebuild the index after adding documents to data/raw/
./run.sh eval     # run the RAG vs. LLM-only evaluation
./run.sh lora     # run the bonus LoRA fine-tuning experiment
```

Equivalent manual commands (from the project root, venv activated): `python app.py`, `python -m src.embeddings`, `python -m evaluation.evaluate`, `python -m finetuning.lora_finetune`. Evaluation results appear in the console, in `evaluation/results.json`, and in the app's Evaluation tab.

## Deployment (Hugging Face Spaces)

1. Create a new Space (SDK: **Gradio**).
2. Push this repository to the Space (the YAML header of this README configures the Space; `app.py` at the root is the entry point).
3. In the Space settings, add a secret named `HF_TOKEN` with your Hugging Face token (used for the Inference API).
4. On first startup the Space builds the FAISS index from the PDFs in `data/raw/` automatically.

## Status

✅ MVP functional — evaluation results pending.

## Evaluation Results

_TODO — run `python -m evaluation.evaluate` and report the numbers here._
