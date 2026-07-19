# ClearClause

**An AI assistant that explains your legal and administrative documents in plain language — with sources.**

ClearClause is a Retrieval-Augmented Generation (RAG) application that lets users upload a legal or administrative document (terms of service, lease agreement, insurance contract...) and ask questions about it in natural language. The assistant retrieves the relevant clauses and generates a plain-language answer, always citing the exact source passage.

Final project — GenAI & Machine Learning Bootcamp, Developer Institute.

## Problem

Most people accept or sign legal documents without truly understanding them, due to dense legal language and lack of time. ClearClause makes these documents queryable and understandable, while staying grounded in the source text (no hallucinated legal advice).

## Features

**MVP**
- PDF ingestion of legal/administrative documents
- Semantic search over document chunks (embeddings + FAISS)
- Plain-language answer generation with cited source passages
- Simple chat interface (Gradio)

**Additional / Bonus**
- RAG vs. LLM-only comparison on the same questions
- Evaluation dashboard with accuracy metrics against a ground-truth Q&A set
- LoRA fine-tuning experiment on legal Q&A data

## Tech Stack

- **Language:** Python
- **Embeddings:** sentence-transformers
- **Vector store:** FAISS
- **Generation:** Hugging Face Inference API
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces

## Project Structure

```
clearclause-rag/
├── data/
│   ├── raw/            # source PDFs
│   └── processed/      # chunked/preprocessed text
├── src/
│   ├── ingestion.py    # PDF loading + chunking
│   ├── embeddings.py   # embedding generation + FAISS index (WIP)
│   └── rag_pipeline.py # retrieval + generation (WIP)
├── evaluation/
│   ├── qa_dataset.json # ground-truth Q&A set (WIP)
│   └── evaluate.py     # RAG vs. baseline evaluation (WIP)
├── app/
│   └── app.py          # Gradio interface (WIP)
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
4. Add source PDF documents to `data/raw/`.

## Status

🚧 Work in progress — MVP under active development.

## Evaluation Results

_TODO — to be filled in once the evaluation pipeline is complete._
