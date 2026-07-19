#!/usr/bin/env bash
# Single entry point for the ClearClause project.
#
#   ./run.sh            launch the app (default)
#   ./run.sh setup      create the venv and install dependencies
#   ./run.sh index      (re)build the FAISS index from data/raw/
#   ./run.sh eval       run the RAG vs. LLM-only evaluation
#   ./run.sh lora       run the bonus LoRA fine-tuning experiment
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python

find_python() {
    # Prefer 3.12: newest version verified to work with all dependencies.
    for candidate in python3.12 python3.11 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return
        fi
    done
    echo "python3"
}

pip_install() {
    # Venvs created by uv ship without pip; bootstrap it on demand.
    "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null
    "$PY" -m pip install -q "$@"
}

setup() {
    if [ ! -x "$PY" ]; then
        echo ">> Creating virtual environment..."
        "$(find_python)" -m venv venv
    fi
    echo ">> Installing dependencies..."
    pip_install -r requirements.txt
    if [ ! -f .env ]; then
        cp .env.example .env
        echo ">> Created .env — add your Hugging Face token to it."
    fi
    echo ">> Setup done."
}

[ -x "$PY" ] || setup

case "${1:-app}" in
    setup) setup ;;
    app)   exec "$PY" app.py ;;
    index) exec "$PY" -m src.embeddings ;;
    eval)  exec "$PY" -m evaluation.evaluate ;;
    lora)  pip_install -r finetuning/requirements-lora.txt
           exec "$PY" -m finetuning.lora_finetune ;;
    *)     echo "Usage: ./run.sh [setup|app|index|eval|lora]" >&2; exit 1 ;;
esac
