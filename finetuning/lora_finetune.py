"""
lora_finetune.py

Bonus experiment: LoRA fine-tuning of a lightweight open-source model
on legal Q&A data.

The idea: instead of retraining all the weights of a language model
(billions of parameters), LoRA (Low-Rank Adaptation) freezes the base
model and learns two small low-rank matrices per targeted layer. The
trainable parameters drop to a fraction of a percent of the model,
which makes fine-tuning possible on a laptop CPU/Apple Silicon — at
the cost of a smaller capacity to change the model's behavior.

This experiment fine-tunes a small instruct model on the project's
legal Q&A pairs, then compares the base model's answers with the
fine-tuned model's answers on a held-out question.

Extra dependencies (not needed by the main app):
    pip install -r finetuning/requirements-lora.txt

Run from the project root:
    python -m finetuning.lora_finetune
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Small enough to fine-tune on a laptop, good enough to follow
# instructions. Swap for a bigger model if a GPU is available.
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QA_DATASET_PATH = "evaluation/qa_dataset.json"
ADAPTER_DIR = "finetuning/lora_adapter"
HELD_OUT_QUESTIONS = 2  # kept out of training for the before/after check

SYSTEM_PROMPT = (
    "You are a legal assistant. Answer questions about legal and "
    "administrative documents in plain, simple language."
)


def load_pairs(path: str = QA_DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_chat_text(tokenizer, question: str, answer: str | None) -> str:
    """
    Format one Q&A pair with the model's own chat template. Training
    and inference must use the same template, otherwise the model sees
    a prompt format it was never trained on.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(tokenizer, pairs: list[dict]) -> Dataset:
    texts = [to_chat_text(tokenizer, p["question"], p["expected_answer"]) for p in pairs]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    return Dataset.from_dict({"text": texts}).map(tokenize, remove_columns=["text"])


def generate(model, tokenizer, question: str) -> str:
    prompt = to_chat_text(tokenizer, question, answer=None)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    pairs = load_pairs()
    train_pairs = pairs[:-HELD_OUT_QUESTIONS]
    held_out = pairs[-HELD_OUT_QUESTIONS:]
    print(f"Training on {len(train_pairs)} Q&A pairs, holding out {len(held_out)}.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

    # Baseline answers BEFORE fine-tuning, for the comparison below.
    print("\nGenerating baseline (pre-fine-tuning) answers...")
    baseline_answers = [generate(model, tokenizer, p["question"]) for p in held_out]

    lora_config = LoraConfig(
        r=8,                # rank of the low-rank update matrices
        lora_alpha=16,      # scaling factor (alpha/r scales the update)
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # attention projections only
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer, train_pairs)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir="finetuning/checkpoints",
            num_train_epochs=5,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            use_cpu=not torch.backends.mps.is_available(),
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    print(f"\nLoRA adapter saved to {ADAPTER_DIR}")

    # Before/after comparison on the held-out questions.
    print("\n=== Before / after fine-tuning (held-out questions) ===")
    model.eval()
    for pair, before in zip(held_out, baseline_answers):
        after = generate(model, tokenizer, pair["question"])
        print(f"\nQ: {pair['question']}")
        print(f"  expected : {pair['expected_answer'][:160]}")
        print(f"  base     : {before[:160]}")
        print(f"  fine-tuned: {after[:160]}")

    print(
        "\nNote: with only ~20 training pairs this is a methodology "
        "demonstration, not a production fine-tune. The interesting part "
        "is the pipeline (data formatting, LoRA config, before/after "
        "evaluation), which scales unchanged to a larger legal Q&A corpus."
    )


def load_finetuned():
    """Utility for notebooks: reload base model + trained adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    return model, tokenizer


if __name__ == "__main__":
    if not Path(QA_DATASET_PATH).exists():
        raise SystemExit(f"{QA_DATASET_PATH} not found — run from the project root.")
    main()
