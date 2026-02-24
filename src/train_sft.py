"""
Supervised Fine-Tuning (SFT) training script.

Implements SFT with the same hyperparameters as Table 5 in
“Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning”
(`2510.25992v1.pdf`):

- learning_rate = 5e-6
- global batch size ≈ 64
- scheduler = cosine
- warmup_ratio = 0.3
- num_train_epochs = 3
- dtype = bf16 (if available)

This script is intended to be run as:

    python -m src.train_sft \
      --output-dir checkpoints/sft

After training, point `configs/models_config.json["models"]["sft"]["model_path"]`
to the final checkpoint (e.g. `checkpoints/sft` or `checkpoints/sft/checkpoint-XXXX`)
so that the existing evaluation pipeline can pick it up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"


@dataclass
class SFTConfig:
    model_name: str
    output_dir: str
    dataset_name: str
    max_train_samples: int | None
    max_eval_samples: int | None

    # Paper hyperparameters (Table 5)
    learning_rate: float = 5e-6
    global_batch_size: int = 64
    num_train_epochs: int = 3
    warmup_ratio: float = 0.3
    lr_scheduler_type: str = "cosine"


def build_chat_example(example: Dict) -> str:
    """
    Turn a s1K-style example into a chat-style training text.
    Assumes fields: "problem" and "solution" (as in s1K-1.1).
    """
    problem = example.get("problem") or example.get("question") or ""
    solution = example.get("solution") or ""

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You should think step-by-step.",
        },
        {
            "role": "user",
            "content": problem,
        },
        {
            "role": "assistant",
            "content": solution,
        },
    ]

    # The actual chat template will be applied via tokenizer.apply_chat_template
    # in the mapping function below.
    return messages  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) training")
    default_model = get_base_model()

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Base model name or path (default: base_model from configs/models_config.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sft",
        help="Directory to save SFT checkpoints (default: checkpoints/sft)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset name for SFT (default: simplescaling/s1K-1.1)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional: limit number of training samples (for debugging)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=512,
        help="Optional: limit number of eval samples",
    )

    args = parser.parse_args()

    cfg = SFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    raw_train = load_dataset(cfg.dataset_name, split="train")
    raw_eval = load_dataset(cfg.dataset_name, split="validation", streaming=False)

    if cfg.max_train_samples is not None:
        raw_train = raw_train.select(range(min(cfg.max_train_samples, len(raw_train))))
    if cfg.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(cfg.max_eval_samples, len(raw_eval))))

    def formatting_func(example: Dict) -> Dict[str, str]:
        messages = build_chat_example(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        return {"text": text}

    train_dataset = raw_train.map(formatting_func)
    eval_dataset = raw_eval.map(formatting_func)

    # 3. Data collator: only learn on assistant tokens
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Approximate per-device batch + grad_accum to get global ≈ 64 for 1 GPU.
    per_device_train_batch_size = 1
    gradient_accumulation_steps = cfg.global_batch_size // per_device_train_batch_size

    # 4. Training arguments (match paper)
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),  # use bf16 if available
        report_to="none",
    )

    # 5. Initialize SFT trainer
    trainer = SFTTrainer(
        model=cfg.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=2048,
    )

    # 6. Train
    trainer.train()

    # Save final model (useful path for evaluation pipeline)
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()

