"""
Supervised Fine-Tuning (SFT) training script.

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
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"


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
    print(f"Default model: {default_model}")

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

    # 1. Load tokenizer first to get response template
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    #raw_eval = load_dataset(args.dataset_name, split="test", streaming=False)

    EVAL_SIZE=60
    TRAIN_SIZE=len(dataset) - EVAL_SIZE

    raw_train = dataset.select(range(TRAIN_SIZE))
    raw_eval = dataset.select(range(TRAIN_SIZE, (TRAIN_SIZE + EVAL_SIZE)))

    if args.max_train_samples is not None:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    if args.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))

    def formatting_func(example: Dict) -> Dict[str, str]:
        messages = build_chat_example(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = raw_train.map(formatting_func)
    eval_dataset = raw_eval.map(formatting_func)

    # 3. Configure SFTConfig with paper hyperparameters (Table 5)
    # Paper parameters:
    # - learning_rate = 5e-6
    # - global batch size ≈ 64
    # - scheduler = cosine
    # - warmup_ratio = 0.3
    # - num_train_epochs = 3
    # - dtype = bf16 (if available)
    
    # Calculate batch size for global batch ≈ 64
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 64  # Global batch size = 64
    
 
    training_config = SFTConfig(
        # Paper hyperparameters (Table 5)
        learning_rate=5e-6,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        
        # Output and logging
        output_dir=args.output_dir,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        
        # Data type (paper uses bf16)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # SFT-specific parameters (TRL v0.28.0)
        # completion_only_loss=True masks non-assistant tokens
        completion_only_loss=True,
        
        # Sequence length (max_seq_length might be in SFTConfig in v0.28.0)
        # Try without it first, or check if it's a different parameter name
    )

    # 4. Initialize SFT trainer with SFTConfig
    trainer = SFTTrainer(
        model=args.model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 5. Train
    trainer.train()

    # 6. Save final model (useful path for evaluation pipeline)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

