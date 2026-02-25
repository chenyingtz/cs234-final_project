"""
RLVR (outcome-based GRPO) training script.


- batch size ≈ 128 (prompts)
- learning_rate = 5e-7
- rollout temperature = 1.0
- rollout number (num_generations) = 8
- KL loss coeff (beta) = 0.0
- dtype = bf16 (if available)

This script can be run:
1. After SRL training, using the SRL checkpoint as initialization:
   python -m src.train_rlvr_grpo \\
     --init-from checkpoints/srl/step_500 \\
     --output-dir checkpoints/srl_rlvr

2. From the base model (if --init-from is not provided):
   python -m src.train_rlvr_grpo \\
     --output-dir checkpoints/srl_rlvr

After training, point `configs/models_config.json["models"]["srl_rlvr"]["model_path"]`
to the RLVR checkpoint directory (e.g. `checkpoints/srl_rlvr`) so that the
existing evaluation pipeline can pick it up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import torch

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"


@dataclass
class RLVRConfig:
    init_from: str
    output_dir: str
    dataset_name: str
    max_train_samples: int | None
    max_eval_samples: int | None

    # Paper hyperparameters (Table 6)
    learning_rate: float = 5e-7
    batch_size: int = 128
    num_generations: int = 8
    num_train_epochs: int = 3  # paper uses steps; epochs is a practical proxy
    beta: float = 0.0  # KL coeff


def build_prompt(example: Dict[str, Any]) -> str:
    """
    Build a simple instruction-style prompt from a s1K-style example.
    Assumes fields: "problem" and "solution" (as in s1K-1.1).
    RLVR reward will check final answer correctness.
    """
    problem = example.get("problem") or example.get("question") or ""
    return (
        "You are a helpful math assistant. Solve the following problem step by step, "
        "then give the final answer clearly.\n\n"
        f"Problem:\n{problem}\n\nAnswer:"
    )


def extract_final_answer(text: str) -> str:
    """
    Heuristic extractor for final scalar answer from model output.
    For a real project, you may want to reuse the AIME-style extractor in utils.py.
    """
    # Very simple heuristic: take the last number in the string.
    import re

    numbers = re.findall(r"-?\\d+\\.?\\d*", text)
    return numbers[-1] if numbers else ""


def accuracy_reward_func(samples: list[Dict[str, Any]]) -> list[float]:
    """
    Simple RLVR reward: 1.0 if final answer matches ground truth, else 0.0.

    The GRPOTrainer in TRL v0.28.0 will call this with a list of dicts containing:
      - "prompt": str
      - "completion": str (model-generated completion)
      - The dataset's "completion" column contains the ground truth solution
    """
    rewards: list[float] = []
    for s in samples:
        # Model-generated completion
        model_completion: str = s.get("completion", "")
        
        # Ground truth is stored in the dataset's "completion" column
        # In TRL v0.28.0, the original dataset row might be in metadata
        # or we need to compare against the dataset's completion column
        # For now, we'll extract from the sample's original data
        # Note: This may need adjustment based on how TRL v0.28.0 passes data
        ground_truth: str = ""
        if "metadata" in s and isinstance(s["metadata"], dict):
            ground_truth = s["metadata"].get("completion", "")
        elif "original_completion" in s:
            ground_truth = s["original_completion"]
        
        # Extract final answers
        gt = extract_final_answer(ground_truth)
        pred = extract_final_answer(model_completion)

        rewards.append(1.0 if gt and pred == gt else 0.0)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR GRPO training (SRL → RLVR)")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to SRL checkpoint to initialize RLVR (e.g. checkpoints/srl/step_500). "
             "If not provided or empty, defaults to base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/srl_rlvr",
        help="Directory to save RLVR checkpoints (default: checkpoints/srl_rlvr)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset name (default: simplescaling/s1K-1.1)",
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
        default=256,
        help="Optional: limit number of eval samples",
    )

    args = parser.parse_args()

    cfg = RLVRConfig(
        init_from=args.init_from,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    base_model = get_base_model()
    print(f"Base model (for tokenizer): {base_model}")
    
    # Default to base model if --init-from is not provided or is empty
    if not args.init_from or args.init_from.strip() == "":
        init_from = base_model
        print(f"Initializing RLVR from base model: {init_from}")
    else:
        init_from = args.init_from
        print(f"Initializing RLVR from SRL checkpoint: {init_from}")
    
    # Update cfg with the determined init_from path
    cfg.init_from = init_from

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load dataset
    # Note: s1K-1.1 only has "train" split, so we split it manually
    dataset = load_dataset(cfg.dataset_name, split="train")
    
    # Split train/val from the dataset (same approach as train_sft.py)
    EVAL_SIZE = 60
    TRAIN_SIZE = len(dataset) - EVAL_SIZE
    
    raw_train = dataset.select(range(TRAIN_SIZE))
    raw_eval = dataset.select(range(TRAIN_SIZE, (TRAIN_SIZE + EVAL_SIZE)))

    if cfg.max_train_samples is not None:
        raw_train = raw_train.select(range(min(cfg.max_train_samples, len(raw_train))))
    if cfg.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(cfg.max_eval_samples, len(raw_eval))))

    def map_to_prompts(example: Dict[str, Any]) -> Dict[str, Any]:
        # GRPOTrainer expects specific column names in TRL v0.28.0
        # Use standard column names: "prompt" and "completion"
        return {
            "prompt": build_prompt(example),
            "completion": example.get("solution", ""),  # Use "completion" instead of "solution"
        }

    train_dataset = raw_train.map(map_to_prompts)
    eval_dataset = raw_eval.map(map_to_prompts)

    # 2. Model loading config
    # Note: Quantization cannot be used for fine-tuning base models
    # Quantized models require PEFT adapters for training
    # We only use quantization if explicitly needed for checkpoint loading (disabled by default)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }
    
    # Quantization is disabled to allow full fine-tuning
    # If memory is constrained, consider using gradient checkpointing or smaller batch sizes instead
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    # )
    # model_kwargs["quantization_config"] = quantization_config

    # 3. GRPO training configuration (match Table 6)
    # Paper hyperparameters (Table 6):
    # - batch size ≈ 128 (prompts)
    # - learning_rate = 5e-7
    # - rollout temperature = 1.0
    # - rollout number (num_generations) = 8
    # - KL loss coeff (beta) = 0.0
    # - dtype = bf16 (if available)
    
    per_device_train_batch_size = 1
    gradient_accumulation_steps = cfg.batch_size // per_device_train_batch_size

    # GRPOConfig in TRL v0.28.0 extends TrainingArguments
    # Note: model is passed to GRPOTrainer, not to GRPOConfig
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        logging_steps=10,
        log_level="info",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        num_generations=cfg.num_generations,
        max_completion_length=512,
        temperature=1.0,
        beta=cfg.beta,  # KL coeff
        model_init_kwargs=model_kwargs,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # 4. Initialize GRPOTrainer
    # Note: In TRL v0.28.0:
    # - tokenizer is not passed directly (loaded automatically from model)
    # - prompt_column and completion_column are not parameters
    # - Dataset should have "prompt" and "completion" columns (standard names)
    trainer = GRPOTrainer(
        model=cfg.init_from,
        reward_funcs=[accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 5. Start training
    trainer.train()


if __name__ == "__main__":
    main()

