"""
RLVR (outcome-based GRPO) training script with LoRA.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

- batch size ≈ 128 (prompts)
- learning_rate = 5e-7
- rollout temperature = 1.0
- rollout number (num_generations) = 8
- KL loss coeff (beta) = 0.0
- dtype = bf16 (if available)
- LoRA rank = 16 (default)
- LoRA alpha = 32 (default)

This script can be run:
1. After SRL training, using the SRL checkpoint as initialization:
   python -m src.train_rlvr_grpo \\
     --init-from checkpoints/srl/step_500 \\
     --output-dir checkpoints/srl_rlvr

2. From the base model (if --init-from is not provided):
   python -m src.train_rlvr_grpo \\
     --output-dir checkpoints/srl_rlvr

After training, point `configs/models_config.json["models"]["srl_rlvr"]["model_path"]`
to the RLVR checkpoint directory (e.g. `checkpoints/srl_rlvr` or `checkpoints/srl_rlvr_merged`) 
so that the existing evaluation pipeline can pick it up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
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
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning",
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

    # 2. Load model and apply LoRA (if enabled)
    print(f"Loading model: {init_from}")
    model = AutoModelForCausalLM.from_pretrained(
        init_from,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not args.no_lora:
        # Configure LoRA for Qwen models
        # Target attention and MLP layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # For Qwen models, also target MLP layers if they exist
        try:
            # Check if model has gate_proj, up_proj, down_proj (typical for Qwen)
            first_layer = next(iter(model.model.layers)) if hasattr(model, 'model') else None
            if first_layer and hasattr(first_layer, 'mlp'):
                if hasattr(first_layer.mlp, 'gate_proj'):
                    target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        except:
            pass
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        print(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"Target modules: {target_modules}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Using full fine-tuning (LoRA disabled)")

    # 3. Model loading config for GRPOConfig (if needed)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }

    # 4. GRPO training configuration (match Table 6)
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

    # 5. Initialize GRPOTrainer
    # Note: In TRL v0.28.0:
    # - tokenizer is not passed directly (loaded automatically from model)
    # - prompt_column and completion_column are not parameters
    # - Dataset should have "prompt" and "completion" columns (standard names)
    # - Pass the model object directly (not model path) when using LoRA
    trainer = GRPOTrainer(
        model=model,  # Use the LoRA-enabled model
        reward_funcs=[accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Start training
    trainer.train()

    # 7. Save final model (useful path for evaluation pipeline)
    if not args.no_lora:
        # For LoRA, save the adapter weights
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        
        # Also merge and save the full model (for easier evaluation)
        # This creates a model that can be loaded without PEFT
        merged_model = model.merge_and_unload()
        merged_output_dir = cfg.output_dir + "_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Saved LoRA adapter to: {cfg.output_dir}")
        print(f"Saved merged model (for evaluation) to: {merged_output_dir}")
        print(f"Note: Use '{merged_output_dir}' in models_config.json for evaluation")
    else:
        # For full fine-tuning, save normally
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved model to: {cfg.output_dir}")


if __name__ == "__main__":
    main()

