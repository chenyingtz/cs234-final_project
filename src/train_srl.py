"""
SRL training entrypoint: load config, prepare model, run GRPO training with resume support.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo_trainer import GRPOTrainer
from .utils import set_seed


def get_device() -> torch.device:
    """Return best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SRL GRPO training")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (overrides other args)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument("--data", type=str, default="data/srl_instances.jsonl", help="SRL instances JSONL")
    parser.add_argument("--output-dir", type=str, default="checkpoints/srl", help="Checkpoint output dir")
    parser.add_argument("--num-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (prompts)")
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt (G)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="GRPO clip epsilon")
    parser.add_argument("--eps-std", type=float, default=1e-4, help="Min reward std for dynamic filter")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--val-data", type=str, default=None, help="Validation JSONL path for best-checkpoint selection")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        mapping = {
            "model": "model", "data": "data", "output_dir": "output_dir",
            "num_steps": "num_steps", "batch_size": "batch_size", "group_size": "group_size",
            "max_new_tokens": "max_new_tokens", "temperature": "temperature",
            "lr": "lr", "clip_epsilon": "clip_epsilon", "eps_std": "eps_std",
            "kl_coef": "kl_coef", "checkpoint_every": "checkpoint_every", "seed": "seed",
            "resume": "resume", "val_data": "val_data", "eval_every": "eval_every",
        }
        for k, v in cfg.items():
            attr = mapping.get(k, k)
            if hasattr(args, attr):
                setattr(args, attr, v)

    set_seed(args.seed)
    device = get_device()

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = AutoModelForCausalLM.from_pretrained(args.resume, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.resume, trust_remote_code=True)
        step_file = Path(args.resume) / "trainer_step.txt"
        if step_file.exists():
            start_step = int(step_file.read_text().strip())
            print(f"Resuming from step {start_step}, {args.num_steps - start_step} steps remaining")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    #model.gradient_checkpointing_enable()
    model = model.to(device)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        ref_model=None,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        kl_coef=0.0,
        clip_epsilon=args.clip_epsilon,
        eps_std=args.eps_std,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    trainer.train(
        data_path=args.data,
        num_steps=args.num_steps,
        device=device,
        start_step=start_step,
        val_data_path=getattr(args, "val_data", None),
        eval_every=getattr(args, "eval_every", 50),
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
