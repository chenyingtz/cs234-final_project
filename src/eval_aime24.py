"""
Evaluate models on AIME24 via lm-evaluation-harness.
Supports: greedy accuracy, Avg@N (N=1, 32) with HuggingFace or vLLM backend.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_lm_eval(
    model: str,
    model_path: str | None = None,
    backend: str = "hf",
    tasks: str = "aime24",
    batch_size: int = 1,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    max_gen_toks: int = 4096,
    output_path: str | None = None,
    device: str = "cuda",
    limit: int | None = None,
) -> str:
    """
    Run lm_eval. Returns output directory path.
    - backend: "hf" for HuggingFace local, "vllm" for local-completions (vLLM server)
    - For greedy: do_sample=False, num_return_sequences=1
    - For Avg@N: do_sample=True, num_return_sequences=N, temperature=1.0
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        n_str = f"avg{num_return_sequences}" if do_sample and num_return_sequences > 1 else "greedy"
        output_path = f"results/aime24_{n_str}_{timestamp}"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if backend == "hf":
        model_arg = model_path if model_path else model
        # Resolve local checkpoint paths to absolute so HuggingFace loads from disk (not Hub)
        if model_arg.startswith(("checkpoints/", "checkpoints\\", "./", ".\\")) or (model_arg.startswith("checkpoints") and len(Path(model_arg).parts) > 1):
            model_arg = str(Path(model_arg).resolve())
        else:
            p = Path(model_arg)
            if p.exists() and p.is_dir():
                model_arg = str(p.resolve())
        is_local = Path(model_arg).is_absolute() or model_arg.startswith(("checkpoints", "./", ".\\"))
        model_args_str = f"pretrained={model_arg},trust_remote_code=True"
        if is_local:
            model_args_str += ",local_files_only=True"
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args_str,
            "--tasks", tasks,
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", output_path,
            "--log_samples",
        ]
        # Use num_return_sequences: HF model.generate() does not accept 'n'
        gen_kwargs = f"temperature={temperature},do_sample={do_sample},num_return_sequences={num_return_sequences},max_gen_toks={max_gen_toks}"
    elif backend == "vllm":
        base_url = "http://127.0.0.1:8000/v1/completions"
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", f"model={model},base_url={base_url},tokenized_requests=False,trust_remote_code=True",
            "--tasks", tasks,
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", output_path,
            "--log_samples",
        ]
        gen_kwargs = f"temperature={temperature},do_sample={do_sample},n={num_return_sequences},max_gen_toks={max_gen_toks}"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    cmd.extend(["--gen_kwargs", gen_kwargs])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    with open(Path(output_path) / "eval_cmd.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")

    subprocess.run(cmd, check=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate on AIME24 via lm-eval")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name (for vLLM) or HF path")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (overrides --model for HF)")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Evaluation backend")
    parser.add_argument("--mode", choices=["greedy", "avg1", "avg32"], default="greedy",
        help="greedy=deterministic; avg1=1 sample; avg32=32 samples")
    parser.add_argument("--output-dir", type=str, default=None, help="Results output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task (for quick test runs)")
    parser.add_argument("--max-gen-toks", type=int, default=2048, help="Max tokens to generate per answer (default 2048; lower = faster)")
    args = parser.parse_args()

    if args.mode == "greedy":
        temperature = 0.0
        do_sample = False
        n = 1
    elif args.mode == "avg1":
        temperature = 1.0
        do_sample = True
        n = 1
    elif args.mode == "avg32":
        temperature = 1.0
        do_sample = True
        n = 32
    else:
        raise ValueError(args.mode)

    out = run_lm_eval(
        model=args.model,
        model_path=args.model_path,
        backend=args.backend,
        temperature=temperature,
        do_sample=do_sample,
        num_return_sequences=n,
        max_gen_toks=args.max_gen_toks,
        output_path=args.output_dir,
        device=args.device,
        limit=args.limit,
    )
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
