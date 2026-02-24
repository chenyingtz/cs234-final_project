"""
Data prep pipeline: s1K-1.1 -> step-wise SRL training instances.
Parse expert solution into numbered steps; create N-1 instances per solution (steps 2..N).
Output JSONL: id, problem, steps, k, prompt, target_step.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from .prompts import build_srl_user_prompt
from .utils import save_jsonl


def parse_expert_steps(solution: str) -> list[str] | None:
    """
    Parse expert solution into numbered steps like "1. ...", "2. ...".
    Returns list of step strings, or None if parsing fails.
    """
    if not solution or not isinstance(solution, str):
        return None

    # Split by "N. " at line start; keep chunks
    parts = re.split(r"(?m)^\s*(\d+)\.\s*", solution)
    if len(parts) < 2:
        return None
    # parts[0] = leading text (often empty), parts[1]=num, parts[2]=text, parts[3]=num, ...
    steps = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            step_text = parts[i + 1].strip()
            if step_text:
                steps.append(step_text)
    if len(steps) < 2:
        return None
    return steps


def create_srl_instances(
    dataset_name: str = "simplescaling/s1K-1.1",
    split: str = "train",
    output_path: str | Path = "data/srl_instances.jsonl",
    question_key: str = "question",
    solution_key: str = "solution",
    max_examples: int | None = None,
) -> int:
    """
    Load s1K-1.1, parse steps, create SRL instances for steps 2..N.
    SRL: context = problem + expert steps 1..k-1, target = expert step k.
    Returns count of created instances.
    """
    ds = load_dataset(dataset_name, split=split)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    instances = []
    instance_id = 0

    for idx, item in enumerate(ds):
        if max_examples and len(instances) >= max_examples:
            break

        problem = item.get(question_key) or item.get("question", "")
        solution = item.get(solution_key) or item.get("solution", "")

        if not problem or not solution:
            continue

        steps = parse_expert_steps(solution)
        if not steps:
            continue

        # Create instances for steps 2..N (step index k=2 to k=N, 1-indexed)
        for k in range(2, len(steps) + 1):
            previous_steps = steps[: k - 1]
            target_step = steps[k - 1]
            prompt = build_srl_user_prompt(problem, previous_steps)

            instances.append({
                "id": f"s1k_{idx}_k{k}",
                "problem": problem,
                "steps": steps,
                "k": k,
                "prompt": prompt,
                "target_step": target_step,
            })
            instance_id += 1

            if max_examples and len(instances) >= max_examples:
                break

    save_jsonl(output_path, instances)
    return len(instances)


def main():
    parser = argparse.ArgumentParser(description="Prepare SRL training data from s1K-1.1")
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1", help="HuggingFace dataset")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", default="data/srl_instances.jsonl", help="Output JSONL path")
    parser.add_argument("--max-examples", type=int, default=None, help="Max instances (for testing)")
    args = parser.parse_args()

    n = create_srl_instances(
        dataset_name=args.dataset,
        split=args.split,
        output_path=args.output,
        max_examples=args.max_examples,
    )
    print(f"Created {n} SRL instances -> {args.output}")


if __name__ == "__main__":
    main()
