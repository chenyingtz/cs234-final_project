"""Shared utilities for SRL pipeline."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator


def set_seed(seed: int) -> None:
    """Deterministic seeding for reproducibility."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(path: str | Path, items: list[dict[str, Any]]) -> None:
    """Write items to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl_list(path: str | Path) -> list[dict[str, Any]]:
    """Load entire JSONL file as list."""
    return list(load_jsonl(path))


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_aime_answer(text: str) -> int | None:
    """
    Extract AIME-style final integer answer (0-999) from model output.
    Handles common formats: \\boxed{123}, \\boxed{ 123 }, ANSWER: 123, etc.
    """
    import re

    if not text or not isinstance(text, str):
        return None

    m = re.search(r"\\boxed\s*\{([^}]+)\}", text)
    if m:
        s = m.group(1).strip()
        try:
            v = int(s)
            if 0 <= v <= 999:
                return v
        except ValueError:
            pass

    m = re.search(r"answer\s*(?::|is)\s*(\d{1,3})\b", text, re.I)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 999:
            return v

    candidates = re.findall(r"\b(\d{1,3})\b", text)
    for c in reversed(candidates):
        v = int(c)
        if 0 <= v <= 999:
            return v

    return None
