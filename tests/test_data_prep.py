"""Unit tests for data prep step parsing."""

import pytest
from src.data_prep import parse_expert_steps


def test_parse_numbered_steps():
    solution = """1. First step here.
2. Second step continues.
3. Third step."""
    steps = parse_expert_steps(solution)
    assert steps is not None
    assert len(steps) >= 2
    assert "First step" in steps[0]
    assert "Second step" in steps[1]


def test_parse_bold_headers():
    solution = """1. **Rewrite the function:** We can use identities.
2. **Analyze:** Consider the integral."""
    steps = parse_expert_steps(solution)
    assert steps is not None
    assert len(steps) >= 2
