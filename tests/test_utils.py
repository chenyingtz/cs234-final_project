"""Unit tests for utils."""

import pytest
from src.utils import extract_aime_answer


def test_extract_boxed():
    assert extract_aime_answer(r"The answer is \boxed{42}") == 42
    assert extract_aime_answer(r"$\boxed{123}$") == 123
    assert extract_aime_answer(r"\boxed{ 7 }") == 7


def test_extract_answer_keyword():
    assert extract_aime_answer("ANSWER: 042") == 42
    assert extract_aime_answer("The answer is 100.") == 100
    assert extract_aime_answer("answer: 999") == 999


def test_extract_last_integer():
    text = "We get x=5 and final answer 100."
    assert extract_aime_answer(text) == 100


def test_aime_range_0_999():
    """AIME answers must be 0-999."""
    assert extract_aime_answer(r"\boxed{0}") == 0
    assert extract_aime_answer(r"\boxed{999}") == 999
    assert extract_aime_answer(r"\boxed{1000}") is None  # out of range
    assert extract_aime_answer("Thus 42 and 1000.") == 42  # last valid in range


def test_invalid_or_empty():
    assert extract_aime_answer("") is None
    assert extract_aime_answer(r"\boxed{abc}") is None
    assert extract_aime_answer("No numbers here.") is None
