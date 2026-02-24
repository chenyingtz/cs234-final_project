"""Unit tests for SRL parsing and reward."""

import pytest
from src.reward import (
    parse_srl_output,
    is_valid_srl_output,
    compute_srl_reward,
    SequenceMatcherReward,
    INVALID_REWARD,
)


class TestParseSrlOutput:
    def test_valid_single_step(self):
        text = "<think>\nLet me think about this step.\n</think>\nThe next step is to factor the equation."
        think, action = parse_srl_output(text)
        assert "Let me think" in think
        assert "factor the equation" in action

    def test_invalid_no_think(self):
        think, action = parse_srl_output("Just some output without tags.")
        assert think == "" and action == ""

    def test_invalid_empty_string(self):
        think, action = parse_srl_output("")
        assert think == "" and action == ""

    def test_invalid_none(self):
        think, action = parse_srl_output(None)
        assert think == "" and action == ""

    def test_invalid_missing_close_tag(self):
        # No </think> tag
        think, action = parse_srl_output("<think>reasoning here")
        assert think == "" and action == ""

    def test_valid_action_extracted_correctly(self):
        text = "<think>x</think>\nApply the formula: x = 5."
        think, action = parse_srl_output(text)
        assert think == "x"
        assert "Apply" in action

    def test_valid_multiline_think(self):
        text = "<think>\nLine 1.\nLine 2.\nLine 3.\n</think>\nFinal action step."
        think, action = parse_srl_output(text)
        assert "Line 1" in think
        assert "Line 3" in think
        assert action == "Final action step."

    def test_invalid_multiple_numbered_steps(self):
        # Second line starts with "2. " — should be rejected as multiple steps
        text = "<think>ok</think>\n1. First step\n2. Second step"
        think, action = parse_srl_output(text)
        assert think == "" and action == ""

    def test_valid_single_numbered_step(self):
        # Single numbered step is fine
        text = "<think>ok</think>\n1. The prime factorization is 2^3 * 3."
        think, action = parse_srl_output(text)
        assert action != ""
        assert "prime" in action

    def test_action_whitespace_normalized(self):
        text = "<think>ok</think>\n  lots   of   spaces   "
        think, action = parse_srl_output(text)
        assert "  " not in action  # internal whitespace normalized

    def test_empty_action_after_think(self):
        text = "<think>reasoning</think>\n   "
        think, action = parse_srl_output(text)
        # Empty action should give empty result
        assert action == ""


class TestIsValidSrlOutput:
    def test_valid(self):
        assert is_valid_srl_output("<think>reasoning</think>\nThe action step.") is True

    def test_invalid_no_tags(self):
        assert is_valid_srl_output("no tags") is False

    def test_invalid_empty(self):
        assert is_valid_srl_output("") is False

    def test_invalid_empty_action(self):
        assert is_valid_srl_output("<think>ok</think>") is False


class TestSequenceMatcherReward:
    def test_identical_strings(self):
        fn = SequenceMatcherReward()
        assert fn("same text", "same text") == pytest.approx(1.0)

    def test_completely_different(self):
        fn = SequenceMatcherReward()
        r = fn("completely different", "xyz abc")
        assert 0.0 <= r <= 1.0
        assert r < 0.5

    def test_partial_overlap(self):
        fn = SequenceMatcherReward()
        r = fn("hello world", "hello there")
        assert r > 0.3

    def test_reward_in_range(self):
        fn = SequenceMatcherReward()
        for a, b in [("abc", "xyz"), ("", "something"), ("long text here", "long")]:
            r = fn(a, b)
            assert 0.0 <= r <= 1.0, f"reward out of range for ({a!r}, {b!r}): {r}"

    def test_empty_both(self):
        fn = SequenceMatcherReward()
        assert fn("", "") == 0.0  # no expert step = malformed, score 0

    def test_empty_model_step(self):
        fn = SequenceMatcherReward()
        assert fn("", "expert step") == 0.0

    def test_empty_expert_step(self):
        # Empty expert step = malformed instance, score 0 regardless of model output
        fn = SequenceMatcherReward()
        assert fn("model output", "") == 0.0

    def test_symmetry(self):
        # SequenceMatcher ratio is symmetric
        fn = SequenceMatcherReward()
        assert fn("abc def", "def abc") == pytest.approx(fn("def abc", "abc def"))

    def test_longer_common_prefix_scores_higher(self):
        fn = SequenceMatcherReward()
        r_close = fn("The answer is 42", "The answer is 43")
        r_far = fn("The answer is 42", "Something completely unrelated")
        assert r_close > r_far


class TestComputeSrlReward:
    def test_valid_output_perfect_match(self):
        text = "<think>reasoning</think>\nThe answer is 42."
        r = compute_srl_reward(text, "The answer is 42.")
        assert r == pytest.approx(1.0)

    def test_valid_output_partial_match(self):
        text = "<think>reasoning</think>\nThe answer is 42."
        r = compute_srl_reward(text, "The answer is 100.")
        assert 0.0 < r < 1.0

    def test_valid_output_no_match(self):
        text = "<think>reasoning</think>\nCompletely unrelated output."
        r = compute_srl_reward(text, "Factor the polynomial x^2 + 2x + 1.")
        assert 0.0 <= r <= 1.0

    def test_invalid_format_gives_minus_one(self):
        assert compute_srl_reward("no think tags", "expert") == INVALID_REWARD

    def test_empty_output_gives_minus_one(self):
        assert compute_srl_reward("", "expert step") == INVALID_REWARD

    def test_empty_expert_step_gives_invalid(self):
        # Malformed data instance — should be -1 so dynamic sampling filters it out
        assert compute_srl_reward("<think>ok</think>\nsome step", "") == INVALID_REWARD
        assert compute_srl_reward("no tags", "") == INVALID_REWARD

    def test_reward_always_minus_one_or_in_range(self):
        cases = [
            ("<think>ok</think>\nstep", "step"),
            ("no tags", "step"),
            ("", "expert"),
            ("<think>ok</think>\n", "step"),
        ]
        for text, expert in cases:
            r = compute_srl_reward(text, expert)
            assert r == INVALID_REWARD or 0.0 <= r <= 1.0, f"unexpected reward {r} for {text!r}"
