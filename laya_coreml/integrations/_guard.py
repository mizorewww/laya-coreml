"""The score-question rule shared by the guardrail wrappers."""

from typing import Any, Dict


def score_violation_probability(answer: Dict[str, Any], levels: int = 0) -> float:
    """Probability that a `score` answer is at or above the middle of its scale.

    `score` is the expected level (0..k-1), not a probability, so a guard threshold cannot be
    compared with it directly. `levels` is the number of levels the question defines; it is only
    used when the answer carries no `probabilities`, and then the normalised expected level stands
    in for the distribution.
    """
    probs = answer.get("probabilities") or {}
    k = len(probs) or levels
    if k < 2:
        return 0.0
    if probs:
        return sum(float(probs.get(str(i), 0.0)) for i in range(k // 2, k))
    return answer.get("score", 0.0) / (k - 1)
