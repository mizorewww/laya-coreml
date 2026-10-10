"""Exceptions shared by the framework integrations."""

from typing import Any, Dict


class LayaLowConfidenceError(ValueError):
    """Raised when a routing decision falls below the confidence threshold and no fallback is set."""

    def __init__(
        self, message: str, confidence: float, threshold: float, raw_decision: Dict[str, Any]
    ):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision
