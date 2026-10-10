"""Compatibility import; prompt semantics live in the official host runtime."""

from .runtime import RuntimeMixin


class PromptMixin(RuntimeMixin):
    @staticmethod
    def _to_internal(qdef):
        RuntimeMixin._check_question("question", qdef)
        return RuntimeMixin._to_internal(qdef)
