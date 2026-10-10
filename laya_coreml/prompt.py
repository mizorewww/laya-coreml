"""Laya input semantics, adapted from laya-mlx; see NOTICE."""

import json

from .common import (
    QTYPES,
    build_sequence,
    option_layout,
    render_options,
    resolve_noul_labels,
    serialize_state,
)


class PromptMixin:
    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if not isinstance(kind, str) or kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        instructions = qdef["instructions"]
        if instructions is None or (
            isinstance(instructions, (str, dict, list)) and not instructions
        ):
            raise ValueError("instructions must not be empty or None")
        if isinstance(instructions, str) and not instructions.strip():
            raise ValueError("instructions must not be blank")
        if not isinstance(instructions, (str, dict, list, int, float)):
            raise ValueError("instructions must be JSON-serializable text or structured data")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
            if any(level is None for level in criteria):
                raise ValueError("Score criteria must not contain a null level")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        elif criteria is not None:
            criteria = {str(k).lower(): v for k, v in criteria.items()}
            if not set(criteria) <= {"false", "true"}:
                raise ValueError(
                    "Noul criteria must be keyed only 'false'/'true'; use labels to change display text"
                )
        if "labels" in qdef:
            if kind != "noul":
                raise ValueError("labels is only supported for noul questions")
            resolve_noul_labels(qdef["labels"])
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions, ensure_ascii=False)
        q = {"t": kind, "ins": instructions, "crit": criteria}
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        return q

    @staticmethod
    def _question(qid, definition):
        if (
            qid is None
            or not isinstance(qid, (str, int))
            or (isinstance(qid, str) and not qid.strip())
        ):
            raise ValueError("Question id must be a nonempty string or integer")
        try:
            return PromptMixin._to_internal(definition)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Question {qid!r}: {error}") from error

    def prepare(self, state, questions):
        """Construct CPU inputs and per-question token-budget diagnostics."""
        if state is None:
            raise ValueError("state must not be None")
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        if not questions:
            return [], []
        state_ids = self.tok(
            serialize_state(state).replace(self.tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        parallel = option_layout(self.cfg) == "parallel"
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._question(qid, definition)
            ids, markers, stats, state_stats, *layout = build_sequence(
                self.tok,
                state,
                q,
                self.cfg.get("max_len", 512),
                self.cfg.get("head_max_len", 192),
                truncate_left=isinstance(state, list),
                state_ids=state_ids,
                return_stats=True,
                return_truncation_stats=True,
                return_layout=parallel,
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "options": stats,
                    "state_stats": state_stats,
                }
            )
            if parallel:
                items[-1]["layout"] = layout[0]
            internal.append(q)
        return items, internal
