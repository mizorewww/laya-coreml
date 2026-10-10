# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-coreml.
"""Opt-in embedding shortlist for high-cardinality choice questions.

Choice options share one ``head_max_len`` budget, so a large label set leaves only a few
tokens per label. ``predict_shortlist`` embeds the state and each option with a
caller-supplied ``embed_fn``, keeps the top ``k``, and runs a single ``predict`` (or
``system_one``) on that reduced criteria set.

``Agent.predict`` and ``Agent.system_one`` are separate: they still score every criterion
they are given. This module does not change ``DecisionModel.__call__`` and does not add a
second decision-model pass.

The coarse-to-fine pattern is the one the upstream README recommends and the one reported in
https://github.com/NandhaKishorM/laya/issues/102. Ranking here is cosine similarity on
whatever vectors ``embed_fn`` returns. Issue #102's BANKING77 figures belong to that
report; this module does not measure them.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .common import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20
DEFAULT_TOURNAMENT_GROUP = 16


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
) -> List[Any]:
    """Return the top-``k`` choice labels for ``state``.

    ``embed_fn`` maps a list of strings to an array of shape ``(len(texts), dim)``.
    It is called once, with the query text first and then one string per option in
    criteria order. Option strings match ``render_options`` for a choice question.

    When ``k`` is at least the number of labels, every label is returned in its
    original order and ``embed_fn`` is not called.

    Ties keep the earlier label. A zero vector scores 0: it ranks above negative
    cosine scores and below positive scores.
    """
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Shortlist each choice question, then call ``predict`` or ``system_one`` once.

    Non-choice questions are forwarded unchanged. A choice whose label count is
    ``<= k`` is forwarded unchanged and does not call ``embed_fn``. The caller's
    ``questions`` dict is not mutated.

    The returned dict is the model result plus a ``shortlist`` entry. Probabilities
    on a shortlisted choice are over the kept labels only. ``shortlist[qid]`` holds
    ``labels`` (rank order), ``scores`` (cosine, or ``None`` when nothing was
    dropped), ``k``, ``n``, and ``passthrough``.

    Extra keyword arguments are forwarded to ``predict`` / ``system_one`` (for
    example ``model=`` on a ``Router``).
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    checked = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError("question %r is a choice but has no criteria" % (qid,))
        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked, qdef.get("instructions")
        )
        meta[qid] = {
            "labels": list(labels),
            "scores": scores,
            "k": checked,
            "n": n,
            "passthrough": passthrough,
        }
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = updated

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError("predict/system_one must return a dict, got %s" % type(result).__name__)
    out = dict(result)
    out["shortlist"] = meta
    return out


def predict_tournament(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    group_size: int = DEFAULT_TOURNAMENT_GROUP,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Narrow each large choice question by elimination, then call ``predict`` once more.

    A choice with more than ``group_size`` labels is cut, in criteria order, into groups of
    near-equal size and at most ``group_size``. One ``predict`` call answers every group of
    every such question -- the groups go in as separate questions, so they share one forward
    pass -- and each group's answer goes through to the next round. Rounds repeat until no
    choice has more than ``group_size`` labels left; at the default 16, up to 256 labels take
    one round. Unlike ``predict_shortlist`` this needs no embedder, and each label is read
    with the option budget of a ``group_size``-label question, not of the whole label set.

    The final call answers every question of the request, with each choice that went through
    a round cut to its finalists. Non-choice questions and choices of at most ``group_size``
    labels go to it unchanged, so when nothing needs a round it is the only call. The
    caller's ``questions`` dict is not mutated.

    The returned dict is the final call's result plus a ``tournament`` entry.
    ``tournament[qid]`` holds ``labels`` (the finalists, in criteria order), ``n`` (the original
    label count) and ``rounds`` for each choice question. Probabilities, confidences and
    ``usage`` come from the final call, so a tournament choice's probabilities are over its
    finalists only.

    Extra keyword arguments are forwarded to every call (for example ``model=`` on a ``Router``).
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer of at least 2, got %r" % (group_size,))
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, qdef in questions.items():
        if isinstance(qdef, dict) and qdef.get("type") == "choice":
            if "criteria" not in qdef:
                raise ValueError("question %r is a choice but has no criteria" % (qid,))
            labels = [key for key, _value in _criteria_items(qdef["criteria"])]
            meta[qid] = {"labels": labels, "n": len(labels), "rounds": 0}

    def cut(qid, labels):
        return dict(questions[qid], criteria=_subset_criteria(questions[qid]["criteria"], labels))

    while True:
        groups = []
        for qid, entry in meta.items():
            labels = entry["labels"]
            parts = -(-len(labels) // group_size)
            if parts > 1:
                groups += [
                    (qid, labels[i * len(labels) // parts : (i + 1) * len(labels) // parts])
                    for i in range(parts)
                ]
        if not groups:
            break
        round_questions = {str(i): cut(qid, labels) for i, (qid, labels) in enumerate(groups)}
        answers = _call_predict(agent, state, round_questions, **predict_kwargs)["answers"]
        winners: Dict[str, List[Any]] = {}
        for i, (qid, _labels) in enumerate(groups):
            winners.setdefault(qid, []).append(answers[str(i)]["choice"])
        for qid, labels in winners.items():
            meta[qid]["labels"] = labels
            meta[qid]["rounds"] += 1

    final = dict(questions)
    for qid, entry in meta.items():
        if entry["rounds"]:
            final[qid] = cut(qid, entry["labels"])
    result = _call_predict(agent, state, final, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError("predict/system_one must return a dict, got %s" % type(result).__name__)
    out = dict(result)
    out["tournament"] = meta
    return out


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked]
    labels = [keys[int(i)] for i in order]
    scores = [float(sims[int(i)]) for i in order]
    return labels, scores, False, n


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer, got %r" % (k,))
    return k


def _criteria_items(criteria):
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError("choice criteria must be a dict or list, got %s" % type(criteria).__name__)
    if not items:
        raise ValueError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise ValueError("choice criteria label %r is duplicated" % (key,))
        seen.add(key)
    return items


def _option_texts(items) -> List[str]:
    crit = {key: value for key, value in items}
    rendered = render_options({"t": "choice", "ins": "", "crit": crit})
    texts = [piece if isinstance(piece, str) else str(piece) for piece in rendered]
    if len(texts) != len(items):
        raise ValueError("could not render every choice option")
    return texts


def _query_text(state, instructions) -> str:
    body = serialize_state(state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return "%s\n%s" % (instructions, body)


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn, texts: Sequence[str]) -> np.ndarray:
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError(
            "embed_fn must return an array of shape (%d, dim), got %s"
            % (len(texts), tuple(arr.shape))
        )
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(query))
    if qn == 0.0 or docs.shape[0] == 0:
        return np.zeros(docs.shape[0], dtype=np.float64)
    dn = np.linalg.norm(docs, axis=1)
    denom = dn * qn
    ok = denom > 0.0
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if np.any(ok):
        sims[ok] = np.clip(np.dot(docs[ok], query) / denom[ok], -1.0, 1.0)
    return sims


def _call_predict(agent, state, questions, **predict_kwargs):
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide predict or system_one")
    return fn(state, questions, **predict_kwargs)
