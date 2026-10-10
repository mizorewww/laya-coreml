"""Opt-in embedding shortlist and tournament for high-cardinality choice questions.

Choice options share one ``head_max_len`` budget, so a large label set leaves only a few
tokens per label. ``predict_shortlist`` embeds the state and each option with a
caller-supplied ``embed_fn``, keeps the top ``k``, and runs a single ``predict`` (or
``system_one``) on that reduced criteria set. ``predict_tournament`` needs no embedder: the
decision model answers the labels in groups small enough to keep their tokens, and the group
winners meet in one more ``predict``.

``Agent.predict`` and ``Agent.system_one`` are separate: they still score every criterion
they are given. This module does not change ``DecisionModel.forward``.
``predict_shortlist`` adds no second decision-model pass; ``predict_tournament`` adds one
per elimination round.

The coarse-to-fine pattern is the one the README recommends and the one reported in
https://github.com/NandhaKishorM/laya/issues/102. Ranking here is cosine similarity on
whatever vectors ``embed_fn`` returns. Issue #102's BANKING77 figures belong to that
report; this module does not measure them.
"""

import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .common import encode_text, render_options, serialize_state

DEFAULT_SHORTLIST_K = 20
DEFAULT_TOURNAMENT_GROUP = 16


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
    return_scores: bool = False,
) -> Any:
    """Return the top-``k`` choice labels for ``state``.

    ``embed_fn`` maps a list of strings to an array of shape ``(len(texts), dim)``.
    It is called once, with the query text first and then one string per option in
    criteria order. Option strings match ``render_options`` for a choice question.

    When ``k`` is at least the number of labels, every label is returned in its
    original order and ``embed_fn`` is not called.

    Ties keep the earlier label. Ranking is a signed cosine, not a similarity floor:
    a label that scores 0 -- no signal at all, or a non-finite vector treated as one
    -- does outrank an earlier label that scored negative, and ``k`` drops the
    negative labels first.

    With ``return_scores=True`` the return is the ``(labels, scores)`` pair, where
    ``scores`` holds the signed cosine per kept label in rank order -- the same
    values ``predict_shortlist`` reports in its ``shortlist`` metadata. ``scores``
    is ``None`` when nothing was dropped, exactly as in that metadata.
    """
    labels, scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    if return_scores:
        return labels, scores
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
    ``labels``, ``scores``, ``k``, ``n``, and ``passthrough``. ``labels`` is the rank
    order a shortlist produced, or the criteria order itself when ``passthrough`` is
    set and no ranking ran; ``scores`` is the signed cosine of each kept label in that
    order -- negative included, never clamped to 0 -- or ``None`` when nothing was
    dropped.

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
    ``tournament[qid]`` holds ``labels`` (the finalists, in criteria order), ``n`` (the
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


def embed_fn_from_agent(
    agent: Any,
    max_length: int = 512,
    batch_size: int = 32,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Mean-pool the checkpoint encoder already loaded on ``agent``.

    The callable embeds a list of strings with ``agent.tok`` and ``agent.model.encoder``.
    It does not run the decision head and does not download weights. A dedicated
    bi-encoder passed as ``embed_fn`` will usually shortlist better; this helper is
    for callers who only have the Laya checkpoint in memory.

    Padding positions are excluded from the mean. The encoder's train/eval flag is
    left as the caller set it (a loaded ``Agent`` is already in eval).
    Each call uses the current ``agent.device``, including after CPU fallback.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer, got %r" % (max_length,))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer, got %r" % (batch_size,))

    if not hasattr(agent.model, "encoder"):
        raise NotImplementedError(
            "This Core ML bundle exposes decision outputs only; supply an external embed_fn."
        )

    import torch

    tok = agent.tok
    encoder = agent.model.encoder

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if text is None else str(text) for text in texts]
        hidden = _hidden_size(encoder)
        if not rows:
            return np.zeros((0, hidden), dtype=np.float32)
        device = agent.device
        parts: List[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = encode_text(
                tok,
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.inference_mode():
                hidden_states = encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            parts.append(pooled.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    return embed_fn


def cached_embed_fn(
    embed_fn: Callable[[Sequence[str]], Any],
    maxsize: int = 4096,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Cache ``embed_fn`` output per input string, under an LRU bound.

    ``predict_shortlist`` embeds the query plus every option text on each call. When the
    same option set is shortlisted on every request -- a fixed intent or label list, as
    in the README's BANKING77 example -- the option rows do not change between calls,
    yet they are re-embedded every time. Wrapping the embedder once::

        embed_fn = cached_embed_fn(embed_fn_from_agent(agent))

    leaves the first call unchanged and reduces each repeat call to embedding the new
    query alone.

    Lookups are exact string matches. Texts missing from the cache are deduplicated and
    embedded in a single ``embed_fn`` call, so a cold cache costs the same number of
    batched calls as the unwrapped function. Rows are stored as float32; the cache holds
    at most ``maxsize`` strings and then evicts the least recently used entry, bounding
    memory at about ``maxsize * dim * 4`` bytes. Nothing is cached when ``embed_fn``
    raises or returns a bad shape.

    The wrapper is safe to share between threads: the lock covers only cache reads and
    writes, never the embedding call. The returned callable carries ``cache_info()`` --
    a dict with ``size``, ``maxsize``, ``hits`` and ``misses`` -- and ``cache_clear()``.
    Clear the cache if the model or weights behind ``embed_fn`` change.
    """
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
        raise ValueError("maxsize must be a positive integer, got %r" % (maxsize,))

    rows_by_text: OrderedDict[str, np.ndarray] = OrderedDict()
    lock = threading.Lock()
    counts = {"hits": 0, "misses": 0}

    def cached(texts: Sequence[str]) -> np.ndarray:
        keys = ["" if text is None else str(text) for text in texts]
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        with lock:
            found: Dict[str, np.ndarray] = {}
            hits = 0
            for key in keys:
                row = rows_by_text.get(key)
                if row is not None:
                    rows_by_text.move_to_end(key)
                    found[key] = row
                    hits += 1
            counts["hits"] += hits
            counts["misses"] += len(keys) - hits
            missing = [key for key in dict.fromkeys(keys) if key not in found]
        if missing:
            raw = embed_fn(list(missing))
            if hasattr(raw, "detach"):
                raw = raw.detach().float().cpu().numpy()
            fresh = np.asarray(raw, dtype=np.float32)
            if fresh.ndim != 2 or fresh.shape[0] != len(missing) or fresh.shape[1] < 1:
                raise ValueError(
                    "embed_fn must return an array of shape (%d, dim), got %s"
                    % (len(missing), tuple(fresh.shape))
                )
            fresh = np.nan_to_num(fresh, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            if rows_by_text:
                # Rows stored under one dimensionality cannot be stacked against rows of
                # another: a changed embedder (or one whose dimension drifts between calls)
                # would otherwise surface rows of mixed width, which `_rank` reads as one
                # matrix. Deciding by identity is what the docstring already asks of the
                # caller ("clear the cache if the model changes") -- this refuses the call
                # instead of returning a matrix that silently mixes both.
                want = next(iter(rows_by_text.values())).shape[0]
                if fresh.shape[1] != want:
                    raise ValueError(
                        "embed_fn returned dim %d, but the cache holds dim %d; "
                        "call cache_clear() if the model behind embed_fn changed"
                        % (fresh.shape[1], want)
                    )
            with lock:
                for key, row in zip(missing, fresh):
                    rows_by_text[key] = row
                    rows_by_text.move_to_end(key)
                    while len(rows_by_text) > maxsize:
                        rows_by_text.popitem(last=False)
                    found[key] = row
        return np.stack([found[key] for key in keys])

    def cache_info() -> Dict[str, int]:
        with lock:
            return {
                "size": len(rows_by_text),
                "maxsize": maxsize,
                "hits": counts["hits"],
                "misses": counts["misses"],
            }

    def cache_clear() -> None:
        with lock:
            rows_by_text.clear()
            counts["hits"] = 0
            counts["misses"] = 0

    cached.cache_info = cache_info
    cached.cache_clear = cache_clear
    return cached


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


def _hidden_size(encoder) -> int:
    size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return 0
    return size
