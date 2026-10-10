"""Laya prompt construction and calibration, adapted from upstream (see NOTICE)."""

import json
import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        )
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [
            str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label
        + ": "
        + (
            render_criterion(false_crit)
            if false_crit not in (None, "")
            else "no, the statement does not hold"
        ),
        true_label
        + ": "
        + (
            render_criterion(true_crit)
            if true_crit not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def build_prefix(tok, q: Dict, head_max_len: int = 192, option_order=None, *, return_stats=False):
    """Build the question-only prefix, before state tokens and final truncation."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    if return_stats:
        return (
            ids,
            markers,
            {
                "options": len(opt_ids),
                "options_distinct": len({tuple(o) for o in opt_ids}),
                "tokens_per_option": per,
            },
        )
    return ids, markers


def parallel_layout(markers: List[int], head_len: int, length: int) -> Dict[str, List[int]]:
    """Position ids and option ids that make the encoder blind to option order.

    In the sequential layout option `s` sits at positions after option `s-1`, and every option
    attends to every other, so where an option is listed changes its embedding: the slot logits
    of five identical options spread by 3.59 (`tests/test_option_order.py`). Here every option
    starts at the same position, the first one after the instruction's [SEP], and the head's
    closing [SEP] and the state continue after the longest option. Together with
    `parallel_option_masks`, which stops an option from attending to another, reordering the
    options only reorders the marker embeddings.

    `option_ids` is 0 for shared tokens (instruction, closing [SEP], state, padding) and `s + 1`
    for the tokens of slot `s`, which runs from its [MASK] up to the next marker or, for the last
    slot, up to the [SEP] at `head_len - 1`.
    """
    if not markers:
        return {"position_ids": list(range(length)), "option_ids": [0] * length}
    start = markers[0]
    spans = list(zip(markers, markers[1:] + [head_len - 1]))
    position_ids, option_ids = list(range(start)), [0] * start
    for s, (a, b) in enumerate(spans):
        position_ids += range(start, start + b - a)
        option_ids += [s + 1] * (b - a)
    after = start + max(b - a for a, b in spans)
    position_ids += range(after, after + length - len(position_ids))
    option_ids += [0] * (length - len(option_ids))
    return {"position_ids": position_ids, "option_ids": option_ids}


def uses_parallel_layout(cfg):
    layout = cfg.get("option_layout", "sequential")
    if layout not in ("sequential", "parallel"):
        raise ValueError(f"Unsupported option_layout: {layout!r}")
    return layout == "parallel"


def finish_sequence(tok, prefix, markers, state_ids, max_len, truncate_left=False):
    """Append the same state slice in cached and uncached preparation."""
    room = max(0, max_len - len(prefix) - 1)
    kept = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    ids = (list(prefix) + kept + [tok.sep_token_id])[:max_len]
    return (
        ids,
        [m for m in markers if m < max_len],
        {
            "state_tokens": len(state_ids),
            "state_tokens_used": len(kept),
            "state_tokens_dropped": len(state_ids) - len(kept),
            "truncated": len(kept) < len(state_ids),
        },
    )


def build_sequence(
    tok,
    state,
    q,
    max_len=512,
    head_max_len=192,
    option_order=None,
    truncate_left=False,
    state_ids=None,
    return_stats=False,
    return_truncation_stats=False,
    return_layout=False,
):
    """Build the upstream sequence; optional diagnostics describe the actual token budgets."""
    prefix, markers, stats = build_prefix(tok, q, head_max_len, option_order, return_stats=True)
    if state_ids is None:
        state_ids = tok(
            serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
    original_markers = list(markers)
    ids, markers, state_stats = finish_sequence(
        tok, prefix, markers, state_ids, max_len, truncate_left
    )
    result = (ids, markers)
    if return_stats:
        result += (stats,)
    if return_truncation_stats:
        result += (state_stats,)
    if return_layout:
        layout = parallel_layout(original_markers, len(prefix), max(len(prefix), len(ids)))
        result += ({k: v[: len(ids)] for k, v in layout.items()},)
    return result


def collapsed_options(qids, items) -> Dict[str, Dict[str, Optional[int]]]:
    """The questions whose options no longer have a token span each, from per-item stats.

    `total` is the number of options the question defines, not the number of markers that
    reached the sequence: a report counted from the markers would say "43/58" about a request
    where 28 options never made it into the input at all.
    """
    out = {}
    for qid, item in zip(qids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            out[qid] = {
                "total": stats["options"],
                "distinct": stats["options_distinct"],
                "tokens_per_option": stats["tokens_per_option"],
            }
    return out


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. The README's gating section relies on the property that goes with it:
    of the answers returned at confidence c, about c of them are right. That property is
    conditional, and the condition is not met by default -- it holds only after the temperatures
    have been fitted and validated on held-out data for this checkpoint and this option count.
    The shipped checkpoints are over-confident: `choice:11+` is a ~10x sharpener that returns a
    point mass at 1.0, so a threshold applied to them selects below model accuracy (issue #394).

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published
# as 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest
# calibration needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(hi, max(lo, t))


def read_temperatures(cfg: Dict):
    """Calibration temperatures from an agent config: clamped working copies plus the raw values.

    Returns (temperature, temperature_by_options, temperature_raw, temperature_by_options_raw).
    Only the clamped values are ever applied; the raw ones stay visible for inspection, and a
    RuntimeWarning names every bucket that had to be clamped.
    """
    raw = cfg.get("temperature", [1.0, 1.0, 1.0])
    raw_by_options = cfg.get("temperature_by_options", {})
    if len(raw) != 3 or any(
        not math.isfinite(float(t)) or float(t) <= 0 for t in [*raw, *raw_by_options.values()]
    ):
        raise ValueError("Calibration temperatures must be finite and positive")
    temperature = [clamp_temperature(t) for t in raw]
    by_options = {k: clamp_temperature(v) for k, v in raw_by_options.items()}
    rejected = [
        "%s=%.4g" % (k, float(v))
        for k, v in raw_by_options.items()
        if clamp_temperature(v) != float(v)
    ]
    rejected += [
        "temperature[%d]=%.4g" % (i, float(t))
        for i, t in enumerate(raw)
        if clamp_temperature(t) != float(t)
    ]
    if rejected:
        warnings.warn(
            "laya-coreml: this checkpoint ships temperatures outside [%g, %g] which would "
            "distort confidence; clamping %s. Treat confidence from the affected buckets "
            "as uncalibrated." % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
            RuntimeWarning,
            stacklevel=2,
        )
    return temperature, by_options, raw, raw_by_options


def option_layout(cfg):
    layout = cfg.get("option_layout", "sequential")
    if layout not in ("sequential", "parallel"):
        raise ValueError(f"Unknown option_layout: {layout!r}")
    return layout
