"""Confidence validation, calibrated thresholds, and abstention gating helpers (#361).

Pure Python: safe to import without PyTorch so that Router and structured
decisions stay lightweight and free of torch import overhead.
"""

import math
from typing import Any, Dict, List, Optional


def answer_confidence_value(answer: Dict[str, Any]) -> Optional[float]:
    """The `answer_confidence` of `answer`, or None if it did not report a usable one.

    `answer_confidence` is `max(p)` -- the probability mass on the answer being reported. It is
    the quantity temperature scaling fits, the quantity every calibration figure in this
    repository is computed on, and the quantity the `min_confidence` gate is defined against
    (#361). That makes it the right number for a threshold a caller compares against; it is
    not on its own a claim that the number is right. Reading it as "about c of the answers
    returned at c are correct" holds only after temperatures have been fitted and validated on
    held-out data for that checkpoint and question shape -- the shipped checkpoints are
    over-confident as shipped, and `laya-multilingual` ships with no fitted temperatures at all
    (README, Calibration). `common.answer_confidence` states the quantity; this states when it
    carries that property.

    `confidence` is a different quantity on a different scale -- normalized entropy, which
    `common.confidence_from_probs` calls "not calibrated" and #394 says does not transfer across
    option counts. So this deliberately does **not** fall back to it: reporting the entropy
    number under the `answer_confidence` name would be worse than reporting nothing, because the
    name is what a caller filters on.

    A `bool` is not a confidence and a NaN is not a decision, so both give None rather than 0.0.
    """
    conf = answer.get("answer_confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf):
        return float(conf)
    return None


def _gate_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """The number the abstention gate compares against `min_confidence`, or None if there is none.

    `answer_confidence` first, falling back to the entropy `confidence` so an answer that carries
    only the older field is still gated rather than silently passed.

    None means *no usable number*, never an unusable one. An earlier version returned whatever
    `confidence` held when the validity test failed, so an answer with no `answer_confidence` and a
    NaN `confidence` returned NaN. `flag_low_confidence` was indifferent to that -- `NaN < x` is
    False and `None is not None` is also False, so it flagged either way -- but a caller that
    *reads* the number could not tell "nothing to gate on" from "a gate ran on a NaN", and
    :func:`apply_confidence_gate` reports those two differently on purpose.
    """
    conf = answer_confidence_value(answer)
    if conf is not None:
        return conf
    fallback = answer.get("confidence")
    if (
        isinstance(fallback, (int, float))
        and not isinstance(fallback, bool)
        and math.isfinite(fallback)
    ):
        return float(fallback)
    return None


def _check_one_threshold(v: Any) -> float:
    if (
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0.0
        or v > 1.0
    ):
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


# The vocabulary a per-bucket map may be keyed by, reproduced here rather than imported from
# `laya.common` because this module must import without PyTorch. `tests/test_conformal_abstention.py`
# holds the copy against `common.temp_bucket` in both directions, so a band added on one side fails
# there instead of silently rejecting (or accepting) the wrong keys here.
BUCKET_QTYPES = ("choice", "score", "noul")
BUCKET_SIZES = ("2", "3-5", "6-10", "11+")
#: The one non-bucket key a map may carry: the threshold for every bucket the map does not name.
DEFAULT_BUCKET_KEY = "default"


def bucket_keys() -> set:
    """Every option-count bucket a `min_confidence` map can name -- the answer shapes
    :func:`_option_bucket` can produce, and the same strings `common.temp_bucket` spells."""
    return {"%s:%s" % (qt, size) for qt in BUCKET_QTYPES for size in BUCKET_SIZES}


def check_min_confidence_map(m: Dict[Any, Any]) -> Dict[str, float]:
    """Validate a per-bucket abstention-threshold map (#394).

    Keys are the option-count buckets in `common.temp_bucket`'s spelling -- every
    :func:`bucket_keys` member (``"choice:2"``, ``"choice:3-5"``, ``"score:6-10"``, ``"noul:2"``
    and so on) -- plus the optional ``"default"`` used for any bucket the map does not name.
    Anything else raises: a key no answer can produce is not an unused entry, it is a bucket the
    caller believes is gated and is not, because :func:`resolve_min_confidence` falls past it to
    ``"default"`` and then to 0.0 -- gate nothing. Values are floats in [0.0, 1.0]. One confidence
    threshold does not transfer across option counts (#394); this lets a caller gate each bucket
    at the level its calibration actually earns. Fit one with
    :func:`laya.calibrate.fit_abstention_thresholds`, whose keys are this vocabulary by
    construction.
    """
    if not isinstance(m, dict) or not m:
        raise ValueError(
            "a min_confidence map must be a non-empty dict of bucket -> float, got %r" % (m,)
        )
    allowed = bucket_keys() | {DEFAULT_BUCKET_KEY}
    out: Dict[str, float] = {}
    for key, val in m.items():
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(
                "min_confidence map keys must be a bucket like 'choice:3-5' or "
                "'default', got %r" % (key,)
            )
        out[key] = _check_one_threshold(val)
    return out


def check_min_confidence(v: Any):
    """Validate opt-in abstention threshold `min_confidence` (#361, #394).

    Either a real number in [0.0, 1.0] (one threshold for every answer; booleans rejected even
    though `isinstance(True, int)`), or a per-bucket mapping (see :func:`check_min_confidence_map`)
    so the threshold can differ by option count. Returns the value in its validated form -- a
    `float` for the scalar case, a `dict[str, float]` for the mapping case -- which the gate
    functions below both accept.
    """
    if isinstance(v, dict):
        return check_min_confidence_map(v)
    return _check_one_threshold(v)


# Bucket spelling mirrors `common.temp_bucket` but is reproduced here so this module stays
# torch-free (it must import without PyTorch). The answer already carries its type name and, via
# `probabilities`, its option count, so no checkpoint config is needed. `BUCKET_QTYPES` is the one
# list of types it reads, and `tests/test_conformal_abstention.py` compares what this returns to
# `common.temp_bucket` for every option count, so the two bands cannot drift apart unnoticed.
def _option_bucket(answer: Dict[str, Any]) -> Optional[str]:
    qt = answer.get("type")
    if qt not in BUCKET_QTYPES:
        return None
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        k = len(probs)
    elif qt == "noul":
        k = 2
    else:
        return None
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (qt, size)


def resolve_min_confidence(
    answer: Dict[str, Any], thresholds: Dict[str, float], default: float = 0.0
) -> float:
    """The threshold this answer's option-count bucket is gated at, under a per-bucket map.

    Falls back to the map's ``"default"`` entry, then to `default` (0.0 -- gate nothing), for a
    bucket the map does not name, so an unconfigured bucket never abstains by surprise. A map that
    went through :func:`check_min_confidence_map` can only miss here by omitting a bucket: a key
    mis-spelled well enough that no answer produces it is refused at validation, because it would
    otherwise reach this fall-through and leave that bucket gated at nothing while its caller
    believes it is gated.
    """
    key = _option_bucket(answer)
    if key is not None and key in thresholds:
        return thresholds[key]
    return thresholds.get("default", default)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: float) -> None:
    """Opt-in abstention marker (#361): flag answers whose confidence falls below `min_confidence`.

    Reads `answer_confidence` (`max(p)`, the quantity the calibration figures describe and the one
    that does not drift with the number of options), falling back to `confidence` if
    `answer_confidence` is absent.
    The raw answer and confidence stay intact; `low_confidence: True` is added when the answer
    falls below the threshold, and removed if a previously-flagged answer now clears it (e.g.
    when a result dict is reused or re-evaluated with a different threshold).

    `min_confidence` is either a float (one threshold for every answer) or a per-bucket mapping
    (#394), in which case each answer is gated at the threshold of its own option-count bucket via
    :func:`resolve_min_confidence`.
    """
    is_map = isinstance(min_confidence, dict)
    if not is_map and min_confidence == 0.0:
        return
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            conf = _gate_confidence(a)
            if conf is None:
                a.pop("low_confidence", None)
                continue
            thr = resolve_min_confidence(a, min_confidence) if is_map else min_confidence
            if conf < thr:
                a["low_confidence"] = True
            else:
                a.pop("low_confidence", None)


#: The states :func:`apply_confidence_gate` reports, and the only ones. There is deliberately no
#: "no gate was configured" member: when `min_confidence` is not set the function writes nothing at
#: all, so a caller tells "this run had no gate" from "this answer cleared the gate" by whether
#: `abstention` is present. A sentinel for the unconfigured case would put that same information
#: back into the payload for every caller, which is the thing the gate is meant to avoid.
GATE_PASSED = "passed"  # a gate ran and this answer's confidence cleared it
GATE_ABSTAINED = "abstained"  # a gate ran and this answer's confidence fell below it
GATE_UNEVALUATED = "unevaluated"  # a gate ran and this answer carried no usable confidence

GATE_STATES = (GATE_PASSED, GATE_ABSTAINED, GATE_UNEVALUATED)


def apply_confidence_gate(
    results: List[Dict[str, Any]], min_confidence: Optional[float] = None
) -> None:
    """Report the confidence gate's state, on the answers a gate was actually applied to.

    A gate is a policy, and a policy whose application cannot be observed is not one. With
    `min_confidence` set, this writes ``abstention`` -- one of :data:`GATE_STATES` -- onto every
    answer, plus ``abstention_threshold``, so a caller can answer three questions it otherwise
    cannot:

    * what fraction of decisions abstained, rather than inferring it from whether
      ``low_confidence`` happened to be set;
    * how many answers the gate could not decide on, which a boolean cannot express at all;
    * what threshold produced these results -- ``flag_low_confidence`` consumes the threshold and
      drops it, so without this a batch run with per-class thresholds cannot be re-split.

    ``GATE_UNEVALUATED`` is the case a boolean cannot express: the gate ran and the answer carried
    no usable confidence, so the gate could not decide. Reporting that as a pass is the same lie
    as reporting it as a flag.

    **With `min_confidence` unset, this writes nothing.** No ``abstention``, no
    ``abstention_threshold``, no flag. That is the whole contract: an ungated call returns exactly
    the payload it returned before, and the presence of the field -- not a fourth value read out
    of it -- is what tells a caller the gate ran. Call it unconditionally, once per call, in place
    of an `if min_confidence is not None:` guard: that guard is what leaves a path reporting
    nothing at all, which is the state this function exists to distinguish.

    The flag itself stays :func:`flag_low_confidence`'s -- this delegates rather than
    re-implementing the rule, so the boolean and the reported state cannot drift apart.

    A `min_confidence` of exactly ``0.0`` *was* set, so states are reported, and
    :func:`flag_low_confidence` treats ``0.0`` as a no-op because nothing can fall below it. Every
    answer carrying a usable confidence therefore reads ``passed``, and the threshold echo is what
    distinguishes that from a real pass at a real threshold.
    """
    if min_confidence is None:
        return
    is_map = isinstance(min_confidence, dict)
    flag_low_confidence(results, min_confidence)
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            if not is_map and min_confidence == 0.0:
                a.pop("low_confidence", None)
            if a.get("low_confidence"):
                a["abstention"] = GATE_ABSTAINED
            elif _gate_confidence(a) is None:
                a["abstention"] = GATE_UNEVALUATED
            else:
                a["abstention"] = GATE_PASSED
            # With a per-bucket map, echo the threshold this answer's bucket was actually gated at.
            a["abstention_threshold"] = float(
                resolve_min_confidence(a, min_confidence) if is_map else min_confidence
            )
