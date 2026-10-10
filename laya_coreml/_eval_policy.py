"""Opt-in, weight-free quality gates over the slices already in an eval report."""

import json
import math
from typing import Any, Dict, List, Optional

from . import evals

_DIMENSIONS = ("language", "model", "qid", "tag")
_LIMITS = ("min", "max", "max_drop", "max_increase")
_RELATIVE = ("max_drop", "max_increase")


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def load_policy(path: str) -> Dict[str, Any]:
    """Read and validate a small, versioned JSON gate before a checkpoint is loaded."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            policy = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise evals.EvalError("invalid gate policy %r: %s" % (path, exc)) from exc
    if not isinstance(policy, dict) or set(policy) != {"version", "rules"}:
        raise evals.EvalError("gate policy needs exactly 'version' and 'rules'")
    if type(policy["version"]) is not int or policy["version"] != 1:
        raise evals.EvalError("gate policy version must be 1")
    rules = policy["rules"]
    if not isinstance(rules, list) or not rules:
        raise evals.EvalError("gate policy 'rules' must be a non-empty list")
    for index, rule in enumerate(rules, 1):
        where = "gate policy rule %d" % index
        if not isinstance(rule, dict):
            raise evals.EvalError("%s must be an object" % where)
        limits = set(rule) & set(_LIMITS)
        if (
            set(rule) - {"slice", "metric", "min_count"} - set(_LIMITS)
            or len(limits) != 1
            or not {"slice", "metric", "min_count"} <= set(rule)
        ):
            raise evals.EvalError("%s needs slice, metric, min_count and exactly one limit" % where)
        selector = rule["slice"]
        if (
            not isinstance(selector, dict)
            or len(selector) != 1
            or next(iter(selector)) not in _DIMENSIONS
            or not isinstance(next(iter(selector.values())), str)
            or not next(iter(selector.values())).strip()
        ):
            raise evals.EvalError(
                "%s slice must name one non-empty language/model/qid/tag value" % where
            )
        if not isinstance(rule["metric"], str) or not rule["metric"].strip():
            raise evals.EvalError("%s metric must be a non-empty string" % where)
        if type(rule["min_count"]) is not int or rule["min_count"] < 1:
            raise evals.EvalError("%s min_count must be a positive integer" % where)
        limit_name = next(iter(limits))
        limit = rule[limit_name]
        if not _finite_number(limit) or (limit_name in _RELATIVE and limit < 0):
            raise evals.EvalError(
                "%s %s must be finite%s"
                % (where, limit_name, " and non-negative" if limit_name in _RELATIVE else "")
            )
    return policy


def needs_baseline(policy: Dict[str, Any]) -> bool:
    return any(set(rule) & set(_RELATIVE) for rule in policy["rules"])


def _count(report: evals.EvalReport, dimension: str, value: str, metric: str) -> int:
    cases = report.cases if isinstance(report.cases, list) else []
    count = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        if dimension == "tag":
            tags = case.get("tags")
            values = tags if isinstance(tags, (list, tuple)) else []
        else:
            values = [case.get(dimension)]
        if value not in [str(item) for item in values if item is not None]:
            continue
        if evals.is_confidence_metric(metric):
            confidence, correct = case.get("confidence"), case.get("correct")
            valid = _finite_number(confidence) and isinstance(correct, bool)
        else:
            scores = case.get("scores")
            score = scores.get(metric) if isinstance(scores, dict) else None
            valid = _finite_number(score)
        count += bool(valid)
    return count


def _value(report: evals.EvalReport, dimension: str, value: str, metric: str) -> Optional[float]:
    slices = report.slices if isinstance(report.slices, dict) else {}
    dimension_slices = slices.get(dimension)
    if not isinstance(dimension_slices, dict):
        return None
    group = dimension_slices.get(value)
    if not isinstance(group, dict):
        return None
    result = group.get(metric)
    if not _finite_number(result):
        return None
    return float(result)


def _incomparable_coverage_definitions(
    report: evals.EvalReport, baseline: Any, metric: str
) -> Optional[str]:
    """Why a relative rule on `metric` must not subtract these two reports, or None.

    Delegates to `evals.coverage_definition_conflict` so that this gate and `EvalReport.compare`
    cannot drift apart about what "comparable" means -- they are the same rule, not two copies of
    it. Both the candidate and the baseline are checked: see that function for why a stale
    candidate is the more dangerous of the two.
    """
    if not evals.is_coverage_metric(metric):
        return None
    conflict = evals.coverage_definition_conflict(report.config, baseline)
    if conflict is None:
        return None
    return "%s on a relative limit" % conflict


def check_policy(
    report: evals.EvalReport, policy: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Return actionable failures. Missing measurements never count as a pass."""
    failures: List[str] = []
    base_report = None
    if baseline is not None:
        base_report = evals.EvalReport(
            config=evals._identity_of(baseline),
            slices=baseline.get("slices", {}),
            cases=baseline.get("cases", []),
        )
    if report.config.get("errored"):
        failures.append("slice gate: candidate has skipped/errored cases")
    if base_report is not None and base_report.config.get("errored"):
        failures.append("slice gate: baseline has skipped/errored cases")
    if needs_baseline(policy):
        if base_report is None:
            return failures + ["slice gate: relative rule needs a baseline"]
        missing = [
            key
            for key in evals._IDENTITY_KEYS
            if report.config.get(key) is None or base_report.config.get(key) is None
        ]
        if missing:
            failures.append(
                "slice gate: baseline and candidate need %s identity" % ", ".join(missing)
            )
        comparable, reasons = report.comparable_to(baseline)
        if not comparable:
            failures.append("slice gate: baseline is not comparable: " + "; ".join(reasons))
    for index, rule in enumerate(policy["rules"], 1):
        dimension, value = next(iter(rule["slice"].items()))
        metric = rule["metric"]
        label = "slice gate rule %d (%s=%s, %s)" % (index, dimension, value, metric)
        current = _value(report, dimension, value, metric)
        current_count = _count(report, dimension, value, metric)
        minimum = rule["min_count"]
        if current is None:
            failures.append("%s: metric or slice is missing" % label)
            continue
        if current_count < minimum:
            failures.append(
                "%s: candidate count %d is below min_count %d" % (label, current_count, minimum)
            )
            continue
        comparator = next(name for name in _LIMITS if name in rule)
        limit = rule[comparator]
        if comparator in _RELATIVE:
            # the raw baseline document, not `base_report`: that is rebuilt as an
            # `EvalReport` and carries only what `_identity_of` kept
            stale = _incomparable_coverage_definitions(report, baseline, metric)
            if stale is not None:
                failures.append("%s: %s" % (label, stale))
                continue
            previous = _value(base_report, dimension, value, metric)
            previous_count = _count(base_report, dimension, value, metric)
            if previous is None:
                failures.append("%s: baseline metric or slice is missing" % label)
                continue
            if previous_count < minimum:
                failures.append(
                    "%s: baseline count %d is below min_count %d" % (label, previous_count, minimum)
                )
                continue
            drift = previous - current if comparator == "max_drop" else current - previous
            if drift > limit and not math.isclose(drift, limit, rel_tol=1e-12, abs_tol=1e-12):
                failures.append(
                    "%s: candidate=%.4f (n=%d), baseline=%.4f (n=%d), %s=%.4f exceeded "
                    "by %.4f"
                    % (
                        label,
                        current,
                        current_count,
                        previous,
                        previous_count,
                        comparator,
                        limit,
                        drift - limit,
                    )
                )
        elif (comparator == "min" and current < limit) or (comparator == "max" and current > limit):
            failures.append(
                "%s: candidate=%.4f (n=%d), %s=%.4f failed"
                % (label, current, current_count, comparator, limit)
            )
    return failures
