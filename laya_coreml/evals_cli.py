"""`laya-evals`: run a labelled evaluation, validate a dataset, or compare a report.

Exit codes: 0 on success, 1 when a threshold or a baseline tolerance fails, 2 on a usage error.

    laya-evals validate research/evals/fixture.jsonl
    laya-evals run research/evals/fixture.jsonl --model english --min-accuracy 0.8 --max-ece 0.1
    laya-evals run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 --json out.json
    laya-evals run data.jsonl --score-within 0.25 --min score_within_0.25=0.9

`laya eval ...` dispatches here from the main CLI, so both spellings work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _eval_policy, evals, evidence
from .evals import EvalError


class RouterRunner:
    """Adapt a `Router` to the harness: one predict, or a batch of identical questions."""

    def __init__(self, router: Any):
        self.router = router

    def predict(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        if min_confidence is not None:
            return self.router.predict(state, questions, model=model, min_confidence=min_confidence)
        return self.router.predict(state, questions, model=model)

    def predict_batch(
        self,
        states: Sequence[Any],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        sort_by_length: bool = False,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        requests = [{"state": state, "questions": questions, "model": model} for state in states]
        kwargs = {"batch_size": batch_size, "sort_by_length": sort_by_length}
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        return self.router.predict_batch(requests, **kwargs)


class OnnxRunner:
    """Adapt a single-checkpoint `ONNXAgent` to the harness, like `RouterRunner` does for a Router.

    The agent serves one checkpoint, so a per-example `model` that names a different one is an
    error rather than a silent no-op: the report would otherwise claim to score a fleet it never
    ran. `predict_batch` delegates to the agent when it has one and falls back to one predict per
    state otherwise, so the harness's batching path works on every ONNXAgent build.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def _check_model(self, model: Optional[str]) -> None:
        if model is not None and model != self.agent.model_id:
            raise EvalError(
                "the ONNX runner serves only %r, but this example asks for %r; "
                "run them separately or drop --onnx" % (self.agent.model_id, model)
            )

    def predict(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._check_model(model)
        if min_confidence is not None:
            return self.agent.predict(state, questions, min_confidence=min_confidence)
        return self.agent.predict(state, questions)

    def predict_batch(
        self,
        states: Sequence[Any],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        sort_by_length: bool = False,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        self._check_model(model)
        agent_batch = getattr(self.agent, "predict_batch", None)
        extra = {}
        if min_confidence is not None:
            extra["min_confidence"] = min_confidence
        if agent_batch is not None:
            if sort_by_length:
                # Asked for, so it has to reach the agent's own grouping. The per-state fallback
                # below cannot honour it: there is no batch to reorder.
                return agent_batch(
                    list(states), questions, batch_size=batch_size, sort_by_length=True, **extra
                )
            return agent_batch(list(states), questions, batch_size=batch_size, **extra)
        return [self.agent.predict(state, questions, **extra) for state in states]


def _parse_pairs(pairs: Optional[Sequence[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pair in pairs or []:
        name, _, raw = pair.partition("=")
        if not name or not raw:
            raise EvalError("expected NAME=VALUE, got %r" % pair)
        try:
            number = float(raw)
        except ValueError:
            raise EvalError("%r is not a number in %r" % (raw, pair))
        # float() accepts "nan", and a NaN limit or tolerance would disable the gate it names.
        if math.isnan(number):
            raise EvalError("%r is not a number in %r" % (raw, pair))
        out[name.strip()] = number
    return out


def _limit(raw: str) -> float:
    """argparse `type=` for `--min-accuracy` and `--max-ece`: refuse "nan" as `_parse_pairs` does."""
    try:
        number = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not a number" % raw)
    # float() accepts "nan", and a NaN limit would disable the gate it names.
    if math.isnan(number):
        raise argparse.ArgumentTypeError("%r is not a number" % raw)
    return number


def _parse_revisions(pairs: Optional[Sequence[str]]) -> Tuple[Optional[str], Dict[str, str]]:
    """Read `--revision` into the two forms `Router` takes: one commit, or one per checkpoint.

    A bare value is the commit for every checkpoint this run loads; `NAME=SHA` pins one
    checkpoint, which is what an auto-routing run needs, because the three checkpoints are three
    Hub repositories and one commit cannot exist in all of them.
    """
    shared: Optional[str] = None
    per_model: Dict[str, str] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if sep:
            name, value = name.strip(), value.strip()
            if not name or not value:
                raise EvalError("expected NAME=REVISION, got %r" % pair)
            per_model[name] = value
        else:
            value = pair.strip()
            if not value:
                raise EvalError("--revision wants a commit SHA or NAME=SHA, got %r" % pair)
            if shared is not None and shared != value:
                raise EvalError(
                    "--revision names two commits for every checkpoint: %r, %r" % (shared, value)
                )
            shared = value
    return shared, per_model


def _score_within(tolerances: Optional[Sequence[float]]) -> List[evals.Evaluator]:
    """Build the `ScoreWithin` evaluators `--score-within` asks for, in the order given.

    A non-finite or negative tolerance would name a metric that is always 1.0 or always 0.0, and a
    `--min` gate on that metric would then decide nothing, so the CLI rejects them up front.
    """
    out: List[evals.Evaluator] = []
    for tolerance in tolerances or []:
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise EvalError("--score-within needs a finite tolerance >= 0, got %g" % tolerance)
        out.append(evals.ScoreWithin(tolerance))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-evals", description="Evaluate a Laya checkpoint on a labelled dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check a dataset file without running a model")
    validate.add_argument(
        "dataset",
        help="JSONL file to check; blank lines and lines starting with '#' are "
        "ignored, and a file that leaves no examples is an error",
    )

    run = sub.add_parser("run", help="evaluate a dataset and apply thresholds")
    run.add_argument(
        "dataset",
        help="labelled JSONL file; each row carries a state, the questions to answer "
        "on it, and the expected answer per question id",
    )
    run.add_argument("--model", help="force a checkpoint instead of auto-routing")
    run.add_argument("--device", help="torch device, e.g. cpu or cuda")
    run.add_argument(
        "--onnx",
        metavar="PATH",
        help="evaluate an ONNX export through ONNXAgent instead of the torch Router; "
        "--model then names the checkpoint directory or Hub id the export came "
        "from (default convaiinnovations/laya)",
    )
    run.add_argument(
        "--calibration",
        metavar="PATH",
        help="path to a JSON calibration map for ONNXAgent (requires --onnx)",
    )
    run.add_argument(
        "--revision",
        action="append",
        metavar="SHA | NAME=SHA",
        help="pin the checkpoint commit: a bare SHA applies to every checkpoint this "
        "run loads, NAME=SHA pins one (repeatable). Unpinned runs fetch the "
        "checkpoint's default branch; the report records the commit that answered "
        "either way",
    )
    run.add_argument(
        "--batch-size", type=int, help="examples per forward pass when questions match"
    )
    run.add_argument(
        "--sort-by-length",
        action="store_true",
        dest="sort_by_length",
        help="with --batch-size N where 1 < N < the run, group similarly sized "
        "examples into the same forward pass so each pads to a shorter maximum; "
        "scores the same answers, in the same order",
    )
    run.add_argument(
        "--min-confidence",
        dest="min_confidence",
        type=float,
        metavar="THRESHOLD",
        help="abstention threshold on `answer_confidence` (#361): every answer below "
        "it is returned flagged with `low_confidence: True` and "
        '`abstention: "abstained"`. The gate writes those state fields and '
        'leaves `answer["choice"] / ["noul"] / ["score"]` as the raw argmax, '
        "so the metrics are identical at every threshold -- what changes is the "
        "report's config, which names the threshold and asserts it was sent. "
        "Accepted range is core's -- `laya.confidence.check_min_confidence` -- "
        "not a copy of it here, and a runner that predates the gate is refused "
        "with a named error rather than publishing a `min_confidence` it never "
        "applied",
    )
    run.add_argument(
        "--on-error",
        choices=("fail", "skip"),
        default="fail",
        help="'fail' (the default) stops the run when a runner call raises; 'skip' "
        "lists every row it could not score under the report's config.errored "
        "and reports the metrics for the rows that returned",
    )
    run.add_argument("--baseline", help="a baseline report JSON to compare against")
    run.add_argument(
        "--tolerance",
        action="append",
        metavar="METRIC=VALUE",
        help="allowed absolute drift from the baseline; repeatable",
    )
    run.add_argument(
        "--min-accuracy",
        type=_limit,
        help="minimum accuracy (choice, else noul) for the whole dataset",
    )
    run.add_argument("--max-ece", type=_limit, help="maximum expected calibration error")
    run.add_argument(
        "--score-within",
        dest="score_within",
        type=float,
        action="append",
        metavar="TOL",
        help="also report score_within_TOL, the fraction of score answers within TOL "
        "of the label; repeatable, and added to the default metrics",
    )
    run.add_argument(
        "--min", action="append", metavar="METRIC=VALUE", help="minimum for any metric"
    )
    run.add_argument(
        "--max", action="append", metavar="METRIC=VALUE", help="maximum for any metric"
    )
    run.add_argument(
        "--slice",
        action="append",
        choices=("language", "model", "qid", "tag"),
        help="also report this slice dimension; repeatable",
    )
    run.add_argument(
        "--gate-policy",
        metavar="FILE",
        help="apply opt-in per-slice quality rules from a JSON policy",
    )
    run.add_argument("--json", dest="json_out", help="write the full report JSON here")
    run.add_argument("--markdown", dest="markdown_out", help="write a Markdown summary here")

    compare = sub.add_parser("compare", help="compare a report JSON against a baseline")
    compare.add_argument("report", help="a report JSON written by `laya-evals run --json`")
    compare.add_argument(
        "--baseline",
        required=True,
        help="the reviewed report JSON this one is checked against; its dataset "
        "and question fingerprints must match",
    )
    compare.add_argument(
        "--tolerance",
        action="append",
        metavar="METRIC=VALUE",
        help="allowed absolute drift; repeatable",
    )
    compare.add_argument(
        "--gate-policy",
        metavar="FILE",
        help="apply opt-in per-slice quality rules from a JSON policy",
    )
    evidence_cmd = sub.add_parser(
        "evidence",
        help="read-only evidence inspection over a checkpoint config and an eval report",
        epilog="example: laya-evals evidence --checkpoint ./my-checkpoint --report report.json",
    )
    evidence_cmd.add_argument(
        "--checkpoint", required=True, help="checkpoint directory containing rl_agent_config.json"
    )
    evidence_cmd.add_argument("--report", default=None, help="an existing laya-evals report JSON")

    return parser


def _load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_thresholds(
    overall: Dict[str, float], mins: Dict[str, float], maxs: Dict[str, float]
) -> List[str]:
    failures: List[str] = []
    if "choice_accuracy" not in overall and "noul_accuracy" in overall:
        overall = dict(overall, choice_accuracy=overall["noul_accuracy"])
    for name, limit in mins.items():
        value = overall.get(name)
        if value is None:
            failures.append("metric %r is not in the report" % name)
        elif math.isnan(value):
            failures.append("metric %r is NaN" % name)
        elif value < limit:
            failures.append("%s=%.4f is below the minimum %.4f" % (name, value, limit))
    for name, limit in maxs.items():
        value = overall.get(name)
        if value is None:
            failures.append("metric %r is not in the report" % name)
        elif math.isnan(value):
            failures.append("metric %r is NaN" % name)
        elif value > limit:
            failures.append("%s=%.4f is above the maximum %.4f" % (name, value, limit))
    return failures


def _print_deltas(deltas: Dict[str, Dict[str, Any]]) -> None:
    for metric, delta in sorted(deltas.items()):
        if delta.get("missing"):
            print("%-18s baseline=%.4f missing from the report" % (metric, delta["baseline"]))
            continue
        if delta.get("incomparable"):
            # The diff is NaN by construction here, and "diff=+nan" on its own is not a
            # diagnosis: print the reason the comparison was refused, not just its symptom.
            print(
                "%-18s baseline=%.4f value=%.4f not compared: %s"
                % (metric, delta["baseline"], delta["value"], delta["incomparable"])
            )
            continue
        print(
            "%-18s baseline=%.4f value=%.4f diff=%+.4f (tol %.4f)"
            % (metric, delta["baseline"], delta["value"], delta["diff"], delta["tolerance"])
        )


def _comparability_failure(report: evals.EvalReport, baseline: Dict[str, Any]) -> Optional[str]:
    """The refusal to compare two runs that are not the same measurement, as one failure line.

    The metric gate answers "did the numbers move"; it cannot answer "were these the same
    numbers", because `EvalReport.compare` reads `overall` and only `overall`. So a baseline
    recorded against one dataset passes a candidate scored on another, with identical
    arithmetic. The gate still prints its deltas -- the reviewer wants to see both facts -- but
    an incomparable pair is not a pass.
    """
    ok, reasons = report.comparable_to(baseline)
    if ok:
        return None
    return "baseline is not comparable: " + "; ".join(reasons)


def _cmd_validate(args) -> int:
    dataset = evals.Dataset.from_jsonl(args.dataset)
    questions = sorted({qid for example in dataset.examples for qid in example.questions})
    print(
        "%d examples, %d question id(s): %s" % (len(dataset), len(questions), ", ".join(questions))
    )
    return 0


def _warn_no_value(extra: Sequence[evals.Evaluator], report: evals.EvalReport) -> None:
    """Say so when a metric the caller asked for scored nothing.

    `evaluate` drops a metric no answer applied to, so a requested `--score-within` over a
    choice-only dataset would otherwise change the command and not the report.
    """
    missing = [evaluator.name for evaluator in extra if evaluator.name not in report.overall]
    if not missing:
        return
    score_cases = sum(
        1 for case in report.cases if (case.get("answer") or {}).get("type") == "score"
    )
    for name in missing:
        print(
            "laya-evals: %s has no value: %d of %d answered case(s) are score answers, and the "
            "metric needs one whose label is a number. A threshold naming it reports it missing."
            % (name, score_cases, len(report.cases)),
            file=sys.stderr,
        )


def _cmd_run(args) -> int:
    dataset = evals.Dataset.from_jsonl(args.dataset)
    policy = _eval_policy.load_policy(args.gate_policy) if args.gate_policy else None
    if policy and _eval_policy.needs_baseline(policy) and not args.baseline:
        raise EvalError("--gate-policy has a relative rule; pass --baseline")
    baseline = _load_report(args.baseline) if args.baseline and policy else None
    if args.model:
        for example in dataset.examples:  # --model is authoritative over per-row model
            example.model = args.model
    extra = _score_within(args.score_within)  # before the checkpoint loads: a bad flag is cheap
    revision, revisions = _parse_revisions(args.revision)
    router = None
    if args.onnx:
        raise EvalError("ONNX evaluation requires official laya; this distribution uses Core ML")
    else:
        if args.calibration:
            raise EvalError(
                "--calibration requires --onnx; for the torch Router calibrate via fit_temperatures"
            )
        import laya_coreml as laya

        try:
            pins: Dict[str, Any] = {}
            if revision is not None:
                pins["revision"] = revision
            if revisions:
                pins["revisions"] = revisions
            router = laya.Router(device=args.device, preload=False, **pins)
        except ValueError as exc:
            # `Router` already rejects a checkpoint name it does not know -- normalising case,
            # surrounding spaces and aliases on the way -- with the option list in the message.
            # This only turns that into `laya-evals: unknown model 'englishg'; choose one of …`
            # instead of a traceback, so a mistyped pin fails as a usage error and never starts a
            # download.
            raise EvalError(str(exc)) from exc
        runner = RouterRunner(router)
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "device": args.device,
        # The path above is the name, not the data. Hashing the bytes here is what makes
        # a committed baseline reviewable: the dataset can be edited in place, moved or
        # refetched under the same name, and a reviewer comparing two reports needs the
        # report -- not a file mtime -- to say so.
        "dataset_sha256": evals.file_fingerprint(args.dataset),
    }
    if policy:
        config["gate_policy"] = policy
    if args.onnx:
        config["onnx"] = args.onnx
        if args.calibration:
            config["calibration"] = args.calibration
    if extra:
        config["score_within"] = [evaluator.tolerance for evaluator in extra]
    report = evals.evaluate(
        runner,
        dataset,
        evaluators=evals.default_evaluators() + extra,
        batch_size=args.batch_size,
        on_error=args.on_error,
        config=config,
        sort_by_length=args.sort_by_length,
        min_confidence=args.min_confidence,
    )
    # Which commit answered belongs in the artifact a baseline is, and it can only be read after
    # the run: `preload=False` means no checkpoint is resident before the first row.
    # `loaded_revisions` reports the commit each resident agent came from -- the pin when there is
    # one, the commit the default branch resolved to when there is not, None for a local path.
    if router is not None:
        report.config = dict(report.config, revisions=getattr(router, "loaded_revisions", {}))

    if extra:
        _warn_no_value(extra, report)

    mins = _parse_pairs(args.min)
    maxs = _parse_pairs(args.max)
    if args.min_accuracy is not None:
        key = "choice_accuracy" if "choice_accuracy" in report.overall else "noul_accuracy"
        mins[key] = args.min_accuracy
    if args.max_ece is not None:
        maxs["ece"] = args.max_ece
    tolerances = _parse_pairs(args.tolerance)
    # The gate this run actually applied, recorded with the numbers it produced. A baseline whose
    # numbers came from a different gate is a different claim, and `docs/staged-adoption.md`
    # asks for exactly this alongside the checkpoint version and the evaluation set.
    report.config = dict(
        report.config, thresholds={"min": mins, "max": maxs, "baseline_tolerance": tolerances}
    )

    failures = _check_thresholds(report.overall, mins, maxs)
    if args.baseline:
        if baseline is None:
            baseline = _load_report(args.baseline)
        ok, deltas = report.compare(baseline, tolerances)
        _print_deltas(deltas)
        if not ok:
            failures.append("baseline comparison failed")
        # Checked after the deltas are printed, so a reviewer sees what moved as well as why the
        # two runs are not the same experiment.
        refusal = _comparability_failure(report, baseline)
        if refusal:
            failures.append(refusal)
    if policy:
        failures.extend(_eval_policy.check_policy(report, policy, baseline))

    for name in sorted(report.overall):
        print("%-18s %.4f" % (name, report.overall[name]))
    for dimension in args.slice or []:
        for value, metrics in sorted(report.slices.get(dimension, {}).items()):
            print(
                "  %s=%s  %s"
                % (
                    dimension,
                    value,
                    " ".join("%s=%.4f" % (k, v) for k, v in sorted(metrics.items())),
                )
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report.to_json(), handle, ensure_ascii=False, indent=2, sort_keys=True)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write(report.to_markdown())

    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1
    return 0


def _cmd_evidence(args) -> int:
    try:
        result = evidence.inspect_checkpoint(args.checkpoint, args.report)
    except (FileNotFoundError, ValueError) as exc:
        print("laya-evals: %s" % exc, file=sys.stderr)
        return 2
    print(evidence.format_summary(result))
    return 0


def _cmd_compare(args) -> int:
    # `_identity_of` rather than a bare `config` slice, because a report may carry its identity
    # at the top level -- `research/evals/act_head_eval.py` puts `schema` there -- and
    # `EvalReport.comparable_to` reads the candidate's identity out of `config`. Filtering to the
    # four known keys dropped a top-level `schema` before the report was built, so a candidate
    # that disagreed with its baseline passed the gate while the same disagreement stated in
    # `config` was refused. The baseline has always gone through `_identity_of`; this makes the
    # candidate side symmetric.
    policy = _eval_policy.load_policy(args.gate_policy) if args.gate_policy else None
    document = _load_report(args.report)
    report = evals.EvalReport(
        config=evals._identity_of(document),
        **{k: v for k, v in document.items() if k in ("overall", "slices", "cases")},
    )
    if policy and report.config.get("gate_policy") not in (None, policy):
        print(
            "laya-evals: warning: --gate-policy differs from the report's recorded gate_policy; "
            "this verdict uses --gate-policy",
            file=sys.stderr,
        )
    baseline = _load_report(args.baseline)
    ok, deltas = report.compare(baseline, _parse_pairs(args.tolerance))
    _print_deltas(deltas)
    # `compare` is the offline re-check of a report a reviewer already read, so the same
    # comparability refusal has to apply here: a saved report that cannot say which experiment
    # it came from must not pass a gate either.
    refusal = _comparability_failure(report, baseline)
    failures = [refusal] if refusal else []
    if policy:
        failures.extend(_eval_policy.check_policy(report, policy, baseline))
    for failure in failures:
        print("FAIL: " + failure, file=sys.stderr)
    return 0 if ok and not failures else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    # Same reason as `laya.cli.main`: redirected output is encoded with the locale's codec by
    # default (cp1252 on Windows), which cannot hold a non-Latin question id, tag or language, so
    # `laya-evals validate data.jsonl > out.txt` died with a traceback and exit 1 -- the code kept
    # for a quality failure. `laya eval` lands here before `cli.main` reaches its own copy.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "evidence":
            return _cmd_evidence(args)
        return _cmd_compare(args)
    except EvalError as exc:
        # A malformed dataset, an unreadable report, or a mistyped pin is a usage error, not a
        # quality result. Returning 1 for these made "my dataset is broken" and "the model
        # regressed" indistinguishable to a CI job, which is the one distinction the documented
        # exit codes exist to draw. (argparse already exits 2 for a bad flag on its own.)
        print("laya-evals: %s" % exc, file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        # A dataset or report path that does not exist is the caller's mistake. Narrowed twice on
        # purpose. Bare `OSError` would swallow a Hub outage. `FileNotFoundError` still would too:
        # huggingface_hub raises LocalEntryNotFoundError -- a failed *download*, subclassing
        # FileNotFoundError -- when a checkpoint is not cached and cannot be fetched, and "the
        # network is down" is an environment fault, not a usage error. Distinguishing the two by
        # name is deliberate; a checkpoint that cannot be downloaded stays an unhandled failure,
        # which is what it was before.
        from huggingface_hub.errors import LocalEntryNotFoundError

        if isinstance(exc, LocalEntryNotFoundError):
            raise
        print("laya-evals: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
