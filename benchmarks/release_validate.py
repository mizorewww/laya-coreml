"""Check portable distribution bundles against the pinned original FP32 fixture."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from laya_coreml import load
from laya_coreml.artifacts import file_digest, verify_files
from laya_coreml.inputs import collate_items

from .common import calibrated, environment, save, softmax


def validate(directory, reference_path, repeats):
    manifest = json.loads((directory / "coreml_config.json").read_text())
    verify_files(directory, manifest["files"])
    name = manifest["source"].split("/")[-1]
    reference = json.loads(reference_path.read_text())["models"][name]
    if manifest["source_weights_sha256"] != reference["source_weights_sha256"]:
        raise ValueError("Bundle and golden reference originate from different weights")
    start = time.perf_counter()
    agent = load(directory, local_files_only=True)
    report = {
        "environment": environment(),
        "repository": manifest["repository"],
        "format": manifest["format"],
        "shape": agent.shape,
        "compute_units": agent.compute_units,
        "load_seconds": time.perf_counter() - start,
        "source_weights_sha256": manifest["source_weights_sha256"],
        "package_sha256": manifest["package_sha256"],
        "reference_sha256": file_digest(reference_path),
        "bundle_files_verified": True,
        "cases": [],
        "passed": False,
    }
    total = agree = 0
    drift = action_drift = 0.0
    repeat_cases = []
    for entry in reference["cases"]:
        # Compare only source fixture rows that fit this graph without truncation.
        # The runtime now uses the graph's true context for upstream predict_long.
        if any(
            len(i["ids"]) > agent.shape["max_length"]
            or len(i["markers"]) > agent.shape["max_options"]
            for i in entry["items"]
        ):
            report["cases"].append({"name": entry["name"], "skipped": "outside bundle capacity"})
            continue
        items, _ = agent.prepare(entry["state"], entry["questions"])
        if [{key: item[key] for key in ("ids", "markers", "qtype")} for item in items] != entry[
            "items"
        ]:
            raise AssertionError("Prompt tokens changed during packaging")
        correct = 0
        case_drift = case_action = 0.0
        for start in range(0, len(items), agent.batch_size):
            chunk = items[start : start + agent.batch_size]
            batch = collate_items(chunk, agent.tok.pad_token_id, shape=agent.shape)
            logits, actions = agent.forward(batch)
            if not np.isfinite(logits).all() or not np.isfinite(actions).all():
                raise FloatingPointError("Non-finite release-bundle outputs")
            for row, item in enumerate(chunk):
                prob = calibrated(agent, logits[row], item)
                expected = calibrated(agent, entry["logits"][start + row], item)
                correct += int(prob.argmax() == expected.argmax())
                case_drift = max(case_drift, float(np.abs(prob - expected).max()))
                case_action = max(
                    case_action,
                    float(
                        np.abs(
                            softmax(actions[row]) - softmax(entry["action_logits"][start + row])
                        ).max()
                    ),
                )
        result = agent.predict(entry["state"], entry["questions"])
        if any(
            result["usage"][key] != entry["result"]["usage"][key]
            for key in ("input_tokens", "output_tokens")
        ):
            raise AssertionError("Token usage changed during packaging")
        repeat_cases.append((entry["state"], entry["questions"], result))
        report["cases"].append(
            {
                "name": entry["name"],
                "questions": len(items),
                "argmax_agreements": correct,
                "probability_max_abs_error": case_drift,
                "action_probability_max_abs_error": case_action,
            }
        )
        total += len(items)
        agree += correct
        drift, action_drift = max(drift, case_drift), max(action_drift, case_action)
    if not repeat_cases:
        raise ValueError("No fixture fits this bundle")
    stable = True
    for i in range(repeats):
        state, questions, expected = repeat_cases[i % len(repeat_cases)]
        stable &= agent.predict(state, questions) == expected
    report.update(
        questions=total,
        argmax_agreements=agree,
        probability_max_abs_error=drift,
        action_probability_max_abs_error=action_drift,
        stability={"calls": repeats, "identical_rounded_results": stable},
        gate={
            "all_argmax_agree": True,
            "max_probability_error": 0.02,
            "max_action_probability_error": 0.02,
            "finite": True,
            "stable": True,
        },
        passed=total > 0 and total == agree and drift <= 0.02 and action_drift <= 0.02 and stable,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+")
    p.add_argument("--reference", type=Path, default=Path("benchmarks/results/reference.json"))
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/release-validation.json")
    )
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    reports = []
    for directory in args.directories:
        report = validate(directory, args.reference, args.repeats)
        reports.append(report)
        save(directory / "validation.json", report)
        save(args.output, reports)
        print(
            f"{directory.name}: {report['argmax_agreements']}/{report['questions']}, "
            f"probability error {report['probability_max_abs_error']:.6g}, "
            f"stable={report['stability']['identical_rounded_results']}, passed={report['passed']}",
            flush=True,
        )
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
