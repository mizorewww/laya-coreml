"""v0.4.17 boundary regressions that run without native Core ML or model weights."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laya_coreml.agent import Agent
from laya_coreml.calibrate import fit_abstention_thresholds
from laya_coreml.evals import Dataset, EvalError, Example, evaluate
from laya_coreml.evals_cli import _check_thresholds
from laya_coreml.hooks import calibrate_reliability, validate_timeout
from laya_coreml.revisions import verify_digests
from laya_coreml.shortlist import embed_fn_from_agent

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["a/b:c", "nested/C:evil", "nested\\C:evil"])
def test_digest_refuses_nested_drive_anchors_before_reading(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe.*path"):
        verify_digests(str(tmp_path), {name: "0" * 64})


@pytest.mark.parametrize("value", ["0.1", None, [], object(), True, False])
def test_abstention_error_names_invalid_target_before_fitting(value):
    with pytest.raises(ValueError, match="target_error"):
        fit_abstention_thresholds([], [1.0, 1.0, 1.0], {}, target_error=value)


@pytest.mark.parametrize("value", [[], {}, object(), 10**400])
def test_hook_timeout_normalizes_non_numeric_and_overflow_errors(value):
    with pytest.raises(ValueError, match="hooks_timeout"):
        validate_timeout(value)


def test_reliability_cutoffs_preserve_numpy_score_precision_and_json_types():
    scores = np.array([0.123456, 0.654321], dtype=np.float32)
    accept, escalate = calibrate_reliability(scores, np.array([False, True]))
    assert accept == float(scores[1])
    assert type(accept) is float and type(escalate) is float
    json.dumps([accept, escalate], allow_nan=False)


@pytest.mark.parametrize("rows", [[], [{"answers": {}}] * 3])
def test_evaluation_refuses_missing_or_extra_batch_results(rows):
    q = {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}
    data = Dataset([Example(state=str(i), questions=q, expected={"q": "a"}) for i in range(2)])
    runner = SimpleNamespace(predict_batch=lambda *args, **kwargs: iter(rows))
    with pytest.raises(EvalError, match="every row needs its own result"):
        evaluate(runner, data, batch_size=2)


def test_explicit_choice_threshold_does_not_accept_noul_accuracy():
    failures = _check_thresholds({"noul_accuracy": 1.0}, {"choice_accuracy": 0.9}, {})
    assert failures and "choice_accuracy" in failures[0]


def _definitions(path):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            out.update({m.name: m for m in node.body if isinstance(m, ast.FunctionDef)})
    return out


@pytest.mark.parametrize(
    "name",
    [
        "_scan_answer_values",
        "_scan_answer_index",
        "_start_evidence",
        "_decode_answers",
        "predict_long",
    ],
)
def test_changed_host_methods_match_pinned_upstream_ast(name):
    source = ROOT / ".upstream/laya/agent.py"
    if not source.exists():
        pytest.skip("checkout pinned upstream into .upstream")
    original = _definitions(source)[name]
    native = _definitions(ROOT / "laya_coreml/runtime.py")[name]
    assert ast.dump(native, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def _scan_agent():
    """Run the real hook chain with deterministic host-only encoder/forward fixtures."""
    agent = Agent.__new__(Agent)
    agent.cfg = {"max_len": 128, "head_max_len": 20}

    class Tok:
        mask_token = "[MASK]"
        pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    agent.tok = Tok()
    agent.shape = {"max_length": 128}
    agent.temperature_raw = [1.0, 1.0, 1.0]
    agent._init_host()
    agent._forward = lambda batch: (
        np.zeros((len(batch["input_ids"]), 2)),
        np.full((len(batch["input_ids"]), 2), 0.5),
    )
    agent._decode_answers = lambda logits, act, items, ids, internal, row, **kw: {
        qid: {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9}
        for qid in ids
    }
    return agent


@pytest.mark.parametrize("partial", [False, True])
def test_scan_refuses_missing_caller_answers_after_end_hooks(partial):
    agent = _scan_agent()

    def remove(ctx):
        for row in ctx.results[:1] if partial else ctx.results:
            del row["answers"]["q"]

    with pytest.raises(ValueError, match="answer 'q' is missing from"):
        agent.predict_long(
            "x" * 300,
            {"q": {"type": "noul", "instructions": "?"}},
            window=64,
            stride=32,
            on_predict_end=remove,
        )


def test_scan_allows_end_hook_to_remove_all_start_added_answers():
    agent = _scan_agent()

    def add(ctx):
        ctx.questions = dict(ctx.questions, extra={"type": "noul", "instructions": "?"})

    def remove(ctx):
        for row in ctx.results:
            del row["answers"]["extra"]

    result = agent.predict_long(
        "x" * 300,
        {"q": {"type": "noul", "instructions": "?"}},
        window=64,
        stride=32,
        on_predict_start=add,
        on_predict_end=remove,
    )
    assert set(result["answers"]) == {"q"}
    assert result["answers"]["q"]["window"]["index"] == 0


@pytest.mark.parametrize(
    "agent",
    [
        None,
        object(),
        SimpleNamespace(tok=object()),
        SimpleNamespace(model=object()),
        SimpleNamespace(tok=object(), model=SimpleNamespace(encoder=None)),
    ],
)
def test_shortlist_names_missing_encoder_inputs(agent):
    with pytest.raises(TypeError, match="agent must provide tok and model.encoder"):
        embed_fn_from_agent(agent)


def test_shortlist_retains_coreml_encoder_output_boundary():
    with pytest.raises(NotImplementedError, match="external embed_fn"):
        embed_fn_from_agent(SimpleNamespace(tok=object(), model=object()))


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_json", "invalid_utf8", "deep"])
def test_native_cli_reports_question_path_before_loading(kind, tmp_path, monkeypatch, capsys):
    import sys

    from laya_coreml import cli

    bad = tmp_path / "bad.json"
    if kind == "directory":
        bad.mkdir()
    elif kind != "missing":
        bad.write_bytes({"invalid_json": b"{", "invalid_utf8": b"\xff", "deep": b"[" * 2000}[kind])
    loaded = []
    monkeypatch.setattr("laya_coreml.agent.load", lambda *a, **kw: loaded.append((a, kw)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["laya-coreml", "predict", "model", "--state", "hello", "--questions", str(bad)],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert loaded == []
    error = capsys.readouterr().err
    assert "--questions" in error and str(bad) in error


def test_native_cli_accepts_bom_before_prediction(tmp_path, monkeypatch, capsys):
    import sys

    from laya_coreml import cli

    questions = {"q": {"type": "noul", "instructions": "你好?"}}
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8-sig")
    seen = []

    def load(model_dir, **kwargs):
        seen.append((model_dir, kwargs))

        def predict(state, qs):
            seen.append((state, qs))
            return {"answers": {}}

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr("laya_coreml.agent.load", load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "laya-coreml",
            "predict",
            "model",
            "--state",
            "hello",
            "--questions",
            str(path),
            "--compute-units",
            "cpu",
            "--offline",
            "--revision",
            "pinned",
        ],
    )
    cli.main()
    assert seen == [
        ("model", {"compute_units": "cpu", "local_files_only": True, "revision": "pinned"}),
        ("hello", questions),
    ]
    assert json.loads(capsys.readouterr().out) == {"answers": {}}
