"""Parallel-layout parity, permutation sensitivity and actual Core ML execution."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from laya_coreml import Agent
from laya_coreml.common import build_sequence, parallel_layout
from laya_coreml.convert import convert
from laya_coreml.inputs import collate_items


class WordTokenizer:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

    def __init__(self):
        self.words = {}

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.words.setdefault(w, len(self.words) + 10) for w in text.split()]}


def batch(order=None, parallel=True, state_length=12, shape=None):
    tok = WordTokenizer()
    q = {
        "t": "choice",
        "ins": "Which department handles this request?",
        "crit": {
            "refund": "money back please",
            "tech": "a bug",
            "sales": "pricing",
            "other": "none of these fit",
        },
    }
    build_sequence(tok, "hello", q)
    ids, markers, *layout = build_sequence(
        tok, "hello " * state_length, q, max_len=64, option_order=order, return_layout=parallel
    )
    item = {"ids": ids, "markers": markers, "qtype": 0}
    if parallel:
        item["layout"] = layout[0]
    shape = shape or {
        "batch_size": 2,
        "max_length": 64,
        "min_length": 16,
        "max_options": 4,
        "flexible": True,
        "lengths": [16, 32, 64],
    }
    return collate_items([item], 0, shape=shape)


def run(model, inputs):
    with torch.inference_mode():
        return tuple(
            x.numpy() for x in model(**{k: torch.from_numpy(v) for k, v in inputs.items()})
        )


def test_layout_truncation_and_mixed_rows():
    layout = parallel_layout([4, 6, 9], 11, 14)
    assert layout["position_ids"] == [0, 1, 2, 3, 4, 5, 4, 5, 6, 4, 7, 8, 9, 10]
    assert layout["option_ids"] == [0] * 4 + [1, 1, 2, 2, 2, 3] + [0] * 4
    t = WordTokenizer()
    q = {"t": "choice", "ins": "pick", "crit": dict.fromkeys(map(str, range(20)))}
    full = build_sequence(t, "hello", q, return_layout=True)
    short = build_sequence(t, "hello", q, max_len=16, return_layout=True)
    assert short[2] == {k: v[:16] for k, v in full[2].items()}
    items = [{"ids": [1], "markers": [0], "qtype": 0}] * 2
    items[0] = dict(items[0], layout={"position_ids": [0], "option_ids": [0]})
    with pytest.raises(ValueError, match="mix"):
        collate_items(
            items, 0, shape={"batch_size": 2, "max_length": 16, "max_options": 2, "flexible": False}
        )


def test_all_permutations_and_trace(checkpoint):
    _, model = checkpoint
    model.parallel = True
    expected = run(model, batch())
    traced = torch.jit.trace(model, tuple(torch.from_numpy(v) for v in batch().values()))
    for order in itertools.permutations(range(4)):
        inputs = batch(list(order))
        actual = run(model, inputs)
        np.testing.assert_allclose(actual[0][0], expected[0][0, list(order)], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-5, rtol=2e-5)
        for x, y in zip(run(traced, inputs), actual):
            np.testing.assert_allclose(x, y, atol=2e-5, rtol=2e-5)
    for length in (0, 30):
        inputs = batch(state_length=length)
        for x, y in zip(run(traced, inputs), run(model, inputs)):
            np.testing.assert_allclose(x, y, atol=2e-5, rtol=2e-5)
    model.parallel = False
    sequential = run(model, batch(parallel=False))[0][0]
    reversed_logits = run(model, batch([3, 2, 1, 0], parallel=False))[0][0]
    assert np.max(abs(sequential[::-1] - reversed_logits)) > 1e-4


def test_upstream_v041_parity(checkpoint):
    transformers = pytest.importorskip("transformers")
    source, model = checkpoint
    path = Path(__file__).parents[1] / ".upstream/laya/common.py"
    if not path.exists():
        pytest.skip("Checkout upstream 1adc59f common.py for reference parity")
    spec = importlib.util.spec_from_file_location("upstream_v041_common", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    cfg = transformers.ModernBertConfig(
        **json.loads((source / "encoder/config.json").read_text()),
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
    )
    reference = upstream.DecisionModel(transformers.ModernBertModel(cfg), head_layers=2).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    # The unused dummy batch row has all markers masked; compare real questions only.
    for parallel in (False, True):
        model.parallel = parallel
        inputs = {k: v[:1] for k, v in batch(parallel=parallel).items()}
        ref_inputs = {
            k: torch.from_numpy(v).bool()
            if k in ("attention_mask", "marker_mask")
            else torch.from_numpy(v).long()
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            expected = reference(**ref_inputs)
        for x, y in zip(run(model, inputs), expected):
            np.testing.assert_allclose(x, y.numpy(), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("precision", ["float32", "float16"])
@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runtime requires macOS")
def test_parallel_coreml_conversion_and_runtime(checkpoint, precision):
    source, model = checkpoint
    config_path = source / "rl_agent_config.json"
    cfg = json.loads(config_path.read_text())
    cfg["option_layout"] = "parallel"
    config_path.write_text(json.dumps(cfg))
    model.parallel = True
    output = source / "parallel-export"
    convert(source, output, batch_size=2, max_options=4, precision=precision)
    for units in ("cpu", "cpu_gpu"):
        agent = Agent(output, compute_units=units)
        for length in (0, 12, 30):
            expected = run(model, batch(state_length=length, shape=agent.shape))
            for order in ([0, 1, 2, 3], [3, 1, 0, 2]):
                inputs = batch(order, state_length=length, shape=agent.shape)
                actual = agent.forward(inputs)
                np.testing.assert_allclose(
                    actual[0][0], expected[0][0, order], atol=0.015, rtol=0.015
                )
                np.testing.assert_allclose(actual[1], expected[1], atol=0.015, rtol=0.015)
        q = {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}}
        items, _ = agent.prepare("sample", q)
        assert "layout" in items[0]
        assert (
            agent.predict("sample", q, min_confidence=0.0)["answers"]["q"]["abstention"] == "passed"
        )
    cfg["option_layout"] = "sequential"
    (output / "rl_agent_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="re-export"):
        Agent(output)


def test_unknown_layout_rejected(checkpoint):
    from laya_coreml.torch_model import DecisionModel

    source, _ = checkpoint
    with pytest.raises(ValueError, match="option_layout"):
        DecisionModel(
            json.loads((source / "encoder/config.json").read_text()),
            {"head_layers": 2, "option_layout": "typo"},
            64,
        )


def test_parallel_requires_layout_and_legacy_ane_rejects_it(checkpoint):
    from experiments.ane_engineering.model import ConvBody

    _, model = checkpoint
    with pytest.raises(ValueError, match="Sequential"):
        run(model, batch())
    model.parallel = True
    with pytest.raises(ValueError, match="require"):
        run(model, batch(parallel=False))
    with pytest.raises(ValueError, match="sequential"):
        ConvBody(model, 64)
