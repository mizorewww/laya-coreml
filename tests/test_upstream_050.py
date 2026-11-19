"""v0.5.0 boundary regressions that run without native Core ML or model weights.

Upstream 0.5.0 makes vegaml's engine the default for `laya.Agent()` / `laya.load()` and re-exports
vegaml's continuous learner as `laya.Learner`. This distribution ships no vegaml engine, so the
default stays the classic checkpoint (upstream's own fallback without vegaml) and `Learner` is not
exported.
"""

import sys

import pytest

import laya_coreml
from laya_coreml import agent as agent_module
from laya_coreml.agent import Agent, load, wants_vega


def test_wants_vega_matches_upstream_for_explicit_choices():
    # Upstream's own expectations from tests/test_vega.py that do not depend on vegaml being installed.
    assert wants_vega((), {"vegaml": False}) is False
    assert wants_vega((), {"vegaml": "4b"}) is True
    assert wants_vega((), {"vegaml": True}) is True
    assert wants_vega(("convaiinnovations/laya",), {}) is False
    assert wants_vega((), {"model_id_or_path": "multilingual"}) is False
    assert wants_vega((), {"backend": "onnx"}) is False
    assert wants_vega((), {"subfolder": "multilingual"}) is False


def test_no_checkpoint_named_stays_classic_without_vegaml():
    # Upstream falls back to the classic encoder when vegaml is missing; here it always is.
    assert wants_vega() is False
    assert wants_vega((), {}) is False
    assert wants_vega((), {"device": "cpu", "hooks": []}) is False


def test_bare_agent_idiom_stays_a_bare_agent():
    assert type(Agent.__new__(Agent)) is Agent


def test_load_with_no_checkpoint_named_resolves_the_default(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def resolve(model_id_or_path, **kwargs):
        seen.append((model_id_or_path, kwargs["subfolder"]))
        raise Stop

    monkeypatch.setattr(agent_module, "resolve_checkpoint", resolve)
    for call in (lambda: load(), lambda: load(None), lambda: load(vegaml=False)):
        with pytest.raises(Stop):
            call()
    assert seen == [("convaiinnovations/laya", None)] * 3


@pytest.mark.parametrize("value", [True, "4b"])
def test_load_still_refuses_an_explicit_vegaml_ask(value, monkeypatch):
    monkeypatch.setitem(sys.modules, "coremltools", None)
    with pytest.raises(ValueError, match="requires official laya"):
        load(vegaml=value)


def test_learner_is_not_exported():
    assert "Learner" not in laya_coreml.__all__
    with pytest.raises(AttributeError):
        laya_coreml.Learner  # noqa: B018
