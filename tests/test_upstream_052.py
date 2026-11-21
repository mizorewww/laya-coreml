"""v0.5.2 boundary regressions that run without native Core ML or model weights.

Upstream 0.5.2 re-exports vegaml's teacher for the continuous learner as `laya.Escalator` and a
self-hosted vLLM teacher as `laya.vllm_teacher`. Like `laya.Learner`, both belong to vegaml, which
this distribution does not ship, so neither is exported. No host code changed.
"""

import pytest

import laya_coreml


@pytest.mark.parametrize("name", ["Escalator", "vllm_teacher", "Learner"])
def test_vegaml_learner_names_are_not_exported(name):
    assert name not in laya_coreml.__all__
    with pytest.raises(AttributeError):
        getattr(laya_coreml, name)


def test_learn_module_stays_unshipped():
    with pytest.raises(ImportError):
        __import__("laya_coreml.learn")
