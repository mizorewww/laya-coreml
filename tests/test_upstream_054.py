"""v0.5.4 boundary regressions that run without native Core ML or model weights.

Upstream 0.5.4 imports `laya.Learner`, `laya.Escalator` and `laya.vllm_teacher` from the separately
licensed vegaml-learn package instead of vegaml. All three belong to vega's continuous learner,
which this distribution does not ship, so none is exported. No host code changed.
"""

import pytest

import laya_coreml


@pytest.mark.parametrize("name", ["Learner", "Escalator", "vllm_teacher", "learn"])
def test_continuous_learner_names_are_not_exported(name):
    assert name not in laya_coreml.__all__
    with pytest.raises(AttributeError):
        getattr(laya_coreml, name)


def test_learn_module_stays_unshipped():
    with pytest.raises(ImportError):
        __import__("laya_coreml.learn")


def test_vegaml_learn_is_not_a_dependency():
    from importlib.metadata import PackageNotFoundError, requires

    try:
        requirements = requires("laya-coreml") or []
    except PackageNotFoundError:
        pytest.skip("laya-coreml is not installed as a distribution")
    assert not any(r.lower().replace("_", "-").startswith("vegaml") for r in requirements)
