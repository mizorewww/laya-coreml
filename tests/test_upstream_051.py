"""v0.5.1 boundary regressions that run without native Core ML or model weights.

Upstream 0.5.1 hardens `laya.policy` manifest checks and makes `VegaAgent` hash `questions.json`;
neither module is shipped here. Its host change is the uvicorn log-level note: uvicorn before 0.45
does not lowercase the name, so the resolver must hand uvicorn a lowercase, stripped value.
"""

import pytest

import laya_coreml
from laya_coreml import serve


@pytest.mark.parametrize("raw", ["INFO", " Warning ", "debug"])
def test_log_level_reaches_uvicorn_lowercased_and_stripped(monkeypatch, raw):
    pytest.importorskip("uvicorn")
    monkeypatch.setenv("LAYA_LOG_LEVEL", raw)
    assert serve._resolve_log_level() == raw.strip().lower()


def test_policy_module_stays_unshipped():
    assert not hasattr(laya_coreml, "policy")
    with pytest.raises(ImportError):
        __import__("laya_coreml.policy")
