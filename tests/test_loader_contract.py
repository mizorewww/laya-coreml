"""Backend boundary: aliases, artifact revisions, digests and export capacities."""

import json
from pathlib import Path

import pytest

from laya_coreml.agent import Agent, load
from laya_coreml.hub import BUNDLE_REVISIONS, resolve_checkpoint


def test_official_reviewed_pin_maps_to_reviewed_conversion(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path),
    )
    monkeypatch.setenv("LAYA_REVISION", "reviewed")
    assert resolve_checkpoint("convaiinnovations/laya", subfolder="multilingual") == tmp_path
    assert calls[0][0] == "aac6fef/laya-multilingual-coreml"
    assert calls[0][1]["revision"] == BUNDLE_REVISIONS[calls[0][0]]
    resolve_checkpoint("aac6fef/laya-coreml")
    assert calls[1][1]["revision"] == BUNDLE_REVISIONS["aac6fef/laya-coreml"]
    with pytest.raises(ValueError, match="No reviewed Core ML conversion"):
        resolve_checkpoint("convaiinnovations/laya", revision="unreviewed-branch")


def test_explicit_coreml_revision_and_subfolder_are_preserved(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path),
    )
    folder = resolve_checkpoint(
        "custom/coreml", revision="my-sha", subfolder="nested", token="test", local_files_only=True
    )
    assert folder == tmp_path / "nested"
    assert calls[0][1]["revision"] == "my-sha"
    assert calls[0][1]["token"] == "test"
    assert calls[0][1]["local_files_only"] is True
    assert all(pattern.startswith("nested/") for pattern in calls[0][1]["allow_patterns"])


def test_local_alias_name_and_digest_opt_out(monkeypatch, tmp_path):
    import laya_coreml.agent as module

    folder = tmp_path / "english"
    folder.mkdir()
    (folder / "rl_agent_config.json").write_text("{}")
    (folder / "coreml_config.json").write_text(json.dumps({"format": "laya-coreml"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAYA_SHA256_DIGESTS", '{"english":{"missing":"bad"}}')
    calls = []

    class Dummy:
        def __init__(self, directory, **kwargs):
            calls.append((directory, kwargs))

    monkeypatch.setattr(module, "Agent", Dummy)
    agent = load("english", expected_sha256={})
    assert calls[0][0] == Path("english")
    assert calls[0][1]["expected_sha256"] == {}
    assert agent.model_id == "english"
    assert agent.revision is None


@pytest.mark.parametrize(
    "kwargs", [{"fast": True}, {"compile": True}, {"backend": "onnx"}, {"compile_cache": True}]
)
def test_unsupported_backends_fail_before_download(kwargs):
    with pytest.raises(ValueError, match="Core ML"):
        load("does-not-exist", **kwargs)


def test_short_export_uses_true_context_for_official_scanner():
    agent = Agent.__new__(Agent)
    agent.cfg = {"max_len": 1024, "head_max_len": 64}
    agent.shape = {"max_length": 96}
    agent.temperature_raw = [1.0, 1.0, 1.0]
    agent._init_host()
    assert agent.cfg["max_len"] == 96


def test_http_mcp_device_contract_is_coreml_specific(monkeypatch):
    from fastapi.testclient import TestClient

    from laya_coreml.mcp.device import device_report, env_device
    from laya_coreml.mcp.tools import laya_status
    from laya_coreml.serve import _apply_thread_limit, create_app

    monkeypatch.setenv("LAYA_DEVICE", " CPU\n")
    assert env_device() == "cpu"
    assert device_report() == {"device": "cpu", "torch_cuda": False, "torch_version": None}

    class FakeRouter:
        loaded = ["english"]
        loaded_revisions = {"english": "artifact-sha"}
        _agents = {"english": type("A", (), {"device": "coreml:cpu_ne"})()}

    router = FakeRouter()
    body = TestClient(create_app(router=router)).get("/health").json()
    status = laya_status(router=router, loaded=router.loaded)
    assert body["device"] == status["device"] == "coreml:cpu_ne"
    from laya_coreml import __version__

    assert status["package_versions"]["laya"] == __version__
    assert body["checkpoint_devices"] == status["checkpoint_devices"]
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    with pytest.raises(ValueError, match="Core ML"):
        env_device()
    monkeypatch.setenv("LAYA_THREADS", "8")
    with pytest.raises(ValueError, match="Core ML thread pools"):
        _apply_thread_limit()


def test_router_cli_explicitly_refuses_training(capsys):
    from laya_coreml.router_cli import main

    assert main(["train", "--help"]) == 2
    assert "Training requires official laya" in capsys.readouterr().err
