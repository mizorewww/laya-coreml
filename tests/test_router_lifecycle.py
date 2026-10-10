"""Registry and lifecycle races use events, not timing-dependent sleeps."""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from laya_coreml import Router


@pytest.fixture
def blocked_builder(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def build(source, **kwargs):
        calls.append(source)
        entered.set()
        assert release.wait(5), "test did not release builder"
        return {"source": source}

    monkeypatch.setattr("laya_coreml.agent.load", build)
    return entered, release, calls


def test_hot_load_status_and_unrelated_unload_during_cold_build(blocked_builder):
    entered, release, _ = blocked_builder
    r = Router(max_loaded=3)
    hot = r.attach("english", object())
    with ThreadPoolExecutor(3) as pool:
        cold = pool.submit(r.load, "multilingual")
        try:
            assert entered.wait(2)
            assert pool.submit(r.load, "english").result(2) is hot
            assert pool.submit(lambda: r.loaded).result(2) == ["english"]
            pool.submit(r.unload, "english").result(2)
        finally:
            release.set()
        assert cold.result(2)["source"]
    assert r.loaded == ["multilingual"]


def test_preload_does_not_hold_global_lock(blocked_builder):
    entered, release, _ = blocked_builder
    r = Router()
    with ThreadPoolExecutor(2) as pool:
        cold = pool.submit(r.preload, ["english"])
        try:
            assert entered.wait(2)
            assert pool.submit(lambda: r.loaded).result(2) == []
        finally:
            release.set()
        assert cold.result(2) is r


def test_attach_during_build_wins(blocked_builder):
    entered, release, calls = blocked_builder
    r = Router()
    with ThreadPoolExecutor(2) as pool:
        cold = pool.submit(r.load, "english")
        try:
            assert entered.wait(2)
            attached = object()
            assert pool.submit(r.attach, "english", attached).result(2) is attached
        finally:
            release.set()
        assert cold.result(2) is attached
    assert len(calls) == 1


@pytest.mark.parametrize("all_models", [False, True])
def test_unload_waits_for_own_build_then_removes_it(blocked_builder, all_models):
    entered, release, _ = blocked_builder
    r = Router()
    unloading = threading.Event()

    def unload():
        unloading.set()
        r.unload(None if all_models else "english")

    with ThreadPoolExecutor(2) as pool:
        cold = pool.submit(r.load, "english")
        try:
            assert entered.wait(2)
            drop = pool.submit(unload)
            assert unloading.wait(2)
            assert not drop.done()
        finally:
            release.set()
        cold.result(2)
        drop.result(2)
    assert not r.loaded


def test_replace_source_during_build_never_installs_old_model(blocked_builder):
    entered, release, calls = blocked_builder
    r = Router(models={"custom": "old/repo"})
    with ThreadPoolExecutor(2) as pool:
        cold = pool.submit(r.load, "custom")
        try:
            assert entered.wait(2)
            pool.submit(r.register, "custom", "new/repo").result(2)
        finally:
            release.set()
        assert cold.result(2) == {"source": "new/repo"}
    assert calls == ["old/repo", "new/repo"]
    assert r.load("custom") == {"source": "new/repo"}


def test_unregister_during_build_prevents_resurrection(blocked_builder):
    entered, release, _ = blocked_builder
    r = Router(models={"custom": "old/repo"})
    with ThreadPoolExecutor(2) as pool:
        cold = pool.submit(r.load, "custom")
        try:
            assert entered.wait(2)
            pool.submit(r.unregister, "custom").result(2)
        finally:
            release.set()
        with pytest.raises(ValueError, match="unknown model"):
            cold.result(2)
    assert not r.loaded and not r.registered and not r._loading


def test_failed_build_unblocks_waiters_and_can_retry(monkeypatch):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    r = Router()

    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("download failed")

    monkeypatch.setattr("laya_coreml.agent.load", fail)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(r.load, "english")
        try:
            assert entered.wait(2)
            event = r._loading["english"].done
            original_wait = event.wait

            def observed_wait(*args):
                waiting.set()
                return original_wait(*args)

            monkeypatch.setattr(event, "wait", observed_wait)
            second = pool.submit(r.load, "english")
            assert waiting.wait(2)
        finally:
            release.set()
        for future in (first, second):
            with pytest.raises(RuntimeError, match="download failed"):
                future.result(2)
    assert not r._loading and not r.loaded
    monkeypatch.setattr("laya_coreml.agent.load", lambda *a, **kw: object())
    assert r.load("english") is r.load("english")


def test_registry_aliases_lru_and_source_validation(monkeypatch):
    monkeypatch.setattr("laya_coreml.agent.load", lambda source, **kw: {"source": source})
    r = Router(models={" Papers ": "example/papers"}, default="papers")
    assert r.route("123").model == "papers"
    assert r.route("hello", model="PAPERS")["repo"] == "example/papers"
    assert r.register("papers", "example/new", "Classifier") == "papers"
    assert r.registered == {"papers": {"source": "example/new", "description": "Classifier"}}
    r.load("papers")
    r.load("en")
    assert r.loaded == ["english"]
    for name in ("auto", "bad/name", "", 7):
        with pytest.raises(ValueError):
            r.register(name, "example/model")
    with pytest.raises(ValueError):
        r.register("papers", [])
    assert r.registered["papers"]["source"] == "example/new"
    for name in ("papers", "en"):
        with pytest.raises(ValueError):
            r.unregister(name)
    r.default = "multilingual"
    r.unregister("papers")
    with pytest.raises(ValueError, match="unknown model"):
        r.load("papers")


def test_attached_custom_agent_and_unload_preserves_external_reference(monkeypatch):
    cleared = []
    monkeypatch.setattr(gc, "collect", lambda: cleared.append(True))
    r = Router()
    agent = object()
    assert r.attach("mine", agent) is agent
    assert r.registered["mine"]["source"] is None
    r.unload("mine")
    assert cleared == [True]
    assert agent is not None
    with pytest.raises(ValueError, match="no source"):
        r.load("mine")


def test_eviction_clears_cache(monkeypatch):
    cleared = []
    monkeypatch.setattr(gc, "collect", lambda: cleared.append(True))
    monkeypatch.setattr("laya_coreml.agent.load", lambda *a, **kw: object())
    r = Router()
    r.load("english")
    assert not cleared
    r.load("multilingual")
    assert cleared == [True]


def test_unload_releases_agents_but_preserves_caller_references(monkeypatch):
    import weakref

    class HeldAgent:
        pass

    monkeypatch.setattr("laya_coreml.agent.load", lambda *a, **kw: HeldAgent())
    router = Router()
    held = router.load("english")
    reference = weakref.ref(held)
    router.unload("english")
    assert reference() is held
    del held
    gc.collect()
    assert reference() is None
