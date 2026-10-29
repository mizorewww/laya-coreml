"""Timed-out callbacks must retain their serialization lock until they finish."""

import contextvars
import threading

import pytest

from laya_coreml import Router
from laya_coreml.hooks import PredictContext, dispatch, normalise_hooks


@pytest.fixture
def hook_threads(monkeypatch):
    threads = []
    real_thread = threading.Thread

    def tracked_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(threading, "Thread", tracked_thread)
    yield threads
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive(), "hook worker did not stop after test cleanup"


@pytest.mark.parametrize("raise_errors", [True, False])
def test_router_serial_hooks_remain_serial_after_timeout(raise_errors, hook_threads):
    started = threading.Event()
    release = threading.Event()
    calls = []

    class Hook:
        def on_route(self, ctx):
            calls.append(ctx.states[0])
            if ctx.states[0] == "blocked":
                started.set()
                release.wait(10)

    router = Router(hooks=[Hook()], hooks_concurrent=False, hooks_timeout=1)
    router.hooks_raise = raise_errors

    def expect_timeout(state, **kwargs):
        expectation = (
            pytest.raises(TimeoutError, match="exceeded")
            if raise_errors
            else pytest.warns(RuntimeWarning, match="exceeded")
        )
        with expectation:
            router.route(state, model="english", **kwargs)

    try:
        expect_timeout("blocked")
        assert started.is_set()
        expect_timeout("expired", hooks_timeout=0.05)
        assert calls == ["blocked"]
    finally:
        release.set()
        for thread in hook_threads:
            thread.join(5)

    # A hook whose deadline expired while waiting must not execute after the lock frees.
    assert calls == ["blocked"]
    assert router.route("after", model="english").model == "english"
    assert calls == ["blocked", "after"]


def test_timeout_bounds_lock_wait_and_never_runs_expired_hook(hook_threads):
    lock = threading.Lock()
    called = threading.Event()
    returned = threading.Event()
    errors = []
    hooks = normalise_hooks(on_predict_start=lambda ctx: called.set())
    ctx = PredictContext(states=["s"], questions={})

    def caller():
        try:
            dispatch(hooks, "on_predict_start", ctx, lock=lock, timeout=0.05)
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    lock.acquire()
    thread = threading.Thread(target=caller)
    thread.start()
    try:
        assert returned.wait(5), "the hook deadline did not bound its lock wait"
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert not called.is_set()
    finally:
        lock.release()
        for worker in hook_threads:
            worker.join(5)
    assert not called.is_set()


@pytest.mark.parametrize("timeout", [None, 1])
def test_serial_hook_preserves_context_async_result_and_releases_on_error(timeout, hook_threads):
    lock = threading.RLock()
    marker = contextvars.ContextVar("hook_test_marker", default="missing")
    token = marker.set("request")
    ctx = PredictContext(states=["s"], questions={})

    async def rewrite(context):
        context.states = [marker.get()]

    def fail(context):
        raise ValueError("hook failed")

    try:
        dispatch(
            normalise_hooks(on_predict_start=rewrite),
            "on_predict_start",
            ctx,
            lock=lock,
            timeout=timeout,
        )
        assert ctx.states == ["request"]
        with pytest.raises(ValueError, match="hook failed"):
            dispatch(
                normalise_hooks(on_predict_start=fail),
                "on_predict_start",
                ctx,
                lock=lock,
                timeout=timeout,
            )
        # A later call can still acquire the lock after an exception.
        dispatch(
            normalise_hooks(on_predict_start=rewrite),
            "on_predict_start",
            ctx,
            lock=lock,
            timeout=timeout,
        )
    finally:
        marker.reset(token)


@pytest.mark.parametrize("timeout", [None, 1])
def test_serial_hook_can_reenter_without_a_nested_deadline(timeout, hook_threads):
    lock = threading.RLock()
    ctx = PredictContext(states=["s"], questions={})

    def inner(context):
        context.states = ["nested"]

    def outer(context):
        dispatch(normalise_hooks(on_predict_start=inner), "on_predict_start", context, lock=lock)

    dispatch(
        normalise_hooks(on_predict_start=outer), "on_predict_start", ctx, lock=lock, timeout=timeout
    )
    assert ctx.states == ["nested"]
