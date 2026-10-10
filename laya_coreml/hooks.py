"""Opt-in prediction hooks: observe or shape every decision without forking.

A hook on `hooks=` is an object implementing any subset of the lifecycle methods on `Hook`;
a plain callable is only accepted on the single-event `on_predict_start=` / `on_predict_end=`
parameters, because `hooks=` reads for method names and a bare function has none. Hooks are
configured on `Agent` / `Router` and can be overridden per call. Everything here is pure
Python: importing `laya` must not start pulling torch.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import math
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Union


@dataclass(eq=False)
class PredictContext:
    """Mutable state passed to every hook for one call.

    `states` / `questions` may be rewritten by `on_predict_start`; `results` may be rewritten
    by `on_predict_end`. A start hook can call `skip()` to short-circuit inference with a
    cached result.

    Identity semantics (`eq=False`): two contexts are never equal, and a context is hashable
    by identity, so a hook can keep one in a set without comparing agent/router internals.
    """

    states: List[Any]
    questions: Dict[str, Any]
    run_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )  # shared by every hook of one call
    results: Optional[List[Dict[str, Any]]] = None
    decision: Optional[Dict[str, Any]] = None  # Router: the RouteDecision dict
    model: Optional[str] = None  # resolved checkpoint name
    agent: Any = None
    router: Any = None
    max_len: Optional[int] = None  # per-call overrides; None = agent config
    head_max_len: Optional[int] = None
    usage: Optional[Dict[str, int]] = None  # aggregated input/output tokens
    started_at: float = field(default_factory=time.perf_counter)
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None

    def skip(self, results: List[Dict[str, Any]]) -> None:
        """Set cached results from a start hook; inference is skipped, end hooks still run.

        `results` replaces the whole call, so it carries one entry per state in `ctx.states` --
        the shape `predict_batch` returns -- in that order. A hook fires once per call, and a
        call can carry many states. The one other accepted shape is a single entry for the
        whole call, which is what `predict_long` takes as the document's answer (its scan
        hands the hook every window as a state, and refuses anything but one result).

        The count is checked here, against this contract, so a wrong one fails inside the
        hook under the caller's `hooks_raise` policy instead of downstream: `Router.predict`
        indexed `results[0]` of an empty list (an `IndexError`, which serve maps to 500),
        and `predict_batch` returned a shorter list than it was given states, quietly
        dropping rows the caller was about to zip against.
        """
        states = self.states
        if (
            isinstance(states, (list, tuple))
            and isinstance(results, (list, tuple))
            and len(results) not in (1, len(states))
        ):
            raise ValueError(
                "ctx.skip() takes one result for the whole call or one per state in "
                "ctx.states (%d); got %d" % (len(states), len(results))
            )
        self.results = results


class Hook(Protocol):
    """Optional lifecycle methods. Implement any subset; missing methods are skipped."""

    def on_predict_start(self, ctx: PredictContext) -> None: ...
    def on_predict_end(self, ctx: PredictContext) -> None: ...
    def on_route(self, ctx: PredictContext) -> None: ...
    def on_load(self, ctx: PredictContext) -> None: ...
    def on_evict(self, ctx: PredictContext) -> None: ...
    def on_error(self, ctx: PredictContext) -> None: ...


class BaseHook:
    """No-op base class: subclass it and override only the events you need.

    `Hook` is the structural protocol; `BaseHook` is the concrete convenience when you would
    rather subclass than implement methods by shape. Every method does nothing by default.
    """

    def on_predict_start(self, ctx: PredictContext) -> None:
        pass

    def on_predict_end(self, ctx: PredictContext) -> None:
        pass

    def on_route(self, ctx: PredictContext) -> None:
        pass

    def on_load(self, ctx: PredictContext) -> None:
        pass

    def on_evict(self, ctx: PredictContext) -> None:
        pass

    def on_error(self, ctx: PredictContext) -> None:
        pass


PredictHook = Callable[[PredictContext], None]
HookArg = Union[Hook, Sequence[Hook], None]
PredictHookArg = Union[PredictHook, Sequence[PredictHook], None]

HOOK_EVENTS = (
    "on_predict_start",
    "on_predict_end",
    "on_route",
    "on_load",
    "on_evict",
    "on_error",
)


class _StartAdapter:
    """Wrap a plain `on_predict_start` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_start must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_start(self, ctx: PredictContext) -> None:
        return self.fn(ctx)


class _EndAdapter:
    """Wrap a plain `on_predict_end` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_end must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_end(self, ctx: PredictContext) -> None:
        return self.fn(ctx)


def _as_sequence(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def normalise_hooks(
    hooks: HookArg = None,
    on_predict_start: PredictHookArg = None,
    on_predict_end: PredictHookArg = None,
) -> List[Any]:
    """Flatten `hooks` and the two convenience callables into one ordered hook list.

    Installed hooks are normalised once at construction; per-call arguments are appended
    after them, so per-call hooks always run last.
    """
    result: List[Any] = []
    for hook in _as_sequence(hooks):
        if isinstance(hook, type):
            raise TypeError(
                "hooks entries must be instances, not classes; got %s. Instantiate it first."
                % hook.__name__
            )
        if not any(hasattr(hook, event) for event in HOOK_EVENTS):
            raise TypeError(
                "hooks entries must implement at least one of %s; got %s. Pass a plain "
                "callable as on_predict_start=/on_predict_end= instead."
                % (", ".join(HOOK_EVENTS), type(hook).__name__)
            )
        for event in HOOK_EVENTS:
            method = getattr(hook, event, None)
            if method is not None and not callable(method):
                raise TypeError(
                    "hooks entry %s.%s must be callable, got %s"
                    % (type(hook).__name__, event, type(method).__name__)
                )
        result.append(hook)
    for fn in _as_sequence(on_predict_start):
        result.append(_StartAdapter(fn))
    for fn in _as_sequence(on_predict_end):
        result.append(_EndAdapter(fn))
    return result


_DEFAULT_HOOKS: List[Any] = []
_DEFAULT_HOOKS_LOCK = threading.Lock()
_SKIP_DEFAULTS = contextvars.ContextVar("laya_skip_default_hooks", default=False)


def default_hooks() -> List[Any]:
    """The process-wide hooks, a copy, in order. Empty unless set via `set_default_hooks`."""
    with _DEFAULT_HOOKS_LOCK:
        return list(_DEFAULT_HOOKS)


def set_default_hooks(hooks=None, on_predict_start=None, on_predict_end=None) -> None:
    """Replace the process-wide default hooks.

    Defaults run before installed and per-call hooks for every `Agent`, `Router` and
    `ONNXAgent` in the process, so a tracer or metrics hook does not have to be threaded
    through every construction. Accepts the same arguments as the `hooks=` parameter.
    """
    normalised = normalise_hooks(hooks, on_predict_start, on_predict_end)
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS[:] = normalised


def add_default_hook(hook: HookArg) -> None:
    """Append one hook or a sequence to the process-wide defaults."""
    normalised = normalise_hooks(hook)
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS.extend(normalised)


def clear_default_hooks() -> None:
    """Remove every process-wide default hook."""
    with _DEFAULT_HOOKS_LOCK:
        _DEFAULT_HOOKS.clear()


def compose_hooks(installed, hooks=None, on_predict_start=None, on_predict_end=None) -> List[Any]:
    """Effective hook list for one call: defaults, then installed, then per-call hooks.

    Reads the process-wide defaults at call time, so hooks set after construction still apply.
    """
    defaults = [] if _SKIP_DEFAULTS.get() else default_hooks()
    return defaults + list(installed) + normalise_hooks(hooks, on_predict_start, on_predict_end)


def validate_timeout(value: Optional[float]) -> Optional[float]:
    """Return *value* as a positive float, or ``None`` for no limit.

    A non-positive timeout is rejected here rather than left to
    ``thread.join``: ``join(0)`` and ``join(-1)`` return before the hook has
    started, so the outcome of a fast hook with such a value is a race.
    Non-finite values are rejected too: ``join(nan)`` raises ``ValueError`` and
    ``join(inf)`` raises ``OverflowError`` instead of waiting.
    """
    if value is None:
        return None
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "hooks_timeout must be a positive finite number or None; got %r" % (value,)
        )
    return timeout


_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_LOOP_LOCK = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop thread, for running coroutines when the caller already has a loop."""
    global _BACKGROUND_LOOP
    with _BACKGROUND_LOOP_LOCK:
        if _BACKGROUND_LOOP is None or not _BACKGROUND_LOOP.is_running():
            _BACKGROUND_LOOP = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_BACKGROUND_LOOP.run_forever, daemon=True, name="laya-async-hooks"
            )
            thread.start()
        return _BACKGROUND_LOOP


def run_coroutine_sync(
    coro: Awaitable[Any], loop: Optional[asyncio.AbstractEventLoop] = None
) -> Any:
    """Run an awaitable to completion from synchronous code.

    Uses `asyncio.run` when the calling thread has no running loop. When it does (a caller
    inside an async function, or a framework that already runs a loop), the coroutine is run on
    a dedicated background loop so the calling thread can block on it without deadlocking. Pass
    `loop` to use a specific loop instead of the background one.

    A supplied `loop` must be running somewhere, and must not be the calling thread's own
    loop. Both are checked: the first would otherwise block forever with no coroutine ever
    scheduled, and the second would block the only thread that could run the coroutine.
    """
    if loop is not None:
        if not loop.is_running():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise ValueError("run_coroutine_sync: the loop passed is not running")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is running:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise ValueError(
                "run_coroutine_sync: the loop passed is running in the calling thread; "
                "blocking on it would deadlock"
            )
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, _background_loop()).result()


class AsyncHook:
    """Wrap an async hook so its coroutine event methods run to completion in the sync core.

    The wrapped object may implement any subset of the events as `async def` methods, or as
    plain callables that return awaitables. Each event is run with `run_coroutine_sync`, so it
    works whether the caller is synchronous or already inside an event loop.

        from laya_coreml import AsyncHook

        class Remote(BaseHook):
            async def on_predict_end(self, ctx):
                await ship(ctx.results)

        agent = laya.load("convaiinnovations/laya", hooks=[AsyncHook(Remote())])

    Pass `loop` to funnel every coroutine onto a specific loop; otherwise a background loop is
    started on demand when the caller already has one.
    """

    __slots__ = ("hook", "loop")

    def __init__(self, hook: Any, loop: Optional[asyncio.AbstractEventLoop] = None):
        if isinstance(hook, type):
            raise TypeError("AsyncHook wraps an instance, not a class; got %s" % hook.__name__)
        if not any(hasattr(hook, event) for event in HOOK_EVENTS):
            raise TypeError(
                "AsyncHook wraps an object implementing at least one of %s; got %s"
                % (", ".join(HOOK_EVENTS), type(hook).__name__)
            )
        self.hook = hook
        self.loop = loop

    def _run(self, event: str, ctx: PredictContext) -> None:
        method = getattr(self.hook, event, None)
        if method is None:
            return
        result = method(ctx)
        if inspect.isawaitable(result):
            run_coroutine_sync(result, loop=self.loop)

    def on_predict_start(self, ctx: PredictContext) -> None:
        self._run("on_predict_start", ctx)

    def on_predict_end(self, ctx: PredictContext) -> None:
        self._run("on_predict_end", ctx)

    def on_route(self, ctx: PredictContext) -> None:
        self._run("on_route", ctx)

    def on_load(self, ctx: PredictContext) -> None:
        self._run("on_load", ctx)

    def on_evict(self, ctx: PredictContext) -> None:
        self._run("on_evict", ctx)

    def on_error(self, ctx: PredictContext) -> None:
        self._run("on_error", ctx)

    def __repr__(self) -> str:
        return "AsyncHook(%r)" % (self.hook,)


class HookRegistry:
    """Mixin giving a runtime-mutable hook list.

    `Agent`, `Router` and `ONNXAgent` use it so hooks can be added, removed or scoped after
    construction. Mutation is guarded by `_hooks_mutex`; a call reads a snapshot of the list,
    so adding or removing a hook never disturbs a call in flight.
    """

    hooks = ()
    _hooks_mutex = None

    def add_hook(self, hook: HookArg) -> "HookRegistry":
        """Install one hook or a sequence of them. Returns self for chaining."""
        self._extend_hooks(normalise_hooks(hook))
        return self

    def remove_hook(self, hook: Any) -> bool:
        """Remove a hook by identity. Returns True if it was installed."""
        return self._remove_hooks(lambda installed: installed is hook) > 0

    @contextmanager
    def hooks_installed(self, *hooks: HookArg):
        """Install hooks for the duration of the `with` block, then remove them.

            with agent.hooks_installed(tracer):
                agent.system_one(state, questions)

        Each argument is one hook or a sequence of them, matching `add_hook` and the
        `hooks=` parameter, so `hooks_installed([tracer, meter])` works too.
        """
        added = normalise_hooks([hook for arg in hooks for hook in _as_sequence(arg)])
        self._extend_hooks(added)
        try:
            yield self
        finally:
            self._remove_instances(added)

    def _remove_instances(self, hooks: Sequence[Any]) -> int:
        """Remove one occurrence of each of `hooks`, the most recent match by identity.

        Removing by identity alone took every copy, so a hook the application had already
        installed was removed along with the block's own and stayed gone -- `docs/hooks/api.md`
        promises the block "restores the previous list on exit". Restoring a snapshot instead is
        wrong in two other ways: with two overlapping blocks the first exit reinstates its
        snapshot and so removes the second block's hook, and a hook added with `add_hook` inside
        the block is discarded because it is not in the snapshot either. Taking one occurrence
        per hook the block added leaves everything else -- including anything added inside the
        block -- in place.

        The most recent match is the one to drop: `_extend_hooks` appends, so a hook that was
        already installed sits earlier in the list than the block's copy.
        """
        if not hooks:
            return 0
        removed = 0
        with self._hooks_mutex_for_registry():
            current = list(self.hooks)
            for hook in hooks:
                for i in range(len(current) - 1, -1, -1):
                    if current[i] is hook:
                        del current[i]
                        removed += 1
                        break
            self.hooks = current
        return removed

    def _extend_hooks(self, hooks: Sequence[Any]) -> None:
        if not hooks:
            return
        with self._hooks_mutex_for_registry():
            self.hooks = list(self.hooks) + list(hooks)

    def _remove_hooks(self, predicate: Callable[[Any], bool]) -> int:
        with self._hooks_mutex_for_registry():
            current = list(self.hooks)
            remaining = [hook for hook in current if not predicate(hook)]
            self.hooks = remaining
            return len(current) - len(remaining)

    def _hooks_mutex_for_registry(self) -> threading.Lock:
        lock = self._hooks_mutex
        if lock is None:
            # Only reachable for an instance built without __init__ (tests); real instances
            # get their mutex in the constructor.
            lock = self._hooks_mutex = threading.Lock()
        return lock


def aggregate_usage(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the per-state usage blocks so a hook sees one total for the call."""

    def total(key: str) -> int:
        return sum(int((r.get("usage") or {}).get(key, 0) or 0) for r in results)

    return {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens")}


def _hook_name(method: Any) -> str:
    return (
        getattr(method, "__qualname__", None) or getattr(method, "__name__", None) or repr(method)
    )


def _call_hook(method: Any, ctx: PredictContext, timeout: Optional[float]) -> None:
    """Call one hook method, running its result if it is awaitable, under an optional timeout."""
    timeout = validate_timeout(timeout)

    def invoke():
        result = method(ctx)
        if inspect.isawaitable(result):
            run_coroutine_sync(result)

    if timeout is None:
        invoke()
        return

    box: List[BaseException] = []

    # Run the hook in a copy of the caller's context, so a `contextvars` value (a request id,
    # a tracing span) set by the caller is visible to the hook even though it runs on another
    # thread. The no-timeout path runs inline and inherits the context already.
    context = contextvars.copy_context()

    def runner():
        try:
            context.run(invoke)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller thread
            box.append(exc)

    thread = threading.Thread(target=runner, daemon=True, name="laya-hook-timeout")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # The thread keeps running: Python cannot interrupt it. The timeout bounds the request.
        raise TimeoutError("laya: hook %s exceeded %gs" % (_hook_name(method), timeout))
    if box:
        raise box[0]


def dispatch(
    hooks: Sequence[Any],
    event: str,
    ctx: PredictContext,
    *,
    raise_errors: bool = True,
    lock: Any = None,
    timeout: Optional[float] = None,
) -> None:
    """Call `event` on every hook that implements it.

    `raise_errors=False` warns and continues, for hooks (telemetry) that must not fail a
    request. `lock` serialises dispatch for hooks that are not safe to run concurrently.
    `timeout` bounds each hook call in seconds; an overrunning hook raises `TimeoutError`, or
    warns when `raise_errors` is False. Python threads cannot be interrupted, so an overrunning
    hook keeps running in the background: the timeout protects the request, not the process.
    A hook that returns an awaitable is run to completion before moving on.
    """
    for hook in hooks:
        method = getattr(hook, event, None)
        if method is None:
            continue
        try:
            if lock is not None:
                with lock:
                    _call_hook(method, ctx, timeout)
            else:
                _call_hook(method, ctx, timeout)
        except Exception as exc:  # noqa: BLE001 -- policy depends on raise_errors
            if raise_errors:
                raise
            warnings.warn(
                "laya: hook %s.%s failed: %s" % (type(hook).__name__, event, exc),
                RuntimeWarning,
                stacklevel=2,
            )
