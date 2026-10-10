"""LlamaIndex integration for Laya System 1 decision engine.

Provides sub-35ms, non-autoregressive query routing, single/multi-tool selection,
and real-time guardrails for LlamaIndex RAG pipelines and RouterQueryEngine.

Replaces slow, token-generating LLM selectors (`LLMSingleSelector`, `LLMMultiSelector`)
with calibrated, typed decisions executed in a single forward pass without token cost.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `laya-serve`) without requiring PyTorch on edge clients.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

# Optional LlamaIndex base classes
try:
    from llama_index.core.schema import QueryBundle
    from llama_index.core.selectors import BaseSelector, SelectorResult, ToolSelection
    from llama_index.core.tools import ToolMetadata

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:
    _LLAMA_INDEX_AVAILABLE = False

    class BaseSelector:  # type: ignore
        """Lightweight shim when llama-index-core is not installed."""

        def select(self, choices: Sequence[Any], query: Any) -> Any:
            raise NotImplementedError

        async def aselect(self, choices: Sequence[Any], query: Any) -> Any:
            raise NotImplementedError

    @dataclass
    class ToolMetadata:  # type: ignore
        description: str
        name: Optional[str] = None

    @dataclass
    class ToolSelection:  # type: ignore
        index: int
        reason: Optional[str] = None

    @dataclass
    class SelectorResult:  # type: ignore
        selections: List[ToolSelection]

    @dataclass
    class QueryBundle:  # type: ignore
        query_str: str

        def __str__(self) -> str:
            return self.query_str


# The per-call control rules are shared with the other two integrations: they end at the same
# `predict` call and the same laya-serve request body, and three copies of the rule is how two of
# them came to forward only `model`.
from ..confidence import _gate_confidence  # noqa: E402
from ._controls import budget_kwargs as _budget_kwargs  # noqa: E402
from ._controls import decision_kwargs as _decision_kwargs  # noqa: E402
from ._controls import hook_kwargs as _hook_kwargs
from ._controls import predict_kwargs as _predict_kwargs  # noqa: E402
from ._controls import reject_remote_hooks as _reject_remote_hooks

# One class for every integration, so `except LayaLowConfidenceError` catches all of them.
from ._errors import LayaLowConfidenceError  # noqa: E402


def _gated_confidence(answer: Dict[str, Any]) -> float:
    """The threshold number for one answer, defaulting to 1.0 when it reports none.

    Every selector and router in this module gates `confidence_threshold` through this, and it
    delegates to core's `_gate_confidence` so the wrappers and `flag_low_confidence` read the
    same quantity: `answer_confidence` first, falling back to the entropy `confidence` so an
    answer that carries only the older field is still gated rather than silently passed. An
    answer with no usable number at all is treated as fully confident, exactly as before.
    """
    conf = _gate_confidence(answer or {})
    return conf if conf is not None else 1.0


def _extract_query_str(query: Union[str, QueryBundle, Any]) -> str:
    """Extract raw query string from query input."""
    if isinstance(query, str):
        return query
    if hasattr(query, "query_str"):
        return str(query.query_str)
    return str(query)


def _format_choices_criteria(
    choices: Sequence[Union[ToolMetadata, Dict[str, Any], Any]],
) -> Dict[str, str]:
    """Format candidate tool choices into a Laya choice criteria dictionary."""
    criteria: Dict[str, str] = {}
    for i, choice in enumerate(choices):
        key = f"choice_{i}"
        name = getattr(choice, "name", None)
        desc = getattr(choice, "description", None)

        if isinstance(choice, dict):
            name = choice.get("name", name)
            desc = choice.get("description", desc)
        elif isinstance(choice, str):
            desc = choice

        if not desc:
            desc = name or f"Option {i}"

        if name:
            criteria[key] = f"{name}: {desc}"
        else:
            criteria[key] = str(desc)

    return criteria


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward bearer credentials across an origin or HTTPS downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        old_port = old.port or (443 if old.scheme.lower() == "https" else 80)
        new_port = new.port or (443 if new.scheme.lower() == "https" else 80)
        if (
            old.scheme.lower() != new.scheme.lower()
            or (old.hostname or "").lower() != (new.hostname or "").lower()
            or old_port != new_port
        ):
            raise urllib.error.URLError("refusing cross-origin or HTTPS-downgrade redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _call_remote(
    base_url: str,
    state: Any,
    questions: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Send decision request to a remote laya-serve HTTP instance using standard library urllib.

    `max_len` / `head_max_len` travel in the body; laya-serve applies them up to its
    `LAYA_MAX_TOKEN_BUDGET` ceiling and answers a larger value with 422. `lang` / `min_confidence`
    ride in the same body (they are laya-serve `BODY_CONTROLS` too): the language codes the state
    is routed and answered in, and core's abstention gate.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/v1/systemone"):
        url = f"{url}/v1/systemone"

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model
    if max_len is not None:
        payload["max_len"] = max_len
    if head_max_len is not None:
        payload["head_max_len"] = head_max_len
    if lang is not None:
        payload["lang"] = lang
    if min_confidence is not None:
        payload["min_confidence"] = min_confidence

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(_SameOriginRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Laya server error {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Laya server at {url}: {e}") from e


_DEFAULT_ROUTER = None
_DEFAULT_ROUTER_LOCK = threading.Lock()


def _get_default_router():
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        with _DEFAULT_ROUTER_LOCK:
            if _DEFAULT_ROUTER is None:
                from ..router import Router

                _DEFAULT_ROUTER = Router()
    return _DEFAULT_ROUTER


def _execute_decision(
    state: Any,
    questions: Dict[str, Any],
    agent: Optional[Any] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
    hooks: Optional[Any] = None,
    on_predict_start: Optional[Any] = None,
    on_predict_end: Optional[Any] = None,
    hooks_raise: Optional[bool] = None,
    hooks_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """One decision on the local runner or a laya-serve node, with the per-call controls kept.

    Which controls either path accepts is `._controls`' rule, shared with the LangChain and
    CrewAI wrappers -- they end at the same `predict` call and the same request body.
    """
    hook_kwargs = _hook_kwargs(hooks, on_predict_start, on_predict_end, hooks_raise, hooks_timeout)
    if base_url:
        _reject_remote_hooks(hook_kwargs, base_url)
        budget = _budget_kwargs(max_len, head_max_len)
        decision = _decision_kwargs(lang, min_confidence)
        return _call_remote(
            base_url, state, questions, api_key=api_key, model=model, **budget, **decision
        )
    runner = agent if agent is not None else _get_default_router()
    kwargs = _predict_kwargs(model, max_len, head_max_len, lang, min_confidence)
    kwargs.update(hook_kwargs)
    return runner.predict(state, questions, **kwargs)


class LayaSingleSelector(BaseSelector):
    """Sub-35ms single-choice selector for LlamaIndex RouterQueryEngine.

    Replaces LLMSingleSelector to route queries to candidate query engines or tools
    in a single non-autoregressive forward pass with zero token generation cost.
    Supports confidence threshold gating and automatic fallback selection.
    """

    def __init__(
        self,
        instructions: str = "Which tool or query engine is best suited to answer this query?",
        confidence_threshold: float = 0.0,
        fallback_index: Optional[int] = None,
        raise_on_low_confidence: bool = False,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ):
        self.instructions = instructions
        self.confidence_threshold = confidence_threshold
        self.fallback_index = fallback_index
        self.raise_on_low_confidence = raise_on_low_confidence
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.lang = lang
        self.min_confidence = min_confidence
        self.hooks = hooks
        self.on_predict_start = on_predict_start
        self.on_predict_end = on_predict_end
        self.hooks_raise = hooks_raise
        self.hooks_timeout = hooks_timeout
        self.last_decision: Optional[Dict[str, Any]] = None

    def _select(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        """Select a single choice synchronously."""
        if not choices:
            raise ValueError("No choices provided to select from.")

        query_str = _extract_query_str(query)
        criteria = _format_choices_criteria(choices)

        questions = {
            "selector": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            query_str,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        self.last_decision = res

        ans = res.get("answers", {}).get("selector", {})
        chosen_key = ans.get("choice")
        conf = _gated_confidence(ans)

        # Map "choice_i" back to index i
        chosen_idx: int = 0
        if chosen_key and chosen_key.startswith("choice_"):
            try:
                chosen_idx = int(chosen_key.split("_")[1])
            except (ValueError, IndexError):
                chosen_idx = 0
        elif chosen_key in criteria:
            chosen_idx = list(criteria.keys()).index(chosen_key)

        # Confidence gating
        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            if self.fallback_index is not None:
                reason = (
                    f"Selected fallback index {self.fallback_index} because confidence "
                    f"({conf:.3f}) was below threshold ({self.confidence_threshold:.3f})."
                )
                return SelectorResult(
                    selections=[ToolSelection(index=self.fallback_index, reason=reason)]
                )
            if self.raise_on_low_confidence:
                raise LayaLowConfidenceError(
                    f"Confidence {conf:.3f} below threshold {self.confidence_threshold:.3f} "
                    f"for query: {query_str!r}",
                    confidence=conf,
                    threshold=self.confidence_threshold,
                    raw_decision=res,
                )

        choice_obj = choices[chosen_idx]
        choice_name = getattr(choice_obj, "name", None) or f"choice_{chosen_idx}"
        reason = f"Selected '{choice_name}' via Laya System 1 decision (confidence: {conf:.3f})."
        return SelectorResult(selections=[ToolSelection(index=chosen_idx, reason=reason)])

    def select(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        """Public synchronous selection method."""
        return self._select(choices, query)

    async def _aselect(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        """Select a single choice asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self._select, choices, query)

    async def aselect(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        """Public asynchronous selection method."""
        return await self._aselect(choices, query)


class LayaMultiSelector(BaseSelector):
    """Sub-35ms multi-choice selector for LlamaIndex RouterQueryEngine.

    Replaces LLMMultiSelector to evaluate and select multiple query engines or tools
    for composite questions requiring information from multiple sources.
    """

    def __init__(
        self,
        instructions: str = "Which tools or query engines are relevant to answer this query?",
        max_outputs: Optional[int] = None,
        probability_threshold: float = 0.25,
        default_index: int = 0,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ):
        self.instructions = instructions
        self.max_outputs = max_outputs
        self.probability_threshold = probability_threshold
        self.default_index = default_index
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.lang = lang
        self.min_confidence = min_confidence
        self.hooks = hooks
        self.on_predict_start = on_predict_start
        self.on_predict_end = on_predict_end
        self.hooks_raise = hooks_raise
        self.hooks_timeout = hooks_timeout
        self.last_decision: Optional[Dict[str, Any]] = None

    def _select(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        if not choices:
            raise ValueError("No choices provided to select from.")

        query_str = _extract_query_str(query)
        criteria = _format_choices_criteria(choices)

        questions = {
            "selector": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            query_str,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        self.last_decision = res

        ans = res.get("answers", {}).get("selector", {})
        probs = ans.get("probabilities", {})

        selected: List[ToolSelection] = []
        key_list = list(criteria.keys())

        # Sort candidate choices by descending probability
        ranked = sorted(
            key_list,
            key=lambda k: probs.get(k, 0.0),
            reverse=True,
        )

        for key in ranked:
            prob = probs.get(key, 0.0)
            if prob >= self.probability_threshold:
                try:
                    idx = (
                        int(key.split("_")[1]) if key.startswith("choice_") else key_list.index(key)
                    )
                except (ValueError, IndexError):
                    idx = key_list.index(key)

                choice_obj = choices[idx]
                choice_name = getattr(choice_obj, "name", None) or f"choice_{idx}"
                selected.append(
                    ToolSelection(
                        index=idx,
                        reason=f"Selected '{choice_name}' (probability: {prob:.3f} >= {self.probability_threshold:.3f}).",
                    )
                )
                if self.max_outputs is not None and len(selected) >= self.max_outputs:
                    break

        # Fallback to top option if none met threshold
        if not selected and ranked:
            top_key = ranked[0]
            top_idx = (
                int(top_key.split("_")[1])
                if top_key.startswith("choice_")
                else key_list.index(top_key)
            )
            choice_obj = choices[top_idx]
            choice_name = getattr(choice_obj, "name", None) or f"choice_{top_idx}"
            prob = probs.get(top_key, 0.0)
            selected.append(
                ToolSelection(
                    index=top_idx,
                    reason=f"Selected top choice '{choice_name}' as fallback (probability: {prob:.3f}).",
                )
            )

        return SelectorResult(selections=selected)

    def select(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        return self._select(choices, query)

    async def _aselect(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        return await asyncio.to_thread(self._select, choices, query)

    async def aselect(
        self,
        choices: Sequence[Union[ToolMetadata, Any]],
        query: Union[QueryBundle, str],
    ) -> SelectorResult:
        return await self._aselect(choices, query)


class LayaQueryRouter:
    """Direct, high-performance query dispatcher for LlamaIndex query engines.

    Allows routing queries directly to target engines or callable tools without
    the overhead of standard LLM-based query engine routers.
    """

    def __init__(
        self,
        query_engines: Dict[str, Any],
        descriptions: Optional[Dict[str, str]] = None,
        instructions: str = "Which query engine should handle this request?",
        confidence_threshold: float = 0.0,
        fallback_key: Optional[str] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        lang: Optional[str] = None,
        min_confidence: Optional[float] = None,
        hooks: Optional[Any] = None,
        on_predict_start: Optional[Any] = None,
        on_predict_end: Optional[Any] = None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ):
        self.query_engines = query_engines
        self.descriptions = descriptions or {}
        self.instructions = instructions
        self.confidence_threshold = confidence_threshold
        self.fallback_key = fallback_key or next(iter(query_engines.keys()), None)
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.lang = lang
        self.min_confidence = min_confidence
        self.hooks = hooks
        self.on_predict_start = on_predict_start
        self.on_predict_end = on_predict_end
        self.hooks_raise = hooks_raise
        self.hooks_timeout = hooks_timeout
        self.last_decision: Optional[Dict[str, Any]] = None

    def route(self, query: Union[QueryBundle, str]) -> str:
        """Determine which query engine key should handle the query in ~33ms."""
        query_str = _extract_query_str(query)
        criteria = {
            key: self.descriptions.get(key, f"Engine for {key}")
            for key in self.query_engines.keys()
        }

        questions = {
            "route": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            query_str,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            lang=self.lang,
            min_confidence=self.min_confidence,
            hooks=self.hooks,
            on_predict_start=self.on_predict_start,
            on_predict_end=self.on_predict_end,
            hooks_raise=self.hooks_raise,
            hooks_timeout=self.hooks_timeout,
        )
        self.last_decision = res

        ans = res.get("answers", {}).get("route", {})
        chosen = ans.get("choice")
        conf = _gated_confidence(ans)

        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            if self.fallback_key:
                return self.fallback_key

        if chosen in self.query_engines:
            return chosen

        return self.fallback_key or next(iter(self.query_engines.keys()))

    def query(self, query: Union[QueryBundle, str], **kwargs: Any) -> Any:
        """Route the query to the winning engine and execute engine.query()."""
        key = self.route(query)
        engine = self.query_engines[key]
        if hasattr(engine, "query"):
            return engine.query(query, **kwargs)
        if callable(engine):
            return engine(query, **kwargs)
        raise TypeError(
            f"Engine registered for key {key!r} is not callable and has no .query() method."
        )

    async def aquery(self, query: Union[QueryBundle, str], **kwargs: Any) -> Any:
        """Asynchronously route the query and execute engine.aquery() or engine.query()."""
        key = await asyncio.to_thread(self.route, query)
        engine = self.query_engines[key]
        if hasattr(engine, "aquery"):
            return await engine.aquery(query, **kwargs)
        if hasattr(engine, "query"):
            return await asyncio.to_thread(engine.query, query, **kwargs)
        if callable(engine):
            if asyncio.iscoroutinefunction(engine):
                return await engine(query, **kwargs)
            return await asyncio.to_thread(engine, query, **kwargs)
        raise TypeError(
            f"Engine registered for key {key!r} is not callable and has no .query() method."
        )
