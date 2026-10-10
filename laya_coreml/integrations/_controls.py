"""Per-call decision controls shared by the framework integrations.

Every integration wrapper ends at one of two places: a local `runner.predict(state, questions,
**overrides)` call, or a POST to a laya-serve node. Which overrides those two accept is a single
rule, and three independent copies of it drift -- this module is the copy the LangChain, CrewAI
and LlamaIndex wrappers import.

The names are listed once, in `PREDICT_CONTROLS` and `HOOK_CONTROLS`, and the signatures below
are checked against them by `tests/test_langchain.py`, `tests/test_crewai.py` and
`tests/test_llamaindex.py`, which also assert every wrapper's constructor accepts all of them. A
control added here without reaching a wrapper's `__init__` fails that wrapper's suite rather than
being dropped silently.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# The token-budget overrides `Agent.predict` / `Router.predict` take per call, and that
# laya-serve accepts in the request body up to its `LAYA_MAX_TOKEN_BUDGET` ceiling.
PREDICT_CONTROLS = ("max_len", "head_max_len")

# The per-call language and abstention controls. Unlike `task` / `lang_guess` -- Router-only
# routing keywords a direct `Agent.predict` does not accept -- both `lang` and `min_confidence`
# are read by `Agent.predict`/`system_one` AND `Router.predict`, and laya-serve takes each in the
# request body, so the same kwargs are safe on the local and the remote path alike. `lang` routes
# non-English text and selects the answering checkpoint's per-language calibration;
# `min_confidence` is core's abstention gate (#361).
DECISION_CONTROLS = ("lang", "min_confidence")

# The per-call hook family. These are Python callables and flags that run inside `predict`, so
# they exist only on the local path -- see `reject_remote_hooks`.
HOOK_CONTROLS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")


def budget_kwargs(
    max_len: Optional[int] = None, head_max_len: Optional[int] = None
) -> Dict[str, Any]:
    """The two token budgets, with the unset ones omitted.

    Absent rather than `None`: on the local path a `None` budget would override the checkpoint's
    own default with nothing, and on the remote path an older stand-in `_call_remote` -- including
    the ones these integrations' own tests install -- must keep accepting the call.
    """
    return {
        key: value
        for key, value in (("max_len", max_len), ("head_max_len", head_max_len))
        if value is not None
    }


def decision_kwargs(
    lang: Optional[str] = None, min_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """The language and abstention overrides, with the unset ones omitted.

    Absent rather than `None`: a `None` `lang` would shadow the checkpoint's own detection and a
    `None` `min_confidence` would override the runner's abstention default with nothing. Both use
    the `is not None` test rather than truthiness -- an empty `lang` is core's documented "fall
    through to detection", and a `0.0` abstention gate is a real decision (abstain over nothing),
    not an absence.
    """
    return {
        key: value
        for key, value in (("lang", lang), ("min_confidence", min_confidence))
        if value is not None
    }


def predict_kwargs(
    model: Optional[str] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    lang: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """The per-request overrides a local runner accepts, with the unset ones omitted."""
    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    kwargs.update(budget_kwargs(max_len, head_max_len))
    kwargs.update(decision_kwargs(lang, min_confidence))
    return kwargs


def hook_kwargs(
    hooks: Optional[Any] = None,
    on_predict_start: Optional[Any] = None,
    on_predict_end: Optional[Any] = None,
    hooks_raise: Optional[bool] = None,
    hooks_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """The per-call hook overrides, with the unset ones omitted.

    Core reads `None` as "inherit whatever the runner was built with", so an unset hook has to be
    absent rather than passed as `None`. Note the `is not None` tests: `hooks=[]` means "no hooks
    for this call", and `hooks_raise=False` means "keep deciding after a hook fails" -- both are
    decisions a caller made, not absences.
    """
    given = {
        "hooks": hooks,
        "on_predict_start": on_predict_start,
        "on_predict_end": on_predict_end,
        "hooks_raise": hooks_raise,
        "hooks_timeout": hooks_timeout,
    }
    return {k: v for k, v in given.items() if v is not None}


def reject_remote_hooks(given: Dict[str, Any], base_url: Optional[str]) -> None:
    """Refuse hooks on a remote node rather than dropping them silently.

    A hook is a Python callable that runs inside `predict` -- it can cache a decision, gate one or
    rewrite its state. `laya-serve` has no way to receive or run one, so a node with a `base_url`
    and hooks configured would report success while never calling them.
    """
    if base_url and given:
        raise ValueError(
            "%s run in the local runner and cannot be sent to a laya-serve endpoint; "
            "install them where serve runs, or drop them" % ", ".join(sorted(given))
        )
