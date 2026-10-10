"""Core ML device preferences for the official HTTP/MCP protocol.

Compute-unit labels describe allowed execution units, not per-operation placement.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_KEY = "LAYA_DEVICE"


def env_device() -> str | None:
    """LAYA_DEVICE (empty or unset = None), ready for Router(device=...).

    Lower-cased, not verbatim. torch's device parser is case-sensitive -- it accepts ``cuda``
    and rejects ``CUDA`` with "Expected one of cpu, cuda, ..." -- so a value this function
    returned unchanged could fail the Router build while `laya_status` reported the
    lower-cased form as the working device. The same normalisation `resolve_device` applies
    is applied here, so the value handed to torch and the value reported are one string.

    Splitting on the device *type* and the optional index keeps that true for ``CUDA:0``:
    torch wants the type lower-cased and does not care about the index, so
    ``CUDA:0 -> cuda:0``.

    The normalised value is then parsed by torch (``_check_torch_device``): a value torch
    cannot read raises ``ValueError`` naming ``LAYA_DEVICE``, before the string can reach
    ``Router(device=...)`` or be reported as the device in use. ``cuda:`` (trailing colon),
    ``gpu`` and ``cuda: 0`` (space after the colon) are the common typos; ``CUDA:0`` is not
    one, because it is normalised first.
    """
    value = os.environ.get(_ENV_KEY)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    kind, sep, index = value.partition(":")
    kind = kind.lower()
    normalised = "%s%s%s" % (kind, sep, index) if sep else kind
    _check_torch_device(normalised)
    return normalised


def _check_torch_device(value: str) -> None:
    """Validate the distribution's supported device preferences without importing torch."""
    if value not in ("cpu", "mps"):
        raise ValueError("LAYA_DEVICE=%r is unavailable in Core ML; use 'cpu' or 'mps'" % value)


def resolve_device(force: str | None = None) -> str:
    """Configured execution preference; Core ML chooses actual placement per operation."""
    return force or env_device() or "mps"


def device_report() -> dict:
    """Retain protocol keys without claiming a PyTorch/CUDA runtime."""
    return {"device": resolve_device(), "torch_cuda": False, "torch_version": None}


def agent_device(agent: Any) -> str | None:
    """The real device a loaded agent computes on (str), or None if unreadable.

    Reads ``Agent.device``, a ``torch.device`` that already reflects the
    silent GPU -> CPU fallback done at build time. Accepts a plain string too,
    so tests can fake an agent without torch.
    """
    if agent is None:
        return None
    device = getattr(agent, "device", None)
    if device is None:
        return None
    kind = getattr(device, "type", None)
    if isinstance(kind, str) and kind:
        return kind
    if isinstance(device, str) and device:
        return device
    return None


def router_agent(router: Any, name: str) -> Any | None:
    """The agent a router currently holds for checkpoint ``name``, or None.

    Read-only on purpose: it reads the private ``_agents`` mapping (the same
    dictionary ``Router.load`` populates, keyed by the normalised name) and
    never calls ``load()``, because ``load`` has side effects: it reorders the
    LRU for a resident checkpoint and rebuilds an evicted one (hundreds of MB)
    -- unacceptable for a device-label read.
    """
    if router is None:
        return None
    agents = getattr(router, "_agents", None)
    if not isinstance(agents, dict):
        return None
    if name in agents:
        return agents[name]
    # Routers key _agents by the normalised name (Router.load normalises the
    # same way); try the core normaliser for aliased inputs, without heavy
    # imports at module level.
    try:
        from laya_coreml.router import normalise_name

        key = normalise_name(name)
    except Exception:
        return None
    return agents.get(key) if key != name else None
