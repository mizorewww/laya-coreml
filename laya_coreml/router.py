# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-coreml.
"""Route a request to the Laya checkpoint best suited to it.

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical questions
per model -- see the repository's benchmark notebook):

  english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
  multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions
                                                                workflows

Why routing is worth it -- accuracy by language family:

                      english   multilingual
  MASSIVE intent  en    0.783       0.657        <- English checkpoint wins
  MASSIVE intent  non-en 0.306      0.451
  XNLI            en    0.860       0.843
  XNLI            non-en 0.521      0.731        <- +21 points for multilingual
  English suites        0.684       0.619

The English checkpoint does not gently degrade off English, it collapses: on 20-option MASSIVE
intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random guessing -- and it
reports high confidence while doing so (ECE 0.855 on Hindi). Script detection is therefore the
primary routing signal.

`typed-decisions` is never selected automatically unless you opt in with
`auto_task_detection=True` or pass `task="typed_decisions"`: it is fine-tuned on four specific
synthetic workflows and should not be a silent default.
"""

import gc
import os
import re
import threading
from typing import Any, Dict, List, Optional, Union

from .confidence import apply_confidence_gate, check_min_confidence
from .lang import analyse

# Published Core ML bundles; original PyTorch repositories are not runtime inputs.
DEFAULT_MODELS = {
    "english": "aac6fef/laya-coreml",
    "multilingual": "aac6fef/laya-multilingual-coreml",
    "typed-decisions": "aac6fef/laya-typed-decisions-coreml",
}


def _repo_str(spec):
    return spec


# Aliases people are likely to type.
_ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {
        "discrepancy_severity",
        "disposition",
        "duplicate",
        "matches_order",
        "urgency",
    },
    "security_incidents": {
        "credential_compromise",
        "disposition",
        "severity",
        "true_positive",
        "urgency",
    },
}


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


_ENGLISH_SUBTAGS = ("en", "eng", "english")

# Codes that are valid `$LANG` values but name no language, so they answer nothing about the
# state. `C`, `POSIX` and `C.UTF-8` are what minimal images ship -- `C.UTF-8` is the default
# `LANG` in the official Python image, which is where `laya-serve` runs -- and the ISO 639-2
# special codes say the same thing in the standard's own vocabulary: `und` undetermined,
# `zxx` no linguistic content, `mul` multiple languages. They abstain, which is what the blank
# case below already does, rather than forcing the multilingual checkpoint on English text.
_LANGUAGE_AGNOSTIC_CODES = ("c", "posix", "und", "zxx", "mul")


def _english_from_code(value: Any) -> Optional[bool]:
    """True/False for a language code, or None when the code identifies nothing.

    Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`, the
    POSIX `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`. `None` here means "no usable
    hint", which is what lets a language-identification model abstain -- and it is also what a
    code that names no language returns, so `LANG=C` falls through to detection instead of
    pinning every request to one checkpoint.
    """
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]  # en_US.UTF-8 -> en_US
    primary = code.replace("_", "-").split("-", 1)[0]  # en_US -> en
    if not primary or primary in _LANGUAGE_AGNOSTIC_CODES:
        return None
    return primary in _ENGLISH_SUBTAGS


def canonical_name(name):
    if not isinstance(name, str):
        raise ValueError("Checkpoint name must be a string")
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key == "auto" or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", key):
        raise ValueError(f"Invalid checkpoint name: {name!r}")
    return key


class _InFlightBuild:
    def __init__(self):
        self.done = threading.Event()
        self.error = None


class Router:
    """Lazily loads Laya checkpoints and sends each request to the right one.

        from laya_mlx import Router

        r = Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # -> english
        r.predict(state, questions, model="typed-decisions")                     # explicit

    Models are downloaded and built on first use. `max_loaded` caps how many stay resident
    (least-recently-used is evicted), because all three together are ~1.16B parameters.

    For a server or a demo, preload instead: a cold load costs seconds, while detection costs
    microseconds, so anything that alternates languages at `max_loaded=1` reloads on every
    request.

        r = Router(preload=True)                    # all three resident, routing is free
        r = Router(preload=True, compute_units="cpu_gpu")
        r.preload(["english", "multilingual"])      # or just the two you serve
    """

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        compute_units: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        max_loaded: int = 1,
        default: str = "multilingual",
        auto_task_detection: bool = False,
        preload: bool = False,
    ):
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._loading = {}
        self._generation = {}
        self.descriptions = {}
        self.models = dict(DEFAULT_MODELS)
        for name, source in (models or {}).items():
            self.models[canonical_name(name)] = self._source(source)
        self.compute_units = compute_units
        self.revision = revision
        self.local_files_only = local_files_only
        self.max_loaded = max(1, int(max_loaded))
        self.default = self.resolve(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []  # least-recently-used first
        # Shared state only: never hold this lock while constructing or waiting.
        # A separate build lock serializes allocations, preserving peak memory bounds.
        if preload:
            self.preload()

    # ------------------------------------------------------------------ registry
    @staticmethod
    def _source(source):
        if source is None:
            return None  # attach-only registration
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if isinstance(source, str) and source.strip():
            return os.path.expanduser(source)
        raise ValueError("Checkpoint source must be a nonempty Core ML repo/path")

    def resolve(self, name):
        """Resolve a built-in alias or a checkpoint registered on this Router."""
        key = canonical_name(name)
        with self._lock:
            if key not in self.models:
                raise ValueError(f"unknown model {name!r}; choose one of {sorted(self.models)}")
        return key

    def register(self, name, source, description=None):
        """Register a source lazily; replacing it invalidates resident and in-flight old builds."""
        key, source = canonical_name(name), self._source(source)
        if description is not None:
            description = str(description)
        with self._lock:
            changed = key not in self.models or self.models[key] != source
            self.models[key] = source
            if description is not None:
                self.descriptions[key] = description
            freed = False
            if changed:
                self._generation[key] = self._generation.get(key, 0) + 1
                freed = self._drop_locked(key)
        if freed:
            gc.collect()
        return key

    def unregister(self, name):
        """Remove a custom checkpoint; built-ins and the current default cannot be removed."""
        with self._lock:
            key = self.resolve(name)
            if key in DEFAULT_MODELS or key == self.default:
                raise ValueError("Cannot unregister a built-in or the default checkpoint")
            del self.models[key]
            self.descriptions.pop(key, None)
            self._generation[key] = self._generation.get(key, 0) + 1
            freed = self._drop_locked(key)
        if freed:
            gc.collect()

    @property
    def registered(self):
        with self._lock:
            return {
                key: {
                    "source": _repo_str(source) if source is not None else None,
                    "description": self.descriptions.get(key),
                }
                for key, source in self.models.items()
                if key not in DEFAULT_MODELS
            }

    # ------------------------------------------------------------------ loading
    def load(self, name):
        """Share builds per checkpoint without blocking resident checkpoints or status reads."""
        while True:
            with self._lock:
                key = self.resolve(name)
                if key in self._agents:
                    self._touch(key)
                    return self._agents[key]
                inflight = self._loading.get(key)
                if inflight is None:
                    inflight = self._loading[key] = _InFlightBuild()
                    break
            inflight.done.wait()
            if inflight.error is not None:
                raise inflight.error

        try:
            with self._build_lock:
                while True:
                    with self._lock:
                        self.resolve(key)
                        if key in self._agents:
                            self._touch(key)
                            return self._agents[key]
                        source = self.models[key]
                        if source is None:
                            raise ValueError(
                                f"Checkpoint {key!r} has no source; attach or register it again"
                            )
                        generation = self._generation.get(key, 0)
                        kwargs = dict(
                            compute_units=self.compute_units,
                            revision=self.revision,
                            local_files_only=self.local_files_only,
                        )
                    from .agent import load

                    built = load(source, **kwargs)
                    with self._lock:
                        self.resolve(key)  # unregister during construction must not resurrect it
                        if key in self._agents:  # attach during construction takes precedence
                            self._touch(key)
                            return self._agents[key]
                        stale = generation != self._generation.get(key, 0)
                        if not stale:
                            self._agents[key] = built
                            self._touch(key)
                            freed = self._evict_locked()
                    if stale:
                        del built
                        gc.collect()
                        continue
                    if freed:
                        gc.collect()
                    return built
        except BaseException as error:
            inflight.error = error
            raise
        finally:
            with self._lock:
                self._loading.pop(key, None)
                inflight.done.set()

    def _touch(self, key):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _drop_locked(self, key):
        existed = key in self._agents
        self._agents.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        return existed

    def _evict_locked(self):
        freed = False
        while len(self._order) > self.max_loaded:
            freed = self._drop_locked(self._order[0]) or freed
        return freed

    def _evict(self):
        with self._lock:
            freed = self._evict_locked()
        if freed:
            gc.collect()

    def attach(self, name, agent):
        """Attach an existing Agent, registering an unknown name without a reload source."""
        key = canonical_name(name)
        with self._lock:
            self.models.setdefault(key, None)
            replaced = key in self._agents
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        if replaced:
            gc.collect()
        return agent

    def preload(self, names=None):
        """Preload incrementally without holding the lifecycle lock over cold builds."""
        with self._lock:
            if names is None:
                names = [key for key, source in self.models.items() if source is not None]
            names = [self.resolve(name) for name in names]
            self.max_loaded = max(self.max_loaded, len(set(names) | set(self._agents)))
        for name in names:
            self.load(name)
        return self

    def unload(self, name=None):
        """Drop router references, waiting only for the requested checkpoint's builds.

        In-flight predictions and callers retain their own Agent references. Garbage
        collection releases unreachable objects; Core ML manages its native caches.
        """
        key = self.resolve(name) if name is not None else None
        while True:
            with self._lock:
                pending = (
                    ([self._loading[key]] if key in self._loading else [])
                    if key
                    else list(self._loading.values())
                )
                if not pending:
                    if key is None:
                        freed = bool(self._agents)
                        self._agents.clear()
                        self._order.clear()
                    else:
                        freed = self._drop_locked(key)
                    break
            for inflight in pending:
                inflight.done.wait()
        if freed:
            gc.collect()

    @property
    def loaded(self):
        with self._lock:
            return list(self._order)

    # ------------------------------------------------------------------ routing
    def route(self, state, questions=None, model=None, task=None, lang=None):
        with self._lock:
            return self._route(state, questions, model, task, lang)

    def _route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
        explicit `lang` > detected script/language > default.
        """
        if model is not None:
            key = self.resolve(model)
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason="explicit model=%r" % model,
                detection=None,
                workflow=None,
            )

        if task is not None:
            key = self.resolve(
                "typed-decisions"
                if str(task).lower().replace("-", "_") == "typed_decisions"
                else task
            )
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason="explicit task=%r" % task,
                detection=None,
                workflow=None,
            )

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions",
                repo=_repo_str(self.models["typed-decisions"]),
                reason="question ids match the %r typed-decisions workflow" % workflow,
                detection=None,
                workflow=workflow,
            )

        resolved = _english_from_code(lang)
        if resolved is not None:
            key = "english" if resolved else "multilingual"
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason="explicit lang=%r" % lang,
                detection=None,
                workflow=workflow,
            )

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = (
                "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                % (det["script"], 100 * float(det["non_latin_fraction"]))
            )
        elif not det["is_english"]:
            key = "multilingual"
            if det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                # Unidentified Latin-script language: routed on the non-English letters alone,
                # because no stopword list here covers it.
                reason = (
                    "Latin script, language not identified but %.0f%% non-English letters; "
                    "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"]))
                )
        elif det["language_undecided"]:
            key = self.default
            reason = "Latin language undecided; using default (%s)" % key
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(
            model=key,
            repo=_repo_str(self.models[key]),
            reason=reason,
            detection=det,
            workflow=workflow,
        )

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        *,
        min_confidence=None,
    ) -> Dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        The result is the usual `system_one` payload plus a `routing` key recording the decision.
        """
        if min_confidence is not None:
            min_confidence = check_min_confidence(min_confidence)
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.system_one(state, questions)
        apply_confidence_gate([result], min_confidence)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (
            self.loaded,
            self.max_loaded,
            self.default,
        )
