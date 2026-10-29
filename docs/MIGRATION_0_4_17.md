# Upstream v0.4.17 source migration

The current source target is official Laya **v0.4.17**,
`fe9a6c35023a7d94102e7feaaeaae8857d9d1b07`, replacing v0.4.12
`ffc142012af63d015fa19f05f29f9c5ff329a2cc` on Core ML base `7b36334`.
Package version remains **0.5.1**. No release is requested or qualified by this migration.

The source delta contains 26 Python files, 1,038 additions and 166 deletions.
Twenty-four files have applicable host changes. The upstream version bump and
ONNX duplicate are accounted for below. Changes to other SDKs, training/research
and deployment artifacts are outside this Python inference port.

## Behavior

- Language detection and `Router.route` / `route_batch` accept `explain=True`;
  `laya-coreml-router --explain` reports bounded source paths, stages and reasons
  without adding input text. Explicit routing overrides still bypass detection.
  Unicode letter handling and English/Vietnamese heuristics follow upstream.
- Long scans record answer identities and prediction snapshots before end hooks.
  Reordered answers retain their original window; cloned, shared or edited answers
  omit attribution. Reliability annotations remain allowed. Missing caller answers
  raise a named error; consistently removed start-added questions remain allowed.
- Reliability calibration validates lengths and finite numeric scores, accepts NumPy
  scalars, returns JSON-compatible float cutoffs and retains threshold precision.
  Option renaming preserves labels with empty semantic descriptions.
- Integration wrappers validate remote mappings, answer shapes and numeric fields
  through their existing errors/fallback paths. Existing real LlamaIndex compatibility
  remains covered by the installed-framework tests.
- HTTP body reads default to a ten-second `LAYA_BODY_READ_TIMEOUT`; a 408 releases
  admission. Authentication disables FastAPI documentation/schema routes. Forward
  token budgets apply even when callers specify a batch size. Invalid tournament
  labels return 422.
- `RemoteRouter(user_agent=...)` or `LAYA_USER_AGENT` supplies a validated Latin-1
  User-Agent. The default uses this distribution's version, `laya/0.5.1`.
- Router CLI question/batch files and evidence readers accept UTF-8 BOMs and report
  paths on read, JSON, encoding or nesting failures. Evaluation validates row shape,
  result cardinality and exact requested metrics. Invalid calibration targets and
  nested Windows drive paths receive named errors.

## Core ML boundaries

The direct upstream three-way port preserves NumPy tensor transport, converted
artifact/revision mapping, `compute_units`, compiled models, graph capacities,
adaptive batches and bounded shape caches. Shared host methods live in `runtime.py`;
upstream `cli.py` maps to `router_cli.py`. The separate conversion/inference CLI
retains its existing flags, accepts UTF-8 BOMs in question files, and reports named
read/parse errors before loading a model. Package/version exports remain 0.5.1.

`embed_fn_from_agent` now names missing tokenizer/model inputs before importing
PyTorch. A model that exposes no encoder still raises the existing
`NotImplementedError` directing callers to an external `embed_fn`: Core ML exports
provide decision outputs, not encoder embeddings. PyTorch/ONNX backend switches,
training and thread controls remain explicitly unsupported. No new backend is added.

## Complete source inventory

The machine-readable [inventory](UPSTREAM_PORT.json) records every changed upstream
Python file. The [API audit](UPSTREAM_0417_API_AUDIT.json) checks signatures and root
exports; behavioral evidence comes from the original assertions and differential tests.

| Upstream file | Port location | Scope |
|---|---|---|
| `laya/__init__.py` | `laya_coreml/__init__.py` | Upstream package version only; retain laya-coreml 0.5.1. |
| `laya/_eval_policy.py` | `laya_coreml/_eval_policy.py` | Unreadable or excessively nested policy files report EvalError. |
| `laya/agent.py` | `laya_coreml/runtime.py` | Track answer identity and prediction snapshots before end hooks; preserve original scan spans after reordering, omit edited/cloned/shared attribution, detect missing answers, normalize probabilities in place. |
| `laya/calibrate.py` | `laya_coreml/calibrate.py` | Name invalid target_error values while preserving comparable numeric types. |
| `laya/cli.py` | `laya_coreml/router_cli.py` | Routing --explain; BOM-aware questions/batch inputs; path-specific JSON, encoding, depth and OS errors. |
| `laya/common.py` | `laya_coreml/common.py` | Scalar confidence clipping avoids NumPy dispatch while retaining vector entropy order. |
| `laya/confidence.py` | `laya_coreml/confidence.py` | Jev probability coercion handles OverflowError. |
| `laya/evals.py` | `laya_coreml/evals.py` | Validate dataset row shape/read errors, reject bool confidence, require one batch result per row, report incomparable regression metrics. |
| `laya/evals_cli.py` | `laya_coreml/evals_cli.py` | Explicit accuracy thresholds require the exact metric named. |
| `laya/evidence.py` | `laya_coreml/evidence.py` | BOM-aware evidence artifacts and path-specific nesting failures. |
| `laya/hooks.py` | `laya_coreml/hooks.py` | Validate timeout coercion and reliability data; preserve full soft stability and NumPy calibration cutoffs; keep semantic labels when renaming empty descriptions. |
| `laya/integrations/_guard.py` | `laya_coreml/integrations/_guard.py` | Validate response/answer/numeric/probability shapes with named malformed-response errors. |
| `laya/integrations/agno.py` | `laya_coreml/integrations/agno.py` | Fail closed on malformed remote and nested decisions. |
| `laya/integrations/crewai.py` | `laya_coreml/integrations/crewai.py` | Fail closed on malformed remote and nested decisions. |
| `laya/integrations/langchain.py` | `laya_coreml/integrations/langchain.py` | Reject non-object remote response and answers. |
| `laya/integrations/llamaindex.py` | `laya_coreml/integrations/llamaindex.py` | Validate response, choice and ranked probabilities while preserving native port framework compatibility. |
| `laya/integrations/tools.py` | `laya_coreml/integrations/tools.py` | Malformed tool decisions/probabilities take the existing error or fallback path. |
| `laya/lang.py` | `laya_coreml/lang.py` | Unicode letter compatibility table, bisected script ranges, English/Vietnamese heuristics and opt-in diagnostic stage/reason/source paths. |
| `laya/mcp/remote.py` | `laya_coreml/mcp/remote.py` | Validate explicit/environment User-Agent and send it on HTTP calls; default uses local package version. |
| `laya/mcp/server.py` | `laya_coreml/mcp/server.py` | Document LAYA_USER_AGENT. |
| `laya/onnx_agent.py` | `laya_coreml/runtime.py (shared host behavior only)` | ONNX implementation excluded; shared long-attribution and probability changes are implemented once in runtime.py. |
| `laya/revisions.py` | `laya_coreml/revisions.py` | Reject nested Windows drive-anchored digest paths. |
| `laya/router.py` | `laya_coreml/router.py` | Forward opt-in explain on route/route_batch without changing explicit override routing. |
| `laya/serve.py` | `laya_coreml/serve.py` | Bound body read time and release admission on 408; hide schema/docs when authenticated; enforce batch token budget even with caller batch size; report invalid tournament labels as 422. |
| `laya/shortlist.py` | `laya_coreml/shortlist.py` | Validate missing tok/model before importing the fitting backend; preserve the explicit Core ML encoder-output refusal and external embed_fn path. |
| `laya/structured.py` | `laya_coreml/structured.py` | Reject boolean numeric bounds; ignore non-mapping answer entries during projection/details. |

## Assertions and validation scope

`tests/test_official_contract.py` retains the original host/differential gates and
adds language diagnostics, CLI language guesses, confidence, CrewAI, LangChain,
LlamaIndex and the shared long-attribution helper. The helper uses upstream's
unchanged `_Tok` / `make_real_agent` host fixture and all 49 checks. Local
`tests/test_upstream_0417.py` adds changed-method AST equality, missing-answer scans,
NumPy reliability precision, timeout coercion, calibration targets, evaluation
cardinality, exact metrics, unsafe digest paths, the Core ML embedding boundary,
and native CLI BOM/error handling before model loading.

The namespace adapter preserves expected host values and public model names.
Existing exclusions remain: ONNX assertions; Torch device/thread/status checks;
training CLI help; backend-switch implementations. The new confidence suite also
excludes its ONNX decoder alternative and six PyTorch/TileLang padding/source
checks, retaining all applicable host assertions. Full upstream calibration,
revision-pinning, long-prediction and shortlist suites mix backend/model fixtures;
changed host behavior is covered locally and by the shared attribution helper.

The idle fixture adapter and `tests/test_idle_test_adapter.py` remain unchanged:
the same 200 ms minimum is measured on the worker, with all eight combinations,
non-overlap and eventual unload preserved. See [CI_TIMING.md](CI_TIMING.md).
Assertions that inspect upstream docs/examples require the complete pinned checkout.

The CI matrix retains Linux and macOS conversion, development, demo, publishing,
reference, HTTP/MCP, structured and installed LlamaIndex extras; no gate is removed.
Local validation results are recorded below. Native Core ML
conversion/inference and real-model fixture/stability checks require macOS and
are not claimed from Linux host results. Remote CI and independent review results
are recorded below.

## Historical evidence

[UPSTREAM_0412_PORT.json](UPSTREAM_0412_PORT.json) preserves the previous manifest.
The v0.4.12 API audit, progress and qualification reports, 0.5.1 release receipts,
model precision/latency reports and Snake benchmarks retain their original scope.
They do not qualify this new source revision. Current documentation links to
those artifacts as historical evidence, without rewriting their recorded results.

## Local validation

Static checks used the project `uv` environment with CPython 3.12.14; the locally
cached sibling Ruff 0.15.8 was reused. No system packages were modified.

| Command | Result |
|---|---|
| `../laya-mlx/.venv/bin/ruff check .` | Passed. |
| `../laya-mlx/.venv/bin/ruff format --check .` | Passed, 115 files. |
| `.venv/bin/python -m compileall -q laya_coreml tests scripts` | Passed. |
| `.venv/bin/python scripts/audit_public_api.py --package laya_coreml` | 345 entries, zero unclassified differences. |
| Isolated stdlib native CLI smoke, with loader mocked | Six cases passed: five read/encoding/JSON-parsing failures never load; a BOM file reaches prediction with native options intact. The corresponding pytest regressions are included for CI. |
| `git diff --check` | Passed. |

Direct changed-definition AST comparison against upstream matches all applicable
host definitions except the existing native batching, CLI and encoder-output
boundaries. Native agent/ANE/export/compiled-shape implementations, package metadata
and `tests/test_idle_test_adapter.py` remain unchanged from the migration base.

**Pytest, build/Twine and clean inference-only wheel validation were not run locally.**
The Torch 2.7 CPU wheel download and requested dependency installation remained slow;
a parallel dependency install timed out acquiring the NumPy cache lock. The unqualified
PyPI resolver also attempted CUDA wheels before failing, so no successful compatible
project installation is claimed. Owned install processes were stopped before handoff.
Use the CI installation order (finish Torch CPU installation first, then install the
project extras) or explicitly constrain the CPU wheel for local reproduction.
Full-checkpoint stability, clean inference-only wheel smoke, and performance/energy
benchmarks were not rerun. This is a tested source migration, not release qualification.

## Remote CI and independent review

[Run 40825520186](https://github.com/Omicron0314/laya-coreml/actions/runs/40825520186)
passed on source commit `23a317b62b77227e944cefe9c68bab55ca6f41a0`:

| Gate | Linux | macOS 26 |
|---|---|---|
| Complete pytest suite | 386 passed, 3 skipped | 389 passed |
| Public API audit | 345 entries, zero unclassified differences | Same |
| Ruff lint/format, wheel/sdist build, strict Twine | Passed | Passed |

The Linux skips are real Core ML conversion/dynamic-padding inference and the
FP32/FP16 parallel-layout conversion/runtime cases; all three passed on macOS.
Warnings on macOS are retained in the CI log. This does not revalidate the full
published checkpoints, ANE energy measurements or performance claims.

Two independent source reviews found no blocking migration defect or hidden
assertion weakening. They checked complete upstream coverage, native runtime/cache
boundaries, the long-attribution helper, confidence exclusions and the unchanged
idle deadline adapter. The native CLI still delegates question-schema validation
to prediction after model loading; its new pre-load validation covers file reading,
encoding and JSON parsing only. That existing schema-validation limitation is not
an omitted v0.4.17 change.

This CI receipt was added in a documentation-only follow-up; implementation,
tests and CI configuration remain identical to the tested source commit.
