# Changelog

Each release names the upstream [Laya](https://github.com/NandhaKishorM/laya) revision whose host
behavior it follows. Detailed records live in [docs/](docs/README.md).

## 0.7.2 — 2026-11-18

Follows upstream **v0.4.37** (`d1282e79`). [Compatibility record](docs/UPSTREAM.md).
Host behavior and model bundles are unchanged; no reconversion is needed.

- Upstream v0.4.36 and v0.4.37 change only upstream's README, version strings and packaging. The
  Python host code and tests are unchanged, so laya-coreml's behavior is the same as 0.7.1.
- The pin, API audit and compatibility record now name v0.4.37. Both deliberate divergences remain,
  since upstream still has those defects.
- Validation: full test suite, API audit, lint and build checks on macOS. Real-bundle checks were
  not re-run for this release.

## 0.7.1 — 2026-11-18

Follows upstream **v0.4.35** (`b1d65695`). [Compatibility record](docs/UPSTREAM.md).
Host behavior and model bundles are unchanged; no reconversion is needed.

- Upstream v0.4.35 adds HTTP routing snapshots to its TypeScript SDK only. The Python host code is
  unchanged, so laya-coreml's behavior is the same as 0.7.0.
- The pin, API audit and compatibility record now name v0.4.35; upstream's two new `hooks_api`
  export checks run unchanged.
- Validation: full test suite, API audit, lint and build checks on macOS. Real-bundle checks were
  not re-run for this release.

## 0.7.0 — 2026-11-16

Follows upstream **v0.4.34** (`80a1da1f`). [Compatibility record](docs/UPSTREAM.md).
Model bundles are unchanged; no reconversion is needed.

- **Routing without inference:** `POST /v1/route` on `laya-coreml-serve`, and `explain` on the MCP
  `laya_route` and `laya_route_batch` tools, return the decision and its language diagnostic.
- **Router vega lane:** `Router(vega=agent)` sends images, audio, video, over-long states and
  questions with fitted heads to an already built vega agent. laya-coreml does not include the
  vegaml engine: `Agent(vegaml=True)`, `load(vegaml=True)` and `Router(vega=True)` raise, and the
  lane is off by default.
- **LangChain:** `LayaRouter(decision_policy="native")` follows an answer's `auto`/`review` route.
  `LayaToolSelector` accepts a per-bucket `min_confidence` map.
- **Stricter validation (may raise where 0.6.1 accepted input):** a non-integer `batch_size`,
  malformed checkpoint source pairs, decisions naming an agent or choice outside the list,
  non-array `anyOf`/`oneOf`, wide integers in calibration bins, and media keys in HTTP bodies.
- **Fixes:** temperatures, confidences and guard values written as integers wider than a double no
  longer crash a load or a decision; BOM-prefixed eval datasets and reports are read;
  `OptionStabilityHook` no longer counts a rename probe identical to the original question;
  an over-nested `LAYA_SHA256_DIGESTS` is a configuration error.
- **Faster:** calibration fitting is vectorized, and routing skips hook dispatch when no hook
  implements `on_route`.
- The server logs one warning per request when a state was truncated.
- Docs and tests: one evergreen `docs/UPSTREAM.md` replaces the per-version migration records;
  local tests that duplicated the pinned upstream suites were removed.
- Validation: full test suite on macOS; the three published adaptive FP16 bundles re-checked
  against the golden fixture (63/63 selected answers, 100 stable calls each). Performance figures
  were not re-measured for this release.

## 0.6.1 — 2026-10-29

- **Fix:** `predict_batch` no longer groups integer or boolean keys with their string equivalents
  (`1` vs `"1"`), which returned the wrong key types for later requests.
- **Fix:** with `hooks_concurrent=False`, a timed-out hook keeps the serialization lock until it
  exits, and a hook whose deadline passes while waiting is never called.
- Both are ported from laya-mlx 0.6.0 and documented as divergences from upstream v0.4.17.
- The compatibility policy now allows documented, tested fixes for upstream defects.

## 0.6.0 — 2026-10-29

Follows upstream **v0.4.17** (`fe9a6c35`).
Model bundles are unchanged; no reconversion is needed.

- **Language routing:** `detect_language(..., explain=True)` and `Router.route(..., explain=True)`
  report why a language was chosen; updated Unicode script lookup and English/Vietnamese detection.
- **Long documents:** answer attribution survives end hooks that reorder results; missing
  answers raise, and replaced or edited answers carry no misleading window attribution.
- **Stricter validation (may raise where 0.5.1 accepted input):** reliability calibration
  precision, malformed integration responses, evaluation cardinality and metric checks, confidence
  overflow and digest paths.
- **HTTP and CLI:** request-body deadlines, batch token budgets, configurable remote User-Agent,
  BOM-aware input files; the native CLI reports read/encoding/JSON errors before loading a model.
- CI gates on the public API audit; the release-validation reference targets v0.4.17.
- Validation: full test suite on Linux and macOS 26; published adaptive FP16 bundles re-checked
  against the golden fixture. Performance figures were not re-measured for this release.

## 0.5.1 — 2026-10-20

Follows upstream **v0.4.12** (`ffc14201`). [Release notes](docs/RELEASE_0_5_1.md) ·
[qualification](docs/QUALIFICATION_0_5_1.md).

- Fixes a native GIL-state crash on macOS 26 when the shape cache admitted new shapes: extra
  execution instances now reuse the base model's compiled artifact.
- Publishes six adaptive bundles (English, multilingual and typed-decisions, FP16 and FP32).

## 0.5.0 — tagged 2026-10-20, not published

Upstream v0.4.12 migration.
The tag failed the macOS release gate and was never uploaded to PyPI; 0.5.1 replaces it.

- Native B=1/2/4/16 adaptive batch bundles, dense sequence buckets and a bounded shape cache.
- Batch, long-text, structured decisions, hooks, calibration, evaluation, HTTP/MCP and integrations
  aligned with v0.4.12.

## 0.4.0 — 2026-10-10

Aligns host behavior with upstream **v0.4.1**.

- `Router()` keeps two models resident and falls back to multilingual; scalar choice labels,
  `option_order`, `predict_batch`, `predict_long`, hooks, calibration and evaluation follow upstream.

## 0.3.0 — 2026-10-10

Ports upstream v0.4.1 runtime features: custom bundle registration, `min_confidence`,
`predict_tournament` and parallel option layout conversion.

## 0.2.0 — 2026-10-02

Syncs upstream `4aa6761`: prompt validation, custom `noul` labels, conversation-tail truncation,
option-collision diagnostics and `answer_confidence`. Publishes the calibration temperature clamp
that PyPI 0.1.0 lacked.

## 0.1.0 — 2026-09-20

First release: Core ML runtime, ANE and Snake bundles, terminal demo and benchmarks.
[Artifacts](docs/RELEASE.md).
