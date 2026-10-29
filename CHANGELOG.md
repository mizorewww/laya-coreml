# Changelog

Each release names the upstream [Laya](https://github.com/NandhaKishorM/laya) revision whose host
behavior it follows. Detailed records live in [docs/](docs/README.md).

## Unreleased

- **Fix:** `predict_batch` no longer groups integer or boolean keys with their string equivalents
  (`1` vs `"1"`), which returned the wrong key types for later requests.
- **Fix:** with `hooks_concurrent=False`, a timed-out hook keeps the serialization lock until it
  exits, and a hook whose deadline passes while waiting is never called.
- Both are ported from laya-mlx 0.6.0 and documented as divergences from upstream v0.4.17.
- The compatibility policy now allows documented, tested fixes for upstream defects.

## 0.6.0 — 2026-10-29

Follows upstream **v0.4.17** (`fe9a6c35`). [Migration record](docs/MIGRATION_0_4_17.md).
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

Upstream v0.4.12 migration ([notes](docs/RELEASE_0_5_0.md), [qualification](docs/QUALIFICATION_0_4_12.md)).
The tag failed the macOS release gate and was never uploaded to PyPI; 0.5.1 replaces it.

- Native B=1/2/4/16 adaptive batch bundles, dense sequence buckets and a bounded shape cache.
- Batch, long-text, structured decisions, hooks, calibration, evaluation, HTTP/MCP and integrations
  aligned with v0.4.12.

## 0.4.0 — 2026-10-10

Aligns host behavior with upstream **v0.4.1** ([migration guide](docs/MIGRATION_0_4.md)).

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
