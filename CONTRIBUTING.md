# Contributing to Laya-CoreML

## Compatibility policy

Official [Laya](https://github.com/NandhaKishorM/laya) behavior, defaults, validation, return
structures and tests are the source of truth. Keep host changes traceable to upstream and limit
adaptations to the Core ML tensor, artifact, device and resource boundaries. Document every
material incompatibility in [docs/MIGRATION_0_4_17.md](docs/MIGRATION_0_4_17.md) (current) or
[docs/MIGRATION_0_4.md](docs/MIGRATION_0_4.md) (carried forward). Keep explicit failures for
unsupported backends. Never change an expected value only to make a test pass.

A bug that upstream still has may be fixed here (or ported from laya-mlx) if it is reproduced
against the pinned upstream code, the fix keeps upstream's public API, new regression tests fail
before and pass after, and the change is listed under "Deliberate divergences from upstream" in
the current migration document.
[AGENTS.md](AGENTS.md) states the same policy for coding agents.

## Development setup

```bash
gh repo clone mizorewww/laya-coreml
cd laya-coreml
uv sync --extra convert --extra dev --extra demo --extra publish --extra reference \
  --extra serve --extra mcp --extra structured --extra llamaindex
source .venv/bin/activate

# Upstream reference used by parity and contract tests (the commit CI pins):
gh repo clone NandhaKishorM/laya .upstream
git -C .upstream checkout fe9a6c35023a7d94102e7feaaeaae8857d9d1b07
```

Model bundles are downloaded separately and are never committed.

## Tests

```bash
ruff check . && ruff format --check .
python scripts/audit_public_api.py --package laya_coreml   # must report 0 unclassified
git diff --exit-code -- docs/UPSTREAM_0417_API_AUDIT.json  # audit must be committed
python -X faulthandler -m pytest -q
```

CI runs the same suite on Linux (portable tests; native conversion and inference are skipped) and
macOS 26 (everything, including tiny-model Core ML conversion and runtime). Full-checkpoint
validation runs locally on Apple hardware. [CI_TIMING.md](docs/CI_TIMING.md) explains the upstream
idle-window test adaptation.

## Real bundles and benchmarks

Real-bundle checks compare against pinned upstream FP32 golden fixtures: selected answers,
calibrated-probability drift, complete public outputs and repeated stability calls. Adaptive bundles
use `reference-v0412-fp32.json`, which includes the current output fields. The default
`reference.json` predates `answer_confidence` and the `usage` truncation fields, so it only fits
`release_validate`, which compares logits and selected answers.

```bash
# Adaptive bundles (the 0.5.1+ qualification path), once per model and precision:
hf download aac6fef/laya-multilingual-coreml-adaptive-fp16 --local-dir models/check/multilingual
python -m benchmarks.validate models/check/multilingual --name laya-multilingual --repeats 100 \
  --reference benchmarks/results/reference-v0412-fp32.json \
  --output benchmarks/results/validation-laya-multilingual-float16.json

# Single-graph, ANE and Snake bundles prepared by scripts/prepare_hub.py:
python -m benchmarks.release_validate models/hub/* --repeats 100
```

Performance and energy campaigns are documented with their commands in
[BENCHMARKS.md](BENCHMARKS.md) and [docs/ANE_BENCHMARKS.md](docs/ANE_BENCHMARKS.md). The ANE research
exporter, palettization and probes live in [experiments/ane_engineering](experiments/ane_engineering)
and are described in [docs/ANE_ENGINEERING.md](docs/ANE_ENGINEERING.md).

## Releasing

Before tagging, all of these must pass on the release commit:

1. Lint, the API audit and the full test suite (above), locally and in CI.
2. `python -m build` and `twine check --strict dist/*`.
3. A clean inference-only wheel smoke test: install the built wheel into a fresh virtual
   environment and run `scripts/wheel_smoke.py --model <bundle> ...` from outside the source tree.
4. Real-bundle fixture and stability checks (`benchmarks.validate` / `benchmarks.release_validate`).

Then set the same version in `pyproject.toml`, `laya_coreml/__init__.py` and `uv.lock` (`uv lock`),
add a [CHANGELOG.md](CHANGELOG.md) section, and push a matching tag:

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

The release workflow reruns CI, checks that the tag and all three versions agree, builds and
validates the wheel and sdist, publishes to PyPI with the `PYPI_API_TOKEN` secret, and creates a
GitHub release. Any failure stops the release before publishing. Never put credentials in files,
commits, logs or release notes.

Publishing new model bundles to Hugging Face uses `scripts/prepare_hub.py` and
`scripts/publish_adaptive_hub.py`; see [docs/RELEASE.md](docs/RELEASE.md) for the procedure and
[docs/RELEASE_0_5_1.md](docs/RELEASE_0_5_1.md) for the latest publication receipts.
