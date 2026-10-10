# Laya-CoreML 0.1.0 release artifacts

Install the public runtime on Apple Silicon:

```bash
python -m pip install 'laya-coreml[demo]==0.1.0'
```

The wheel contains inference, Hub loading, the terminal Snake demo and its replay
renderer. Model weights are downloaded separately. PyTorch is an optional export
dependency; MLX is an optional comparison dependency. Neither is required to run
the released models. See the [PyPI project](https://pypi.org/project/laya-coreml/).

## Pinned model releases

All six repositories are public. Each includes the exported Core ML package,
tokenizer, configuration, model card, license, attribution and validation report.
ANE packages also contain the exact host embedding and action-head tensors.
They are standalone inference bundles.

| Hugging Face repository | Release revision |
|---|---|
| [aac6fef/laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | `fff78b2d9750c6b748fe8c90fcbf8bed0a1522a9` |
| [aac6fef/laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | `8139e9089273319512c730218903784074133187` |
| [aac6fef/laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | `39d6a9b3d0f67f06da74fbade6121ea134cbdb21` |
| [aac6fef/laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | `7503714810747a879b1e310074a9398bc573e739` |
| [aac6fef/laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | `e580bca9c1a8f5e6084afedc49cba4a166dfc6f2` |
| [aac6fef/laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | `28d24fa8d67a3264556b23391ec6c3fd98573056` |

For example:

```bash
hf download aac6fef/laya-multilingual-coreml-ane \
  --revision 39d6a9b3d0f67f06da74fbade6121ea134cbdb21 \
  --local-dir models/ane
laya-coreml-snake --model ./models/ane
```

Or load a pinned Hub snapshot directly:

```python
import laya_coreml as laya

agent = laya.load(
    "aac6fef/laya-multilingual-coreml-ane",
    revision="39d6a9b3d0f67f06da74fbade6121ea134cbdb21",
)
```

The release inventory, package hashes, shapes and complete bundle sizes are in
[hub-release.json](../benchmarks/results/hub-release.json). The W8 bundle is about
557 MB including host embeddings and tokenizer; 129 MB describes only its Core
ML body. W8 is weight palettization with FP16 compute, not integer activation
quantization.

## Model and package verification

Each published package was checked against its archived validated export before
upload. The portable ANE bundles preserve the original Core ML body and extract
six host tensors with exact element equality. Packaging-time inference then
repeated the upstream golden-fixture checks:

| Bundle | Matching selected answers | Maximum calibrated-probability drift | Repeated calls |
|---|---:|---:|---:|
| English FP16 | 63/63 | 0.003308 | 100 stable |
| Multilingual FP16 | 63/63 | 0.001578 | 100 stable |
| Typed Decisions FP16 | 63/63 | 0.002727 | 100 stable |
| Snake GPU, B3/L64 | 12/12 fitting | 0.001308 | 100 stable |
| ANE FP16, L96 | 59/59 fitting | 0.002925 | 100 stable |
| ANE W8, L96 | 59/59 fitting | 0.014393 | 100 stable |

The short exports reject fixtures beyond their input capacity. Their fitting
subsets are explicit; they are not full-context equivalence claims. All use the
same 0.02 probability-drift threshold. The general checkpoints cover 189/189
questions; the six bundles together cover 319 fitting comparisons, with overlap
between checkpoints. See [release-validation.json](../benchmarks/results/release-validation.json).

A fresh Python 3.12 environment installed the built wheel and demo extra, with
no Torch, MLX or Transformers. All six local bundles loaded and produced stable
predictions while Python socket connections were blocked. A public, pinned ANE
Hub snapshot was then loaded, followed by a second load from cache with network
connections blocked. No offline section attempted a connection. This check also
exercised the Core ML symlink materialization fix described in
[USAGE.md](USAGE.md). Results: [wheel-runtime-smoke.json](../benchmarks/results/wheel-runtime-smoke.json).

Version **0.1.0 was published to PyPI on 2026-09-20**. Both public distribution
files were downloaded and their SHA256 digests matched the local release
artifacts. A second new environment then installed `laya-coreml[demo]==0.1.0`
directly from the public PyPI index with caching disabled. The README's direct
Hub-loading example ran successfully, returned zero output tokens, and produced
the same answer on three repeat calls with network connections blocked during
prediction. Both installed CLI entry points also started successfully.
See [publication receipt](../benchmarks/results/pypi-release.json) and
[public-install inference check](../benchmarks/results/pypi-install-smoke.json).

## Reproduce the release checks

The local release gate passed **64 tests**, Ruff lint/format checks, and strict
Twine package-metadata validation. Tests include conversion semantics, bundle
cache integrity, game invariants, offline model resolution and exact replay of
every board/action in the published 855-move recording.

The build backend is pinned to Hatchling 1.31.0. Its Core Metadata 2.4 output
passes the release's Twine 6 checker; an initial build using the newer default
2.5 metadata was rejected. See [Hatchling's version history](https://hatch.pypa.io/dev/history/hatchling/)
for the default change. The published artifacts use the compatible build and
are checked again before upload. The exact wheel and source-distribution SHA256
digests are recorded in [distribution-files.json](../benchmarks/results/distribution-files.json).

From a source checkout with the original validated artifacts available:

```bash
uv sync --extra convert --extra dev --extra compare --extra research --extra demo --extra publish
.venv/bin/python scripts/prepare_hub.py
.venv/bin/python -m benchmarks.release_validate models/hub/* \
  --output benchmarks/results/release-validation.json
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
.venv/bin/python -m build
.venv/bin/twine check --strict dist/*
```

`prepare_hub.py` refuses to overwrite existing bundle directories. The standalone
wheel check runs with `scripts/wheel_smoke.py --hub-check` in a fresh environment
from outside the source tree, after downloading or preparing the six bundles.
See [SNAKE_BENCHMARKS.md](SNAKE_BENCHMARKS.md) for complete game-loop measurements
and [LAUNCH.md](LAUNCH.md) for the exact recorded media sources.


## 0.2.0 upstream synchronization

This release ports applicable prompt validation, custom noul labels, Unicode
instructions, conversation-tail truncation, option-collapse diagnostics and
answer-confidence reporting through upstream `4aa6761`. It also publishes the
temperature clamp that was present in source 0.1.1 but absent from PyPI 0.1.0.
Core ML graph signatures, weights and ANE capacity checks are unchanged.
Historical benchmark results above describe 0.1.0, not a new performance run.
Mixed-question latency and Intel Mac support have not been validated by this sync.

Push a `v<version>` tag after updating `pyproject.toml`, `laya_coreml/__init__.py`
and `uv.lock`. The Release workflow runs the reusable CI checks, verifies all
three versions against the tag, builds and validates wheel/sdist, publishes to
PyPI using the repository secret `PYPI_API_TOKEN`, and creates a GitHub release
with the same distributions. CI failure prevents publishing.


## v0.3.0 runtime feature release

Upstream runtime baseline: `1adc59f7e371deb601fcfa18a14e25db238addcc` (v0.4.1).
The published model bundles retain their original weights and sequential graphs.
New runtime APIs and parallel graph conversion ship in the Python package; no model
Hub upload is required for this release.

```bash
uv sync --extra convert --extra dev --extra demo --extra publish --extra reference
uv run --no-sync pytest -q
uv run --no-sync python -m benchmarks.release_validate \
  models/hub/laya-coreml models/hub/laya-multilingual-coreml \
  models/hub/laya-typed-decisions-coreml models/hub/laya-multilingual-coreml-ane \
  --repeats 100 --output benchmarks/results/upstream-v041-validation.json
uv run --no-sync python -m build --outdir dist/v0.3.0
uv run --no-sync python -m twine check --strict dist/v0.3.0/*
```

The test reference is upstream `laya/common.py` checked out at the pinned revision
under `.upstream/laya/common.py`. The real model reference is the existing pinned
FP32 fixture in `benchmarks/results/reference.json`; it describes sequential weights.
MacOS CI validates tiny-model Core ML conversion and runtime, while full checkpoint
validation runs locally on Apple hardware. Test failures block tag publication.
