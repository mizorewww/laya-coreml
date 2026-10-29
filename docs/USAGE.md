# Laya-CoreML usage reference

The [README](../README.md) covers installation and a first prediction. This page is the
complete reference: choosing a bundle, offline use, the Python API, routing, CLI and conversion.

Laya-CoreML runs on Apple Silicon with macOS 15+ and Python 3.11–3.13. The local
release checks use M3 Max / macOS 27.2. Older macOS releases and iOS deployment
have not been tested here. Exported ML Programs target macOS 15 / iOS 18.

```bash
python -m pip install laya-coreml
# Terminal demo and recording renderer:
python -m pip install 'laya-coreml[demo]'
```

Inference installs Core ML Tools, NumPy, Tokenizers, Safetensors and Hugging Face
Hub. It does not require PyTorch, Transformers or MLX. The optional `[convert]`
extra installs PyTorch for exporting the original checkpoints.

## Select a model

| Hugging Face bundle under `aac6fef/` | Device default | Total input capacity | Batch / options | Intended use |
|---|---|---:|---:|---|
| [laya-multilingual-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-multilingual-coreml-adaptive-fp16) | CPU + GPU | 1024 tokens | 1, 2, 4, 16 / 32 | **Recommended** general multilingual, 322M |
| [laya-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-coreml-adaptive-fp16) | CPU + GPU | 512 tokens | 1, 2, 4, 16 / 32 | English Laya, 421M |
| [laya-typed-decisions-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-typed-decisions-coreml-adaptive-fp16) | CPU + GPU | 1024 tokens | 1, 2, 4, 16 / 32 | Upstream typed-decisions checkpoint |
| [laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | CPU + GPU | 512 tokens | 1 / 32 | English Laya, 421M |
| [laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | CPU + GPU | 1024 tokens | 1 / 32 | General multilingual, 322M |
| [laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | CPU + GPU | 1024 tokens | 1 / 32 | Upstream typed-decisions checkpoint |
| [laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | CPU + GPU | 64 tokens | 3 / 4 | Batched compact Snake prompts |
| [laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | CPU + ANE | 96 tokens | 1 / 32 | Short decisions, FP16 body |
| [laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | CPU + ANE | 96 tokens | 1 / 32 | Approximate W8 palette weights, FP16 compute |

Each adaptive bundle also has an `-fp32` repository for closer numerical agreement. Adaptive
bundles combine native batch-1/2/4/16 graphs with dense sequence buckets, so batches and mixed
lengths need fewer calls; they use more resident memory than the single-graph bundles. Their
pinned revisions are listed in [RELEASE_0_5_1.md](RELEASE_0_5_1.md#public-artifacts). The original
single-graph bundles are what `Router()` and the `convaiinnovations/*` aliases resolve to; their
revisions are pinned in `laya_coreml/hub.py` and listed, with the ANE and Snake bundles, in
[RELEASE.md](RELEASE.md).

The ANE packages include their own host embedding/action weights and the unchanged
Core ML body. They do not need an original checkpoint directory. The 96-token
capacity includes the question, option descriptions, special markers and state.
These short exports use upstream state truncation at their actual graph context limit.
`predict_long` scans that budget; oversized question heads and option counts still fail.
The general-purpose models use their full checkpoint context limit;
option descriptions also use the original question-prefix budget.

## Python API

```python
import laya_coreml as laya

agent = laya.load("aac6fef/laya-multilingual-coreml")
result = agent.predict(
    "The customer asks for a refund of a duplicate payment.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, refunds, and duplicate charges.",
                "technical": "Broken features, errors, and product troubleshooting.",
                "sales": "Pricing, upgrades, and new purchases.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request?",
            "criteria": ["low", "medium", "high"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        },
    },
)
print(result["answers"])
print(result["usage"])  # output_tokens is always 0
```

`choice` returns a selected label and a probability for every label. `score`
returns the expected zero-based category index, its legend and probabilities.
`noul` returns the probability of true. Answers also include upstream confidence
and action-head probability fields. These estimates can be wrong; the library's
validation measures conversion fidelity, not application accuracy.

`predict` and `system_one` are aliases. A dictionary state is serialized using the
original input conventions. Questions are processed in insertion order. Single-graph
exports use batch one, so multiple questions need multiple model calls. Adaptive bundles batch
up to 16 questions natively, and the Snake GPU export batches three. The bidirectional encoder does not
cache contextual state across different questions.

## Download once, then work offline

```bash
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/ane
```

```python
agent = laya.load("./models/ane", local_files_only=True)
# Or use a previously downloaded shared Hub cache:
agent = laya.load("aac6fef/laya-multilingual-coreml-ane", local_files_only=True)
```

Remote IDs download before initialization unless `local_files_only=True`.
Subsequent prediction uses only local arrays and files. To reproduce an exact
remote snapshot, pass `revision="<Hub commit SHA>"`; adaptive-bundle revisions are in
[RELEASE_0_5_1.md](RELEASE_0_5_1.md#public-artifacts). The terminal game always uses local/cached weights and
fails with a download command when they are missing.

Hugging Face's shared cache stores model files as symbolic links. On the tested
macOS release, Core ML can copy those links into its temporary compiled model
and lose the weight file. The loader automatically materializes just the Core
ML package as regular files under `~/.cache/laya-coreml/packages/`, verifies its
content hash, and reuses that copy. Set `LAYA_COREML_CACHE` to change this cache
root. This uses extra disk space, not extra model downloads. A directory created
with `hf download --local-dir` already contains regular files and needs no copy.

The loader recognizes the package format and selects its default compute units.
Override using `compute_units="cpu_gpu"`, `"cpu_ne"`, `"cpu"` or `"all"` for an
explicit experiment. `cpu_ne` allows CPU work and ANE work; it does not guarantee
every operation executes on ANE. Choosing it for the ordinary SDPA export does
not turn that export into the dedicated ANE graph.

## CLI

Save the question dictionary to `questions.json`, then run:

```bash
laya-coreml predict ./models/ane --offline \
  --state 'The customer requests a refund.' --questions questions.json
```

`laya-coreml predict` accepts a local directory or a Hub ID, an optional
`--revision`, and `--compute-units`. `--offline` prevents Hub access.

## Convert from source

For the ordinary Core ML path:

```bash
pip install 'laya-coreml[convert]'
laya-coreml convert laya-multilingual models/custom-multilingual
```

The ANE research exporter lives in the Git checkout, outside the inference wheel:

```bash
git clone https://github.com/mizorewww/laya-coreml
cd laya-coreml
pip install -e '.[convert,dev,research]'
python -m experiments.ane_engineering.probe --source laya-multilingual \
  --kind body --length 96 --output models/ane96
```

See [conversion findings](CONVERSION.md), [ANE engineering](ANE_ENGINEERING.md)
and [Snake controls and recording](SNAKE_DEMO.md) for the corresponding workflows.

## Calibration

Following upstream v0.3.5, fitted calibration temperatures are clamped to `[0.5, 5.0]`. The
shipped `choice:11+` bucket is 0.1006, which would otherwise sharpen logits about 10× and report a
coin flip as near-certainty. Raw values stay available as `agent.temperature_raw` and
`agent.temperature_by_options_raw`, and a `RuntimeWarning` names each clamped bucket at load.
Fitting new temperatures needs the `calibration` extra (PyTorch); loading a calibration file does not.

## Language routing

```python
import laya_coreml as laya

router = laya.Router()
result = router.predict(state, questions)
router.register("local", "./models/my-coreml-bundle", description="Custom classifier")
result = router.predict(state, questions, model="local")
router.unload("local")
router.unregister("local")
```

`Router()` uses the published English, multilingual and typed-decisions single-graph bundles.

| Option | Default | Meaning |
|---|---|---|
| `default` | `"multilingual"` | Fallback for undecided text. Use `"english"` for mostly-English traffic. |
| `max_loaded` | `2` | Resident bundles. |
| `auto_task_detection` | `False` | Detect typed-decisions workflows automatically. |
| `compute_units`, `revision`, `local_files_only` | — | Passed to bundle loading. An attached ANE bundle keeps its CPU+NE default. |

`route()` inspects a request without loading weights; explicit `model`, `task` and `lang` override
detection. `detect_language(state, explain=True)` and `route(..., explain=True)` report why a
language was chosen. Detection is heuristic and handles acronyms, capitalization and mixed scripts
as upstream does.

Register custom bundles with `models={"custom": "./bundle"}` at construction, or with `register`,
`attach`, `preload` and `unload`. Sources are local bundles or Hub IDs. Resident bundles stay usable
while another loads, and duplicate loads share one construction. Unloading drops the router's
reference without invalidating agents held elsewhere. Core ML manages its native caches; the router
never deletes cached model files.

## Hooks and batching

`Agent` and `Router` accept `hooks=[...]` objects implementing any of `on_route`, `on_load`,
`on_evict`, `on_predict_start`, `on_predict_end` and `on_error`; async methods run to completion.
`hooks_timeout` bounds each call, but Python threads cannot be interrupted, so an overrunning hook
keeps running in the background. With `hooks_concurrent=False` a timed-out hook keeps the
serialization lock until it exits, the timeout includes waiting for that lock, and a hook whose
deadline passes while waiting is never called. `Router(hooks_raise=False)` turns hook errors and
timeouts into warnings.

`predict_batch` groups requests with identical question schemas into shared forward passes.
Grouping keeps Python key types and option order, so integer `1` and string `"1"` stay distinct.

## Confidence gate

`Agent`, `ANEAgent` and `Router` accept `min_confidence=0.7` or a bucket map such as
`{"choice:2": 0.8, "choice:3-5": 0.7, "default": 0.6}`. Buckets are `2`, `3-5`, `6-10` and `11+`,
prefixed with `choice:`, `score:` or `noul:`. Answers gain `abstention`, `abstention_threshold` and,
below threshold, `low_confidence`. The gate never replaces an answer or calls a fallback. Shipped
confidence values need held-out calibration before they can be read as a probability of being correct.

## Large choice sets

```python
result = laya.predict_tournament(router, state, questions, group_size=16)
# An external embedder maps a list of strings to an (N, D) array.
result = laya.predict_shortlist(router, state, questions, embed_fn, k=16)
```

A tournament needs no embedding model; `result["tournament"]` records finalists, original counts
and rounds. Core ML bundles do not expose encoder embeddings, so a shortlist needs your own
`embed_fn`, and `shortlist_choice` supports caller-managed ranking. Keep `group_size` or `k` within
the bundle's option capacity. Probabilities and usage describe the **final pass only**.

## Email helpers

`laya_coreml.email.clean_email_body` and `email_state` clean English, Portuguese, Spanish and
French mail footers and quoted replies; `laya.email_questions()` builds a matching question preset.
They apply only when called.

## Command-line tools and services

| Command | Extra | Purpose |
|---|---|---|
| `laya-coreml predict` / `convert` | — / `convert` | One prediction; checkpoint export |
| `laya-coreml-router` | — | Routed predictions from the shell |
| `laya-coreml-serve` | `serve` | Upstream-compatible HTTP service with body-read deadlines and batch token limits |
| `laya-coreml-mcp-server` | `mcp` | Model Context Protocol server |
| `laya-coreml-evals` | — | Evaluation and reliability calibration |
| `laya-coreml-snake` | `demo` | Terminal Snake demo ([guide](SNAKE_DEMO.md)) |

Framework integrations are the `langchain`, `langgraph`, `llamaindex`, `crewai` and `agno` extras.
`structured` adds Pydantic schemas for typed answers.

## Parallel option layout

The converter supports checkpoints trained with `"option_layout": "parallel"`: shared option
positions, option-isolating masks, and matching `position_ids`/`option_ids` inputs through
preparation, conversion and runtime. A missing setting means `sequential`; unknown layouts and
configuration/export mismatches raise. **Do not switch an existing bundle to parallel by editing its
JSON**: the graph and trained weights must match. Legacy ANE exports bake in sequential RoPE and
reject parallel configurations; use the general `convert` path for those checkpoints.

## Upstream compatibility

Host behavior follows [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) v0.4.17
(`fe9a6c35023a7d94102e7feaaeaae8857d9d1b07`); the 345-entry public API audit has no unclassified
differences ([audit](UPSTREAM_0417_API_AUDIT.json)). Native artifact mapping, compute units,
compiled execution and graph/shape caching are Core ML-specific. Encoder embeddings are explicitly
unsupported. [MIGRATION_0_4_17.md](MIGRATION_0_4_17.md) has the full mapping and exclusions;
[MIGRATION_0_4.md](MIGRATION_0_4.md) records incompatibilities carried from earlier migrations.
