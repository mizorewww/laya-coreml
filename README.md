![Laya Core ML playing Snake with real local model probabilities](https://raw.githubusercontent.com/mizorewww/laya-coreml/main/docs/assets/snake-demo.gif)

Version 0.2.0 synchronizes applicable prompt and result fixes through upstream
`4aa6761` (source version 0.3.23). Both Core ML and ANE runtimes accept custom
`noul` display labels (`{"false": "no", "true": "yes"}`), validate questions with
question-specific errors, and preserve Unicode structured instructions. Long
conversation lists retain their newest tokens; strings and objects retain their
beginning. `usage` reports state truncation and, when present, collapsed options.
`answer_confidence` is the highest answer probability; existing `confidence`
semantics remain unchanged. Neither field establishes calibration accuracy.
The checkpoint temperature clamp documented below is included in this release.
Exported graph capacity limits still raise errors rather than silently clipping.

Maintenance follows upstream Laya fixes applicable to these runtimes. New runtime
features and backend-specific optimization proposals require an upstream-aligned
implementation and validation; closing an issue does not establish that its
reported behavior is fixed.


# Laya-CoreML

**Open-weight typed decisions on Apple Silicon. Core ML, Neural Engine, zero generated tokens.**

[PyPI](https://pypi.org/project/laya-coreml/) · [Hugging Face weights](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) · [中文](https://github.com/mizorewww/laya-coreml/blob/main/README.zh-CN.md)

A real Laya model plays Snake locally, with visible probabilities, score, length,
latency and safety interventions. The GIF replays a recorded Core ML run at **1× speed**.
The game uses explicit planner features and a visible cycle safety layer.

The complete active Snake loop sustained **49.1–50.0 decisions/s** across three
uncapped 600-step episodes, with zero deaths and two safety interventions.
[Game-loop timings and paced-rate limits](https://github.com/mizorewww/laya-coreml/blob/main/docs/SNAKE_BENCHMARKS.md)
include rendering serialization; terminal painting is excluded.

**One short multilingual decision: 4.98 ms P50 / 5.31 ms P95 on M3 Max with ANE FP16.**
The same experiment measured **2.78× better whole-system energy per decision** than
compiled MLX FP16. A separately validated W8 palette variant reached 4.88 ms and
3.19× energy improvement. These are single-question results, not full Snake frame
times; the requested 10× improvement was not achieved.

## Run the demo

Apple Silicon · macOS 15+ · Python 3.11–3.13.

```bash
pip install 'laya-coreml[demo]'
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/snake
laya-coreml-snake --model ./models/snake
```

Download once, then play offline. No PyTorch, Transformers or MLX is needed for
inference. The terminal needs 104 columns × 35 rows. Space pauses; ↑/↓ changes
speed; R resets; Q quits. First-time Core ML initialization can take tens of seconds.

[Controls, recording and video export](https://github.com/mizorewww/laya-coreml/blob/main/docs/SNAKE_DEMO.md)
· [Measured stable decision rates](https://github.com/mizorewww/laya-coreml/blob/main/docs/SNAKE_BENCHMARKS.md)
· [Shareable video and recording provenance](https://github.com/mizorewww/laya-coreml/blob/main/docs/LAUNCH.md)

## Ask for a decision

```bash
pip install laya-coreml
```

```python
import laya_coreml as laya

agent = laya.load("aac6fef/laya-multilingual-coreml-ane")
result = agent.predict(
    "The customer requests a refund of a duplicate payment.",
    {
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        }
    },
)
print(result["answers"]["refund"])
```

Laya returns probabilities for **choice**, ordinal **score**, and boolean **noul**
questions. There is no autoregressive decoding or generated JSON to parse. Hub
models download before initialization; subsequent predictions stay local. Pass
`local_files_only=True` to require an existing cache, or load a local directory.

Following upstream v0.3.5, fitted calibration temperatures are clamped to
`[0.5, 5.0]` before use: the shipped `choice:11+` bucket is 0.1006, which would
sharpen logits ~10x and report a coin flip as near-certainty. The checkpoint's
raw values remain available as `agent.temperature_raw` and
`agent.temperature_by_options_raw`, and a `RuntimeWarning` names every clamped
bucket at load.

The ANE bundle has a **96-token total limit**, including question, options and
state. Longer requests raise a capacity error. Use
`aac6fef/laya-multilingual-coreml` for the general-purpose 1024-token model.
[Full API, model selection and offline usage](https://github.com/mizorewww/laya-coreml/blob/main/docs/USAGE.md).

## Measured on M3 Max

40-core GPU, 128 GiB, macOS 27.2. One 91-token question padded to 96, including
prompt preparation, tokenization, arrays, synchronous inference, calibration and
formatting. Loading and warmup are excluded. MLX enables compile, prefix caching
and shape buckets. Six alternating 20-second blocks per implementation produced
**65,598 stable calls**.

| Metric | Compiled MLX FP16 | Core ML ANE FP16 | Core ML ANE W8 |
|---|---:|---:|---:|
| P50 / P95 | 6.94 / 7.39 ms | **4.98 / 5.31 ms** | **4.88 / 5.23 ms** |
| Mean system power estimate | 61.39 W | 30.75 W | 27.39 W |
| System energy / decision | 0.4288 J | **0.1540 J** | **0.1344 J** |
| Speed gain | 1× | **1.39×** | **1.42×** |
| System energy gain | 1× | **2.78×** | **3.19×** |

Energy uses direct SMC PSTR sensor readings, with raw samples and explicit anomaly
rejection. This is an estimate with sensor and background-load uncertainty.
**Speed gain × average power ratio = energy gain**; multiplying energy by speed
again would double-count time. The W8 variant compresses weights while retaining
FP16 compute. It is approximate, and its package-size reduction is not a speed ratio.

[Speed, energy and hardware evidence](https://github.com/mizorewww/laya-coreml/blob/main/docs/ANE_BENCHMARKS.md)
· [Raw measurements](https://github.com/mizorewww/laya-coreml/tree/main/benchmarks/results).

## Available checkpoints

| Hugging Face bundle | Default engine | Capacity | Purpose |
|---|---|---:|---|
| [Laya 421M](https://huggingface.co/aac6fef/laya-coreml) | CPU + GPU | 512 tokens | Original English model |
| [Multilingual 322M](https://huggingface.co/aac6fef/laya-multilingual-coreml) | CPU + GPU | 1024 tokens | General multilingual decisions |
| [Typed Decisions 421M](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | CPU + GPU | 1024 tokens | Original specialized checkpoint |
| [Snake GPU](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | CPU + GPU | B3 / L64 | Batches the three compact game questions |
| [Multilingual ANE](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | CPU + ANE | B1 / L96 | Short decisions, FP16 |
| [Multilingual ANE W8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | CPU + ANE | B1 / L96 | Optional approximate palette compression |

Every bundle includes tokenizer/configuration, model card, provenance, checksums
and packaging-time validation. ANE bundles also include the exact original host
embedding/action tensors they need. No original training checkout is required.

## Port fidelity and limits

The three general-purpose FP16 checkpoints match upstream selected answers on
**189/189 validation questions**. Each passes 100 repeated calls. ANE FP16 L96
passes **59/59 fitting questions**, with maximum calibrated-probability drift
0.002925; W8 passes the same subset with drift 0.014393 under an unchanged 0.02
gate. Six- and four-bit experiments failed that gate and are not published weights.
These are conversion-fidelity fixtures, not proof of general task accuracy.

A separately exported FP16 ANE L1024 graph passes the complete **63/63** fixture,
but an actual 1024-token request takes about **91.7 ms** in its serial screen.
The short ANE result does not establish a long-context advantage. A 600-step
paired Snake check matches **600/600 actions**, with zero deaths and zero shield
interventions; the current ANE adapter's three sequential calls do not establish
a consistent full-game speedup over compiled MLX.

The ordinary SDPA Core ML export and the ANE graph are different implementations.
The ordinary export defaults to CPU+GPU after unrestricted RangeDim GPU shapes
failed local fidelity checks. Changing its device setting alone does not reproduce
the ANE result. The ANE rewrite uses BC1L activations, 1×1 projections and per-head
attention; its plan and a separate Instruments trace support Neural Engine work.
CPU still handles input/output boundaries.

## Documentation and reproducibility

- [Install and Python/CLI API](https://github.com/mizorewww/laya-coreml/blob/main/docs/USAGE.md)
- [Snake demo and media](https://github.com/mizorewww/laya-coreml/blob/main/docs/SNAKE_DEMO.md)
- [Release artifacts and pinned Hub revisions](https://github.com/mizorewww/laya-coreml/blob/main/docs/RELEASE.md)
- [General Core ML benchmarks](https://github.com/mizorewww/laya-coreml/blob/main/BENCHMARKS.md)
- [ANE engineering experiments](https://github.com/mizorewww/laya-coreml/blob/main/docs/ANE_ENGINEERING.md)
- [Mathematical investigation of the 10× target](https://github.com/mizorewww/laya-coreml/blob/main/docs/ANE_MATH.md)
- [Conversion issues and supported shapes](https://github.com/mizorewww/laya-coreml/blob/main/docs/CONVERSION.md)

To export yourself, install `laya-coreml[convert]` and run
`laya-coreml convert laya-multilingual models/custom`. ANE research conversion,
compression and benchmark scripts live in the Git checkout. The inference wheel
contains the portable runtime and optional terminal demo.

Apache-2.0. Independent port of [Laya](https://github.com/NandhaKishorM/laya), by
Convai Innovations and contributors, building on the [MLX sibling project](https://github.com/mizorewww/laya-mlx).
Not an official Convai Innovations or Apple release. See
[NOTICE](https://github.com/mizorewww/laya-coreml/blob/main/NOTICE).


## Upstream v0.4.1 runtime port (v0.3.0)

This release ports the runtime features at upstream `1adc59f` to Core ML. Model
weights remain separate downloads. The runtime still needs no PyTorch, Transformers,
or MLX; PyTorch is used only for conversion and reference testing.

```python
import laya_coreml as laya

router = laya.Router(max_loaded=2)
result = router.predict(state, questions, min_confidence=0.7)
router.register("local", "./models/my-coreml-bundle", description="Custom classifier")
result = router.predict(state, questions, model="local")
router.unload("local")
router.unregister("local")
```

`Router()` uses the published English, multilingual and typed-decisions **Core ML**
bundles. It defaults to multilingual when the language is undecided. Use
`Router(default="english")` to choose English as the fallback. `route()` inspects a
request without loading weights; explicit `model`, `task`, and `lang` override
inference from the input. Automatic typed-workflow detection is opt-in through
`auto_task_detection=True`. Language detection includes the upstream acronym,
capitalization and mixed-script fixes.

Pass `models={"custom": "./bundle"}` at construction, or use `register`, `registered`,
`attach`, `preload`, and `unload`. Sources are local Core ML bundles or Hub repository
IDs. `compute_units`, `revision`, and `local_files_only` are passed to bundle loading.
An attached ANE bundle uses its usual CPU+NE default when `compute_units` is omitted.
Resident-model access and status reads remain available during a cold load; duplicate
loads share construction. Replacing/unregistering a source prevents an old in-flight
build from becoming resident. Unload drops router references without invalidating
agents held by callers or ongoing predictions. Core ML manages native caches; the
router collects unreachable Python objects and never deletes cached model files.

`Agent`, `ANEAgent`, and `Router` accept `min_confidence=0.7` or a bucket mapping such
as `{"choice:2": 0.8, "choice:3-5": 0.7, "default": 0.6}`. Answers gain `abstention`,
`abstention_threshold`, and, when below threshold, `low_confidence`. The gate preserves
the answer and does not automatically call a fallback. Shipped confidence values
still require held-out calibration for any correctness-probability interpretation.

```python
result = laya.predict_tournament(router, state, questions, group_size=16)
# Optional external embedder: maps a list of strings to an (N, D) numeric array.
result = laya.predict_shortlist(router, state, questions, embed_fn, k=16)
```

Tournament elimination handles large choice sets without an embedding model.
`result["tournament"]` records finalists, original counts, and rounds. Probabilities
and usage describe the **final pass only**. Choose `group_size` / shortlist `k` within
the bundle's exported option capacity. `shortlist_choice` is also available for
caller-managed ranking; the Core ML graph does not expose an encoder embedder.

Email helpers (`laya_coreml.email.clean_email_body`, `email_state`, `email_questions`)
include English, Portuguese, Spanish and French cleanup, including the French device
footer fix. Cleaning is explicit and never silently applied to ordinary input.

### Parallel option layout

The converter supports checkpoints trained with `"option_layout": "parallel"`:
shared option positions, option-isolating encoder masks, local attention measured in
position IDs, and the position-free decision head. Preparation, fixed/enumerated
shapes, conversion and runtime use matching `position_ids` and `option_ids` inputs.
Missing configuration means `sequential`; unknown layouts and configuration/export
mismatches raise an error. **Do not switch an existing bundle to parallel by editing
its JSON**: the graph and trained weights must match the layout.

FP32 and FP16 exports are tested on CPU and CPU+GPU, with varying input lengths and
option order, against upstream v0.4.1. Legacy ANE exports have sequential RoPE baked
into the graph and reject parallel configurations; use the general `convert` path
for parallel checkpoints. Existing ANE packages receive the shared prompt, gating,
routing and decision-helper features without requiring reconversion. No new parallel
pretrained weights or parallel ANE acceleration claim is made by this release.

Validation covers all 24 permutations of four options, a sequential sensitivity
control, pinned upstream numerical parity, real Core ML export/load execution,
registry races, routing and helper semantics. CI runs portable tests on Linux and
actual Core ML tests on macOS. Reproduce with:

```bash
uv sync --extra convert --extra dev --extra demo --extra publish --extra reference
# Check out upstream 1adc59f into .upstream for reference parity.
uv run --no-sync pytest -q
```

The three general-purpose bundles match all 189/189 selected answers from the
original FP32 golden fixture. The fixed L96 ANE bundle matches all 59/59 questions
that fit its capacity (four long questions are excluded). Each bundle passes 100
identical repeated calls: 248/248 agreements and 400 stability calls in total.
This measures port fidelity on the fixture, not general task accuracy.
See [the release validation report](benchmarks/results/upstream-v041-validation.json).
