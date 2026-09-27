![Laya Core ML playing Snake with real local model probabilities](https://raw.githubusercontent.com/mizorewww/laya-coreml/main/docs/assets/snake-demo.gif)

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

> **PyPI note:** the wheel currently on PyPI (`0.1.0`) predates the clamp
> above - it applies fitted temperatures unclamped and emits no
> `RuntimeWarning`. Until `0.1.1` is published, install from source for the
> documented behaviour:
>
> ```bash
> pip install 'laya-coreml @ git+https://github.com/mizorewww/laya-coreml'
> ```

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
