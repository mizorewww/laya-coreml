# Install, download and make typed decisions

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
| [laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | CPU + GPU | 512 tokens | 1 / 32 | English Laya, 421M |
| [laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | CPU + GPU | 1024 tokens | 1 / 32 | General multilingual, 322M |
| [laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | CPU + GPU | 1024 tokens | 1 / 32 | Upstream typed-decisions checkpoint |
| [laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | CPU + GPU | 64 tokens | 3 / 4 | Batched compact Snake prompts |
| [laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | CPU + ANE | 96 tokens | 1 / 32 | Short decisions, FP16 body |
| [laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | CPU + ANE | 96 tokens | 1 / 32 | Approximate W8 palette weights, FP16 compute |

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
original input conventions. Questions are processed in insertion order. Most
exports use batch one; multiple questions therefore require multiple model calls.
The Snake GPU export batches up to three. The bidirectional encoder does not
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
remote snapshot, pass `revision="<Hub commit SHA>"`; release commit IDs are in
[RELEASE.md](RELEASE.md). The terminal game always uses local/cached weights and
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

## Official host compatibility

See [the 0.4.0 migration guide](MIGRATION_0_4.md) for batch and long-text inference,
hooks, structured decisions, calibration, defaults and artifact revision mapping.
