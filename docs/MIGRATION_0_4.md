# Migrating to laya-coreml 0.4.0

The compatibility target is official Laya v0.4.1, commit
`1adc59f7e371deb601fcfa18a14e25db238addcc`. Official behavior is authoritative.
0.3.0 was a selective port through laya-mlx; it did not reproduce the complete
host inference contract. 0.4.0 replaces that host layer directly from official
Laya and adds upstream assertions, source comparisons and differential tests.

## Changes from 0.3.0

| Surface | 0.4.0 behavior, matching official Laya |
|---|---|
| `load()` / `Agent()` default | English; aliases such as `ml` and `typed-decisions` resolve through the official table |
| Router defaults | `max_loaded=2`; official routing priorities and multilingual fallback |
| Choice labels | String, numeric and boolean scalar labels; official duplicate and invalid-label validation |
| `option_order` | Validated permutation, applied to the prompt, then probabilities restored to the caller's label order |
| Structured score criteria | Official JSON-rendered legend values |
| Empty questions | Empty answers and exactly `input_tokens=0`, `output_tokens=0` in usage |
| Invalid calibration scalars | Official neutral fallback/clamping and warnings; wrong temperature vector length still fails |
| Registry lifecycle | Official waiting, registration, unload, error propagation and cache behavior |
| Prediction APIs | `predict_batch`, sorting, per-call token budgets, `predict_long`, `decide`, `decide_batch` |
| Hooks | Start/end/error hooks, request rewriting, cache skips, async adapters, concurrency controls and timeouts |
| Calibration | Language overrides, fitted temperatures, binning, abstention fitting, save/load payloads |
| Routing | Hooks, batch/long/structured dispatch, language hints, shared/per-model revisions and digests |
| Tooling | Official routing CLI, HTTP protocol, MCP local/remote tools, evaluation datasets/metrics/regression gates |
| Helpers | Official shortlist/tournament, cached embedding function, email, language and framework adapters |

Existing `predict(state, questions)` calls continue to work. To keep the old
implicit multilingual choice, say `load("multilingual")` explicitly. To retain
one resident model, say `Router(max_loaded=1)` explicitly.

```python
import laya_coreml as laya

agent = laya.load("multilingual")
questions = {
    "priority": {
        "type": "choice",
        "instructions": "Choose a priority",
        "criteria": [1, 2, 3],
        "option_order": [2, 0, 1],
    }
}
results = agent.predict_batch(["Routine request", "Production is down"], questions)
scanned = agent.predict_long("long document ...", questions)
value = agent.decide(
    "Customer asks for a refund",
    {
        "type": "object",
        "properties": {"refund": {"type": "boolean"}},
    },
)
agent.save_calibration("calibration.json")
```

JSON Schema decisions need no extra dependency. Pydantic models use
`pip install 'laya-coreml[structured]'`. Loading/applying calibration and fitting
histogram bins remain inference-only. Temperature optimization and the optional
training reward helpers use PyTorch: `pip install 'laya-coreml[calibration]'`.
Framework adapters are ported from upstream; their SDK dependencies are optional.

## Core ML boundaries

These differences come from the exported model/runtime, rather than alternative
host decision rules:

* Inference uses Core ML arrays. Logical batches are packed into the graph's fixed
  batch/option capacities; padding rows are removed before official decoding.
* The effective `agent.cfg["max_len"]` is bounded by the export's context capacity.
  `predict` applies official truncation at that budget and reports it in usage.
  `predict_long` scans using that same budget. A question head that cannot fit,
  excess option count, or an explicit per-call budget exceeding actual graph
  capacity can still fail. A 96-token ANE graph remains a 96-token graph.
* Existing ANE exports support sequential layout. Parallel-layout checkpoints
  need matching general Core ML exports. Editing JSON cannot change a graph.
* FP16 Core ML and official FP32 outputs have small numerical differences; the
  fixture validates probabilities within 0.02, token identity and selected labels.
* `device="cpu"` selects CPU; `device="mps"` selects CPU+GPU. `compute_units`
  additionally exposes CPU+NE and ALL. CUDA, TileLang, torch.compile and ONNX
  selections raise an explicit error. Backend acceleration APIs, the official
  PyTorch training engine and ONNX tooling are not part
  of this Core ML distribution.
* Decision graphs do not expose encoder embeddings. `embed_fn_from_agent` raises
  `NotImplementedError`; supply an external embedder to shortlist helpers. This
  also applies to the MCP shortlist tool when it needs to prune candidates.
* Base inference still imports neither PyTorch, Transformers nor MLX.

## Artifact names, revisions and integrity

Routing metadata uses official repository names. At the loader boundary these
names select the corresponding converted artifacts:

| Official source | Core ML artifact |
|---|---|
| `convaiinnovations/laya` root | `aac6fef/laya-coreml` |
| `convaiinnovations/laya`, `subfolder="multilingual"` | `aac6fef/laya-multilingual-coreml` |
| `convaiinnovations/laya`, `subfolder="typed-decisions"` | `aac6fef/laya-typed-decisions-coreml` |

The two official standalone family repositories map to the same converted
artifacts. Explicit Core ML repository names and local bundles remain supported.
Downloaded model weights are not included in the Python wheel.

Source repositories and converted repositories have different commit histories.
`LAYA_REVISION=reviewed` and audited explicit source SHAs map to a reviewed
converted SHA. Unsupported source revisions fail rather than selecting unrelated
weights. Explicit Core ML repository revisions are forwarded unchanged.
`agent.revision` reports the actual downloaded artifact snapshot.
`expected_sha256` verifies paths in the converted bundle before model loading;
use converted artifact digests, not a nonexistent source `model.safetensors` file.

The [artifact audit](../benchmarks/results/official-v041-artifact-audit.json)
records ten source revision/subfolder combinations and the matching converted
revisions. All three families have identical source weight SHA-256 values and
`rl_agent_config.json` to both the official reviewed revisions and the current
source revisions audited for this release.

## Validation and maintenance

CI checks out the exact upstream commit above. `test_official_contract.py`
compares official host method ASTs and runs both implementations against identical
inputs and deterministic logits. It covers scalar labels, option ordering,
batching/sorting, long windows, structured decisions, hooks, error behavior and
calibration persistence. Upstream host suites run through a namespace adapter;
assertions and expected values remain unchanged. The two structured suites omit
only their ONNX-specific method-existence assertion. Backend-specific CUDA device,
PyTorch thread-pool and training CLI checks are replaced by Core ML boundary tests. Router construction mocks
are forwarded through the Core ML factory. Linux runs portable checks; macOS
also executes real Core ML conversion and inference.

The [real-model report](../benchmarks/results/official-host-v040-validation.json)
compares general bundles and the fixed L96 ANE bundle with original FP32 fixtures.
The three general models cover 189 questions; ANE covers 59 that fit its graph
(four longer questions excluded). Each graph also repeats 100 calls. This checks
port fidelity, not general task accuracy or confidence calibration quality.

The optional CrewAI SDK currently requires Rich <15 through `instructor`, while
this project's Snake demo requires Rich 15. Install `[crewai]` and `[demo]` in
separate environments; the lockfile declares that conflict explicitly. The base
inference package and the other optional integrations do not require CrewAI.


## HTTP, MCP, evaluation and command-line tools

The host tooling is ported directly from official Laya as well:

```bash
pip install 'laya-coreml[serve,mcp]'
laya-coreml-router --json "Route this request"
LAYA_PRELOAD=0 laya-coreml-serve
LAYA_PRELOAD=0 laya-coreml-mcp-server
laya-coreml-evals --help
```

`laya-coreml-router` exposes the official routing/prediction CLI. The existing
`laya-coreml convert` / `laya-coreml predict` commands remain available for Core ML
export and direct bundle use. The HTTP endpoints, request controls, authentication,
strict Jev projection, MCP tool schemas, remote forwarding, evaluation metrics and
regression gates retain their official behavior. No server is started by importing
the package. Set `LAYA_HOST=127.0.0.1` for a loopback-only HTTP service.

Device preferences accept `LAYA_DEVICE=cpu` or `mps`. Health and MCP status retain
the protocol fields; loaded agents report `coreml:<compute_units>` and PyTorch
version/CUDA fields remain null/false. These labels describe allowed compute
units; Core ML decides placement per operation. A positive `LAYA_THREADS` fails
explicitly because it cannot set Core ML's thread pools. ONNX evaluation and the
training CLI fail explicitly; use official Laya for those backend-specific tools.
