> Current source target: upstream v0.4.17 (`fe9a6c35023a7d94102e7feaaeaae8857d9d1b07`).
> See [the current migration](MIGRATION_0_4_17.md) for behavior, inventory and validation.
> The sections below preserve earlier migrations and qualification evidence.

# Migrating to laya-coreml 0.4.0

## Historical v0.4.12 synchronization

The source target of this historical synchronization was official v0.4.12, commit
`ffc142012af63d015fa19f05f29f9c5ff329a2cc`. This source update is not a published
release. The 341-entry public API audit and real-logit calibration compatibility
campaign have passed. Real optional-framework checks cover LangChain/LangGraph,
LlamaIndex, CrewAI and Agno, including the LlamaIndex compatibility repair below.
Final native performance and package qualification is recorded in
`UPSTREAM_0412_PROGRESS.json`; historical experiments remain separately labeled.
The host update includes calibration correctness fixes, schema singleton constraints,
shortlist/tournament ordering, OptionStabilityHook, agreement/tournament evaluation,
Agno/tool integration and HTTP/MCP updates.

Native agents now expose `config` as the checkpoint `cfg` mapping. Upstream's new
OptionStabilityHook reads `agent.config`; the alias makes it detect native parallel
layouts and avoid meaningless reorder probes without changing checkpoint weights.
Core ML device and thread-pool boundaries remain as described below. The test adapter
excludes ONNX-only runner assertions and PyTorch backend-switching properties, keeps host
assertions unchanged, and identifies the unsupported training CLI by syntax rather than
line numbers. Native boundary tests retain explicit refusal checks.

The strict reference is `benchmarks/results/reference-v0412-fp32.json`: both raw
logits and public outputs disable upstream MPS autocast. The older
`reference-v0412.json` retains correct FP32 raw logits, but its large-batch public
outputs used upstream row-gated autocast; it is retained only for traceability.
Current reports compare full public structures, exact categorical/integer values,
and precision-specific float tolerances. They are finite fidelity checks, not a
claim that the classifier is always correct on arbitrary real-world questions.

## Previous v0.4.1 migration record

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

The v0.4.12 workload matrix exposed repeated Core ML shape-switch overhead:
trimming every fixed-B chunk independently alternated sequence shapes inside one
logical request. The native tensor bridge now pads all chunks to the logical
batch's width before selecting an export bucket. Real token masks, position IDs,
option IDs and marker masks are preserved; extra padding remains invisible to
attention. The host algorithms and returned row order are unchanged.

For the English checkpoint, a two-question request improved from 799.7 ms to
32.5 ms P50 (20 measured calls). The old bundle and a fresh integer_positions_v2
export both exhibited the original slowdown, so re-exporting alone did not fix
it. That B=1 graph was slower than upstream on heterogeneous batch and
stability-hook workloads. The adaptive bundles described below supersede this
initial experiment; existing single-graph artifacts retain their capacity limits. Reproduction and full nested output
comparisons are in the sibling `laya-mlx/benchmarks/features.py` and
`benchmarks/features_report.py`; JSON evidence lives under
`laya-mlx/benchmarks/results/v0412-features/`.

Both packages now expose the upstream `agno` and `langgraph` installation extras.
`scripts/sdk_interop.py` runs the pinned standalone Python SDK's original live
HTTP assertions against native checkpoints. It replaces only fixture checkpoint
construction and namespace imports; transport, strict mode, tournament, routing,
authentication and request-limit assertions remain upstream code.

Native adaptive batch bundles are an additive artifact/runtime capability. They
store separately exported graphs of the same checkpoint and preserve the
upstream logical batch/window semantics. Extra graph loading is lazy and bounded
by the manifest's finite set of capacities. Context-manager cleanup releases the
additional model references. See `docs/CONVERSION.md` for the local assembly
command and the memory/storage tradeoff.

Both native runtimes now expose `warmup(shapes=None)` with the upstream defaults
`((1, 64, 3), (4, 128, 5))`, checkpoint token cap, CLS fill and elapsed-seconds
return. It runs native tensor forwards directly and does not invoke prediction
hooks. MLX `compile=True` now honors `compile_warmup`: automatic warmup is best
effort with a warning on failure, while explicit `warmup()` propagates errors,
matching the upstream distinction. Core ML warmup can also load adaptive batch
graphs; those allocations are part of the documented resident-memory tradeoff.

The final surface audit also covers backend introspection. `backend` identifies
`"coreml"` (`"mlx"` in the sibling port), and `backend_object` is `None` because no
upstream PyTorch Backend object is installed. `dtype_for(rows)` returns the native
dtype / declared export precision; native inference does not use upstream's
row-gated MPS autocast. `set_backend(...)` explicitly raises `NotImplementedError`
instead of falling back to PyTorch. The TileLang-only `accelerate()` follows the
upstream non-CUDA behavior (`False`, or a CUDA requirement error with
`strict=True`); `deaccelerate()` leaves native execution in place. Change native
execution settings by constructing an Agent with the supported native controls.

`scripts/audit_public_api.py` compares public call signatures, original parameter
defaults, inherited host methods and root exports to the pinned upstream. Its
341 entries classify native tensor/model replacements and backend-only modules
explicitly. This surface check complements, rather than replaces, the unchanged
upstream assertion suites and real-model differential tests.

Real-logit calibration checks now exercise temperature fitting with held-out ECE,
histogram binning, atomic save/load and prediction restoration for all three
checkpoints on both backends. Fitted maps match the official fitter exactly when
fed identical native logits. The compatibility fixture uses deterministic
synthetic targets and repeats records to reach the official bucket floors;
these results are not estimates of calibration quality on deployment data.


Mixed-shape performance exposed a large Core ML reshape cost. The final native
cache reuses each physical graph's original instance for its current shape and
admits up to four additional instances by aged frequency. Keys contain the graph
identity and all input shapes; each request still predicts fresh tensor values.
Long-tail misses use the base instance instead of repeatedly loading models.
`shape_cache_size=0` disables the extra cache, and context-manager exit clears it.
Dense sequence buckets include 48, 80 and 112 tokens. B=1/2/4/16 adaptive bundles
are exported in both FP32 and FP16 for all three checkpoints.

The cache trades resident memory for latency. Initial experiments, including a
rejected thrashing LRU policy, remain in benchmark results for traceability.
`v0412-torch27-matrix`, `v0412-phase-chain` and
`v0412-abba-batch-torch27` used the conversion environment's PyTorch 2.7 baseline;
they must not establish final speedup over the PyTorch 2.14 reference environment.
The final sequential campaign uses PyTorch 2.14 for upstream, MLX's own runtime,
and Core ML's conversion/inference environment, with versions recorded per run.
The explicit `reference-v0412-fp32-torch214.json` golden is byte-identical to the
previous pure-FP32 golden (which already used PyTorch 2.14); only the performance
baseline environment needed correction.

Actual optional-framework smoke tests use LangChain/LangGraph, LlamaIndex, CrewAI and Agno with a deterministic local decision runner and forbidden network access. Both native packages pass all eight integration checks. The LlamaIndex adapter fixes an upstream compatibility bug: current llama-index-core exports SingleSelection rather than ToolSelection and requires prompt-mixin abstract methods. Selectors expose no LLM prompt templates; configure typed instructions directly. Both single and multi selectors now inherit the actual framework BaseSelector. See benchmarks/results/v0412-real-integrations.json for versions and scope. Both dependency lockfiles now include the new extras.

Final local qualification is complete for the pinned upstream and measured workloads. See [the consolidated evidence](QUALIFICATION_0_4_12.md) for exact scope, paired timing, original slower runs, package checks and resource limitations. The migration remains unpublished.
