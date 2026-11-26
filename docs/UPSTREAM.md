# Upstream compatibility

Host behavior follows [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) **v0.5.5**
(`8f445c623453f7514f30fe958a73878296aea9af`). CI checks that commit out into `.upstream`, runs its
unchanged host suites against this package on Linux and macOS, and audits the public API against
it. This page is the current record; what each release changed is in the
[changelog](../CHANGELOG.md). The policy for changes is in [AGENTS.md](../AGENTS.md).

## What is native

Prompts, validation, routing, hooks, calibration, evaluation, the routing CLI, HTTP, MCP and the
framework integrations are upstream's host code under the `laya_coreml` namespace, with upstream
`cli.py` as `router_cli.py` and the host half of `agent.py` in `runtime.py`. The differences below
come from the exported model and its runtime, not from different decision rules.

- Inference uses Core ML arrays. Logical batches are packed into the graph's fixed batch and
  option capacities; padding rows are removed before upstream decoding. Adaptive bundles hold
  B=1/2/4/16 graphs of the same checkpoint and a bounded shape cache (`shape_cache_size`).
- `agent.cfg["max_len"]` is bounded by the export's context capacity. `predict` truncates at that
  budget and reports it in `usage`; `predict_long` scans with it. A question head that cannot fit,
  too many options, or a per-call budget above the graph's capacity still fails. A 96-token ANE
  graph remains a 96-token graph.
- ANE exports support the sequential option layout only. Parallel-layout checkpoints need a
  matching general export; editing JSON cannot change a graph.
- FP16 Core ML and upstream FP32 differ slightly; the fixture checks probabilities within 0.02,
  token identity and selected labels.
- `device="cpu"` selects CPU and `device="mps"` selects CPU+GPU. `compute_units` also offers CPU+NE
  and ALL. Health and MCP status report `coreml:<compute_units>`; PyTorch and CUDA fields stay
  null. A positive `LAYA_THREADS` fails because Core ML's thread pools cannot be set.
- `backend` is `"coreml"`, `backend_object` is `None`, `dtype_for()` returns the export precision,
  and `set_backend()` raises. TileLang `accelerate()` returns `False` (raises with `strict=True`).
- Decision graphs do not expose encoder embeddings. `embed_fn_from_agent` names missing inputs,
  then raises `NotImplementedError`; pass an external embedder to the shortlist helpers and the
  MCP shortlist tool.
- Base inference imports neither PyTorch, Transformers nor MLX. Temperature fitting and the reward
  helpers use PyTorch through the `calibration` extra.
- The `crewai` and `demo` extras need different Rich majors and cannot share an environment.

## Artifact names, revisions and integrity

Routing metadata uses upstream repository names. The loader maps them to converted artifacts:

| Upstream source | Core ML artifact |
|---|---|
| `convaiinnovations/laya` root | `aac6fef/laya-coreml` |
| `convaiinnovations/laya`, `subfolder="multilingual"` | `aac6fef/laya-multilingual-coreml` |
| `convaiinnovations/laya`, `subfolder="typed-decisions"` | `aac6fef/laya-typed-decisions-coreml` |

The two upstream standalone family repositories map to the same artifacts. Explicit Core ML
repository names and local bundles also work. Weights are not in the wheel.

Source and converted repositories have different histories. `LAYA_REVISION=reviewed` and audited
source SHAs map to a reviewed converted SHA; unsupported source revisions fail instead of selecting
unrelated weights. Explicit Core ML revisions are forwarded unchanged, and `agent.revision` reports
the snapshot actually downloaded. `expected_sha256` verifies paths in the converted bundle before
loading, so use converted digests, not a source `model.safetensors`. The
[artifact audit](../benchmarks/results/official-v041-artifact-audit.json) records the audited
source and converted revisions.

## Not included

These exist upstream and are refused by name here rather than emulated:

| Upstream feature | Here |
|---|---|
| PyTorch/CUDA/TileLang backends, `torch.compile` controls, ONNX (`onnx_path`, `ONNXAgent`, `--onnx` evaluation) | `ValueError` naming official laya |
| Training and fine-tuning CLI | Not shipped |
| vegaml engine: `Agent(vegaml=...)`, `load(vegaml=...)`, `VegaAgent`, `laya.vega`, `laya.video`, `laya.policy` | `vegaml=None`/`False` are accepted; any other value raises `ValueError`. `VegaAgent` is not exported and `vegaml`/`cryptography` are not dependencies |
| vega as the default engine (0.5.0): `Agent()` / `load()` with no checkpoint named, through `_AgentMeta` | `wants_vega` is ported and returns `True` only for an explicit `vegaml=` value, which is refused. With no checkpoint named, `Agent()` and `load()` load the classic checkpoint, as upstream does when vegaml is not installed, without upstream's "vegaml is not installed" warning, whose install hint does not apply here |
| `laya.Learner`, `laya.Escalator`, `laya.vllm_teacher` and `laya.learn` (vega's continuous learner and its teachers, 0.5.0 and 0.5.2; from 0.5.4 imported from the separately licensed `vegaml-learn` package) | Not exported |
| `laya.policy` manifest hardening (0.5.1): `questions.json` must be hashed, `key_id` must be a string, deeply nested JSON and non-object provenance are refused | `laya.policy` is a vegaml module and is not shipped (see the vegaml row) |
| Router vega lane built by the Router (`Router(vega=True)`, a reader name, or `"auto"` with vegaml installed) | `"auto"` never turns the lane on. `vega=True` raises `ImportError`. A reader name routes like upstream but raises `ImportError` when loaded |
| CPU sliding-window and packed-head fast paths in `DecisionModel`, `LAYA_CUDA_AMP` parsing | PyTorch execution details; exported graphs are unchanged |
| `agno<4` | The extra still pins `agno<2`; newer Agno majors were not tested here |

The lane itself is ported: `Router(vega=agent)` accepts an already built agent, for example
official `laya.Agent(vegaml=True)`, and then routes `image=`/`audio=`/`video=`, states past
`vega_over_tokens` and questions with fitted heads to it exactly as upstream does. Without a lane,
media arguments raise `ValueError`, and `laya-coreml-serve` and `RemoteRouter` refuse media as
upstream.

## Deliberate divergences

These fix defects that upstream v0.5.5 still has, under the policy in [AGENTS.md](../AGENTS.md).
Both were first fixed in laya-mlx ([#26](https://github.com/mizorewww/laya-mlx/pull/26)). Each
keeps upstream's public API and is pinned by its own regression tests.

| Behavior | Upstream v0.5.5 | Core ML | Tests |
|---|---|---|---|
| `Router.predict_batch` grouping | Schemas serialized with `json.dumps`, so key `1` and `"1"` (or `True` and `"true"`) share a group and later requests receive the first request's key types | Grouping signature keeps Python key types and insertion order; equal schemas still share a batch | `tests/test_batch_key_types.py` |
| Serial hook timeout (`hooks_concurrent=False`) | The caller releases the lock when a hook times out while the hook is still running, so later serial hooks can overlap it | The worker holds the lock until the hook exits; `hooks_timeout` includes the lock wait, and a hook whose deadline passes while waiting is never invoked | `tests/test_hook_serialization.py` |

When an upstream release fixes one of these, adopt upstream's version and remove the row.

## v0.5.4 → v0.5.5 file inventory

`git diff v0.5.4 v0.5.5` (`60a6ce91..8f445c62`, 4 commits) touches 4 files (45 additions,
108 deletions). No host code or test changed.

| Upstream file | Port | Scope |
|---|---|---|
| `README.md` | — | Upstream documentation: the paper citation moved to the top, duplicated sections and the top animation removed. |
| `laya/__init__.py`, `pyproject.toml`, `nix/package.nix` | — | Version bumps to 0.5.5. |

The v0.5.3 → v0.5.4 release moved `laya.Learner`, `laya.Escalator` and `laya.vllm_teacher` to the
separately licensed `vegaml-learn` package; they are vegaml-only and not shipped, and
`tests/test_upstream_054.py` pins their absence.
The v0.5.2 → v0.5.3 release ported upstream's unaccented and closed-class Spanish and Portuguese
stopwords and the elided Italian `qual` into `laya_coreml/lang.py`; `tests/test_upstream_053.py`
compares the tables with the pinned checkout. Its vegaml label translation and ONNX changes are
not shipped.
The v0.5.1 → v0.5.2 release re-exported vegaml's `laya.Escalator` and `laya.vllm_teacher`, which
are not exported; `tests/test_upstream_052.py` pins their absence.
The v0.5.0 → v0.5.1 release changed only vegaml modules and the `_resolve_log_level` docstring in
`laya/serve.py`; `tests/test_upstream_051.py` pins the resolver.
The v0.4.37 → v0.5.0 release ported `wants_vega` and `load(model_id_or_path=None)`; see the 0.8.0
entry of the [changelog](../CHANGELOG.md).
The v0.4.35 → v0.4.37 releases changed only upstream's README, version strings and packaging.
The v0.4.34 → v0.4.35 release changed only upstream's TypeScript SDK and added two `hooks_api`
export checks, which run unchanged. The v0.4.17 → v0.4.34 host merge is described in the 0.7.0
entry of the [changelog](../CHANGELOG.md).

Changes to upstream's other SDKs, training and research tools, notebooks and deployment files are
outside this package.

## How upstream assertions run

`tests/test_official_contract.py` compares upstream host method ASTs, runs both implementations on
identical inputs and deterministic logits, and runs 35 pinned upstream suites through
`scripts/run_official_test.py`. The adapter rewrites the `laya` namespace and mock targets;
assertions, expected values and public model names are upstream's. v0.5.1 through v0.5.5 add no
suites run here; v0.5.3's `test_router.py` cases run in the existing `router` suite, and the
`test_policy.py`, `test_vega.py` (including v0.5.4's learner messages) and ONNX changes of these
releases need vegaml or ONNX.

Excluded as backend-specific:

- ONNX assertions in the hooks, hooks API, evals API, structured and confidence suites; `VegaAgent`
  in the hooks API signature loop.
- Six TileLang/PyTorch padding checks in the confidence suite.
- Torch device, thread and status checks in the serve and MCP suites, replaced by Core ML boundary
  tests.
- The training CLI help assertion.
- One `router_vega` test that needs vegaml installed to turn the lane on by itself, and one
  `langchain_native_policy` test that imports `laya.vega`.

Upstream's full calibration, revision-pinning, long-prediction, batch-prediction,
temperature-loading and shortlist suites mix host assertions with Torch or ONNX fixtures and are
not run whole. Their changed host behavior is covered by `tests/test_upstream_0417.py`,
`tests/test_upstream_0434.py` and the native tests. Some upstream assertions read upstream's docs
and examples, so the pinned checkout must be complete.

### Idle-window test adaptation

Upstream's idle-unload fixture uses a 200 ms threshold and expects an observing thread's
`Event.wait(0.1)` to return before the unload. On a busy macOS runner the observer can resume after
the full 200 ms, so the observation fails even when the unload happened on time; the same assertion
failed against unmodified upstream
([run](https://github.com/mizorewww/laya-mlx/actions/runs/39355649931)).

The adapter therefore records inference completion on the worker, and `BlockingRouter.unload`
asserts that at least 0.2 seconds elapsed before it signals `unloaded`. An early unload raises and
also fails the existing bounded wait. All eight batch, cancellation and failure combinations, the
no-overlap assertion and the eventual-unload timeouts remain; there is no retry, skip, xfail,
larger threshold or runtime change. `tests/test_idle_test_adapter.py` drives the transformed
fixture with a controlled clock: unloading at 100 or 199 ms must fail, at 201 ms or later must
pass. The transformer fails if the fixture's expected markers disappear.

## Validation for 0.10.2

Automated port on a macOS runner (Apple silicon), on the release source:

| Check | Result |
|---|---|
| `python -u -X faulthandler -m pytest -q` | 369 passed, including 35 upstream suite wrappers, `tests/test_upstream_054.py` and real conversion and inference of tiny models |
| `scripts/audit_public_api.py --package laya_coreml` | 353 surface entries, 0 unclassified differences |
| `ruff check .`, `ruff format --check .` | Passed |
| `python -m build`, `twine check --strict` | Passed for wheel and sdist |

Not re-run for this release: real-bundle validation (`benchmarks.validate`,
`benchmarks.release_validate`) and the clean inference-only wheel smoke test; they run on Apple
hardware after the automated release. Model bundles and exported graphs are unchanged since 0.7.0,
which re-checked the three published adaptive FP16 bundles against the golden fixture (63/63
selected answers, 100 stable calls each). The golden fixture was generated from upstream v0.4.12.
Performance and energy were not re-measured.

## Reproduce

```bash
uv sync --extra convert --extra dev --extra reference --extra serve --extra mcp --extra structured
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout 8f445c623453f7514f30fe958a73878296aea9af
python scripts/audit_public_api.py --package laya_coreml
python -X faulthandler -m pytest -q
```

Linux runs the portable tests; conversion and inference tests need macOS. Real-bundle checks are
described in [CONTRIBUTING.md](../CONTRIBUTING.md).
