# Core ML 0.5.1 release qualification

The 0.5.0 release gate caught a native GIL-state crash on macOS 26 while the shape
cache constructed another MLModel from the same package. Version 0.5.1 reuses the
base model's compiled artifact through CompiledMLModel and releases cache instances
before their owning base models. Weights, graph/input values, cache capacity and
frequency admission policy are unchanged. The failed 0.5.0 tag was not published.

All 337 local tests pass with actual optional dependencies installed. Remote
Linux/macOS release CI passed. Model output assertions and numeric tolerances are unchanged.
A separate unchanged upstream idle-window timing assertion failed once in CI and
passed on the current revision and rerun; its evidence remains in the run history.

All three checkpoints in FP32 and FP16 pass 63/63 selected answers, full public
outputs and 100 repeated calls per combination. Graph/cache cleanup checks pass.
The six `v051-MODEL-PRECISION-validation.json` reports record the new runtime.

## FP32 performance after the fix

Same machine/checkpoints/inputs, PyTorch 2.14, upstream/Core ML/Core ML/upstream
order, five warmups and 30 timed calls per workload per process. All 18 pooled
P50 and mean-throughput comparisons improve. P95 improves for 16/18; typed-decisions
single and mixed-shape P95 regressions are retained, not discarded. Shared desktop
load is uncontrolled, so there is no universal latency or tail guarantee.

| Model | Workload | P50 speedup | Mean throughput ratio | P95 speedup |
|---|---|---:|---:|---:|
| laya | single | 1.487× | 1.461× | 1.349× |
| laya | batch8 | 1.075× | 1.080× | 1.115× |
| laya | long | 1.254× | 1.252× | 1.230× |
| laya | structured | 1.659× | 1.660× | 1.632× |
| laya | stability | 1.288× | 1.286× | 1.293× |
| laya | mixed_shapes | 1.366× | 1.369× | 1.380× |
| laya-multilingual | single | 1.866× | 1.759× | 1.483× |
| laya-multilingual | batch8 | 1.408× | 1.583× | 2.329× |
| laya-multilingual | long | 1.709× | 1.476× | 1.043× |
| laya-multilingual | structured | 1.975× | 1.965× | 1.847× |
| laya-multilingual | stability | 1.116× | 1.164× | 1.149× |
| laya-multilingual | mixed_shapes | 1.405× | 1.455× | 1.177× |
| laya-typed-decisions | single | 1.144× | 1.044× | 0.804× |
| laya-typed-decisions | batch8 | 1.053× | 1.049× | 1.059× |
| laya-typed-decisions | long | 1.112× | 1.091× | 1.056× |
| laya-typed-decisions | structured | 1.735× | 1.755× | 1.869× |
| laya-typed-decisions | stability | 1.130× | 1.087× | 1.114× |
| laya-typed-decisions | mixed_shapes | 1.152× | 1.105× | 0.984× |

Core ML process peak RSS spans 5.10–8.77 GiB in this campaign, versus 2.54–2.72 GiB
for upstream. It remains more memory-intensive; process RSS is not a complete
device/unified-memory measurement. Previous Core ML measurements reached 13.51 GiB.
Cold model loading and unseen-shape initialization are outside steady-state timing.

[Raw paired samples and summary](https://github.com/mizorewww/laya-mlx/tree/main/benchmarks/results/v051-compiled-paired)
include environment, precision, input/checkpoint/source hashes and every timing sample.
The earlier v0.4.12 report remains development evidence for the previous cache loader.
Final distribution and public-install receipts are in `benchmarks/results/release-v051.json`.

The unchanged upstream idle-window timing test was also reproduced directly against the original upstream code on the same macOS CI runner during MLX post-release checks. See [the retained comparison](https://github.com/mizorewww/laya-mlx/blob/main/docs/UPSTREAM.md#idle-window-timing). This test instability is documented separately from the successful Core ML release pipeline and real-model qualification.

Post-release CI maintenance: the idle-window fixture now measures its original 200 ms deadline on the worker, avoiding a scheduling-dependent observer assertion. The explicit test adaptation and early-unload rejection checks are documented in [the compatibility record](UPSTREAM.md#idle-window-test-adaptation). Production package code is unchanged.
