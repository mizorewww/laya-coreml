# Local benchmark results

M3 Max (40 GPU cores, 128 GB unified memory), macOS 27.2, Python 3.12.13, coremltools 9.0, MLX 0.32.2, NumPy 2.1.3. Measured 2026-09-20.

**This page measures the ordinary SDPA export**, which does not outperform MLX on these workloads. Core ML CPU+GPU is its recommended/default configuration. Automatic and CPU+Neural Engine selection preferred CPU operations in that plan.

The subsequent **[ANE graph rewrite](docs/ANE_BENCHMARKS.md)** improves short-decision speed and measured system energy against compiled MLX FP16. Its FP16/8-bit comparison, shape limits, fidelity tests and hardware evidence are reported separately; the requested 10× improvement was not achieved.

## Short typed decisions

FP16. End-to-end wall time includes prompt construction, tokenization, input arrays, synchronous model execution, calibration and result formatting. Loading and 10 warmup calls are excluded. 100 measured calls per cell; timings and the first prediction are retained in the linked JSON. Jobs ran sequentially with no competing model jobs.

| Model | Backend | 1 question P50 / P95 | 3 questions P50 | 10 questions P50 |
|---|---|---:|---:|---:|
| Multilingual 322M | [Core ML CPU+GPU](benchmarks/results/perf-laya-multilingual-coreml.json) | 11.28 / 17.38 ms | 21.67 ms | 91.20 ms |
| Multilingual 322M | [MLX GPU](benchmarks/results/perf-laya-multilingual-mlx.json) | 7.87 / 9.87 ms | 11.63 ms | 27.46 ms |
| Laya 421M | [Core ML CPU+GPU](benchmarks/results/perf-laya-coreml.json) | 13.71 / 14.31 ms | 40.11 ms | 129.80 ms |
| Laya 421M | [MLX GPU](benchmarks/results/perf-laya-mlx.json) | 13.33 / 13.73 ms | 26.24 ms | 74.38 ms |
| Typed Decisions 421M | [Core ML CPU+GPU](benchmarks/results/perf-laya-typed-decisions-coreml.json) | 13.70 / 14.38 ms | 39.58 ms | 134.60 ms |
| Typed Decisions 421M | [MLX GPU](benchmarks/results/perf-laya-typed-decisions-mlx.json) | 13.37 / 14.39 ms | 27.08 ms | 76.98 ms |

Core ML exports use **B=1**, with short sequences padded to 96 tokens; MLX uses batch_size=16 and unpadded lengths of 91 (multilingual) / 93 (the other models) for one question. Both preserve the same prompt tokens and mask added padding. Multi-question rows compare the shipped APIs: Core ML executes questions sequentially, while MLX batches them. They are not a comparison of equal batch tensor shapes. MLX uses its eager FP16 path; compile/prefix-cache optimizations are not enabled.

These are measurements from one desktop run, not a guarantee of future p95 latency. Clock scaling and background system activity were not controlled. No energy or battery measurement was performed in this ordinary-export campaign. In particular the multilingual run showed a broader latency distribution; all individual samples remain available.

## Compute-unit selection

Multilingual, one short question. GPU row uses 100 samples; other rows use 50. All predictions completed and repeated public results were identical.

| Allowed compute units | P50 | P95 | Plan preference |
|---|---:|---:|---|
| [CPU+GPU](benchmarks/results/perf-laya-multilingual-coreml.json) | 11.28 ms | 17.38 ms | 1345 GPU operations |
| [ALL](benchmarks/results/perf-multilingual-all.json) | 78.04 ms | 80.90 ms | 1318 CPU operations |
| [CPU+NE](benchmarks/results/perf-multilingual-cpu_ne.json) | 81.34 ms | 140.90 ms | 1318 CPU operations |
| [CPU only](benchmarks/results/perf-multilingual-cpu.json) | 82.54 ms | 87.59 ms | 1318 CPU operations |

No operation in these inspected plans preferred the Neural Engine. In the CPU+GPU plan, 1,345 nonconstant operations preferred GPU; 1,787 constants had no device attribution. The ALL / CPU+NE plans attributed 1,318 operations to CPU, with constants and 27 other operations unattributed. These are anticipated `MLComputePlan` assignments, not a runtime hardware trace. Allowing Neural Engine does not establish that it accelerated this model.

## Long input and startup

Multilingual, one 1024-token question, 50 samples:

| Backend | P50 | P95 |
|---|---:|---:|
| coreml | 54.73 ms | 83.31 ms |
| mlx | 51.98 ms | 68.57 ms |

Core ML model construction took 3.4–4.2 seconds in the GPU runs; the first short prediction took 313–475 ms before warmup. These are observed load/first-call times with existing OS caches, not a controlled cold-launch experiment. Load once and warm the model before an interactive loop.

## Actual Snake workload

[Raw paired traces](benchmarks/results/snake.json). Two seeds × 300 live steps. Both backends process each identical state; execution order alternates every tick. The Core ML action advances the game. Core ML uses fixed **B=3, L=64, K=4**; MLX uses batch 3 with eager FP16. Timings include planner feature construction and all three questions; exclude the other backend, game update and UI rendering.

| Seed | Steps | Score / length | Core ML decision P50 / P95 | MLX decision P50 / P95 | Actions match |
|---|---:|---:|---:|---:|---:|
| 101 | 300 | 9 / 15 | 11.89 / 15.45 ms | 11.53 / 14.96 ms | 300/300 |
| 102 | 300 | 10 / 16 | 12.26 / 13.79 ms | 11.69 / 13.59 ms | 300/300 |

**600/600 actions matched; both runs survived, with zero safety interventions.** Maximum difference in displayed move probability was 0.002. The deterministic planner and cycle safety mechanism are present in both policies. This is neither proof of unrestricted Snake intelligence nor a terminal FPS or unlimited-survival test.

## Fidelity and stability

| Export | Selected answers | Max calibrated probability error | Repeated API calls |
|---|---:|---:|---:|
| [laya](benchmarks/results/validation-laya.json) | 63/63 | 0.003307626 | 100 |
| [multilingual](benchmarks/results/validation-multilingual.json) | 63/63 | 0.001577638 | 100 |
| [typed-decisions](benchmarks/results/validation-typed-decisions.json) | 63/63 | 0.002726635 | 100 |
| [multilingual-fp32](benchmarks/results/validation-multilingual-fp32.json) | 63/63 | 0.000000632 | 100 |

All input token sequences matched upstream exactly. All repeated rounded public results were identical and finite. Action-probability error was zero on these fixtures, where the action distributions are strongly saturated; this is not a broad calibration evaluation. Process RSS is recorded, including framework caches; it is not interchangeable with MLX active-memory statistics.

The 63-question suite per model includes eight languages, 512/1024-token limits, empty input, literal mask tokens, structured criteria, 20 options and 20-question calls. The committed [reference](benchmarks/results/reference.json) contains original FP32 logits from pinned upstream Laya. Original weights and their SHA256 are recorded in every export manifest and validation report.

Failed RangeDim GPU experiments are also committed, explicitly marked `passed: false`. See [conversion findings](docs/CONVERSION.md). They are not used in the performance tables.

## Reproduce

```bash
pip install -e '.[convert,dev,compare]'
# Export the three models and fixed Snake package as shown in README.md.
python -m benchmarks.campaign --source-root /path/to/original/checkpoints
python -m benchmarks.report
```

Portable CI covers prompt/shape/trace behavior and packaging. The native Core ML conversion/prediction test and the real-checkpoint reports were executed locally on the Mac; CI does not download large model weights.


## v0.3.0 runtime regression

[Raw revalidation report](benchmarks/results/upstream-v041-validation.json) checks
four unchanged published bundles against `benchmarks/results/reference.json`.
English, multilingual and typed-decisions each match 63/63 selected answers;
the fixed L96 multilingual ANE bundle matches 59/59 fitting questions, excluding
four cases beyond its context capacity. Each bundle passes 100 identical repeated
calls (248/248 agreements, 400 stability calls total). This release does not rerun
or replace the older speed/energy measurements and makes no new throughput claim.

New parallel layout is independently tested on tiny models against upstream v0.4.1,
including all 24 option permutations, traced variable lengths and actual FP32/FP16
Core ML execution on CPU and CPU+GPU. Published bundles use sequential layout;
these tests do not claim task accuracy for newly trained parallel weights.
