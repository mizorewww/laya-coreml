# 0.5.1 — upstream v0.4.12 migration and compiled-instance stability

This release includes the upstream v0.4.12 migration. The initial
0.5.0 tag failed the macOS release gate and was never published to PyPI.
Its tag remains unchanged for traceability; 0.5.1 is the corrected release.

The bounded shape cache now loads extra execution instances from the owning
model's already compiled artifact instead of repeatedly compiling its package.
The previous route triggered a fatal coremltools GIL-state error on macOS 26
when admitting additional shapes. Instance cleanup now precedes releasing the
base models that own the compiled directories. Cache size, admission policy,
input values, graph weights and host outputs are unchanged.

The macOS CI keeps uncaptured native-stage diagnostics and runs all original
conversion and inference assertions. No tests are skipped to publish this fix.
See the publication receipt and new real-model qualification results for final
validation status. Existing model repositories remain unchanged; new adaptive
FP16/FP32 bundles use separate repositories.

Final validation: 337 tests; 3 checkpoints × FP32/FP16 × 63 questions, 100 stable calls per combination. Paired FP32 performance improves P50 and mean throughput on 18/18 workloads and P95 on 16/18; the two tail regressions and memory cost remain documented in [qualification evidence](https://github.com/mizorewww/laya-coreml/blob/main/docs/QUALIFICATION_0_5_1.md).

## Public artifacts

[PyPI 0.5.1](https://pypi.org/project/laya-coreml/0.5.1/) wheel and sdist hashes match the GitHub release assets. The public wheel was reinstalled into an inference-only environment and passed offline checks against all three packaged FP16 models.

All six repositories below are public. Every model manifest file size and SHA256 was checked against Hub metadata or downloaded bytes. The English FP16 bundle was additionally downloaded anonymously at its pinned revision and exercised through public-wheel inference, batch sizes 17 and 2, warmup, repeated calls and offline cached reload. All six local bundles have the real-model correctness/stability evidence described above.

| Model bundle | Immutable revision |
|---|---|
| [aac6fef/laya-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-coreml-adaptive-fp16) | [`97bcf6b78e3489c1ebbd54001d637456f3eb381c`](https://huggingface.co/aac6fef/laya-coreml-adaptive-fp16/tree/97bcf6b78e3489c1ebbd54001d637456f3eb381c) |
| [aac6fef/laya-coreml-adaptive-fp32](https://huggingface.co/aac6fef/laya-coreml-adaptive-fp32) | [`974246b3a87d03e58cb6b95180ea349c2ec99f5a`](https://huggingface.co/aac6fef/laya-coreml-adaptive-fp32/tree/974246b3a87d03e58cb6b95180ea349c2ec99f5a) |
| [aac6fef/laya-multilingual-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-multilingual-coreml-adaptive-fp16) | [`d7927f83d2fd324b2e9cfed72cc2277e6a255387`](https://huggingface.co/aac6fef/laya-multilingual-coreml-adaptive-fp16/tree/d7927f83d2fd324b2e9cfed72cc2277e6a255387) |
| [aac6fef/laya-multilingual-coreml-adaptive-fp32](https://huggingface.co/aac6fef/laya-multilingual-coreml-adaptive-fp32) | [`0acafea451b2474e8ff15f310f5e07fcb45dc7a7`](https://huggingface.co/aac6fef/laya-multilingual-coreml-adaptive-fp32/tree/0acafea451b2474e8ff15f310f5e07fcb45dc7a7) |
| [aac6fef/laya-typed-decisions-coreml-adaptive-fp16](https://huggingface.co/aac6fef/laya-typed-decisions-coreml-adaptive-fp16) | [`bb04dc23c9f85e0402eddc785e54aa5054f07efe`](https://huggingface.co/aac6fef/laya-typed-decisions-coreml-adaptive-fp16/tree/bb04dc23c9f85e0402eddc785e54aa5054f07efe) |
| [aac6fef/laya-typed-decisions-coreml-adaptive-fp32](https://huggingface.co/aac6fef/laya-typed-decisions-coreml-adaptive-fp32) | [`62ab15fe7ad079134d82cced8feaa016f6404840`](https://huggingface.co/aac6fef/laya-typed-decisions-coreml-adaptive-fp32/tree/62ab15fe7ad079134d82cced8feaa016f6404840) |

```bash
pip install laya-coreml==0.5.1
```

```python
import laya_coreml as laya

with laya.load(
    "aac6fef/laya-multilingual-coreml-adaptive-fp16",
    revision="d7927f83d2fd324b2e9cfed72cc2277e6a255387",
) as agent:
    result = agent.predict(
        "The customer requests a refund.",
        {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}},
    )
```
