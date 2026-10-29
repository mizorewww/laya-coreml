"""Generate golden outputs from the pinned, unmodified upstream Laya package.

Run in an environment with upstream-compatible torch and transformers installed.
The committed output lets Core ML validation run without either dependency.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from .cases import parity_cases

UPSTREAM_REVISION = "fe9a6c35023a7d94102e7feaaeaae8857d9d1b07"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/reference.json"))
    a = p.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_REVISION:
        raise ValueError(f"Expected {UPSTREAM_REVISION}, got {commit}")
    sys.path.insert(0, str(a.upstream.resolve()))
    import laya
    import torch
    from laya.common import QTYPES, build_sequence, collate_items

    from laya_coreml.convert import REVISIONS, sha256

    report = {
        "upstream_revision": commit,
        "device": "mps",
        "dtype": "float32",
        "autocast_enabled": False,
        "public_result_precision": "float32 for every row count",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
        "models": {},
    }
    for name, revision in REVISIONS.items():
        agent = laya.load(str(a.model_root / name), device="mps")
        # v0.4.12 otherwise switches public predict() to FP16 for larger batches.
        # Raw logits and public outputs must use the same stated reference precision.
        agent.amp_enabled = False
        assert agent.dtype_for(64) == torch.float32
        if agent.device.type != "mps":
            raise RuntimeError("Reference unexpectedly fell back from MPS")
        cases = []
        for label, state, questions in parity_cases():
            items = []
            for q in questions.values():
                internal = agent._to_internal(q)
                ids, markers = build_sequence(
                    agent.tok, state, internal, agent.cfg["max_len"], agent.cfg["head_max_len"]
                )
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[internal["t"]]})
            batch = collate_items([items], agent.tok.pad_token_id)
            with torch.inference_mode():
                logits, actions = agent.model(
                    **{
                        k: batch[k].to(agent.device)
                        for k in (
                            "input_ids",
                            "attention_mask",
                            "marker_pos",
                            "marker_mask",
                            "qtype",
                        )
                    }
                )
                torch.mps.synchronize()
            cases.append(
                {
                    "name": label,
                    "state": state,
                    "questions": questions,
                    "items": items,
                    "logits": logits.cpu().tolist(),
                    "action_logits": actions.cpu().tolist(),
                    "result": agent.predict(state, questions),
                }
            )
        report["models"][name] = {
            "revision": revision,
            "source_weights_sha256": sha256(a.model_root / name / "model.safetensors"),
            "cases": cases,
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        )
        print(f"{name}: {sum(len(c['items']) for c in cases)} reference questions", flush=True)
        del agent
        gc.collect()
        torch.mps.empty_cache()
    print("Reference SHA256:", hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
