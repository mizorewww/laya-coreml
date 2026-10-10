"""Reproducible FP16/FP32 Core ML export from the original Laya checkpoints."""

import hashlib
import json
import shutil
import time
from pathlib import Path

REVISIONS = {
    "laya": "c5d78730f3493e4fe16d61507ef4b78eef7318cf",
    "laya-multilingual": "052592a15d198d9ad47da779604259b10b47b7aa",
    "laya-typed-decisions": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(source, revision=None):
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or str(source).startswith((".", "~")):
        raise FileNotFoundError(source)
    from huggingface_hub import snapshot_download

    repo_id = str(source) if "/" in str(source) else "convaiinnovations/" + str(source)
    revision = revision or REVISIONS.get(repo_id.removeprefix("convaiinnovations/"))
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[
                "model.safetensors",
                "encoder/config.json",
                "rl_agent_config.json",
                "tokenizer/*",
            ],
        )
    )


def convert(
    source,
    output,
    *,
    max_length=None,
    flexible=True,
    batch_size=1,
    max_options=32,
    precision="float16",
    revision=None,
    attention="sdpa",
    shape_mode="enumerated",
):
    import coremltools as ct
    import numpy as np
    import torch

    from .torch_model import load_model

    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not Path(source).expanduser().is_dir() and revision is None:
        revision = REVISIONS.get(str(source).removeprefix("convaiinnovations/"))
    source_path = resolve_source(source, revision)
    cfg = json.loads((source_path / "rl_agent_config.json").read_text())
    max_length = cfg["max_len"] if max_length is None else max_length
    if not 16 <= max_length <= cfg["max_len"]:
        raise ValueError("max_length must be between 16 and the checkpoint context limit")
    if not 1 <= batch_size <= 64 or not 2 <= max_options <= 255:
        raise ValueError("Expected batch_size 1..64 and max_options 2..255")
    if precision not in ("float16", "float32"):
        raise ValueError("precision must be float16 or float32")
    started = time.perf_counter()
    torch.set_num_threads(8)
    model = load_model(source_path, max_length, attention_implementation=attention)
    length = min(128, max_length) if flexible else max_length
    inputs = {
        "input_ids": torch.zeros((batch_size, length), dtype=torch.int32),
        "attention_mask": torch.ones((batch_size, length), dtype=torch.int32),
        "marker_pos": torch.zeros((batch_size, max_options), dtype=torch.int32),
        "marker_mask": torch.ones((batch_size, max_options), dtype=torch.int32),
        "qtype": torch.zeros((batch_size,), dtype=torch.int32),
    }
    if model.parallel:
        inputs["position_ids"] = torch.arange(length, dtype=torch.int32)[None].expand(
            batch_size, -1
        )
        inputs["option_ids"] = torch.zeros((batch_size, length), dtype=torch.int32)
    print(
        f"Tracing {precision} B={batch_size}, L={length}, K={max_options}, flexible={flexible}",
        flush=True,
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, tuple(inputs.values()), strict=True, check_trace=True)
    if shape_mode not in ("range", "enumerated"):
        raise ValueError("shape_mode must be range or enumerated")
    lengths = (
        sorted(
            {
                v
                for v in (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, max_length)
                if v <= max_length
            }
        )
        if flexible and shape_mode == "enumerated"
        else None
    )
    dimension = ct.RangeDim(16, max_length, default=length) if flexible else length
    sequence_shape = (
        ct.EnumeratedShapes([(batch_size, n) for n in lengths], default=(batch_size, length))
        if lengths and len(lengths) > 1
        else (batch_size, dimension)
    )
    specs = [
        ct.TensorType(
            name=name,
            dtype=np.int32,
            shape=sequence_shape
            if name in ("input_ids", "attention_mask", "position_ids", "option_ids")
            else tuple(value.shape),
        )
        for name, value in inputs.items()
    ]
    converted = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=specs,
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="action_logits", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32,
        skip_model_load=True,
    )
    converted.author = (
        "Laya / Convai Innovations; independent Core ML port by laya-coreml contributors"
    )
    converted.license = "Apache-2.0"
    converted.short_description = (
        "Laya typed decision logits and action logits, without token generation"
    )
    output.mkdir(parents=True)
    try:
        converted.save(str(output / "model.mlpackage"))
        shutil.copytree(source_path / "tokenizer", output / "tokenizer")
        (output / "encoder").mkdir()
        shutil.copy2(source_path / "encoder/config.json", output / "encoder/config.json")
        shutil.copy2(source_path / "rl_agent_config.json", output / "rl_agent_config.json")
        manifest = {
            "format": "laya-coreml",
            "format_version": 1,
            "option_layout": "parallel" if model.parallel else "sequential",
            "source": str(source),
            "revision": revision,
            "source_weights_sha256": sha256(source_path / "model.safetensors"),
            "precision": precision,
            "attention": attention,
            "attention_mask_construction": "integer_positions_v2",
            "minimum_deployment_target": "macOS15 / iOS18",
            "shape": {
                "batch_size": batch_size,
                "max_length": max_length,
                "min_length": 16,
                "default_length": length,
                "max_options": max_options,
                "flexible": flexible,
                "mode": shape_mode if flexible else "fixed",
                "lengths": lengths,
            },
            "versions": {
                "coremltools": ct.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "conversion_seconds": time.perf_counter() - started,
            "files": {
                str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        }
        (output / "coreml_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Saved {output} in {manifest['conversion_seconds']:.1f}s", flush=True)
    return output
