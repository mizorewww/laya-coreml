"""Core ML inference. No MLX, Transformers, or PyTorch dependency at runtime."""

import json
import os
import threading

import numpy as np

from .artifacts import package_for_coreml
from .common import option_layout, read_temperatures
from .hub import DEFAULT_MODEL, resolve_checkpoint
from .runtime import RuntimeMixin
from .tokenizer import Tokenizer

COMPUTE_UNITS = {"all": "ALL", "cpu": "CPU_ONLY", "cpu_gpu": "CPU_AND_GPU", "cpu_ne": "CPU_AND_NE"}


def _refuse_vegaml(vegaml):
    """`vegaml=None`/`False` mean the ordinary checkpoint, as upstream; anything else asks for
    vegaml's engine, which only official laya carries."""
    if vegaml is not None and vegaml is not False:
        raise ValueError(
            "vegaml=%r loads vegaml's engine, which requires official laya; this distribution "
            "uses Core ML" % (vegaml,)
        )


class Agent(RuntimeMixin):
    @property
    def config(self):
        """Checkpoint metadata for upstream hooks, including option layout."""
        return self.cfg

    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        fast=False,
        compile=False,
        revision=None,
        expected_sha256=None,
        lang_temperatures=None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise=True,
        hooks_concurrent=True,
        hooks_timeout=None,
        calibration=None,
        backend=None,
        compile_warmup=True,
        compile_cache=False,
        compile_mode="default",
        vegaml=None,
        *,
        compute_units="cpu_gpu",
        allow_unvalidated_gpu=False,
        local_files_only=False,
        shape_cache_size=4,
    ):
        if (
            fast
            or compile
            or compile_cache
            or compile_mode != "default"
            or backend not in (None, "auto", "coreml")
        ):
            raise ValueError(
                "This distribution uses Core ML; PyTorch/CUDA/ONNX backend options require official laya"
            )
        _refuse_vegaml(vegaml)
        if device is not None:
            if device not in ("cpu", "mps"):
                raise ValueError("Core ML device must be cpu or mps; use compute_units for ANE")
            compute_units = "cpu" if device == "cpu" else "cpu_gpu"
        from .shape_cache import ShapeCache

        self._shape_cache = ShapeCache(shape_cache_size)
        self.model_id = self.model_id_or_path = str(model_id_or_path)
        self.subfolder, self.revision = subfolder, None
        if compute_units not in COMPUTE_UNITS:
            raise ValueError(f"compute_units must be one of {list(COMPUTE_UNITS)}")
        self.model_dir = resolve_checkpoint(
            model_id_or_path,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            subfolder=subfolder,
        )
        from .revisions import snapshot_revision, verify_digests

        verify_digests(str(self.model_dir), expected_sha256)
        self.revision = snapshot_revision(str(self.model_dir))
        self.manifest = json.loads((self.model_dir / "coreml_config.json").read_text())
        if self.manifest.get("format") != "laya-coreml" or self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported Core ML export format")
        self.shape = self.manifest["shape"]
        from .batching import variant_shapes

        self._batch_variants = variant_shapes(self.manifest, self.model_dir)
        self._batch_models = {}
        self._batch_lock = threading.RLock()
        if (
            compute_units == "cpu_gpu"
            and self.shape["flexible"]
            and not self.shape.get("lengths")
            and not allow_unvalidated_gpu
        ):
            raise ValueError(
                "RangeDim + CPU_AND_GPU failed local fidelity and repeatability checks. "
                "Re-export with the default enumerated shapes, or use compute_units='cpu'. "
                "allow_unvalidated_gpu=True is for reproducing the failure only."
            )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        layout = option_layout(self.cfg)
        if layout != self.manifest.get("option_layout", "sequential"):
            raise ValueError(
                "Checkpoint option_layout does not match the exported graph; re-export it"
            )
        (
            self.temperature,
            self.temperature_by_options,
            self.temperature_raw,
            self.temperature_by_options_raw,
        ) = read_temperatures(self.cfg)
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        self.batch_size = self.shape["batch_size"]
        self.pad_to_multiple = 16
        self._init_host(
            lang_temperatures=lang_temperatures,
            calibration=calibration,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
            hooks_raise=hooks_raise,
            hooks_concurrent=hooks_concurrent,
            hooks_timeout=hooks_timeout,
        )
        import coremltools as ct

        self.compute_units = compute_units
        self.device = "coreml:" + compute_units
        self.model = ct.models.MLModel(
            str(package_for_coreml(self.model_dir / "model.mlpackage")),
            compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units]),
        )

    def _select_graph(self, rows):
        """Keep host batching semantics; select only the physical tensor graph."""
        variants = getattr(self, "_batch_variants", {})
        if not variants or rows <= self.shape["batch_size"]:
            return self.shape, self.forward
        capacities = sorted(variants)
        capacity = next((n for n in capacities if n >= rows), capacities[-1])
        shape, path = variants[capacity]
        with self._batch_lock:
            if capacity not in self._batch_models:
                import coremltools as ct

                self._batch_models[capacity] = ct.models.MLModel(
                    str(package_for_coreml(path)),
                    compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[self.compute_units]),
                )
            model = self._batch_models[capacity]

        def predict(batch):
            return self._graph_outputs(model, batch)

        return shape, predict

    def forward(self, batch):
        return self._graph_outputs(self.model, batch)

    def _graph_outputs(self, model, batch):
        cache = getattr(self, "_shape_cache", None)
        outputs = cache.predict(model, batch) if cache is not None else model.predict(batch)
        return np.asarray(outputs["logits"], np.float32), np.asarray(
            outputs["action_logits"], np.float32
        )


RLAgent = Agent


def _is_local_checkpoint_arg(model_id_or_path: str) -> bool:
    """True when `model_id_or_path` should be treated as a local path, not a registry name.

    A bare word with no path separator reads as a name/alias. Only treat it as a path when
    it looks like one (contains a separator, or starts with `.` / `~`) or when the named
    directory actually holds a Laya checkpoint (`rl_agent_config.json`). That way
    `load("laya")` resolves the alias from the repo root, while `load("./laya")` and a
    real checkpoint directory still load from disk.
    """
    if not model_id_or_path:
        return False
    if model_id_or_path.startswith((".", "~")):
        return True
    if os.sep in model_id_or_path or "/" in model_id_or_path or "\\" in model_id_or_path:
        return True
    return os.path.isdir(model_id_or_path) and os.path.isfile(
        os.path.join(model_id_or_path, "rl_agent_config.json")
    )


def load(
    model_id_or_path=DEFAULT_MODEL,
    device=None,
    token=None,
    subfolder=None,
    fast=False,
    compile=False,
    revision=None,
    expected_sha256=None,
    lang_temperatures=None,
    hooks=None,
    on_predict_start=None,
    on_predict_end=None,
    hooks_raise=True,
    hooks_concurrent=True,
    hooks_timeout=None,
    calibration=None,
    backend=None,
    onnx_path=None,
    compile_warmup=True,
    compile_cache=False,
    compile_mode="default",
    vegaml=None,
    *,
    local_files_only=False,
    compute_units=None,
    allow_unvalidated_gpu=False,
    shape_cache_size=4,
):
    from .revisions import snapshot_revision, verify_digests
    from .router import resolve_model_spec

    if (
        fast
        or compile
        or compile_cache
        or compile_mode != "default"
        or backend not in (None, "auto", "coreml")
        or onnx_path is not None
    ):
        raise ValueError(
            "This distribution uses Core ML; PyTorch/CUDA/ONNX backend options require official laya"
        )
    _refuse_vegaml(vegaml)
    model_id_or_path = str(model_id_or_path)
    if subfolder is None and not _is_local_checkpoint_arg(model_id_or_path):
        spec = resolve_model_spec(model_id_or_path)
        if spec is not None:
            model_id_or_path, subfolder = spec
    directory = resolve_checkpoint(
        model_id_or_path,
        revision=revision,
        local_files_only=local_files_only,
        token=token,
        subfolder=subfolder,
    )
    verify_digests(str(directory), expected_sha256)
    manifest = json.loads((directory / "coreml_config.json").read_text())
    host = dict(
        lang_temperatures=lang_temperatures,
        hooks=hooks,
        on_predict_start=on_predict_start,
        on_predict_end=on_predict_end,
        hooks_raise=hooks_raise,
        hooks_concurrent=hooks_concurrent,
        hooks_timeout=hooks_timeout,
    )
    if manifest.get("format") == "laya-coreml-ane":
        from .ane import ANEAgent

        if device is not None:
            if device not in ("cpu", "mps"):
                raise ValueError("Core ML device must be cpu or mps")
            compute_units = "cpu" if device == "cpu" else "cpu_gpu"
        agent = ANEAgent(directory, compute_units=compute_units or "cpu_ne", **host)
    else:
        agent = Agent(
            directory,
            device=device,
            compute_units=compute_units or "cpu_gpu",
            allow_unvalidated_gpu=allow_unvalidated_gpu,
            shape_cache_size=shape_cache_size,
            expected_sha256={},
            **host,
        )
    agent.model_id = agent.model_id_or_path = str(model_id_or_path)
    agent.subfolder, agent.revision = subfolder, snapshot_revision(str(directory))
    if calibration:
        agent.load_calibration(calibration)
    return agent
