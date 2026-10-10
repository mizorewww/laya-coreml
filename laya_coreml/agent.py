"""Core ML inference. No MLX, Transformers, or PyTorch dependency at runtime."""

import json

import numpy as np

from .artifacts import package_for_coreml
from .common import option_layout, read_temperatures
from .hub import DEFAULT_MODEL, resolve_checkpoint
from .prompt import PromptMixin
from .result import ResultMixin
from .tokenizer import Tokenizer

COMPUTE_UNITS = {"all": "ALL", "cpu": "CPU_ONLY", "cpu_gpu": "CPU_AND_GPU", "cpu_ne": "CPU_AND_NE"}


class Agent(PromptMixin, ResultMixin):
    def __init__(
        self,
        model_dir,
        *,
        compute_units="cpu_gpu",
        allow_unvalidated_gpu=False,
        revision=None,
        local_files_only=False,
    ):
        if compute_units not in COMPUTE_UNITS:
            raise ValueError(f"compute_units must be one of {list(COMPUTE_UNITS)}")
        self.model_dir = resolve_checkpoint(
            model_dir, revision=revision, local_files_only=local_files_only
        )
        self.manifest = json.loads((self.model_dir / "coreml_config.json").read_text())
        if self.manifest.get("format") != "laya-coreml" or self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported Core ML export format")
        self.shape = self.manifest["shape"]
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
        import coremltools as ct

        self.compute_units = compute_units
        self.model = ct.models.MLModel(
            str(package_for_coreml(self.model_dir / "model.mlpackage")),
            compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units]),
        )

    def forward(self, batch):
        outputs = self.model.predict(batch)
        return np.asarray(outputs["logits"], np.float32), np.asarray(
            outputs["action_logits"], np.float32
        )


def load(
    model_dir=DEFAULT_MODEL,
    *,
    revision=None,
    local_files_only=False,
    compute_units=None,
    allow_unvalidated_gpu=False,
):
    directory = resolve_checkpoint(model_dir, revision=revision, local_files_only=local_files_only)
    manifest = json.loads((directory / "coreml_config.json").read_text())
    if manifest.get("format") == "laya-coreml-ane":
        from .ane import ANEAgent

        return ANEAgent(directory, compute_units=compute_units or "cpu_ne")
    return Agent(
        directory,
        compute_units=compute_units or "cpu_gpu",
        allow_unvalidated_gpu=allow_unvalidated_gpu,
    )
