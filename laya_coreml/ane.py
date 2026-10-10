"""Host embedding lookup + one ANE graph + CPU action-head runtime."""

import json
import math
from pathlib import Path

import coremltools as ct
import numpy as np
from safetensors import safe_open

from laya_coreml.prompt import PromptMixin
from laya_coreml.result import ResultMixin
from laya_coreml.tokenizer import Tokenizer

from .artifacts import package_for_coreml, verify_files, verify_research_manifest
from .common import option_layout, read_temperatures


class ANEAgent(PromptMixin, ResultMixin):
    def __init__(self, source, package=None, *, length=None, compute_units="cpu_ne"):
        self.source = Path(source)
        if package is None:
            self.manifest = json.loads((self.source / "coreml_config.json").read_text())
            if (self.manifest.get("format"), self.manifest.get("format_version")) != (
                "laya-coreml-ane",
                1,
            ):
                raise ValueError("Unsupported ANE bundle format")
            shape = self.manifest["shape"]
            if shape["batch_size"] != 1 or shape["max_options"] != 32 or shape["flexible"]:
                raise ValueError("ANE runtime requires a fixed B1/K32 bundle")
            if length is not None and length != shape["max_length"]:
                raise ValueError("Requested length does not match the ANE bundle")
            length = shape["max_length"]
            verify_files(self.source, self.manifest["files"])
            package = self.source / "model.mlpackage"
            host_weights = self.source / "host_weights.safetensors"
        else:
            length = 96 if length is None else length
            self.manifest = verify_research_manifest(self.source, package, length=length)
            host_weights = self.source / "model.safetensors"
        self.model_dir = self.source
        self.cfg = json.loads((self.source / "rl_agent_config.json").read_text())
        if option_layout(self.cfg) != "sequential":
            raise ValueError(
                "Legacy ANE bundles use sequential RoPE; export this checkpoint with laya_coreml.convert"
            )
        (
            self.temperature,
            self.temperature_by_options,
            self.temperature_raw,
            self.temperature_by_options_raw,
        ) = read_temperatures(self.cfg)
        self.tok = Tokenizer(self.source / "tokenizer")
        self.batch_size, self.pad_to_multiple = 1, 16
        self.shape = {
            "batch_size": 1,
            "max_length": length,
            "min_length": length,
            "max_options": 32,
            "flexible": False,
            "lengths": None,
        }
        self.compute_units = compute_units
        units = {
            "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
        }
        if compute_units not in units:
            raise ValueError(f"compute_units must be one of {list(units)}")
        self.model = ct.models.MLModel(
            str(package_for_coreml(package)), compute_units=units[compute_units]
        )
        self.encoder_cfg = json.loads((self.source / "encoder/config.json").read_text())
        width = int(self.encoder_cfg["hidden_size"])
        expected_shapes = {
            "embeddings": (1, width, 1, length),
            "full_mask": (1, length, 1, length),
            "local_mask": (1, length, 1, length),
            "type_vectors": (1, width, 1, 1),
            "marker_map": (1, length, 1, 32),
        }
        actual_shapes = {
            feature.name: tuple(feature.type.multiArrayType.shape)
            for feature in self.model.get_spec().description.input
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"Package signature mismatch: expected {expected_shapes}, got {actual_shapes}"
            )
        with safe_open(str(host_weights), framework="numpy") as weights:
            self.embedding = weights.get_tensor("encoder.embeddings.tok_embeddings.weight")
            self.type_embedding = weights.get_tensor("type_emb.weight")
            self.action = {
                key: weights.get_tensor("act_head." + key).astype(np.float32)
                for key in ("0.weight", "0.bias", "2.weight", "2.bias")
            }
        spec = self.model.get_spec()
        self.output_names = [output.name for output in spec.description.output]
        self._erf = np.frompyfunc(math.erf, 1, 1)
        positions = np.arange(length)
        self.window = (
            np.abs(positions[:, None] - positions[None, :])
            <= int(self.encoder_cfg.get("local_attention", 128)) // 2
        )

    def model_inputs(self, batch):
        expected_shapes = {
            "input_ids": (1, self.shape["max_length"]),
            "attention_mask": (1, self.shape["max_length"]),
            "marker_pos": (1, 32),
            "marker_mask": (1, 32),
            "qtype": (1,),
        }
        if set(batch) != set(expected_shapes):
            raise ValueError("Prepared batch fields do not match the fixed ANE signature")
        for name, shape in expected_shapes.items():
            if batch[name].shape != shape or not np.issubdtype(batch[name].dtype, np.integer):
                raise ValueError(f"{name} must have integer dtype and shape {shape}")
        if np.any(batch["input_ids"] < 0) or np.any(batch["input_ids"] >= self.embedding.shape[0]):
            raise ValueError("Token id outside checkpoint vocabulary")
        for name in ("attention_mask", "marker_mask"):
            if not np.isin(batch[name], (0, 1)).all():
                raise ValueError(f"{name} must contain only zero or one")
        if not batch["attention_mask"].any(axis=-1).all():
            raise ValueError("Every batch row needs at least one valid attention key")
        if np.any(batch["qtype"] < 0) or np.any(batch["qtype"] > 2):
            raise ValueError("Question type must be 0, 1 or 2")
        if np.any(batch["marker_pos"] < 0) or np.any(
            batch["marker_pos"] >= self.shape["max_length"]
        ):
            raise ValueError("Marker position outside exported sequence")
        ids, valid = batch["input_ids"], batch["attention_mask"].astype(bool)
        embeddings = self.embedding[ids].transpose(0, 2, 1)[:, :, None, :]
        full = np.broadcast_to(valid[:, None, :], (ids.shape[0], ids.shape[1], ids.shape[1]))
        local = (self.window[None] | ~valid[:, :, None]) & full
        # Core ML BC1S attention scores are [B,key,1,query].
        masks = {"full_mask": full, "local_mask": local}
        result = {
            name: np.where(value.transpose(0, 2, 1)[:, :, None, :], 0, -1e4).astype(np.float16)
            for name, value in masks.items()
        }
        result["embeddings"] = np.ascontiguousarray(embeddings, dtype=np.float16)
        result["type_vectors"] = np.ascontiguousarray(
            self.type_embedding[batch["qtype"]][:, :, None, None], dtype=np.float16
        )
        marker_map = np.zeros((ids.shape[0], ids.shape[1], 1, 32), np.float16)
        for row in range(ids.shape[0]):
            marker_map[row, batch["marker_pos"][row], 0, np.arange(32)] = 1
        result["marker_map"] = marker_map
        return result

    def forward(self, batch):
        outputs = self.model.predict(self.model_inputs(batch))
        # Output names are traced identifiers; shapes uniquely identify these outputs.
        logits = (
            next(v for v in outputs.values() if v.shape[1] == 1).reshape(1, 32).astype(np.float32)
        )
        pooled = (
            next(v for v in outputs.values() if v.shape[1] != 1).reshape(1, -1).astype(np.float32)
        )
        logits = np.where(batch["marker_mask"].astype(bool), logits, -1e4)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        k = np.maximum(batch["marker_mask"].sum(axis=-1), 2).astype(np.float32)
        entropy = -(p * np.log(np.maximum(p, 1e-9))).sum(axis=-1) / np.log(k)
        top = np.sort(p, axis=-1)[:, -2:]
        features = np.stack((top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0), axis=-1)
        action_input = np.concatenate((pooled, features), axis=-1)
        hidden = action_input @ self.action["0.weight"].T + self.action["0.bias"]
        # Only 256 host elements: exact erf GELU, no tanh/sigmoid approximation.
        hidden = hidden * (1 + self._erf(hidden / np.sqrt(2)).astype(np.float32)) / 2
        action = hidden @ self.action["2.weight"].T + self.action["2.bias"]
        return logits, action
