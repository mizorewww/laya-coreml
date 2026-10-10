"""Export-only ModernBERT graph, adapted from laya-mlx; see NOTICE.

Explicit attention/head modules avoid Transformers' tracing-time backend switches.
State-dict names match original Laya checkpoints. No training or weight changes.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .common import option_layout


def attention(q, k, v, mask, implementation):
    if implementation == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    # Alternate lowering retained for controlled conversion experiments. RangeDim
    # GPU failure also affected this implementation; enumerated shapes fix it.
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    scores = torch.where(mask, scores, torch.full_like(scores, -1e4))
    return torch.matmul(F.softmax(scores, dim=-1), v)


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.norm = nn.LayerNorm(
            cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5), bias=cfg.get("norm_bias", False)
        )

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


class Attention(nn.Module):
    def __init__(self, cfg, kind, max_length):
        super().__init__()
        self.implementation = "explicit"
        d = cfg["hidden_size"]
        self.heads, self.dim = cfg["num_attention_heads"], d // cfg["num_attention_heads"]
        self.Wqkv = nn.Linear(d, 3 * d, bias=cfg.get("attention_bias", False))
        self.Wo = nn.Linear(d, d, bias=cfg.get("attention_bias", False))
        params = cfg.get("rope_parameters", {}).get(kind, {})
        if params.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported")
        base = params.get(
            "rope_theta",
            cfg.get("global_rope_theta", 160000.0)
            if kind == "full_attention"
            else cfg.get("local_rope_theta", 10000.0),
        )
        inv = 1.0 / (float(base) ** (torch.arange(0, self.dim, 2).float() / self.dim))
        angles = torch.outer(torch.arange(max_length).float(), inv)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, value, position_ids=None):
        a, b = value.chunk(2, dim=-1)
        length = value.shape[-2]
        if position_ids is None:
            c, s = self.cos[:, :, :length], self.sin[:, :, :length]
        else:
            c = self.cos[0, 0][position_ids.long()][:, None]
            s = self.sin[0, 0][position_ids.long()][:, None]
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    def forward(self, x, mask, position_ids=None):
        b, n, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        value = attention(
            self.rotate(q, position_ids), self.rotate(k, position_ids), v, mask, self.implementation
        )
        return self.Wo(value.transpose(1, 2).reshape(b, n, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["hidden_size"], cfg["intermediate_size"]
        self.Wi = nn.Linear(d, 2 * h, bias=cfg.get("mlp_bias", False))
        self.Wo = nn.Linear(h, d, bias=cfg.get("mlp_bias", False))

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class Layer(nn.Module):
    def __init__(self, cfg, index, kind, max_length):
        super().__init__()
        self.kind = kind

        def norm():
            return nn.LayerNorm(
                cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5), bias=cfg.get("norm_bias", False)
            )

        self.attn_norm = nn.Identity() if index == 0 else norm()
        self.attn = Attention(cfg, kind, max_length)
        self.mlp_norm = norm()
        self.mlp = MLP(cfg)

    def forward(self, x, mask, position_ids=None):
        x = x + self.attn(self.attn_norm(x), mask, position_ids)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, cfg, max_length):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        kinds = cfg.get("layer_types") or [
            "full_attention"
            if i % cfg.get("global_attn_every_n_layers", 3) == 0
            else "sliding_attention"
            for i in range(cfg["num_hidden_layers"])
        ]
        if len(kinds) != cfg["num_hidden_layers"] or set(kinds) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid layer_types")
        self.layers = nn.ModuleList(
            [Layer(cfg, i, kind, max_length) for i, kind in enumerate(kinds)]
        )
        self.final_norm = nn.LayerNorm(
            cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5), bias=cfg.get("norm_bias", False)
        )
        self.window = cfg.get("local_attention", 128)
        # Slice integer positions before building the bool mask. Slicing a
        # constant bool matrix triggers an MPSGraph compiler trap on this OS.
        self.register_buffer(
            "positions", torch.arange(max_length, dtype=torch.int32), persistent=False
        )

    def forward(self, ids, valid, position_ids=None, option_ids=None):
        x = self.embeddings(ids)
        full = valid[:, None, None, :]
        length = ids.shape[1]
        positions = self.positions[:length]
        window = (positions[:, None] - positions[None, :]).abs() <= self.window // 2
        local = torch.logical_and(
            torch.logical_or(window[None, None], torch.logical_not(valid[:, None, :, None])),
            full,
        )
        if option_ids is not None:
            other = torch.logical_and(option_ids[:, :, None] > 0, option_ids[:, None, :] > 0)
            other = torch.logical_and(other, option_ids[:, :, None] != option_ids[:, None, :])
            full = torch.logical_and(full, torch.logical_not(other[:, None]))
            distance = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
            local = torch.logical_and(
                full,
                torch.logical_or(
                    distance <= self.window // 2, torch.logical_not(valid[:, :, None])
                )[:, None],
            )
        for layer in self.layers:
            x = layer(x, full if layer.kind == "full_attention" else local, position_ids)
        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.implementation = "explicit"
        self.heads = max(1, d // 64)
        self.dim = d // self.heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, mask):
        b, n, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(
            b, n, 3, self.heads, self.dim
        )
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        value = attention(q, k, v, mask, self.implementation)
        return self.out_proj(value.transpose(1, 2).reshape(b, n, -1))


class HeadLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = HeadAttention(d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class Head(nn.Module):
    def __init__(self, d, layers):
        super().__init__()
        self.layers = nn.ModuleList([HeadLayer(d) for _ in range(layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecisionModel(nn.Module):
    def __init__(self, cfg, agent_cfg, max_length):
        super().__init__()
        if cfg.get("model_type") != "modernbert" or cfg.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("Only GELU ModernBERT checkpoints are supported")
        d = cfg["hidden_size"]
        self.parallel = option_layout(agent_cfg) == "parallel"
        self.encoder = Encoder(cfg, max_length)
        self.head = Head(d, agent_cfg["head_layers"])
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent_cfg.get("act_costs", {})) + 1),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        position_ids=None,
        option_ids=None,
    ):
        if self.parallel and (position_ids is None or option_ids is None):
            raise ValueError("Parallel checkpoints require position_ids and option_ids")
        if not self.parallel and (position_ids is not None or option_ids is not None):
            raise ValueError("Sequential checkpoints do not accept parallel layout inputs")
        valid, selected = attention_mask.bool(), marker_mask.bool()
        h = (
            self.encoder(input_ids, valid, position_ids, option_ids)
            + self.type_emb(qtype)[:, None, :]
        )
        h = self.head(h, valid[:, None, None, :])
        markers = torch.gather(h, 1, marker_pos.long()[:, :, None].expand(-1, -1, h.shape[-1]))
        logits = self.scorer(markers).squeeze(-1).float()
        logits = torch.where(selected, logits, torch.full_like(logits, -1e4))
        p = F.softmax(logits, dim=-1)
        k = selected.sum(-1).clamp(min=2).float()
        entropy = -(p * p.clamp(min=1e-9).log()).sum(-1) / k.log()
        top = p.topk(2, dim=-1).values
        features = torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy, k / 255.0), dim=-1)
        action = self.act_head(torch.cat((h[:, 0].float(), features), dim=-1))
        return logits, action.float()


def load_model(path, max_length, attention_implementation="explicit"):
    path = Path(path)
    cfg = json.loads((path / "encoder/config.json").read_text())
    agent_cfg = json.loads((path / "rl_agent_config.json").read_text())
    model = DecisionModel(cfg, agent_cfg, max_length)
    if attention_implementation not in ("explicit", "sdpa"):
        raise ValueError("attention_implementation must be explicit or sdpa")
    for module in model.modules():
        if isinstance(module, (Attention, HeadAttention)):
            module.implementation = attention_implementation
    # Original checkpoints, never rounded MLX FP16 exports.
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    return model.eval()
