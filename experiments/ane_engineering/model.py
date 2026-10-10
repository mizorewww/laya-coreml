"""Independent BC1S/Conv transformer prototype for fixed-shape ANE export.

Uses the original checkpoint's weights and affine ordering. Layout follows the
principles in Apple's ml-ane-transformers reference, without its altered
LayerNorm affine ordering or optional clipping. Production code is untouched.
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv_from_linear(linear):
    result = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
    result.weight = nn.Parameter(linear.weight.detach()[:, :, None, None])
    if linear.bias is not None:
        result.bias = nn.Parameter(linear.bias.detach())
    return result


def conv_from_weights(weight, bias=None):
    result = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    result.weight = nn.Parameter(weight.detach()[:, :, None, None])
    if bias is not None:
        result.bias = nn.Parameter(bias.detach())
    return result


class ChannelNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.detach()[None, :, None, None])
        self.bias = None
        if source.bias is not None:
            self.bias = nn.Parameter(source.bias.detach()[None, :, None, None])

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        out = centered * (centered.square().mean(dim=1, keepdim=True) + self.eps).rsqrt()
        out = out * self.weight
        return out if self.bias is None else out + self.bias


class ConvAttention(nn.Module):
    def __init__(self, source, *, rope, length):
        super().__init__()
        self.rope = rope
        self.heads, self.dim = source.heads, source.dim
        weight = source.Wqkv.weight if rope else source.in_proj_weight
        bias = source.Wqkv.bias if rope else source.in_proj_bias
        self.qkv = conv_from_weights(weight, bias)
        self.out = conv_from_linear(source.Wo if rope else source.out_proj)
        if rope:
            self.register_buffer("cos", source.cos[0, 0, :length].T[None, :, None, :])
            self.register_buffer("sin", source.sin[0, 0, :length].T[None, :, None, :])

    def rotate(self, x):
        left, right = x.chunk(2, dim=1)
        return torch.cat(
            (left * self.cos - right * self.sin, right * self.cos + left * self.sin), dim=1
        )

    def forward(self, x, mask):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        output = []
        for qi, ki, vi in zip(
            q.split(self.dim, dim=1), k.split(self.dim, dim=1), v.split(self.dim, dim=1)
        ):
            if self.rope:
                qi, ki = self.rotate(qi), self.rotate(ki)
            scores = torch.einsum("bchq,bkhc->bkhq", qi, ki.transpose(1, 3)) * (self.dim**-0.5)
            probabilities = F.softmax(scores + mask, dim=1)
            output.append(torch.einsum("bkhq,bchk->bchq", probabilities, vi))
        return self.out(torch.cat(output, dim=1))


class ConvMLP(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.Wi, self.Wo = conv_from_linear(source.Wi), conv_from_linear(source.Wo)

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=1)
        return self.Wo(F.gelu(value) * gate)


class ConvEncoderLayer(nn.Module):
    def __init__(self, source, length):
        super().__init__()
        self.kind = source.kind
        self.attn_norm = (
            nn.Identity()
            if isinstance(source.attn_norm, nn.Identity)
            else ChannelNorm(source.attn_norm)
        )
        self.attn = ConvAttention(source.attn, rope=True, length=length)
        self.mlp_norm = ChannelNorm(source.mlp_norm)
        self.mlp = ConvMLP(source.mlp)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class ConvHeadLayer(nn.Module):
    def __init__(self, source, length):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(source.norm1), ChannelNorm(source.norm2)
        self.attn = ConvAttention(source.self_attn, rope=False, length=length)
        self.linear1, self.linear2 = (
            conv_from_linear(source.linear1),
            conv_from_linear(source.linear2),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class ConvBody(nn.Module):
    """Embedding norm, full encoder, type addition, full decision head and scorer.

    Inputs: raw embedding lookup BC1S, additive attention masks BK1Q, type vector
    BC11, and one-hot marker selectors BL1K. Outputs: option logits and CLS hidden.
    Small action-head features and calibrated output formatting remain on the CPU.
    """

    def __init__(self, source, length):
        super().__init__()
        if source.parallel:
            raise ValueError(
                "Legacy ANE export requires sequential layout; use laya_coreml.convert for parallel checkpoints"
            )
        self.embedding_norm = ChannelNorm(source.encoder.embeddings.norm)
        self.layers = nn.ModuleList(
            [ConvEncoderLayer(layer, length) for layer in source.encoder.layers]
        )
        self.final_norm = ChannelNorm(source.encoder.final_norm)
        self.head = nn.ModuleList([ConvHeadLayer(layer, length) for layer in source.head.layers])
        self.scorer = nn.Sequential(
            ChannelNorm(source.scorer[0]),
            conv_from_linear(source.scorer[1]),
            nn.GELU(),
            conv_from_linear(source.scorer[3]),
        )

    def forward(self, embeddings, full_mask, local_mask, type_vectors, marker_map):
        x = self.embedding_norm(embeddings)
        for layer in self.layers:
            x = layer(x, full_mask if layer.kind == "full_attention" else local_mask)
        x = self.final_norm(x) + type_vectors
        for layer in self.head:
            x = layer(x, full_mask)
        markers = torch.einsum("bkhq,bchk->bchq", marker_map, x)
        return self.scorer(markers), x[:, :, :, :1]
