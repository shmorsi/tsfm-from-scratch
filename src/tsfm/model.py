"""Decoder-only transformer for time series, in plain PyTorch.

Pre-norm (RMSNorm) blocks with rotary position embeddings, SwiGLU MLPs and an optional
key/value cache for incremental decoding. The input embedding and output head depend on
the tokenizer:

* ``"bins"``  - token embedding in, logits over the 4,096-token vocabulary out.
* ``"patch"`` - residual MLP on (values, observed-mask) patches in, non-crossing
  quantiles of the next ``horizon`` steps out.

Parameter count of the transformer body is ~ ``12 * n_layers * d_model**2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tsfm.tokenizers import QUANTILES, BinTokenizer, QuantileHead


@dataclass
class ModelConfig:
    tokenizer: Literal["bins", "patch"] = "patch"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: float = 8 / 3  # SwiGLU hidden = ffn_mult * d_model, rounded to a multiple of 64
    dropout: float = 0.0
    rope_base: float = 10_000.0
    max_seq_len: int = 2048
    # bins
    n_bins: int = 4094
    bin_limit: float = 15.0
    # patch
    patch_len: int = 32
    horizon: int = 64
    quantiles: tuple[float, ...] = field(default=QUANTILES)

    @property
    def d_ff(self) -> int:
        return int(64 * round(self.ffn_mult * self.d_model / 64))

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quantiles"] = list(self.quantiles)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        d = dict(d)
        if "quantiles" in d:
            d["quantiles"] = tuple(d["quantiles"])
        return cls(**d)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def rope_tables(
    head_dim: int, max_len: int, base: float, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = torch.arange(max_len, dtype=torch.float32, device=device)
    ang = torch.outer(pos, inv)  # (T, Dh/2)
    return ang.cos(), ang.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x (B, H, T, Dh); cos/sin (T, Dh/2). Rotates (even, odd) pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


@dataclass
class KVCache:
    """Per-layer keys/values plus the key padding mask, filled left to right."""

    k: list[Tensor]
    v: list[Tensor]
    key_pad: Tensor  # (B, T_max) True = padding
    length: int = 0

    @classmethod
    def empty(
        cls, cfg: ModelConfig, batch: int, device: torch.device, dtype: torch.dtype
    ) -> KVCache:
        shape = (batch, cfg.n_heads, cfg.max_seq_len, cfg.head_dim)
        return cls(
            k=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)],
            v=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)],
            key_pad=torch.ones(batch, cfg.max_seq_len, dtype=torch.bool, device=device),
        )

    def repeat_interleave(self, n: int) -> KVCache:
        """Expand each row ``n`` times (for drawing several sample paths)."""
        return KVCache(
            [k.repeat_interleave(n, 0) for k in self.k],
            [v.repeat_interleave(n, 0) for v in self.v],
            self.key_pad.repeat_interleave(n, 0),
            self.length,
        )


def attention_mask(q_pos: Tensor, k_pos: Tensor, key_pad: Tensor) -> Tensor:
    """Boolean mask (B, 1, Tq, Tk), True = may attend. Causal, no padded keys, and every
    query may always attend to itself so fully padded rows never produce NaNs."""
    causal = k_pos[None, :] <= q_pos[:, None]  # (Tq, Tk)
    allowed = causal[None] & ~key_pad[:, None, :]
    allowed = allowed | (k_pos[None, :] == q_pos[:, None])[None]
    return allowed[:, None]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor,
        cache: KVCache | None,
        layer: int,
        start: int,
    ) -> Tensor:
        b, t, _ = x.shape
        h, dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).view(b, t, 3, h, dh).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos[start : start + t], sin[start : start + t])
        k = apply_rope(k, cos[start : start + t], sin[start : start + t])
        if cache is not None:
            cache.k[layer][:, :, start : start + t] = k.to(cache.k[layer].dtype)
            cache.v[layer][:, :, start : start + t] = v.to(cache.v[layer].dtype)
            k = cache.k[layer][:, :, : start + t].to(q.dtype)
            v = cache.v[layer][:, :, : start + t].to(q.dtype)
        drop = self.cfg.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.out(y.transpose(1, 2).reshape(b, t, h * dh))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor,
        cache: KVCache | None,
        layer: int,
        start: int,
    ) -> Tensor:
        x = x + self.drop(self.attn(self.n1(x), cos, sin, mask, cache, layer, start))
        return x + self.drop(self.mlp(self.n2(x)))


class PatchEmbed(nn.Module):
    """Residual MLP on [scaled values, observed mask] of one patch."""

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.hidden = nn.Linear(2 * patch_len, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.skip = nn.Linear(2 * patch_len, d_model)

    def forward(self, values: Tensor, observed: Tensor) -> Tensor:
        z = torch.cat([values, observed.to(values.dtype)], dim=-1)
        return self.out(F.silu(self.hidden(z))) + self.skip(z)


class TSModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.tokenizer == "bins":
            self.bins = BinTokenizer(cfg.n_bins, cfg.bin_limit)
            self.embed: nn.Module = nn.Embedding(self.bins.vocab_size, cfg.d_model)
            self.head: nn.Module = nn.Linear(cfg.d_model, self.bins.vocab_size, bias=False)
        elif cfg.tokenizer == "patch":
            self.embed = PatchEmbed(cfg.patch_len, cfg.d_model)
            self.head = QuantileHead(cfg.d_model, cfg.horizon, cfg.quantiles)
        else:
            raise ValueError(cfg.tokenizer)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style scaled residual init
            if name.endswith(("attn.out.weight", "mlp.down.weight")):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, body_only: bool = False) -> int:
        if body_only:
            return sum(p.numel() for p in self.blocks.parameters())
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: Tensor | None = None,
        *,
        patches: Tensor | None = None,
        observed: Tensor | None = None,
        pad: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """Run the transformer on new positions.

        bins:  ``tokens`` (B, T) ids -> logits (B, T, V)
        patch: ``patches``/``observed`` (B, T, P) -> quantiles (B, T, horizon, Q)
        ``pad`` (B, T) marks padding positions. With ``cache`` the T new positions are
        appended after ``cache.length`` previous ones and the cache is updated."""
        if self.cfg.tokenizer == "bins":
            assert tokens is not None
            x = self.embed(tokens)
        else:
            assert patches is not None
            assert observed is not None
            x = self.embed(patches, observed)
        b, t, _ = x.shape
        start = cache.length if cache is not None else 0
        if start + t > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {start + t} exceeds max_seq_len {self.cfg.max_seq_len}")
        if pad is None:
            pad = torch.zeros(b, t, dtype=torch.bool, device=x.device)
        q_pos = torch.arange(start, start + t, device=x.device)
        if cache is not None:
            cache.key_pad[:, start : start + t] = pad
            key_pad = cache.key_pad[:, : start + t]
            k_pos = torch.arange(0, start + t, device=x.device)
        else:
            key_pad, k_pos = pad, q_pos
        mask = attention_mask(q_pos, k_pos, key_pad)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin, mask, cache, i, start)
        if cache is not None:
            cache.length = start + t
        out: Tensor = self.head(self.norm(x))
        return out
