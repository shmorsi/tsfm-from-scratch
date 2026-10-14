"""Generate configs/scaling/*.yaml: the size grid (patch tokenizer) and the bins model
used for the tokenizer comparison at a fixed 6ND compute budget."""

from __future__ import annotations

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "configs" / "scaling"
GRID = {
    "s0": (128, 4, 4),
    "s1": (192, 6, 6),
    "s2": (320, 8, 5),
    "s3": (512, 10, 8),
    "s4": (768, 14, 12),
}
LRS = {"s0": 1e-3, "s1": 1e-3, "s2": 6e-4, "s3": 4e-4, "s4": 3e-4}
BATCH, P, WINDOW, BINS_WINDOW = 256, 32, 2048, 1024
TOKENS_PER_PARAM = 20  # Chinchilla-style heuristic from LLMs; not established for time series


def data(window: int) -> dict:
    return {
        "corpora": [{"path": "pretrain"}, {"path": "synthetic", "synthetic": True}],
        "val": [{"path": "val"}, {"path": "synthetic_val", "synthetic": True}],
        "window": window,
        "min_context": 32,
        "synthetic_share": 0.1,
        "temperature": 0.5,
        "val_batches": 16,
        "val_batch_size": 128,
    }


def train(steps: int, lr: float, micro: int) -> dict:
    return {
        "steps": steps,
        "batch_size": BATCH,
        "micro_batch": micro,
        "lr": lr,
        "warmup": max(50, steps // 50),
        "min_lr_ratio": 0.1,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "fp16",
        "device": "auto",
        "ckpt_minutes": 10,
        "eval_every": max(100, steps // 20),
        "log_every": 50,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok_per_step = BATCH * WINDOW // P
    for k, (d, n_layers, h) in GRID.items():
        n_body = 12 * n_layers * d * d
        steps = max(100, int(round(TOKENS_PER_PARAM * n_body / tok_per_step / 100) * 100))
        cfg = {
            "name": f"patch_{k}",
            "seed": 0,
            "model": {
                "tokenizer": "patch",
                "d_model": d,
                "n_layers": n_layers,
                "n_heads": h,
                "patch_len": P,
                "horizon": 64,
                "max_seq_len": 128,
                "dropout": 0.0,
            },
            "data": data(WINDOW),
            "train": train(steps, LRS[k], 64 if d >= 512 else 128),
        }
        hdr = f"# Scaling grid {k}: d={d}, L={n_layers}, ~{n_body / 1e6:.1f}M body params, {steps} steps x {tok_per_step} patch tokens.\n"
        (OUT / f"patch_{k}.yaml").write_text(hdr + yaml.safe_dump(cfg, sort_keys=False))
    d, n_layers, h = GRID["s2"]
    n_body = 12 * n_layers * d * d
    s2_steps = max(100, int(round(TOKENS_PER_PARAM * n_body / tok_per_step / 100) * 100))
    steps = int(round(s2_steps * tok_per_step / (BATCH * (BINS_WINDOW - 1)) / 100) * 100)
    cfg = {
        "name": "bins_s2",
        "seed": 0,
        "model": {
            "tokenizer": "bins",
            "d_model": d,
            "n_layers": n_layers,
            "n_heads": h,
            "n_bins": 4094,
            "bin_limit": 15.0,
            "max_seq_len": 2048,
            "dropout": 0.0,
        },
        "data": data(BINS_WINDOW),
        "train": train(steps, LRS["s2"], 64),
    }
    hdr = f"# Tokenizer (a) at the s2 size and the same token budget as patch_s2 ({steps} steps x {BATCH * (BINS_WINDOW - 1)} tokens).\n"
    (OUT / "bins_s2.yaml").write_text(hdr + yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
