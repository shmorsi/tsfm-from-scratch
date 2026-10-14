"""Training loop: AdamW, warmup + cosine LR, gradient accumulation, mixed precision,
time-based checkpoints and exact resume.

Exact resume works because (1) every batch and every dropout mask is a pure function of
``(seed, step, micro_step)``, and (2) the checkpoint holds the model, optimiser and
grad-scaler state at a step boundary. On CPU a resumed run is bit-identical to an
uninterrupted one (tested). On GPU some kernels are not bit-deterministic, so the two
agree only up to that nondeterminism.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tsfm.model import ModelConfig, TSModel
from tsfm.paths import processed_dir, runs_dir
from tsfm.provenance import config_hash, git_state
from tsfm.train.data import MixtureSampler, Packed, step_rng
from tsfm.train.objective import compute_loss, draw_context


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 64
    micro_batch: int = 64
    lr: float = 1e-3
    warmup: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "fp32"  # fp32 | fp16 | bf16
    device: str = "auto"
    ckpt_minutes: float = 10.0
    ckpt_every_steps: int | None = None
    eval_every: int = 200
    log_every: int = 10


@dataclass
class DataConfig:
    corpora: list[dict[str, Any]] = field(default_factory=list)  # {path, synthetic}
    val: list[dict[str, Any]] = field(default_factory=list)
    window: int = 512
    min_context: int = 16
    max_context: int | None = None
    synthetic_share: float = 0.1
    temperature: float = 0.5
    val_batches: int = 8
    val_batch_size: int = 64
    val_seed: int = 12345


@dataclass
class RunConfig:
    name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        t = dict(raw.get("train", {}))
        if "betas" in t:
            t["betas"] = tuple(t["betas"])
        for k in ("lr", "weight_decay", "min_lr_ratio", "grad_clip", "ckpt_minutes"):
            if k in t:
                t[k] = float(t[k])
        return cls(
            name=raw["name"],
            seed=int(raw.get("seed", 0)),
            model=ModelConfig.from_dict(raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**t),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "name": self.name,
            "seed": self.seed,
            "model": self.model.to_dict(),
            "data": asdict(self.data),
            "train": {**asdict(self.train), "betas": list(self.train.betas)},
        }


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: TSModel, cfg: TrainConfig) -> torch.optim.AdamW:
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        fused=False,
    )


def autocast(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def open_corpora(entries: list[dict[str, Any]]) -> tuple[list[Packed], list[bool]]:
    root = Path(os.environ.get("TSFM_PACKED", processed_dir() / "packed"))
    packs = [Packed.open(root / e["path"]) for e in entries]
    return packs, [bool(e.get("synthetic", False)) for e in entries]


def window_len(rc: RunConfig) -> int:
    w = rc.data.window
    if rc.model.tokenizer == "patch" and w % rc.model.patch_len:
        raise ValueError("data.window must be a multiple of patch_len")
    if rc.model.tokenizer == "bins" and w > rc.model.max_seq_len + 1:
        raise ValueError("bins: data.window must be <= max_seq_len + 1")
    if rc.model.tokenizer == "patch" and w // rc.model.patch_len > rc.model.max_seq_len:
        raise ValueError("patch: window / patch_len must be <= max_seq_len")
    return w


def batch_tensors(
    sampler: MixtureSampler, rc: RunConfig, rng: np.random.Generator, n: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    w = window_len(rc)
    x = sampler.sample(rng, n, w)
    ctx = draw_context(rng, x, rc.data.min_context, rc.data.max_context or (w - 1))
    return torch.from_numpy(x).to(device), torch.from_numpy(ctx).to(device)


def tokens_per_window(rc: RunConfig) -> int:
    w = rc.data.window
    return w // rc.model.patch_len if rc.model.tokenizer == "patch" else w - 1


@dataclass
class Trainer:
    rc: RunConfig
    run_dir: Path

    def __post_init__(self) -> None:
        self.device = pick_device(self.rc.train.device)
        torch.manual_seed(self.rc.seed)
        self.model = TSModel(self.rc.model).to(self.device)
        self.opt = make_optimizer(self.model, self.rc.train)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.rc.train.precision == "fp16"
        )
        self.step = 0
        packs, syn = open_corpora(self.rc.data.corpora)
        self.sampler = MixtureSampler(
            packs, syn, self.rc.data.synthetic_share, self.rc.data.temperature
        )
        self.val_sampler: MixtureSampler | None = None
        if self.rc.data.val:
            vp, vs = open_corpora(self.rc.data.val)
            # Validation: every dataset equally weighted (temperature 0), fixed windows.
            self.val_sampler = MixtureSampler(
                vp, vs, synthetic_share=0.5 if any(vs) and not all(vs) else 0.1, temperature=0.0
            )
        self.ckpt_path = self.run_dir / "ckpt.pt"
        self.metrics_path = self.run_dir / "metrics.jsonl"

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.rc.to_dict(),
            "config_hash": config_hash(self.rc.to_dict()),
        }

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt_path.with_suffix(".pt.tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, self.ckpt_path)

    def try_resume(self) -> bool:
        if not self.ckpt_path.exists():
            return False
        st = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        if st["config_hash"] != config_hash(self.rc.to_dict()):
            raise ValueError(f"{self.ckpt_path} was written with a different config")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
        self.step = int(st["step"])
        self._truncate_metrics(self.step)
        return True

    def _truncate_metrics(self, step: int) -> None:
        """Drop log lines written after the checkpoint so a resumed log has no duplicates."""
        if not self.metrics_path.exists():
            return
        keep = [
            ln
            for ln in self.metrics_path.read_text().splitlines()
            if json.loads(ln)["step"] <= step
        ]
        self.metrics_path.write_text("".join(ln + "\n" for ln in keep))

    def log(self, rec: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ steps
    def train_step(self) -> dict[str, float]:
        tc = self.rc.train
        if tc.batch_size % tc.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")
        n_micro = tc.batch_size // tc.micro_batch
        lr = lr_at(self.step, tc)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total, n_tok = 0.0, 0
        for m in range(n_micro):
            torch.manual_seed(int(step_rng(self.rc.seed, self.step, m, stream=1).integers(2**62)))
            x, ctx = batch_tensors(
                self.sampler,
                self.rc,
                step_rng(self.rc.seed, self.step, m),
                tc.micro_batch,
                self.device,
            )
            with autocast(self.device, tc.precision):
                loss, k = compute_loss(self.model, x, ctx)
            self.scaler.scale(loss / n_micro).backward()
            total += float(loss.detach()) / n_micro
            n_tok += k
        self.scaler.unscale_(self.opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip))
        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        return {"loss": total, "lr": lr, "grad_norm": gnorm, "loss_targets": n_tok}

    @torch.no_grad()
    def val_loss(self) -> float | None:
        b = self.val_losses()
        return float(np.mean(b)) if b is not None else None

    @torch.no_grad()
    def val_losses(self) -> list[float] | None:
        """Loss on each fixed validation batch (same windows for every run and step)."""
        if self.val_sampler is None:
            return None
        self.model.eval()
        losses = []
        for i in range(self.rc.data.val_batches):
            rng = np.random.default_rng([self.rc.data.val_seed, i])
            x, ctx = batch_tensors(
                self.val_sampler, self.rc, rng, self.rc.data.val_batch_size, self.device
            )
            with autocast(self.device, self.rc.train.precision):
                loss, _ = compute_loss(self.model, x, ctx)
            losses.append(float(loss))
        return losses

    def run(
        self, *, max_steps_this_session: int | None = None, time_budget_min: float | None = None
    ) -> dict[str, Any]:
        resumed = self.try_resume()
        tc = self.rc.train
        t_start = time.time()
        last_ckpt = time.time()
        steps_done = 0
        while self.step < tc.steps:
            t0 = time.time()
            rec: dict[str, Any] = dict(self.train_step())
            steps_done += 1
            rec["step"] = self.step
            rec["step_seconds"] = time.time() - t0
            if self.step % tc.eval_every == 0 or self.step == tc.steps:
                rec["val_loss"] = self.val_loss()
            if self.step % tc.log_every == 0 or "val_loss" in rec:
                self.log(rec)
            due_time = (time.time() - last_ckpt) / 60 >= tc.ckpt_minutes
            due_step = tc.ckpt_every_steps is not None and self.step % tc.ckpt_every_steps == 0
            stop = (
                max_steps_this_session is not None and steps_done >= max_steps_this_session
            ) or (time_budget_min is not None and (time.time() - t_start) / 60 >= time_budget_min)
            if due_time or due_step or stop or self.step == tc.steps:
                self.save()
                last_ckpt = time.time()
            if stop:
                break
        done = self.step >= tc.steps
        summary = {
            "done": done,
            "step": self.step,
            "resumed": resumed,
            "session_seconds": time.time() - t_start,
            **self.compute_summary(),
        }
        if done:
            self.finalize(summary)
        return summary

    def compute_summary(self) -> dict[str, Any]:
        n_body = self.model.num_params(body_only=True)
        tokens = self.step * self.rc.train.batch_size * tokens_per_window(self.rc)
        seq = tokens_per_window(self.rc)
        return {
            "params_total": self.model.num_params(),
            "params_body": n_body,
            "train_tokens": tokens,
            # C = 6 N D (Kaplan et al.); attention-score FLOPs reported separately.
            "train_flops_6ND": 6 * n_body * tokens,
            "train_flops_attention": 6
            * self.rc.model.n_layers
            * seq
            * self.rc.model.d_model
            * tokens,
        }

    def finalize(self, summary: dict[str, Any]) -> None:
        torch.save(
            {"model": self.model.state_dict(), "config": self.rc.to_dict()},
            self.run_dir / "model.pt",
        )
        batches = self.val_losses()
        summary = {
            **summary,
            "val_loss_final": float(np.mean(batches)) if batches else None,
            "val_loss_batches": batches,
            "config": self.rc.to_dict(),
            "config_hash": config_hash(self.rc.to_dict()),
            **git_state(),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )


def train(config: str | Path, run_dir: Path | None = None, **kw: Any) -> dict[str, Any]:
    rc = RunConfig.load(config)
    run_dir = run_dir or runs_dir() / rc.name
    return Trainer(rc, run_dir).run(**kw)


def iter_metrics(run_dir: Path) -> Iterator[dict[str, Any]]:
    p = run_dir / "metrics.jsonl"
    if p.exists():
        for ln in p.read_text().splitlines():
            yield json.loads(ln)
