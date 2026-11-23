"""Export a patch model's single forward pass to ONNX, optionally quantise it, and check parity.

The exported graph takes what ``forecast_patch`` builds before its first forward pass (scaled
patches, their observed mask and the padding mask) and returns the last token's quantiles,
(batch, horizon, quantiles), in scaled units. Scaling, padding and the roll-out past one
horizon stay in Python (``tsfm.serve.engine``), so the graph has no data-dependent control flow.
Batch size and the number of patches are dynamic.

Parity is checked against PyTorch on random contexts of several lengths: the exported model
must agree to 1e-4 (fp32); the int8 model's error is measured and reported, not assumed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tsfm.model import TSModel


class LastTokenQuantiles(nn.Module):
    """``model(patches, observed, pad)[:, -1]`` as a plain module for export."""

    def __init__(self, model: TSModel):
        super().__init__()
        self.model = model

    def forward(self, patches: Tensor, observed: Tensor, pad: Tensor) -> Tensor:
        out: Tensor = self.model(patches=patches, observed=observed, pad=pad)[:, -1]
        return out


def example_inputs(
    model: TSModel, batch: int = 2, n_patches: int = 8
) -> tuple[Tensor, Tensor, Tensor]:
    p = model.cfg.patch_len
    g = torch.Generator().manual_seed(0)
    patches = torch.randn(batch, n_patches, p, generator=g)
    observed = torch.ones(batch, n_patches, p, dtype=torch.bool)
    pad = torch.zeros(batch, n_patches, dtype=torch.bool)
    return patches, observed, pad


def export(model: TSModel, path: Path) -> Path:
    """Write ``path`` (fp32 ONNX, opset 18) with dynamic batch and patch count."""
    model = model.eval()
    wrapper = LastTokenQuantiles(model).eval()
    batch = torch.export.Dim("batch", min=1, max=1024)
    n = torch.export.Dim("n_patches", min=1, max=model.cfg.max_seq_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example_inputs(model),
        str(path),
        input_names=["patches", "observed", "pad"],
        output_names=["quantiles"],
        dynamic_shapes={
            "patches": {0: batch, 1: n},
            "observed": {0: batch, 1: n},
            "pad": {0: batch, 1: n},
        },
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return path


def quantise(src: Path, dst: Path) -> Path:
    """Dynamic int8 quantisation of the MatMul weights (onnxruntime)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
    )
    return dst


def session(path: Path, threads: int | None = None) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run_onnx(sess: Any, patches: Tensor, observed: Tensor, pad: Tensor) -> np.ndarray:
    out: np.ndarray = sess.run(
        ["quantiles"],
        {
            "patches": patches.numpy().astype(np.float32),
            "observed": observed.numpy(),
            "pad": pad.numpy(),
        },
    )[0]
    return out


def parity(
    model: TSModel, path: Path, lengths: tuple[int, ...] = (1, 4, 16, 64), batch: int = 4
) -> dict[str, float]:
    """Largest absolute and relative difference between ONNX and PyTorch over several shapes."""
    sess = session(path)
    wrapper = LastTokenQuantiles(model).eval()
    worst_abs, worst_rel = 0.0, 0.0
    for n in lengths:
        if n > model.cfg.max_seq_len:
            continue
        patches, observed, pad = example_inputs(model, batch, n)
        pad[0, : n // 2] = True  # a left-padded row, as the server builds them
        observed[0, : n // 2] = False
        with torch.no_grad():
            ref = wrapper(patches, observed, pad).numpy()
        got = run_onnx(sess, patches, observed, pad)
        diff = np.abs(got - ref)
        worst_abs = max(worst_abs, float(diff.max()))
        worst_rel = max(worst_rel, float((diff / (np.abs(ref) + 1e-3)).max()))
    return {"max_abs": worst_abs, "max_rel": worst_rel}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_all(checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    from tsfm.forecast import load_model

    model = load_model(str(checkpoint))
    fp32 = export(model, out_dir / "model.onnx")
    int8 = quantise(fp32, out_dir / "model.int8.onnx")
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": model.num_params(),
        "fp32": {
            "file": fp32.name,
            "bytes": fp32.stat().st_size,
            "sha256": sha256(fp32),
            "parity": parity(model, fp32),
        },
        "int8": {
            "file": int8.name,
            "bytes": int8.stat().st_size,
            "sha256": sha256(int8),
            "parity": parity(model, int8),
        },
        "torch": torch.__version__,
    }
    from tsfm.serve.monitor import synthetic_reference

    np.save(out_dir / "reference_features.npy", synthetic_reference())
    (out_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
