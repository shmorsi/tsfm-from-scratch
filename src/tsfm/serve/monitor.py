"""Production monitoring for the forecasting service: request metrics and input drift.

**Metrics.** Request counts, errors and a latency histogram per backend, served at ``/metrics``
in the Prometheus text exposition format (no client library: the format is a few lines).

**Drift.** Every request's series are summarised by four features (log length, share missing,
log10 of the mean absolute level, and the lag-1 autocorrelation of differences). The service
keeps the most recent ``window`` summaries and compares each feature's distribution with a
reference set by the population stability index,

    PSI = sum_b (p_b - q_b) ln(p_b / q_b)

over ten bins fixed on the reference's deciles. The usual reading: below 0.1 stable, 0.1 to 0.25
a moderate shift, above 0.25 a shift worth acting on. The reference is synthetic Gaussian-process
series like those the model was pretrained beside; a deployment would use its own first weeks.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

FEATURES = ("log_length", "missing_share", "log10_level", "diff_autocorr")
BUCKETS = (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0)


def features(series: np.ndarray) -> np.ndarray:
    x = np.asarray(series, float)
    obs = x[np.isfinite(x)]
    d = np.diff(obs)
    ac = (
        float(np.corrcoef(d[1:], d[:-1])[0, 1])
        if len(d) > 3 and d[1:].std() > 0 and d[:-1].std() > 0
        else 0.0
    )
    return np.array(
        [np.log(len(x)), 1 - len(obs) / len(x), np.log10(np.mean(np.abs(obs)) + 1e-12), ac]
    )


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference)
    q = np.histogram(current, edges)[0] / max(len(current), 1)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


@dataclass
class Monitor:
    reference: np.ndarray  # (n, len(FEATURES))
    window: int = 2000
    recent: deque = field(default_factory=lambda: deque(maxlen=2000))
    requests: dict = field(default_factory=lambda: defaultdict(int))
    errors: int = 0
    latency: dict = field(
        default_factory=lambda: defaultdict(lambda: np.zeros(len(BUCKETS) + 1, int))
    )
    latency_sum: dict = field(default_factory=lambda: defaultdict(float))

    def observe(self, series: Sequence[np.ndarray], backend: str, ms: float) -> None:
        for s in series:
            self.recent.append(features(s))
        self.requests[backend] += 1
        self.latency[backend][np.searchsorted(BUCKETS, ms)] += 1
        self.latency_sum[backend] += ms

    def drift(self) -> dict:
        if len(self.recent) < 200:  # PSI over ten bins has a sampling floor near 9 / n
            return {"status": "warming up", "series_seen": len(self.recent)}
        cur = np.array(self.recent)
        scores = {f: psi(self.reference[:, i], cur[:, i]) for i, f in enumerate(FEATURES)}
        worst = max(scores.values())
        status = (
            "stable"
            if worst < 0.1
            else ("moderate shift" if worst < 0.25 else "shift: review the inputs")
        )
        return {"status": status, "psi": scores, "series_seen": len(cur)}

    def prometheus(self) -> str:
        lines = [
            "# HELP tsfm_requests_total Forecast requests served, by backend.",
            "# TYPE tsfm_requests_total counter",
        ]
        lines += [
            f'tsfm_requests_total{{backend="{b}"}} {n}' for b, n in sorted(self.requests.items())
        ]
        lines += [
            "# HELP tsfm_request_errors_total Rejected or failed requests.",
            "# TYPE tsfm_request_errors_total counter",
            f"tsfm_request_errors_total {self.errors}",
            "# HELP tsfm_latency_ms Forecast latency in milliseconds, by backend.",
            "# TYPE tsfm_latency_ms histogram",
        ]
        for b, counts in sorted(self.latency.items()):
            cum = np.cumsum(counts)
            for edge, c in zip(BUCKETS, cum[:-1], strict=True):
                lines.append(f'tsfm_latency_ms_bucket{{backend="{b}",le="{edge}"}} {int(c)}')
            lines.append(f'tsfm_latency_ms_bucket{{backend="{b}",le="+Inf"}} {int(cum[-1])}')
            lines.append(f'tsfm_latency_ms_sum{{backend="{b}"}} {self.latency_sum[b]:.3f}')
            lines.append(f'tsfm_latency_ms_count{{backend="{b}"}} {int(cum[-1])}')
        d = self.drift()
        for f, v in d.get("psi", {}).items():
            lines.append(f'tsfm_input_psi{{feature="{f}"}} {v:.6f}')
        return "\n".join(lines) + "\n"


def synthetic_reference(n: int = 500, length: int = 512, seed: int = 7) -> np.ndarray:
    from tsfm.data.synthetic import sample_series

    return np.stack([features(sample_series(seed, i, length, 3)[0]) for i in range(n)])
