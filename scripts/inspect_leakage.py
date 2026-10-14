"""Look at leakage hits: is a match a real copy or a coincidence?

For every flagged pretraining series (from results/data/leakage.json) this finds the
matching GIFT-Eval windows and measures how far the agreement extends: the length of the
longest aligned stretch around the hit where the two series agree after the same affine
map (fitted on the hit window). A real copy agrees far beyond the 64-point window; a
coincidence (e.g. a flat segment with a few steps) does not.

    uv run python scripts/inspect_leakage.py --max-series 2000

writes results/data/leakage_inspection_series.jsonl (one line per flagged series and
GIFT-Eval dataset) and the per-pair summary results/data/leakage_inspection.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tsfm.data import leakage as L
from tsfm.data.convert import _list_values, iter_raw_series
from tsfm.paths import gifteval_dir, processed_dir, results_dir


def gift_series(ds_name: str) -> list[tuple[str, np.ndarray]]:
    paths = sorted((gifteval_dir() / ds_name).glob("data-*-of-*.arrow"))
    return [(s.item_id, s.values) for s in iter_raw_series(paths, ds_name)]


def agreement_length(a: np.ndarray, ia: int, b: np.ndarray, ib: int, w: int) -> dict[str, float]:
    """Fit b ~ alpha * a + beta on the hit window, then extend left/right while the
    relative error stays below 1e-3 of b's window std."""
    wa, wb = a[ia : ia + w].astype(np.float64), b[ib : ib + w].astype(np.float64)
    A = np.vstack([wa, np.ones_like(wa)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, wb, rcond=None)
    tol = 1e-3 * max(float(np.std(wb)), 1e-12)
    n = 0
    left = 0
    while ia - left - 1 >= 0 and ib - left - 1 >= 0:
        x, y = a[ia - left - 1], b[ib - left - 1]
        if not (np.isfinite(x) and np.isfinite(y)) or abs(alpha * x + beta - y) > tol:
            break
        left += 1
    right = 0
    while ia + w + right < a.size and ib + w + right < b.size:
        x, y = a[ia + w + right], b[ib + w + right]
        if not (np.isfinite(x) and np.isfinite(y)) or abs(alpha * x + beta - y) > tol:
            break
        right += 1
    n = left + w + right
    distinct = len(np.unique(np.round(wa, 6)))
    z = (wa - wa.mean()) / (wa.std() or 1.0)
    q = np.rint(z * L.Q)
    linear_frac = float(np.mean(np.abs(np.diff(q, 2)) <= 1))
    return {
        "agree_len": n,
        "alpha": float(alpha),
        "beta": float(beta),
        "window_distinct_values": distinct,
        "linear_fraction": linear_frac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-series", type=int, default=30)
    a = ap.parse_args()
    rep = json.loads((results_dir() / "data" / "leakage.json").read_text())
    by_gift: dict[str, list[dict]] = defaultdict(list)
    for r in rep["leaking_series"][: a.max_series]:
        for g in r["matched"]:
            by_gift[g.replace(" (daily)", "")].append(r)
    out = []
    for g, items in by_gift.items():
        gs = gift_series(g)
        index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for si, (_, x) in enumerate(gs):
            h, st = L.window_hashes(x, L.W, 1)
            for hh, ss in zip(h.tolist(), st.tolist(), strict=True):
                index[hh].append((si, ss))
        for r in items:
            split = "val" if (processed_dir() / "val" / r["file"]).exists() else "pretrain"
            t = pq.read_table(processed_dir() / split / r["file"])
            ids = t.column("item_id").to_pylist()
            i = ids.index(r["item_id"])
            vals, offs = _list_values(t.column("target"))
            x = vals[offs[i] : offs[i + 1]]
            h, st = L.window_hashes(x, L.W, L.S)
            best = None
            for hh, ss in zip(h.tolist(), st.tolist(), strict=True):
                for si, gpos in index.get(hh, [])[:3]:
                    info = agreement_length(x, ss, gs[si][1], gpos, L.W)
                    info.update(
                        {
                            "gift_series": gs[si][0],
                            "gift_len": int(gs[si][1].size),
                            "pre_len": int(x.size),
                        }
                    )
                    if best is None or info["agree_len"] > best["agree_len"]:
                        best = info
            out.append(
                {"item_id": r["item_id"], "gift_dataset": g, "hits": r["hits"], "best": best}
            )
            print(json.dumps(out[-1]), flush=True)
    series_file = results_dir() / "data" / "leakage_inspection_series.jsonl"
    series_file.write_text("".join(json.dumps(r) + "\n" for r in out))
    print("wrote", write_summary(series_file))
    return 0


def summarise(rows: list[dict], leaking: list[dict] | None = None) -> dict:
    """Aggregate per (source dataset, GIFT-Eval dataset) and classify.

    "coincidence (straight-line windows)": median linear fraction > 0.5;
    "copy": not linear, and the affine agreement reaches the guaranteed detection length;
    "uncertain (short non-linear overlap)": otherwise. Every flagged series is excluded
    from the corpus whatever its verdict."""
    from collections import defaultdict as dd

    test_hits = {r["item_id"]: r["test_region_hits"] for r in (leaking or [])}
    agg: dict[tuple[str, str], dict[str, dict]] = dd(dict)
    for r in rows:  # one entry per (series, GIFT-Eval dataset): native and daily merged
        agg[(r["item_id"].split("/")[0], r["gift_dataset"])].setdefault(r["item_id"], r["best"])
    out = []
    for (src, gift), per_series in sorted(agg.items()):
        bs = [b for b in per_series.values() if b]
        n_series = len(per_series)
        n_test = sum(1 for i in per_series if test_hits.get(i, 0) > 0)
        if not bs:
            out.append(
                {
                    "source": src,
                    "gifteval": gift,
                    "series": n_series,
                    "series_with_test_window_hits": n_test,
                    "verdict": "daily-aggregate hits only",
                }
            )
            continue
        al = sorted(b["agree_len"] for b in bs)
        lin = sorted(b["linear_fraction"] for b in bs)
        alpha = sorted(b["alpha"] for b in bs)
        med_lin = lin[len(lin) // 2]
        med_al = al[len(al) // 2]
        if med_lin > 0.5:
            verdict = "coincidence (straight-line windows)"
        elif med_al >= L.MIN_SHARED_GUARANTEED:
            verdict = "copy"
        else:
            verdict = "uncertain (short non-linear overlap)"
        out.append(
            {
                "source": src,
                "gifteval": gift,
                "series": n_series,
                "series_with_test_window_hits": n_test,
                "agree_len_median": med_al,
                "agree_len_max": al[-1],
                "linear_fraction_median": med_lin,
                "alpha_median": alpha[len(alpha) // 2],
                "verdict": verdict,
            }
        )
    return {"pairs": out}


def write_summary(jsonl: Path) -> Path:
    from tsfm.provenance import write_result

    rows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.startswith("{")]
    r1 = json.loads((results_dir() / "data" / "leakage_round1.json").read_text())
    return write_result(
        results_dir() / "data" / "leakage_inspection.json",
        summarise(rows, r1["leaking_series"]),
        split="none",
        label="M1-leakage-inspection",
    )


if __name__ == "__main__":
    raise SystemExit(main())
