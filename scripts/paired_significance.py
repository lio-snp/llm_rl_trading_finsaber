from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_run_dir(run_id: str | None, run_dir: str | None) -> Path:
    root = _repo_root()
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p
    if not run_id:
        raise ValueError("Either --run-id or --run-dir is required.")
    return (root / "runs" / run_id).resolve()


def _normal_2sided_p_from_t(t_stat: float) -> float:
    z = abs(float(t_stat))
    return float(math.erfc(z / math.sqrt(2.0)))


def _paired_t_pvalue(diff: np.ndarray) -> tuple[float, float]:
    if diff.size <= 1:
        return 0.0, float("nan")
    std = float(np.std(diff, ddof=1))
    if std <= 1e-12:
        return 0.0, 1.0
    t_stat = float(np.mean(diff) / (std / math.sqrt(diff.size)))
    try:
        from scipy import stats  # type: ignore

        p = float(stats.t.sf(abs(t_stat), df=diff.size - 1) * 2.0)
    except Exception:
        p = _normal_2sided_p_from_t(t_stat)
    return t_stat, p


def _bootstrap_ci(diff: np.ndarray, n_resamples: int, alpha: float, seed: int) -> tuple[float, float]:
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = diff.size
    for _ in range(int(max(100, n_resamples))):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(diff[idx])))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--metrics", nargs="+", default=["Score"], help="Score Sharpe_mean CR_mean ...")
    parser.add_argument("--n-resamples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    root = _repo_root()
    run_dir = _resolve_run_dir(args.run_id, args.run_dir)
    wf_table = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    wf_table = wf_table[wf_table["window_index"].astype(str) != "aggregate"].copy()
    wf_table["Score"] = wf_table["Sharpe_mean"].astype(float) + wf_table["CR_mean"].astype(float)
    metrics: List[str] = [str(m) for m in args.metrics]

    rows = []
    for metric in metrics:
        if metric not in wf_table.columns:
            continue
        for algo, grp in wf_table.groupby("algorithm"):
            g0 = grp[grp["group"] == "G0_baseline"][["window_index", metric]].rename(columns={metric: "base"})
            if g0.empty:
                continue
            for group in ["G1_revise_only", "G2_intrinsic_only", "G3_revise_intrinsic"]:
                gk = grp[grp["group"] == group][["window_index", metric]].rename(columns={metric: "test"})
                merged = g0.merge(gk, on="window_index", how="inner")
                if merged.empty:
                    continue
                diff = (merged["test"].astype(float) - merged["base"].astype(float)).to_numpy(dtype=float)
                n = int(diff.size)
                mean_delta = float(np.mean(diff))
                std_delta = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                t_stat, p_val = _paired_t_pvalue(diff)
                ci_low, ci_high = _bootstrap_ci(
                    diff=diff,
                    n_resamples=int(args.n_resamples),
                    alpha=float(args.alpha),
                    seed=int(args.seed),
                )
                rows.append(
                    {
                        "run_id": run_dir.name,
                        "algorithm": str(algo),
                        "group": group,
                        "metric": metric,
                        "n_windows": n,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["metric", "algorithm", "group"]).reset_index(drop=True)
    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
