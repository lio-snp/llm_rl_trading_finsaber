#!/usr/bin/env python3
"""Plot Step45 it5 final dCR(G3-G0) budget curves for composite and index_selected4."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_MAP = {
    ("composite", 126): Path("runs/20260224_151424_749_a557_demo"),
    ("composite", 252): Path("runs/20260223_035649_952_b6ed_demo"),
    ("composite", 504): Path("runs/20260224_151424_753_ad4f_demo"),
    ("index_selected4", 126): Path("runs/20260224_151424_754_dfc6_demo"),
    ("index_selected4", 252): Path("runs/20260223_070935_643_681f_demo"),
    ("index_selected4", 504): Path("runs/20260224_151424_759_80c5_demo"),
}

PROTOCOLS = ["composite", "index_selected4"]
BUDGETS = [126, 252, 504]
ALGOS = ["a2c", "ppo", "sac", "td3"]

OUT_DIR = Path(
    "docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224/figures"
)


def _load_scientific_figure_pro() -> Any:
    candidates = [
        Path("scripts/scientific_figure_pro.py").resolve(),
        Path.home() / ".codex/skills/scientific-figure-pro/scripts/scientific_figure_pro.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("scientific_figure_pro", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Cannot locate scientific_figure_pro.py helper module.")


def _algo_color_map(sfp: Any) -> dict[str, str]:
    return {
        "a2c": sfp.PALETTE["blue_main"],
        "ppo": sfp.PALETTE["teal"],
        "sac": sfp.PALETTE["red_strong"],
        "td3": sfp.PALETTE["violet"],
    }


def _dcr_stats(run_dir: Path) -> dict[str, tuple[float, float]]:
    wf = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    wf = wf[wf["window_name"] != "aggregate"].copy()
    out: dict[str, tuple[float, float]] = {}
    for algo in ALGOS:
        sub = wf[wf["algorithm"].str.lower() == algo]
        deltas: list[float] = []
        for w in sorted(sub["window_name"].unique()):
            sw = sub[sub["window_name"] == w]
            g0 = sw[sw["group"] == "G0_baseline"]
            g3 = sw[sw["group"] == "G3_revise_intrinsic"]
            if g0.empty or g3.empty:
                continue
            v0 = float(g0.iloc[0]["CR_mean"])
            v3 = float(g3.iloc[0]["CR_mean"])
            if np.isfinite(v0) and np.isfinite(v3):
                deltas.append(v3 - v0)
        if deltas:
            arr = np.asarray(deltas, dtype=float)
            out[algo] = (float(np.mean(arr)), float(np.std(arr)))
        else:
            out[algo] = (float("nan"), float("nan"))
    return out


def _plot_protocol(protocol: str, out_dir: Path, sfp: Any, dpi: int) -> None:
    colors = _algo_color_map(sfp)
    fig, axes = sfp.create_subplots(1, 1, figsize=(7.2, 5.6))
    ax = np.atleast_1d(axes).flatten()[0]

    for algo in ALGOS:
        means = []
        stds = []
        xs = []
        for budget in BUDGETS:
            run = RUN_MAP.get((protocol, budget))
            if run is None:
                continue
            stats = _dcr_stats(run)
            mean, std = stats.get(algo, (float("nan"), float("nan")))
            xs.append(float(budget))
            means.append(mean)
            stds.append(std)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(means, dtype=float)
        ystd = np.asarray(stds, dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[algo],
            label=algo.upper(),
        )
        ax.fill_between(x, y - ystd, y + ystd, color=colors[algo], alpha=0.14, linewidth=0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Training Budget (configured n_full)")
    ax.set_ylabel("dCR (G3 - G0)")
    ax.set_title(
        f"A. Budget Curve (dCR) - {protocol} (it5)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(ncol=2, loc="best")

    base = out_dir / f"fig_step45_it5_budget_curve_dcr_{protocol}"
    sfp.finalize_figure(fig, base, formats=["png", "pdf"], dpi=dpi, pad=0.06)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step45 it5 dCR budget curves.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sfp = _load_scientific_figure_pro()
    if hasattr(sfp, "set_style"):
        sfp.set_style("paper")

    for protocol in PROTOCOLS:
        _plot_protocol(protocol, out_dir, sfp, args.dpi)

    print(f"[ok] {out_dir / 'fig_step45_it5_budget_curve_dcr_composite.png'}")
    print(f"[ok] {out_dir / 'fig_step45_it5_budget_curve_dcr_index_selected4.png'}")


if __name__ == "__main__":
    main()
