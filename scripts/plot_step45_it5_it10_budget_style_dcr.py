#!/usr/bin/env python3
"""Plot Step45 it5/it10 final dCR(G3-G0) in step43 budget-curve style (2x2 panels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_MAP = {
    ("it5", "composite"): Path("runs/20260223_035649_952_b6ed_demo"),
    ("it5", "index_selected4"): Path("runs/20260223_070935_643_681f_demo"),
    ("it10", "composite"): Path("runs/20260223_035649_950_5015_demo"),
    ("it10", "index_selected4"): Path("runs/20260223_152803_249_f8c2_demo"),
}

BUDGET_TICKS = [126, 252, 504]
OUT_DIR = Path(
    "docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224/figures"
)


def _algo_order() -> list[str]:
    return ["a2c", "ppo", "sac", "td3"]


def _algo_color_map() -> dict[str, str]:
    return {
        "a2c": "#1f77b4",
        "ppo": "#2aa198",
        "sac": "#c64747",
        "td3": "#9b4f96",
    }


def _panel_title(it_setting: str, protocol: str) -> str:
    return f"{it_setting} | {protocol}"


def _compute_g3_dcr_mean_std(run_dir: Path) -> dict[str, tuple[float, float]]:
    wf = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    wf = wf[wf["window_name"] != "aggregate"].copy()
    out: dict[str, tuple[float, float]] = {}
    for algo in _algo_order():
        sub = wf[wf["algorithm"].str.lower() == algo]
        deltas = []
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
            arr = np.array(deltas, dtype=float)
            out[algo] = (float(np.mean(arr)), float(np.std(arr)))
        else:
            out[algo] = (float("nan"), float("nan"))
    return out


def _plot_panel(ax: plt.Axes, stats: dict[str, tuple[float, float]], title: str) -> None:
    colors = _algo_color_map()
    for algo in _algo_order():
        mean, std = stats.get(algo, (float("nan"), float("nan")))
        x = np.array(BUDGET_TICKS, dtype=float)
        y = np.array([np.nan, mean, np.nan], dtype=float)
        ystd = np.array([np.nan, std, np.nan], dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[algo],
            label=algo.upper(),
        )
        # Keep same visual convention as step43 budget curve.
        ax.fill_between(x, y - ystd, y + ystd, color=colors[algo], alpha=0.14, linewidth=0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xticks(BUDGET_TICKS)
    ax.set_xlabel("Training Budget (configured n_full)")
    ax.set_ylabel("dCR (G3 - G0)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.2, linestyle=":")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step45 it5/it10 final dCR in budget-curve style.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), sharey=True)
    order = [
        ("it5", "composite"),
        ("it5", "index_selected4"),
        ("it10", "composite"),
        ("it10", "index_selected4"),
    ]
    for ax, (it_setting, protocol) in zip(axes.flatten(), order):
        stats = _compute_g3_dcr_mean_std(RUN_MAP[(it_setting, protocol)])
        _plot_panel(ax, stats, _panel_title(it_setting, protocol))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    png = out_dir / "fig_step45_it5_it10_budget_style_dcr_g3_2x2.png"
    pdf = out_dir / "fig_step45_it5_it10_budget_style_dcr_g3_2x2.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] {png}")
    print(f"[ok] {pdf}")


if __name__ == "__main__":
    main()
