#!/usr/bin/env python3
"""Plot Step45 final dCR(Gk-G0) curves for it5/it10 (2x2 panels)."""

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

OUT_DIR = Path(
    "docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224/figures"
)


def _algo_order() -> list[str]:
    return ["a2c", "ppo", "sac", "td3"]


def _group_order() -> list[str]:
    return ["G1_revise_only", "G2_intrinsic_only", "G3_revise_intrinsic"]


def _group_label(g: str) -> str:
    m = {
        "G1_revise_only": "G1-G0",
        "G2_intrinsic_only": "G2-G0",
        "G3_revise_intrinsic": "G3-G0",
    }
    return m.get(g, g)


def _algo_color_map() -> dict[str, str]:
    return {
        "a2c": "#1f77b4",
        "ppo": "#2aa198",
        "sac": "#c64747",
        "td3": "#9b4f96",
    }


def _build_delta_stats(run_dir: Path) -> dict[str, dict[str, tuple[float, float]]]:
    table = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    table = table[table["window_name"] != "aggregate"].copy()
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for algo in _algo_order():
        sub = table[table["algorithm"].str.lower() == algo].copy()
        if sub.empty:
            continue
        stats: dict[str, tuple[float, float]] = {}
        for group in _group_order():
            deltas = []
            for w in sorted(sub["window_name"].unique()):
                sw = sub[sub["window_name"] == w]
                g0 = sw[sw["group"] == "G0_baseline"]
                gk = sw[sw["group"] == group]
                if g0.empty or gk.empty:
                    continue
                v0 = float(g0.iloc[0]["CR_mean"])
                vk = float(gk.iloc[0]["CR_mean"])
                if np.isfinite(v0) and np.isfinite(vk):
                    deltas.append(vk - v0)
            if deltas:
                arr = np.array(deltas, dtype=float)
                stats[group] = (float(np.mean(arr)), float(np.std(arr)))
            else:
                stats[group] = (float("nan"), float("nan"))
        out[algo] = stats
    return out


def _plot_panel(ax: plt.Axes, stats: dict[str, dict[str, tuple[float, float]]], title: str) -> None:
    colors = _algo_color_map()
    x = np.arange(len(_group_order()), dtype=float)
    labels = [_group_label(g) for g in _group_order()]

    for algo in _algo_order():
        if algo not in stats:
            continue
        means = np.array([stats[algo][g][0] for g in _group_order()], dtype=float)
        stds = np.array([stats[algo][g][1] for g in _group_order()], dtype=float)
        ax.plot(
            x,
            means,
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            color=colors[algo],
            label=algo.upper(),
        )
        ax.fill_between(x, means - stds, means + stds, color=colors[algo], alpha=0.15, linewidth=0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("dCR (Gk - G0)")
    ax.set_xlabel("Group Delta")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.2, linestyle=":")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step45 it5/it10 final group dCR deltas.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharey=True)
    panel_order = [
        ("it5", "composite"),
        ("it5", "index_selected4"),
        ("it10", "composite"),
        ("it10", "index_selected4"),
    ]

    for ax, (it_setting, protocol) in zip(axes.flatten(), panel_order):
        run_dir = RUN_MAP[(it_setting, protocol)]
        stats = _build_delta_stats(run_dir)
        ttl = f"{it_setting} - {protocol}"
        _plot_panel(ax, stats, ttl)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    png_path = out_dir / "fig_step45_it5_it10_final_group_dcr_2x2.png"
    pdf_path = out_dir / "fig_step45_it5_it10_final_group_dcr_2x2.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] {png_path}")
    print(f"[ok] {pdf_path}")


if __name__ == "__main__":
    main()
