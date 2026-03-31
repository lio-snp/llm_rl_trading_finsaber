#!/usr/bin/env python3
"""Plot LESR efficiency report figures from precomputed CSV tables.

This script reads the CSV artifacts under:
docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224
and generates four report-ready figures (PNG + PDF):
1) budget-performance curve
2) minimum budget to pass threshold
3) window-level stability heatmap (budget=252)
4) LESR iteration effectiveness with valid-window ratio overlay
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REPORT_DIR = Path(
    "docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224"
)

STEP43_RUN_MAP = {
    ("composite", 126): Path("runs/20260222_124434_729_8869_demo"),
    ("composite", 252): Path("runs/20260222_125617_032_2276_demo"),
    ("composite", 504): Path("runs/20260222_131025_865_28c9_demo"),
    ("index_selected4", 126): Path("runs/20260222_141122_152_3899_demo"),
    ("index_selected4", 252): Path("runs/20260222_142952_497_8507_demo"),
    ("index_selected4", 504): Path("runs/20260222_145117_276_44de_demo"),
    ("selected4", 126): Path("runs/20260222_132430_572_39c0_demo"),
    ("selected4", 252): Path("runs/20260222_133728_318_14a3_demo"),
    ("selected4", 504): Path("runs/20260222_135231_544_fe4b_demo"),
}

METRIC_FIELD_MAP = {
    "sharpe": "Sharpe_mean",
    "cr": "CR_mean",
    "av": "AV_mean",
}

METRIC_DELTA_LABEL = {
    "sharpe": "dSharpe",
    "cr": "dCR",
    "av": "dAV",
}


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float_or_nan(value: str) -> float:
    if value is None:
        return float("nan")
    v = value.strip()
    if not v or v.upper() == "NA":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _protocol_order() -> list[str]:
    return ["composite", "selected4", "index_selected4"]


def _algo_order() -> list[str]:
    return ["a2c", "ppo", "sac", "td3"]


def _algo_color_map(sfp: Any) -> dict[str, str]:
    return {
        "a2c": sfp.PALETTE["blue_main"],
        "ppo": sfp.PALETTE["teal"],
        "sac": sfp.PALETTE["red_strong"],
        "td3": sfp.PALETTE["violet"],
    }


def _human_protocol(name: str) -> str:
    if name == "composite":
        return "composite"
    if name == "index_selected4":
        return "index_selected4"
    if name == "selected4":
        return "selected4"
    return name


def _build_budget_curve_rows(metric: str) -> list[dict[str, str]]:
    metric_key = metric.lower().strip()
    if metric_key not in METRIC_FIELD_MAP:
        raise ValueError(f"Unsupported curve metric: {metric}")
    field = METRIC_FIELD_MAP[metric_key]
    rows_out: list[dict[str, str]] = []
    for protocol in _protocol_order():
        for budget in [126, 252, 504]:
            run = STEP43_RUN_MAP.get((protocol, budget))
            if run is None:
                continue
            table_path = run / "walk_forward_metrics_table.csv"
            if not table_path.exists():
                continue
            rows = _read_csv(table_path)
            windows = sorted({r["window_name"] for r in rows if r["window_name"] != "aggregate"})
            for algo in _algo_order():
                deltas: list[float] = []
                for w in windows:
                    sub = [r for r in rows if r["window_name"] == w and r["algorithm"].lower().strip() == algo]
                    g = {r["group"]: _to_float_or_nan(r.get(field, "")) for r in sub}
                    if "G0_baseline" in g and "G3_revise_intrinsic" in g:
                        if np.isfinite(g["G0_baseline"]) and np.isfinite(g["G3_revise_intrinsic"]):
                            deltas.append(float(g["G3_revise_intrinsic"] - g["G0_baseline"]))
                if not deltas:
                    continue
                mean = float(np.mean(np.array(deltas, dtype=float)))
                std = float(np.std(np.array(deltas, dtype=float)))
                pass_ratio = float(np.mean(np.array([1.0 if d > 0 else 0.0 for d in deltas], dtype=float)))
                rows_out.append(
                    {
                        "protocol": protocol,
                        "budget": str(int(budget)),
                        "algo": algo,
                        "dmetric_mean": str(mean),
                        "dmetric_std": str(std),
                        "pass_ratio_windows": str(pass_ratio),
                        "window_count": str(len(deltas)),
                    }
                )
    return rows_out


def plot_budget_curve(report_dir: Path, out_dir: Path, sfp: Any, dpi: int, metric: str) -> None:
    metric_key = metric.lower().strip()
    if metric_key not in METRIC_FIELD_MAP:
        raise ValueError(f"Unsupported curve metric: {metric}")

    if metric_key == "sharpe":
        sharpe_csv = report_dir / "budget_curve_step43.csv"
        if sharpe_csv.exists():
            rows_raw = _read_csv(sharpe_csv)
            rows = [
                {
                    "protocol": r["protocol"],
                    "budget": r["budget"],
                    "algo": r["algo"],
                    "dmetric_mean": r["dSharpe_mean"],
                    "dmetric_std": r["dSharpe_std"],
                }
                for r in rows_raw
            ]
        else:
            rows = _build_budget_curve_rows("sharpe")
    else:
        rows = _build_budget_curve_rows(metric_key)

    colors = _algo_color_map(sfp)
    protocols = [p for p in _protocol_order() if any(r["protocol"] == p for r in rows)]

    fig, axes = sfp.create_subplots(1, len(protocols), figsize=(6.2 * len(protocols), 5.2))
    axes = np.atleast_1d(axes)

    for ax, protocol in zip(axes, protocols):
        sub = [r for r in rows if r["protocol"] == protocol]
        budgets = sorted({int(r["budget"]) for r in sub})
        for algo in _algo_order():
            points = sorted(
                [r for r in sub if r["algo"] == algo],
                key=lambda x: int(x["budget"]),
            )
            if not points:
                continue
            x = np.array([int(r["budget"]) for r in points], dtype=float)
            y = np.array([_to_float_or_nan(r["dmetric_mean"]) for r in points], dtype=float)
            ystd = np.array([_to_float_or_nan(r["dmetric_std"]) for r in points], dtype=float)
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
        ax.set_xticks(budgets)
        ax.set_xlabel("Training Budget (configured n_full)")
        ax.set_ylabel(f"{METRIC_DELTA_LABEL[metric_key]} (G3 - G0)")
        ax.set_title(
            f"A. Budget Curve ({METRIC_DELTA_LABEL[metric_key]}) - {_human_protocol(protocol)}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(ncol=2, loc="best")

    out_name = "fig_A_budget_curve_step43" if metric_key == "sharpe" else f"fig_A_budget_curve_step43_{metric_key}"
    sfp.finalize_figure(
        fig,
        out_dir / out_name,
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def plot_min_budget(report_dir: Path, out_dir: Path, sfp: Any, dpi: int) -> None:
    rows = _read_csv(report_dir / "min_budget_target_step43.csv")
    colors = _algo_color_map(sfp)
    protocols = [p for p in _protocol_order() if any(r["protocol"] == p for r in rows)]
    algos = _algo_order()

    # Prepare matrix protocol x algo.
    matrix = np.full((len(protocols), len(algos)), np.nan, dtype=float)
    for i, p in enumerate(protocols):
        for j, a in enumerate(algos):
            rec = next((r for r in rows if r["protocol"] == p and r["algo"] == a), None)
            if rec is None:
                continue
            matrix[i, j] = _to_float_or_nan(rec["min_budget_for_dSharpe_gt_0"])

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(protocols), dtype=float)
    n_algo = len(algos)
    width = 0.76 / n_algo
    cap_level = 560.0

    for j, algo in enumerate(algos):
        off = (j - (n_algo - 1) / 2.0) * width
        vals = matrix[:, j]
        draw_vals = np.where(np.isfinite(vals), vals, cap_level)
        bars = ax.bar(
            x + off,
            draw_vals,
            width=width,
            color=colors[algo],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            label=algo.upper(),
        )
        for i, b in enumerate(bars):
            if not np.isfinite(vals[i]):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    cap_level + 12.0,
                    "NR",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_human_protocol(p) for p in protocols])
    ax.set_ylim(0, 620)
    ax.set_yticks([0, 126, 252, 504, 600])
    ax.set_yticklabels(["0", "126", "252", "504", "NR zone"])
    ax.set_ylabel("Minimum Budget to Reach dSharpe(G3-G0) > 0")
    ax.set_title("B. Minimum Budget To Positive Delta", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    ax.legend(ncol=4, loc="upper left")

    sfp.finalize_figure(
        fig,
        out_dir / "fig_B_min_budget_step43",
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def _window_sort_key(name: str) -> tuple[int, str]:
    # expected format: wf_window_00
    try:
        return int(name.rsplit("_", 1)[-1]), name
    except ValueError:
        return 9999, name


def plot_heatmap(report_dir: Path, out_dir: Path, sfp: Any, dpi: int) -> None:
    rows = _read_csv(report_dir / "heatmap_window_b252_step43.csv")
    protocols = [p for p in _protocol_order() if any(r["protocol"] == p for r in rows)]
    algos = _algo_order()

    all_vals = [_to_float_or_nan(r["dSharpe_G3_G0"]) for r in rows]
    vmax = max(abs(v) for v in all_vals if np.isfinite(v))
    vmax = max(vmax, 1e-6)

    fig, axes = sfp.create_subplots(1, len(protocols), figsize=(5.7 * len(protocols), 6.6))
    axes = np.atleast_1d(axes)
    im_ref = None
    for ax, protocol in zip(axes, protocols):
        sub = [r for r in rows if r["protocol"] == protocol]
        windows = sorted({r["window"] for r in sub}, key=_window_sort_key)
        mat = np.full((len(windows), len(algos)), np.nan, dtype=float)
        for i, w in enumerate(windows):
            for j, a in enumerate(algos):
                rec = next((r for r in sub if r["window"] == w and r["algo"] == a), None)
                if rec is not None:
                    mat[i, j] = _to_float_or_nan(rec["dSharpe_G3_G0"])

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        im_ref = im
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels([a.upper() for a in algos], rotation=0)
        ax.set_yticks(np.arange(len(windows)))
        ax.set_yticklabels(windows)
        ax.set_title(f"C. Stability Heatmap - {_human_protocol(protocol)}", loc="left", fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    text_color = "white" if abs(v) > (0.55 * vmax) else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=text_color)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=list(axes), shrink=0.92, pad=0.015)
        cbar.set_label("dSharpe (G3 - G0)")

    sfp.finalize_figure(
        fig,
        out_dir / "fig_C_stability_heatmap_b252",
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def plot_iteration_effectiveness(report_dir: Path, out_dir: Path, sfp: Any, dpi: int) -> None:
    rows = _read_csv(report_dir / "iteration_effectiveness_step45.csv")
    protocols = [p for p in _protocol_order() if any(r["protocol"] == p for r in rows)]
    it_order = ["it5", "it10"]
    color_map = {"it5": sfp.PALETTE["blue_main"], "it10": sfp.PALETTE["red_strong"]}

    fig, axes = sfp.create_subplots(1, len(protocols), figsize=(6.3 * len(protocols), 5.8))
    axes = np.atleast_1d(axes)

    for ax, protocol in zip(axes, protocols):
        sub_p = [r for r in rows if r["protocol"] == protocol]
        ax2 = ax.twinx()

        for it_name in it_order:
            sub = sorted(
                [r for r in sub_p if r["iter_setting"] == it_name],
                key=lambda x: int(x["iter"]),
            )
            if not sub:
                continue

            x = np.array([int(r["iter"]) for r in sub], dtype=float)
            y = np.array([_to_float_or_nan(r["mean_best_candidate_sharpe"]) for r in sub], dtype=float)
            valid = np.array([_to_float_or_nan(r["valid_window_count"]) for r in sub], dtype=float)
            total = np.array([_to_float_or_nan(r["total_windows"]) for r in sub], dtype=float)
            ratio = np.divide(valid, total, out=np.full_like(valid, np.nan), where=np.isfinite(total) & (total > 0))

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4,
                markersize=5.5,
                color=color_map[it_name],
                label=f"{it_name} best Sharpe",
            )
            ax2.plot(
                x,
                ratio,
                linestyle="--",
                linewidth=1.9,
                alpha=0.72,
                color=color_map[it_name],
                label=f"{it_name} valid ratio",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Candidate Sharpe (window-mean)")
        ax2.set_ylabel("Valid Window Ratio (n / total)")
        ax2.set_ylim(-0.02, 1.05)
        ax.set_title(f"D. Iteration Effectiveness - {_human_protocol(protocol)}", loc="left", fontweight="bold")
        ax.grid(alpha=0.2, linestyle=":")

        # Merge legends from both y-axes.
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    sfp.finalize_figure(
        fig,
        out_dir / "fig_D_iteration_effectiveness",
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LESR efficiency report figures.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory containing the prepared CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures. Default: <report-dir>/figures",
    )
    parser.add_argument(
        "--curve-metric",
        type=str,
        default="sharpe",
        choices=["sharpe", "cr", "av"],
        help="Metric used on Figure A vertical axis: d(metric) = metric(G3)-metric(G0).",
    )
    parser.add_argument("--dpi", type=int, default=350, help="DPI for raster outputs.")
    args = parser.parse_args()

    report_dir: Path = args.report_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else (report_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    sfp = _load_scientific_figure_pro()
    sfp.apply_publication_style(sfp.FigureStyle(font_size=14, axes_linewidth=2.1, use_tex=False))

    plot_budget_curve(report_dir, out_dir, sfp, args.dpi, args.curve_metric)
    plot_min_budget(report_dir, out_dir, sfp, args.dpi)
    plot_heatmap(report_dir, out_dir, sfp, args.dpi)
    plot_iteration_effectiveness(report_dir, out_dir, sfp, args.dpi)

    print(f"[done] figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
