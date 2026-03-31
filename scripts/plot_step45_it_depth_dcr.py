#!/usr/bin/env python3
"""Plot dCR curves for Step45 it5/it10 LESR runs.

Outputs under:
docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPORT_DIR = Path(
    "docs/steps/step45_lesr_iter_depth/artifacts/efficiency_report_20260224"
)

STEP45_RUN_MAP = {
    ("composite", "it5"): Path("runs/20260223_035649_952_b6ed_demo"),
    ("composite", "it10"): Path("runs/20260223_035649_950_5015_demo"),
    ("index_selected4", "it5"): Path("runs/20260223_070935_643_681f_demo"),
    ("index_selected4", "it10"): Path("runs/20260223_152803_249_f8c2_demo"),
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


def _algo_order() -> list[str]:
    return ["a2c", "ppo", "sac", "td3"]


def _protocol_order() -> list[str]:
    return ["composite", "index_selected4"]


def _it_order() -> list[str]:
    return ["it5", "it10"]


def _it_to_num(it_setting: str) -> int:
    return int(it_setting.replace("it", ""))


def _human_protocol(name: str) -> str:
    return "index_selected4" if name == "index_selected4" else "composite"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _algo_color_map(sfp: Any) -> dict[str, str]:
    return {
        "a2c": sfp.PALETTE["blue_main"],
        "ppo": sfp.PALETTE["teal"],
        "sac": sfp.PALETTE["red_strong"],
        "td3": sfp.PALETTE["violet"],
    }


def build_final_dcr_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol in _protocol_order():
        for it_setting in _it_order():
            run = STEP45_RUN_MAP[(protocol, it_setting)]
            wf = pd.read_csv(run / "walk_forward_metrics_table.csv")
            agg = wf[wf["window_name"] == "aggregate"].copy()
            if agg.empty:
                continue
            for algo in _algo_order():
                sub = agg[agg["algorithm"].str.lower() == algo]
                if sub.empty:
                    continue
                g0 = sub[sub["group"] == "G0_baseline"]
                if g0.empty:
                    continue
                g0_cr = float(g0.iloc[0]["CR_mean"])
                out = {
                    "protocol": protocol,
                    "it_setting": it_setting,
                    "it": _it_to_num(it_setting),
                    "algo": algo,
                    "dCR_G1": float("nan"),
                    "dCR_G2": float("nan"),
                    "dCR_G3": float("nan"),
                }
                for gname, key in [
                    ("G1_revise_only", "dCR_G1"),
                    ("G2_intrinsic_only", "dCR_G2"),
                    ("G3_revise_intrinsic", "dCR_G3"),
                ]:
                    gi = sub[sub["group"] == gname]
                    if not gi.empty:
                        out[key] = float(gi.iloc[0]["CR_mean"]) - g0_cr
                rows.append(out)
    return pd.DataFrame(rows)


def _best_candidate_cr(candidates: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    for cand in candidates:
        if not cand.get("valid", False):
            continue
        seed_metrics = cand.get("seed_metrics") or []
        cr_vals = []
        for sm in seed_metrics:
            v = sm.get("CR", None)
            if v is None:
                continue
            vf = float(v)
            if np.isfinite(vf):
                cr_vals.append(vf)
        if cr_vals:
            vals.append(float(np.mean(np.array(cr_vals, dtype=float))))
    if not vals:
        return float("nan")
    return float(np.max(np.array(vals, dtype=float)))


def build_shorttrain_dcr_curve() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol in _protocol_order():
        for it_setting in _it_order():
            run = STEP45_RUN_MAP[(protocol, it_setting)]
            window_iter_best_cr: dict[str, dict[int, float]] = {}
            for wdir in sorted(run.glob("wf_window_*")):
                trace_path = wdir / "llm_iter_trace.json"
                if not trace_path.exists():
                    continue
                trace = _read_json(trace_path)
                m: dict[int, float] = {}
                for it in trace:
                    i = int(it["iteration"])
                    m[i] = _best_candidate_cr(it.get("candidates", []))
                window_iter_best_cr[wdir.name] = m
            if not window_iter_best_cr:
                continue
            all_iters = sorted(
                {
                    i
                    for imap in window_iter_best_cr.values()
                    for i, v in imap.items()
                    if np.isfinite(v)
                }
            )
            if not all_iters:
                continue

            baseline_per_window = {
                w: imap.get(0, float("nan")) for w, imap in window_iter_best_cr.items()
            }
            total_windows = len(window_iter_best_cr)
            for i in all_iters:
                delta_vals = []
                valid_w = 0
                for w, imap in window_iter_best_cr.items():
                    cur = imap.get(i, float("nan"))
                    b0 = baseline_per_window.get(w, float("nan"))
                    if np.isfinite(cur):
                        valid_w += 1
                    if np.isfinite(cur) and np.isfinite(b0):
                        delta_vals.append(float(cur - b0))
                mean_delta = float(np.mean(np.array(delta_vals, dtype=float))) if delta_vals else float("nan")
                rows.append(
                    {
                        "protocol": protocol,
                        "it_setting": it_setting,
                        "it": _it_to_num(it_setting),
                        "iter": i,
                        "mean_best_candidate_dCR_vs_iter0": mean_delta,
                        "valid_window_count": valid_w,
                        "total_windows": total_windows,
                    }
                )
    return pd.DataFrame(rows)


def plot_final_g3_dcr(df: pd.DataFrame, out_dir: Path, sfp: Any, dpi: int) -> None:
    colors = _algo_color_map(sfp)
    fig, axes = sfp.create_subplots(1, len(_protocol_order()), figsize=(12.4, 5.2))
    axes = np.atleast_1d(axes)
    for ax, protocol in zip(axes, _protocol_order()):
        sub = df[df["protocol"] == protocol].copy()
        if sub.empty:
            continue
        for algo in _algo_order():
            s = sub[sub["algo"] == algo].sort_values("it")
            if s.empty:
                continue
            x = s["it"].astype(float).to_numpy()
            y = s["dCR_G3"].astype(float).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2.5, markersize=6, color=colors[algo], label=algo.upper())
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_xticks([5, 10])
        ax.set_xlabel("LESR Iteration Setting")
        ax.set_ylabel("dCR (G3 - G0)")
        ax.set_title(f"Step45 dCR(G3-G0) - {_human_protocol(protocol)}", loc="left", fontweight="bold")
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(ncol=2, loc="best")
    sfp.finalize_figure(
        fig,
        out_dir / "fig_step45_it_depth_dcr_g3_curve",
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def plot_shorttrain_dcr(df: pd.DataFrame, out_dir: Path, sfp: Any, dpi: int) -> None:
    line_color = {"it5": sfp.PALETTE["blue_main"], "it10": sfp.PALETTE["red_strong"]}
    fig, axes = sfp.create_subplots(1, len(_protocol_order()), figsize=(12.4, 5.2))
    axes = np.atleast_1d(axes)
    for ax, protocol in zip(axes, _protocol_order()):
        sub = df[df["protocol"] == protocol].copy()
        if sub.empty:
            continue
        for it_setting in _it_order():
            s = sub[sub["it_setting"] == it_setting].sort_values("iter")
            if s.empty:
                continue
            x = s["iter"].astype(float).to_numpy()
            y = s["mean_best_candidate_dCR_vs_iter0"].astype(float).to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.5,
                markersize=5,
                color=line_color[it_setting],
                label=it_setting,
            )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("LESR Iteration Index")
        ax.set_ylabel("Short-Train dCR vs iter0")
        ax.set_title(f"Short-Train dCR Curve - {_human_protocol(protocol)}", loc="left", fontweight="bold")
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(loc="best")
    sfp.finalize_figure(
        fig,
        out_dir / "fig_step45_shorttrain_dcr_curve",
        formats=["png", "pdf"],
        dpi=dpi,
        pad=0.06,
    )


def _write_df_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step45 it-depth dCR curves.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Report artifact directory.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    out_dir = report_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sfp = _load_scientific_figure_pro()
    sfp.apply_publication_style()

    final_df = build_final_dcr_table().sort_values(["protocol", "it", "algo"]).reset_index(drop=True)
    short_df = build_shorttrain_dcr_curve().sort_values(["protocol", "it_setting", "iter"]).reset_index(drop=True)

    _write_df_csv(final_df, report_dir / "it_depth_dcr_g123.csv")
    _write_df_csv(short_df, report_dir / "it_depth_shorttrain_dcr_curve.csv")

    plot_final_g3_dcr(final_df, out_dir, sfp, args.dpi)
    plot_shorttrain_dcr(short_df, out_dir, sfp, args.dpi)

    print(f"[done] wrote: {report_dir / 'it_depth_dcr_g123.csv'}")
    print(f"[done] wrote: {report_dir / 'it_depth_shorttrain_dcr_curve.csv'}")
    print(f"[done] figures in: {out_dir}")


if __name__ == "__main__":
    main()
