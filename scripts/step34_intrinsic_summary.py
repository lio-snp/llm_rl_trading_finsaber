from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _bootstrap_ci(diff: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = diff.size
    means: List[float] = []
    for _ in range(max(200, n_resamples)):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(diff[idx])))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _collect_reward_delta(run_dir: Path) -> float:
    summary = _load_json(run_dir / "walk_forward_summary.json")
    vals: List[float] = []
    for item in summary.get("windows", []):
        reward_path = _repo_root() / item.get("reward_trace", "")
        payload = _load_json(reward_path)
        rows = (((payload.get("td3") or {}).get("G3_revise_intrinsic")) or [])
        for row in rows:
            cell = row.get("reward_total_minus_env", {})
            vals.append(_safe_float(cell.get("mean", np.nan)))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _collect_action_equal(run_dir: Path) -> float:
    summary = _load_json(run_dir / "walk_forward_summary.json")
    vals: List[float] = []
    for item in summary.get("windows", []):
        diff_path = _repo_root() / item.get("td3_g1_g3_diff", "")
        payload = _load_json(diff_path)
        cell = (((payload.get("td3") or {}).get("summary") or {}).get("action_equal_ratio_mean", np.nan))
        vals.append(_safe_float(cell))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _collect_near_bound_delta(run_dir: Path) -> float:
    summary = _load_json(run_dir / "walk_forward_summary.json")
    g0_vals: List[float] = []
    g3_vals: List[float] = []
    for item in summary.get("windows", []):
        sat_path = _repo_root() / item.get("td3_action_saturation", "")
        payload = _load_json(sat_path)
        g0 = (
            (((payload.get("td3") or {}).get("G0_baseline") or {}).get("summary") or {}).get(
                "near_actor_ratio_mean", np.nan
            )
        )
        g3 = (
            (((payload.get("td3") or {}).get("G3_revise_intrinsic") or {}).get("summary") or {}).get(
                "near_actor_ratio_mean", np.nan
            )
        )
        g0_vals.append(_safe_float(g0))
        g3_vals.append(_safe_float(g3))
    g0_vals = [v for v in g0_vals if np.isfinite(v)]
    g3_vals = [v for v in g3_vals if np.isfinite(v)]
    if not g0_vals or not g3_vals:
        return float("nan")
    return float(np.mean(g3_vals) - np.mean(g0_vals))


def _summarize_run(run_dir: Path) -> Dict[str, float]:
    wf = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    wf = wf[wf["window_index"].astype(str) != "aggregate"].copy()
    wf["Sharpe_mean"] = wf["Sharpe_mean"].astype(float)
    wf["CR_mean"] = wf["CR_mean"].astype(float)
    wf["MDD_mean"] = wf["MDD_mean"].astype(float)

    out: Dict[str, float] = {}
    out["windows"] = int(wf["window_index"].nunique())
    g0 = wf[wf["group"] == "G0_baseline"][["window_index", "Sharpe_mean", "CR_mean", "MDD_mean"]].rename(
        columns={"Sharpe_mean": "s0", "CR_mean": "cr0", "MDD_mean": "mdd0"}
    )
    for gk, key in [
        ("G1_revise_only", "g1"),
        ("G2_intrinsic_only", "g2"),
        ("G3_revise_intrinsic", "g3"),
    ]:
        cur = wf[wf["group"] == gk][["window_index", "Sharpe_mean", "CR_mean", "MDD_mean"]].rename(
            columns={"Sharpe_mean": "s", "CR_mean": "cr", "MDD_mean": "mdd"}
        )
        merged = g0.merge(cur, on="window_index", how="inner")
        d = merged["s"] - merged["s0"]
        out[f"{key}_dsharpe_mean"] = float(d.mean())
        out[f"{key}_pos_ratio"] = float((d > 0).mean())
        out[f"{key}_cr_mean"] = float(merged["cr"].mean())
        out[f"{key}_mdd_mean"] = float(merged["mdd"].mean())
        out[f"{key}_dcr_mean"] = float((merged["cr"] - merged["cr0"]).mean())
        out[f"{key}_dmdd_mean"] = float((merged["mdd"] - merged["mdd0"]).mean())
        if key == "g3":
            diff = d.to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(diff)
            out["g3_ci_low"] = ci_low
            out["g3_ci_high"] = ci_high
    out["reward_total_minus_env_g3_mean"] = _collect_reward_delta(run_dir)
    out["action_equal_ratio_g1g3_mean"] = _collect_action_equal(run_dir)
    out["near_bound_delta_g3_minus_g0"] = _collect_near_bound_delta(run_dir)
    return out


def _attach_protocol_consistency(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    if df.empty:
        df["protocol_consistency_ok"] = []
        return df
    out = df.copy()
    out["protocol_consistency_ok"] = True
    for cand, grp in out.groupby("candidate"):
        vals = grp["g3_dsharpe_mean"].dropna().to_numpy(dtype=float)
        ok = True
        if vals.size >= 2:
            vmax = float(np.max(vals))
            vmin = float(np.min(vals))
            strong_sign_conflict = vmax > 0.0 and vmin < 0.0 and min(abs(vmax), abs(vmin)) > float(eps)
            ok = not strong_sign_conflict
        out.loc[out["candidate"] == cand, "protocol_consistency_ok"] = bool(ok)
    return out


def _pick_best(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, grp in df.groupby("protocol"):
        elig = grp[(grp["early_reject"] == False) & (grp["protocol_consistency_ok"] == True)].copy()  # noqa: E712
        if elig.empty:
            elig = grp[(grp["early_reject"] == False)].copy()  # noqa: E712
        if elig.empty:
            continue
        ordered = elig.sort_values(
            [
                "gate_closed",
                "gate_action",
                "gate_sharpe",
                "g3_dsharpe_mean",
                "g3_pos_ratio",
                "action_equal_ratio_g1g3_mean",
            ],
            ascending=[False, False, False, False, False, True],
        )
        rows.append(ordered.head(1))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _to_markdown_fallback(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows.\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--phase", choices=["tune", "holdout"], required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--action-equal-max", type=float, default=0.85)
    parser.add_argument("--early-reject-action-equal", type=float, default=0.90)
    parser.add_argument("--reward-eps", type=float, default=1e-3)
    parser.add_argument("--consistency-eps", type=float, default=0.03)
    args = parser.parse_args()

    root = _repo_root()
    reg = pd.read_csv(args.registry_csv)
    sub = reg[(reg["phase"] == args.phase) & (reg["status"] == "completed")].copy()
    rows = []
    for _, r in sub.iterrows():
        run_id = str(r["run_id"])
        run_dir = root / "runs" / run_id
        if not run_dir.exists():
            continue
        rec = {
            "protocol": str(r["protocol"]),
            "candidate": str(r["candidate"]),
            "run_id": run_id,
        }
        rec.update(_summarize_run(run_dir))
        rec["early_reject"] = bool(
            np.isfinite(rec["action_equal_ratio_g1g3_mean"])
            and rec["action_equal_ratio_g1g3_mean"] > float(args.early_reject_action_equal)
        )
        rec["gate_reward"] = bool(abs(rec["reward_total_minus_env_g3_mean"]) >= float(args.reward_eps))
        rec["gate_action"] = bool(
            (not rec["early_reject"])
            and np.isfinite(rec["action_equal_ratio_g1g3_mean"])
            and rec["action_equal_ratio_g1g3_mean"] <= float(args.action_equal_max)
            and np.isfinite(rec["near_bound_delta_g3_minus_g0"])
            and rec["near_bound_delta_g3_minus_g0"] <= 0.0
        )
        rec["gate_sharpe"] = bool(rec["g3_dsharpe_mean"] > 0.0 and rec["g3_pos_ratio"] > 0.5)
        rec["gate_closed"] = bool(rec["gate_reward"] and rec["gate_action"] and rec["gate_sharpe"])
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty and args.phase == "tune":
        out = _attach_protocol_consistency(out, float(args.consistency_eps))
    elif not out.empty:
        out["protocol_consistency_ok"] = True
    if not out.empty:
        out = out.sort_values(["protocol", "candidate"]).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = (root / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    md = "# Step34 {} Summary\n\n".format(args.phase.capitalize())
    if out.empty:
        md += "No completed runs found.\n"
    else:
        try:
            md += out.to_markdown(index=False)
        except Exception:
            md += _to_markdown_fallback(out)
        md += "\n"
    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = (root / out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    if args.phase == "tune" and args.selection_csv:
        best = _pick_best(out)
        sel = Path(args.selection_csv)
        if not sel.is_absolute():
            sel = (root / sel).resolve()
        sel.parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(sel, index=False)
        print(f"[ok] wrote selection: {sel}")

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
