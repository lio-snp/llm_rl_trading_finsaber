from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data.finsaber_data import load_finsaber_prices


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_run_dir(run_id: str | None, run_dir: str | None) -> Path:
    root = _repo_root()
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = (root / path).resolve()
        return path
    if not run_id:
        raise ValueError("either --run-id or --run-dir is required")
    return (root / "runs" / run_id).resolve()


def _score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Score"] = out["Sharpe_mean"].astype(float) + out["CR_mean"].astype(float)
    return out


def _first_last_return(df_prices: pd.DataFrame, assets: List[str], start: str, end: str) -> float:
    if df_prices.empty or not assets:
        return 0.0
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    sub = df_prices[(df_prices["date"] >= s) & (df_prices["date"] <= e)]
    if sub.empty:
        return 0.0
    rets: List[float] = []
    for asset in assets:
        cur = sub[sub["symbol"] == asset].sort_values("date")
        if cur.empty:
            continue
        p0 = float(cur["close"].iloc[0])
        p1 = float(cur["close"].iloc[-1])
        if p0 <= 0:
            continue
        rets.append((p1 / p0) - 1.0)
    if not rets:
        return 0.0
    return float(np.mean(rets))


def _classify_regime(ret: float) -> str:
    if ret >= 0.05:
        return "bull"
    if ret <= -0.05:
        return "bear"
    return "sideways"


def _load_price_frame(cfg: dict, root: Path) -> pd.DataFrame:
    if str(cfg.get("data_source", "")).lower() != "finsaber":
        return pd.DataFrame(columns=["date", "symbol", "close"])
    path = cfg.get("finsaber_price_path")
    if not path:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    price_path = (root / path).resolve()
    if not price_path.exists():
        return pd.DataFrame(columns=["date", "symbol", "close"])
    df = load_finsaber_prices(
        price_path,
        assets=None,
        start_date=str(cfg.get("start_date", "1900-01-01")),
        end_date=str(cfg.get("end_date", "2100-01-01")),
    ).rename(columns={"asset": "symbol"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "symbol", "close"]]


def _build_gain_matrix(wf_table: pd.DataFrame) -> pd.DataFrame:
    table = wf_table.copy()
    table = table[table["window_index"].astype(str) != "aggregate"].copy()
    table["window_index_num"] = table["window_index"].astype(int)
    table = _score(table)

    rows: List[dict] = []
    for (widx, wname, algo), grp in table.groupby(["window_index_num", "window_name", "algorithm"]):
        score_map = {str(r["group"]): float(r["Score"]) for _, r in grp.iterrows()}
        g0 = score_map.get("G0_baseline", 0.0)
        g1 = score_map.get("G1_revise_only", 0.0)
        g2 = score_map.get("G2_intrinsic_only", 0.0)
        g3 = score_map.get("G3_revise_intrinsic", 0.0)
        d1 = g1 - g0
        d2 = g2 - g0
        d3 = g3 - g0
        rows.append(
            {
                "window_index": int(widx),
                "window_name": wname,
                "algorithm": algo,
                "Score_G0": g0,
                "Score_G1": g1,
                "Score_G2": g2,
                "Score_G3": g3,
                "Delta_G1_vs_G0": d1,
                "Delta_G2_vs_G0": d2,
                "Delta_G3_vs_G0": d3,
                "all_positive": bool(d1 > 0 and d2 > 0 and d3 > 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["window_index", "algorithm"]).reset_index(drop=True)


def _load_window_regimes(run_dir: Path, root: Path, run_cfg: dict, summary: dict) -> Dict[int, dict]:
    prices = _load_price_frame(run_cfg, root)
    window_meta: Dict[int, dict] = {}
    for item in summary.get("windows", []):
        idx = int(item["window_index"])
        split = item.get("split", {})
        test = split.get("test", {})
        sub_manifest_path = root / item.get("run_manifest", "")
        assets: List[str] = []
        if sub_manifest_path.exists():
            try:
                sub_manifest = json.loads(sub_manifest_path.read_text())
                assets = list(sub_manifest.get("selected_assets", []) or [])
            except Exception:
                assets = []
        if not assets:
            assets = list(run_cfg.get("assets", []) or [])
        ret = _first_last_return(prices, assets, str(test.get("start", "")), str(test.get("end", "")))
        window_meta[idx] = {
            "market_return_equal_weight": ret,
            "regime": _classify_regime(ret),
        }
    return window_meta


def _policy_behavior_rows(root: Path, summary: dict) -> pd.DataFrame:
    rows: List[dict] = []
    for item in summary.get("windows", []):
        idx = int(item["window_index"])
        behavior_path = root / item.get("policy_behavior_summary", "")
        if not behavior_path.exists():
            continue
        try:
            payload = json.loads(behavior_path.read_text())
        except Exception:
            continue
        for algo, algo_payload in payload.items():
            for group, group_payload in algo_payload.items():
                if group.startswith("_"):
                    continue
                summary_row = group_payload.get("summary", {})
                rows.append(
                    {
                        "window_index": idx,
                        "window_name": item.get("window_name"),
                        "algorithm": algo,
                        "group": group,
                        "near_bound_ratio_mean": float(summary_row.get("near_bound_ratio_mean", 0.0)),
                        "unique_action_count_mean": float(summary_row.get("unique_action_count_mean", 0.0)),
                        "sign_flip_rate_mean": float(summary_row.get("sign_flip_rate_mean", 0.0)),
                        "action_entropy_mean": float(summary_row.get("action_entropy_mean", 0.0)),
                        "actor_collapse_detected": bool(summary_row.get("actor_collapse_detected", False)),
                    }
                )
    return pd.DataFrame(rows)


def _build_gain_by_regime(gain: pd.DataFrame) -> pd.DataFrame:
    if gain.empty:
        return gain
    rows: List[dict] = []
    for (algo, regime), grp in gain.groupby(["algorithm", "regime"]):
        rows.append(
            {
                "algorithm": algo,
                "regime": regime,
                "window_count": int(len(grp)),
                "Score_G0_mean": float(grp["Score_G0"].mean()),
                "Score_G1_mean": float(grp["Score_G1"].mean()),
                "Score_G2_mean": float(grp["Score_G2"].mean()),
                "Score_G3_mean": float(grp["Score_G3"].mean()),
                "Delta_G1_vs_G0_mean": float(grp["Delta_G1_vs_G0"].mean()),
                "Delta_G2_vs_G0_mean": float(grp["Delta_G2_vs_G0"].mean()),
                "Delta_G3_vs_G0_mean": float(grp["Delta_G3_vs_G0"].mean()),
                "positive_ratio_G1": float((grp["Delta_G1_vs_G0"] > 0).mean()),
                "positive_ratio_G2": float((grp["Delta_G2_vs_G0"] > 0).mean()),
                "positive_ratio_G3": float((grp["Delta_G3_vs_G0"] > 0).mean()),
                "all_positive_ratio": float(grp["all_positive"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["algorithm", "regime"]).reset_index(drop=True)


def _failure_attribution_md(
    gain: pd.DataFrame,
    policy_behavior: pd.DataFrame,
    run_id: str,
) -> str:
    lines = [
        f"# Failure Attribution ({run_id})",
        "",
        "- Objective: `Score = Sharpe_mean + CR_mean`, and check if G1/G2/G3 are all positive vs G0.",
        "",
        "## Algorithm-Level Attribution",
        "",
    ]
    if gain.empty:
        lines.append("- No gain rows found.")
        return "\n".join(lines)

    for algo, grp in gain.groupby("algorithm"):
        d1 = float(grp["Delta_G1_vs_G0"].mean())
        d2 = float(grp["Delta_G2_vs_G0"].mean())
        d3 = float(grp["Delta_G3_vs_G0"].mean())
        p1 = float((grp["Delta_G1_vs_G0"] > 0).mean())
        p2 = float((grp["Delta_G2_vs_G0"] > 0).mean())
        p3 = float((grp["Delta_G3_vs_G0"] > 0).mean())
        pass_all = bool(d1 > 0 and d2 > 0 and d3 > 0 and p1 > 0.5 and p2 > 0.5 and p3 > 0.5)

        pb = policy_behavior[policy_behavior["algorithm"] == algo] if not policy_behavior.empty else pd.DataFrame()
        near_bound = float(pb["near_bound_ratio_mean"].mean()) if not pb.empty else 0.0
        entropy = float(pb["action_entropy_mean"].mean()) if not pb.empty else 0.0
        collapse = bool((near_bound >= 0.95) and (entropy < 0.2))

        reason = "pass"
        if not pass_all:
            if collapse:
                reason = "policy_collapse_or_bound_saturation"
            elif d2 <= 0 and d3 <= 0:
                reason = "intrinsic_direction_or_scale_not_helpful"
            else:
                reason = "regime_or_window_sensitivity"

        lines.extend(
            [
                f"### {algo}",
                f"- aggregate_delta: G1={d1:.6f}, G2={d2:.6f}, G3={d3:.6f}",
                f"- positive_window_ratio: G1={p1:.3f}, G2={p2:.3f}, G3={p3:.3f}",
                f"- policy_behavior_hint: near_bound={near_bound:.3f}, entropy={entropy:.3f}",
                f"- status: {'PASS' if pass_all else 'FAIL'} ({reason})",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Run folder name under runs/")
    parser.add_argument("--run-dir", default="", help="Absolute/relative run dir path")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_id or None, args.run_dir or None)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    root = _repo_root()

    wf_table_path = run_dir / "walk_forward_metrics_table.csv"
    cfg_path = run_dir / "config.yaml"
    if not wf_table_path.exists():
        raise FileNotFoundError(f"missing {wf_table_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")

    wf_table = pd.read_csv(wf_table_path)
    wf_summary_path = run_dir / "walk_forward_summary.json"
    if wf_summary_path.exists():
        wf_summary = json.loads(wf_summary_path.read_text())
    else:
        wf_summary = {"windows": []}
    run_cfg = yaml.safe_load(cfg_path.read_text())
    run_id = run_dir.name

    gain = _build_gain_matrix(wf_table)
    window_regimes = _load_window_regimes(run_dir, root, run_cfg, wf_summary)
    gain["market_return_equal_weight"] = gain["window_index"].map(
        lambda x: float(window_regimes.get(int(x), {}).get("market_return_equal_weight", 0.0))
    )
    gain["regime"] = gain["window_index"].map(
        lambda x: str(window_regimes.get(int(x), {}).get("regime", "sideways"))
    )

    gain_by_regime = _build_gain_by_regime(gain)
    policy_behavior = _policy_behavior_rows(root, wf_summary)
    failure_md = _failure_attribution_md(gain, policy_behavior, run_id)

    gain_path = run_dir / f"gain_matrix_{run_id}.csv"
    regime_path = run_dir / f"gain_by_regime_{run_id}.csv"
    behavior_path = run_dir / f"policy_behavior_{run_id}.csv"
    failure_path = run_dir / f"failure_attribution_{run_id}.md"

    gain.to_csv(gain_path, index=False)
    gain_by_regime.to_csv(regime_path, index=False)
    policy_behavior.to_csv(behavior_path, index=False)
    failure_path.write_text(failure_md)

    print(f"[ok] wrote: {gain_path}")
    print(f"[ok] wrote: {regime_path}")
    print(f"[ok] wrote: {behavior_path}")
    print(f"[ok] wrote: {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
