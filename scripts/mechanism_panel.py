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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


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


def _first_last_return(df_prices: pd.DataFrame, assets: List[str], start: str, end: str) -> float:
    if df_prices.empty or not assets:
        return 0.0
    sub = df_prices[(df_prices["date"] >= pd.to_datetime(start)) & (df_prices["date"] <= pd.to_datetime(end))]
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
    rel = cfg.get("finsaber_price_path")
    if not rel:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    path = (root / rel).resolve()
    if not path.exists():
        return pd.DataFrame(columns=["date", "symbol", "close"])
    df = load_finsaber_prices(
        path,
        assets=None,
        start_date=str(cfg.get("start_date", "1900-01-01")),
        end_date=str(cfg.get("end_date", "2100-01-01")),
    ).rename(columns={"asset": "symbol"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "symbol", "close"]]


def _mean_stat(rows: List[dict], key: str) -> float:
    vals = [float((r.get(key, {}) or {}).get("mean", np.nan)) for r in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    root = _repo_root()
    run_dir = _resolve_run_dir(args.run_id, args.run_dir)
    run_id = run_dir.name
    cfg = _load_yaml(run_dir / "config.yaml")
    summary = _load_json(run_dir / "walk_forward_summary.json")
    wf_table = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    wf_table = wf_table[wf_table["window_index"].astype(str) != "aggregate"].copy()
    wf_table["Score"] = wf_table["Sharpe_mean"].astype(float) + wf_table["CR_mean"].astype(float)
    prices = _load_price_frame(cfg, root)

    rows: List[dict] = []
    for item in summary.get("windows", []):
        widx = int(item["window_index"])
        split = item.get("split", {})
        test = split.get("test", {})
        sub_manifest = _load_json(root / item.get("run_manifest", ""))
        assets = list(sub_manifest.get("selected_assets", []) or cfg.get("assets", []) or [])
        market_ret = _first_last_return(prices, assets, str(test.get("start", "")), str(test.get("end", "")))
        regime = _classify_regime(market_ret)

        reward_trace = _load_json(root / item.get("reward_trace", ""))
        policy_behavior = _load_json(root / item.get("policy_behavior_summary", ""))
        td3_diff = _load_json(root / item.get("td3_g1_g3_diff", ""))
        td3_sat = _load_json(root / item.get("td3_action_saturation", ""))
        td3_diff_summary = ((td3_diff.get("td3") or {}).get("summary") or {})

        table_w = wf_table[wf_table["window_index"].astype(int) == widx].copy()
        for algo, grp in table_w.groupby("algorithm"):
            g0_score = grp.loc[grp["group"] == "G0_baseline", "Score"]
            g0 = float(g0_score.iloc[0]) if not g0_score.empty else 0.0
            for _, mrow in grp.iterrows():
                group = str(mrow["group"])
                reward_rows = (((reward_trace.get(algo) or {}).get(group)) or [])
                policy_summary = ((((policy_behavior.get(algo) or {}).get(group) or {}).get("summary")) or {})
                td3_sat_summary = ((((td3_sat.get("td3") or {}).get(group) or {}).get("summary")) or {})
                near_bound_ratio = float(policy_summary.get("near_bound_ratio_mean", np.nan))
                near_actor_ratio = float(td3_sat_summary.get("near_actor_ratio_mean", np.nan))
                actor_collapse = td3_sat_summary.get("actor_collapse_detected")
                if algo == "td3" and np.isfinite(near_actor_ratio):
                    # For TD3, actor-bound saturation is the correct collapse proxy.
                    near_bound_ratio = near_actor_ratio
                action_equal = float("nan")
                if algo == "td3" and group in {"G1_revise_only", "G3_revise_intrinsic"}:
                    action_equal = float(td3_diff_summary.get("action_equal_ratio_mean", np.nan))
                rows.append(
                    {
                        "run_id": run_id,
                        "window_index": widx,
                        "window_name": str(item.get("window_name", "")),
                        "regime": regime,
                        "market_return_equal_weight": market_ret,
                        "algorithm": str(algo),
                        "group": group,
                        "Score": float(mrow["Score"]),
                        "Score_delta_vs_G0": float(mrow["Score"] - g0),
                        "reward_env_mean": _mean_stat(reward_rows, "reward_env"),
                        "intrinsic_mean": _mean_stat(reward_rows, "intrinsic"),
                        "reward_total_minus_env_mean": _mean_stat(reward_rows, "reward_total_minus_env"),
                        "intrinsic_effect_ratio_robust_mean": _mean_stat(reward_rows, "intrinsic_effect_ratio_robust"),
                        "near_bound_ratio_mean": near_bound_ratio,
                        "near_actor_ratio_mean": near_actor_ratio,
                        "actor_collapse_detected": actor_collapse,
                        "action_entropy_mean": float(policy_summary.get("action_entropy_mean", np.nan)),
                        "sign_flip_rate_mean": float(policy_summary.get("sign_flip_rate_mean", np.nan)),
                        "action_equal_ratio_mean": action_equal,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["window_index", "algorithm", "group"]).reset_index(drop=True)
    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
