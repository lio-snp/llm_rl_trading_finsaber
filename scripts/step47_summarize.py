from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_meta(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.yaml"
    wf_path = run_dir / "walk_forward_metrics_table.csv"
    if not cfg_path.exists() or not wf_path.exists():
        return None
    cfg = _load_yaml(cfg_path)
    exp = (cfg or {}).get("experiment", {}) or {}
    llm = (cfg or {}).get("llm", {}) or {}
    return {
        "run": run_dir.name,
        "run_dir": run_dir,
        "claim_id": str(exp.get("claim_id", "")),
        "hypothesis_id": str(exp.get("hypothesis_id", "")),
        "budget": int((cfg or {}).get("n_full", 0) or 0),
        "iterations": int(llm.get("iterations", 0) or 0),
        "generation_target": str(llm.get("generation_target", "global_best")),
        "window_setup": str((cfg or {}).get("window_setup", "")),
    }


def _compute_deltas(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    gp = df.groupby(["algorithm", "group"], as_index=False)["Sharpe_mean"].mean()
    rows: List[dict] = []
    for algo in sorted(gp["algorithm"].unique()):
        part = gp[gp["algorithm"] == algo]
        m = {r["group"]: float(r["Sharpe_mean"]) for _, r in part.iterrows()}
        if "G0_baseline" not in m:
            continue
        g0 = m["G0_baseline"]
        rows.append(
            {
                "algorithm": algo,
                "dSharpe_G1": float(m.get("G1_revise_only", g0) - g0),
                "dSharpe_G2": float(m.get("G2_intrinsic_only", g0) - g0),
                "dSharpe_G3": float(m.get("G3_revise_intrinsic", g0) - g0),
            }
        )
    return pd.DataFrame(rows)


def _latest_runs(root: Path, claims: set[str]) -> List[dict]:
    out: List[dict] = []
    for run_dir in sorted((root / "runs").glob("*_demo")):
        meta = _run_meta(run_dir)
        if not meta:
            continue
        if meta["claim_id"] in claims:
            out.append(meta)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Step47 dSharpe deltas and rep-budget suggestion.")
    parser.add_argument("--root", default=".", help="Repo root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    metas = _latest_runs(root, {"step47_backend_control", "step47_selected4_sfamily"})
    if not metas:
        print("No step47 runs found.")
        return 0

    print("== Step47 Runs (with walk_forward_metrics_table.csv) ==")
    for m in metas:
        print(
            f"{m['run']} | claim={m['claim_id']} | hypo={m['hypothesis_id']} | "
            f"budget={m['budget']} | it={m['iterations']} | target={m['generation_target']}"
        )

    print("\n== dSharpe (G1/G2/G3 vs G0) ==")
    for m in metas:
        d = _compute_deltas(m["run_dir"])
        if d.empty:
            continue
        print(f"\n[{m['run']}] {m['hypothesis_id']}")
        print(d.to_string(index=False))

    # Phase1 quick backend check
    lookup: Dict[str, dict] = {m["hypothesis_id"]: m for m in metas}
    h_custom = "h_step47_index_selected4_b252_it5_backend_custom_fixed"
    h_sb3 = "h_step47_index_selected4_b252_it5_backend_sb3_fixed"
    if h_custom in lookup and h_sb3 in lookup:
        d_c = _compute_deltas(lookup[h_custom]["run_dir"]).set_index("algorithm")
        d_s = _compute_deltas(lookup[h_sb3]["run_dir"]).set_index("algorithm")
        inter = d_c.index.intersection(d_s.index)
        if len(inter) > 0:
            diff = (d_s.loc[inter, "dSharpe_G3"] - d_c.loc[inter, "dSharpe_G3"]).rename("delta_sb3_minus_custom")
            print("\n== Phase1 Backend Control Check (G3 dSharpe diff) ==")
            print(diff.reset_index().to_string(index=False))
            non_td3 = diff[[a for a in diff.index if a != "td3"]]
            if len(non_td3) > 0:
                ok = bool((non_td3.abs() <= 0.03).all())
                print(f"non_td3_abs_delta<=0.03 : {ok}")

    # rep budget suggestion for it5 selected4
    print("\n== Rep Budget Suggestion (it5 selected4) ==")
    best_budget = None
    best_score = None
    for b in [252, 504]:
        h_g = f"h_step47_selected4_b{b}_it5_global"
        h_s = f"h_step47_selected4_b{b}_it5_sfamily"
        if h_g not in lookup or h_s not in lookup:
            print(f"budget={b}: missing run(s) for global/sfamily")
            continue
        d_g = _compute_deltas(lookup[h_g]["run_dir"]).set_index("algorithm")
        d_s = _compute_deltas(lookup[h_s]["run_dir"]).set_index("algorithm")
        inter = d_g.index.intersection(d_s.index)
        if len(inter) == 0:
            print(f"budget={b}: no overlapping algos")
            continue
        improve = d_s.loc[inter, "dSharpe_G3"] - d_g.loc[inter, "dSharpe_G3"]
        mean_improve = float(improve.mean())
        pos_cnt = int((improve >= 0.05).sum())
        print(f"budget={b}: mean_improve={mean_improve:+.4f}, algos_improve>=0.05={pos_cnt}/{len(improve)}")
        score = (pos_cnt, mean_improve)
        if best_score is None or score > best_score:
            best_score = score
            best_budget = b

    if best_budget is not None:
        print(f"recommended_rep_budget={best_budget}")
        print(f"run: bash scripts/run_step47_selected4_it5_rep.sh {best_budget}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
