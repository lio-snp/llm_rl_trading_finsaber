from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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


def _score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Score"] = out["Sharpe_mean"].astype(float) + out["CR_mean"].astype(float)
    return out


def _collect_run_ids(root: Path, explicit_run_ids: List[str] | None) -> List[str]:
    if explicit_run_ids:
        return [str(x) for x in explicit_run_ids]
    run_ids: List[str] = []
    for run_dir in sorted((root / "runs").glob("*_demo")):
        manifest = _load_json(run_dir / "run_manifest.json")
        if not manifest:
            continue
        if str(manifest.get("experiment_phase", "")).upper() != "E1":
            continue
        claim = str(manifest.get("claim_id", ""))
        if claim.startswith("mechanism_isolation"):
            run_ids.append(run_dir.name)
    return sorted(run_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument("--out-dir", default="docs/steps/step_08_e1_regime_matrix/artifacts")
    args = parser.parse_args()

    root = _repo_root()
    run_ids = _collect_run_ids(root, args.run_ids)
    if not run_ids:
        raise ValueError("No matching E1 run ids found.")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = []
    g2_rows = []
    regime_rows = []
    for run_id in run_ids:
        run_dir = (root / "runs" / run_id).resolve()
        manifest = _load_json(run_dir / "run_manifest.json")
        wf_path = run_dir / "walk_forward_metrics_table.csv"
        if not wf_path.exists():
            registry_rows.append(
                {
                    "run_id": run_id,
                    "status": "missing_metrics",
                    "window_setup": manifest.get("window_setup", ""),
                    "phase": manifest.get("experiment_phase", ""),
                    "claim_id": manifest.get("claim_id", ""),
                    "hypothesis_id": manifest.get("hypothesis_id", ""),
                }
            )
            continue
        table = pd.read_csv(wf_path)
        table = table[table["window_index"].astype(str) != "aggregate"].copy()
        table = _score(table)
        ccheck = manifest.get("completeness_check", {})
        registry_rows.append(
            {
                "run_id": run_id,
                "status": ccheck.get("status", "unknown"),
                "is_complete": not bool(ccheck.get("excluded_incomplete", True)),
                "window_setup": manifest.get("window_setup", ""),
                "phase": manifest.get("experiment_phase", ""),
                "claim_id": manifest.get("claim_id", ""),
                "hypothesis_id": manifest.get("hypothesis_id", ""),
                "window_count": int(table["window_index"].nunique()),
                "config_fingerprint": manifest.get("config_fingerprint", ""),
                "candidate_fingerprint": json.dumps(manifest.get("candidate_fingerprint", {}), ensure_ascii=False),
            }
        )

        for algo, grp in table.groupby("algorithm"):
            g0 = grp[grp["group"] == "G0_baseline"][["window_index", "Score"]].rename(columns={"Score": "Score_G0"})
            g2 = grp[grp["group"] == "G2_intrinsic_only"][["window_index", "Score"]].rename(columns={"Score": "Score_G2"})
            merged = g0.merge(g2, on="window_index", how="inner")
            if merged.empty:
                continue
            delta = merged["Score_G2"] - merged["Score_G0"]
            g2_rows.append(
                {
                    "run_id": run_id,
                    "algorithm": algo,
                    "g2_delta_score_mean": float(delta.mean()),
                    "g2_positive_window_ratio": float((delta > 0).mean()),
                    "window_count": int(len(merged)),
                }
            )

        regime_path = run_dir / f"gain_by_regime_{run_id}.csv"
        if regime_path.exists():
            reg = pd.read_csv(regime_path)
            reg.insert(0, "run_id", run_id)
            regime_rows.append(reg)

    reg_df = pd.DataFrame(registry_rows).sort_values("run_id").reset_index(drop=True)
    g2_df = pd.DataFrame(g2_rows).sort_values(["run_id", "algorithm"]).reset_index(drop=True)
    regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()

    reg_df.to_csv(out_dir / "run_registry.csv", index=False)
    g2_df.to_csv(out_dir / "g2_direction_summary.csv", index=False)
    regime_df.to_csv(out_dir / "regime_delta_summary.csv", index=False)
    print(f"[ok] wrote: {out_dir / 'run_registry.csv'}")
    print(f"[ok] wrote: {out_dir / 'g2_direction_summary.csv'}")
    print(f"[ok] wrote: {out_dir / 'regime_delta_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
