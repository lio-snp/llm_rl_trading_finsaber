from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


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


def _resolve_run_dirs(root: Path, run_ids: List[str]) -> List[Path]:
    out: List[Path] = []
    for run_id in run_ids:
        run_dir = (root / "runs" / run_id).resolve()
        if run_dir.exists():
            out.append(run_dir)
    return out


def _collect_td3_action_equal_ratio(run_dir: Path) -> float:
    summary_path = run_dir / "walk_forward_summary.json"
    if summary_path.exists():
        payload = _load_json(summary_path)
        vals: List[float] = []
        for item in payload.get("windows", []):
            rel = item.get("td3_g1_g3_diff") or ""
            if not rel:
                continue
            diff_path = (_repo_root() / rel).resolve()
            diff = _load_json(diff_path)
            s = ((diff.get("td3") or {}).get("summary") or {})
            if s:
                vals.append(float(s.get("action_equal_ratio_mean", np.nan)))
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            return float(np.mean(vals))
    diff = _load_json(run_dir / "td3_g1_g3_diff.json")
    s = ((diff.get("td3") or {}).get("summary") or {})
    if s:
        return float(s.get("action_equal_ratio_mean", np.nan))
    return float("nan")


def _collect_registry_row(run_dir: Path) -> dict:
    root = _repo_root()
    run_id = run_dir.name
    cfg = _load_yaml(run_dir / "config.yaml")
    manifest = _load_json(run_dir / "run_manifest.json")
    wf_summary = _load_json(run_dir / "walk_forward_summary.json")

    completeness = manifest.get("completeness_check", {}) or {}
    missing_files = completeness.get("missing_files", [])
    excluded_windows = int(completeness.get("excluded_window_count", 0))
    status = str(completeness.get("status", "unknown"))
    is_complete = bool(
        (run_dir / "run_manifest.json").exists()
        and status == "complete"
        and len(missing_files) == 0
        and excluded_windows == 0
    )

    eval_algos = manifest.get("eval_algorithms") or cfg.get("eval_algorithms") or [cfg.get("algorithm", "")]
    eval_algos = [str(x).lower() for x in eval_algos if str(x).strip()]
    window_setup = str(manifest.get("window_setup", cfg.get("window_setup", "")))
    window_count = int(wf_summary.get("window_count", 1 if (run_dir / "metrics_table.csv").exists() else 0))
    td3_action_equal_ratio = _collect_td3_action_equal_ratio(run_dir)

    return {
        "run_id": run_id,
        "window_setup": window_setup,
        "window_count": window_count,
        "eval_algorithms": ",".join(eval_algos),
        "max_trade_effective": int(manifest.get("max_trade_effective", cfg.get("max_trade", 0))),
        "intrinsic_timing_effective": str(manifest.get("intrinsic_timing_effective", cfg.get("intrinsic_timing", ""))),
        "state_norm_effective": json.dumps(manifest.get("state_norm_effective", {}), ensure_ascii=False),
        "experiment_phase": str(manifest.get("experiment_phase", "")),
        "claim_id": str(manifest.get("claim_id", "")),
        "hypothesis_id": str(manifest.get("hypothesis_id", "")),
        "is_confirmatory": bool(manifest.get("is_confirmatory", False)),
        "config_fingerprint": str(manifest.get("config_fingerprint", "")),
        "candidate_fingerprint": json.dumps(manifest.get("candidate_fingerprint", {}), ensure_ascii=False),
        "completeness_status": status,
        "missing_file_count": int(len(missing_files)),
        "missing_files": ";".join(str(x) for x in missing_files),
        "excluded_window_count": excluded_windows,
        "is_complete": bool(is_complete),
        "is_conclusion_eligible": bool(is_complete),
        "td3_action_equal_ratio_mean": float(td3_action_equal_ratio),
        "run_dir": str(run_dir.relative_to(root)),
    }


def _score_table(wf_table: pd.DataFrame) -> pd.DataFrame:
    out = wf_table.copy()
    out["Score"] = out["Sharpe_mean"].astype(float) + out["CR_mean"].astype(float)
    return out


def _aggregate_gain_rows(run_dir: Path) -> List[dict]:
    path = run_dir / "walk_forward_metrics_table.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    table = table[table["window_index"].astype(str) != "aggregate"].copy()
    if table.empty:
        return []
    table = _score_table(table)

    rows: List[dict] = []
    for algo, grp in table.groupby("algorithm"):
        g0 = grp[grp["group"] == "G0_baseline"][["window_index", "Score"]].rename(columns={"Score": "Score_G0"})
        if g0.empty:
            continue
        for group in ["G1_revise_only", "G2_intrinsic_only", "G3_revise_intrinsic"]:
            gk = grp[grp["group"] == group][["window_index", "Score"]].rename(columns={"Score": "Score_GK"})
            merged = g0.merge(gk, on="window_index", how="inner")
            if merged.empty:
                continue
            delta = merged["Score_GK"] - merged["Score_G0"]
            delta_mean = float(delta.mean())
            pos_ratio = float((delta > 0).mean())

            categories: List[str] = []
            if delta_mean <= 0:
                categories.append("direction_wrong")
            elif pos_ratio <= 0.5:
                categories.append("strength_insufficient")
            else:
                categories.append("pass_candidate")
            if group == "G2_intrinsic_only" and delta_mean <= 0:
                categories.append("intrinsic_direction_or_scale_risk")

            row = {
                "run_id": run_dir.name,
                "window_setup": str(_load_json(run_dir / "run_manifest.json").get("window_setup", "")),
                "algorithm": str(algo),
                "group": str(group),
                "delta_score_mean": delta_mean,
                "positive_window_ratio": pos_ratio,
                "window_count": int(len(merged)),
                "category": "|".join(categories),
                "notes": "",
            }
            rows.append(row)

    td3_action_equal = _collect_td3_action_equal_ratio(run_dir)
    if np.isfinite(td3_action_equal):
        for row in rows:
            if row["algorithm"] == "td3" and row["group"] in {"G1_revise_only", "G3_revise_intrinsic"}:
                if td3_action_equal >= 0.999:
                    row["category"] += "|action_unchanged"
                    row["notes"] = f"td3_action_equal_ratio_mean={td3_action_equal:.6f}"
    return rows


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True, help="Run ids under runs/")
    parser.add_argument("--output-csv", required=True, help="Evidence registry csv path")
    parser.add_argument("--output-md", required=True, help="Failure taxonomy markdown path")
    args = parser.parse_args()

    root = _repo_root()
    run_dirs = _resolve_run_dirs(root, [str(x) for x in args.run_ids])
    if not run_dirs:
        raise ValueError("No valid run dirs found.")

    registry_rows = [_collect_registry_row(run_dir) for run_dir in run_dirs]
    registry_df = pd.DataFrame(registry_rows).sort_values("run_id").reset_index(drop=True)

    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = (root / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(out_csv, index=False)

    failure_rows: List[dict] = []
    for run_dir in run_dirs:
        failure_rows.extend(_aggregate_gain_rows(run_dir))
    failure_df = pd.DataFrame(failure_rows)
    if not failure_df.empty:
        failure_df = failure_df.sort_values(["run_id", "algorithm", "group"]).reset_index(drop=True)

    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = (root / out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Failure Taxonomy",
        "",
        "- Categories:",
        "  - `direction_wrong`: aggregate delta <= 0",
        "  - `strength_insufficient`: aggregate delta > 0 but positive-window ratio <= 0.5",
        "  - `intrinsic_direction_or_scale_risk`: `G2_intrinsic_only` still negative",
        "  - `action_unchanged`: TD3 action-equality remains near 1.0",
        "",
        "## Table",
        "",
        _to_markdown_table(failure_df),
        "",
    ]
    out_md.write_text("\n".join(lines))

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
