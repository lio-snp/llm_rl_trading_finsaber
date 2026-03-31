from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUT_MD = ROOT / "docs" / "steps" / "step_td3_diagnosis" / "report_mt500.md"
OUT_EVIDENCE_MD = ROOT / "docs" / "steps" / "step_td3_diagnosis" / "evidence_matrix_mt500.md"


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _is_target_run(cfg: dict) -> bool:
    eval_algos = [str(x).lower() for x in cfg.get("eval_algorithms", [])]
    groups = set(cfg.get("groups", []))
    return (
        cfg.get("algorithm") == "td3"
        and cfg.get("window_setup") in {"selected_4", "composite"}
        and int(cfg.get("max_trade", 0)) == 500
        and float(cfg.get("intrinsic_w", 0.0)) in {100.0, 300.0}
        and cfg.get("intrinsic_timing") == "pre_action_state"
        and eval_algos == ["td3"]
        and groups == {"G1_revise_only", "G3_revise_intrinsic"}
        and bool((cfg.get("walk_forward") or {}).get("enabled", False))
    )


def _completeness_for_run(run_dir: Path) -> tuple[bool, list[str], list[dict]]:
    missing: list[str] = []
    required_root = [
        run_dir / "run_manifest.json",
        run_dir / "walk_forward_summary.json",
        run_dir / "walk_forward_metrics_table.csv",
    ]
    for p in required_root:
        if not p.exists():
            missing.append(p.name)
    excluded_windows: list[dict] = []
    summary_path = run_dir / "walk_forward_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for w in summary.get("windows", []):
            win_name = str(w.get("window_name", ""))
            checks = [
                run_dir / win_name / "metrics_table.csv",
                run_dir / win_name / "run_manifest.json",
                run_dir / win_name / "reward_trace.json",
                run_dir / win_name / "td3_g1_g3_diff.json",
                run_dir / win_name / "td3_action_saturation.json",
                run_dir / win_name / "state_scale_summary.json",
            ]
            miss = [p.name for p in checks if not p.exists()]
            if miss:
                excluded_windows.append({"window_name": win_name, "missing_files": miss})
    ok = not missing and not excluded_windows
    return ok, missing, excluded_windows


def _load_aggregate_metrics(run_dir: Path) -> dict:
    table = pd.read_csv(run_dir / "walk_forward_metrics_table.csv")
    table = table[table["window_index"].astype(str) == "aggregate"]
    table = table[table["algorithm"] == "td3"]
    by_group = {row["group"]: row for _, row in table.iterrows()}
    g1 = by_group.get("G1_revise_only")
    g3 = by_group.get("G3_revise_intrinsic")
    if g1 is None or g3 is None:
        return {}
    return {
        "G1_Sharpe": float(g1["Sharpe_mean"]),
        "G3_Sharpe": float(g3["Sharpe_mean"]),
        "Sharpe_delta_G3_minus_G1": float(g3["Sharpe_mean"] - g1["Sharpe_mean"]),
        "G1_CR": float(g1["CR_mean"]),
        "G3_CR": float(g3["CR_mean"]),
        "CR_delta_G3_minus_G1": float(g3["CR_mean"] - g1["CR_mean"]),
    }


def _classify_root_cause(action_equal_ratio: float, eval_value_mae: float, reward_total_delta: float) -> str:
    if action_equal_ratio >= 0.999 and abs(reward_total_delta) > 1e-6:
        return "R1"
    if action_equal_ratio < 0.999 and eval_value_mae <= 1e-6:
        return "R2"
    if action_equal_ratio < 0.999 and eval_value_mae > 1e-6:
        return "R3"
    return "R4"


def _load_window_diagnosis(run_dir: Path) -> dict:
    summary_path = run_dir / "walk_forward_summary.json"
    if not summary_path.exists():
        return {}
    wf = json.loads(summary_path.read_text())
    act_vals = []
    mae_vals = []
    reward_delta_vals = []
    intrinsic_vals = []
    diagnosis_counts: dict[str, int] = {}
    for w in wf.get("windows", []):
        win_name = str(w.get("window_name", ""))
        p = run_dir / win_name / "td3_g1_g3_diff.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text()).get("td3", {})
        diag = str(obj.get("diagnosis", ""))
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1
        s = obj.get("summary", {})
        act_vals.append(float(s.get("action_equal_ratio_mean", 0.0)))
        mae_vals.append(float(s.get("eval_value_mae_mean", 0.0)))
        reward_delta_vals.append(float(s.get("eval_reward_total_delta_mean", 0.0)))
        intrinsic_vals.append(float(s.get("intrinsic_mean_delta_mean", 0.0)))
    if not act_vals:
        return {}
    action_mean = float(sum(act_vals) / len(act_vals))
    mae_mean = float(sum(mae_vals) / len(mae_vals))
    reward_delta_mean = float(sum(reward_delta_vals) / len(reward_delta_vals))
    intrinsic_delta_mean = float(sum(intrinsic_vals) / len(intrinsic_vals))
    major_diag = max(diagnosis_counts.items(), key=lambda kv: kv[1])[0]
    return {
        "window_diag_count": len(act_vals),
        "diag_major": major_diag,
        "diag_action_equal_ratio_mean": action_mean,
        "diag_eval_value_mae_mean": mae_mean,
        "diag_eval_reward_total_delta_mean": reward_delta_mean,
        "diag_intrinsic_mean_delta_mean": intrinsic_delta_mean,
        "root_cause": _classify_root_cause(action_mean, mae_mean, reward_delta_mean),
    }


def _window_metrics_pair(win_dir: Path) -> dict:
    p = win_dir / "metrics_table.csv"
    if not p.exists():
        return {}
    table = pd.read_csv(p)
    table = table[table["algorithm"] == "td3"]
    by_group = {row["group"]: row for _, row in table.iterrows()}
    g1 = by_group.get("G1_revise_only")
    g3 = by_group.get("G3_revise_intrinsic")
    if g1 is None or g3 is None:
        return {}
    return {
        "g1_sharpe": float(g1["Sharpe_mean"]),
        "g3_sharpe": float(g3["Sharpe_mean"]),
        "delta_sharpe": float(g3["Sharpe_mean"] - g1["Sharpe_mean"]),
        "g1_cr": float(g1["CR_mean"]),
        "g3_cr": float(g3["CR_mean"]),
        "delta_cr": float(g3["CR_mean"] - g1["CR_mean"]),
    }


def _load_window_evidence_rows(run_dir: Path, setup: str, intrinsic_w: float) -> list[dict]:
    summary_path = run_dir / "walk_forward_summary.json"
    if not summary_path.exists():
        return []
    wf = json.loads(summary_path.read_text())
    rows = []
    for w in wf.get("windows", []):
        win_name = str(w.get("window_name", ""))
        win_dir = run_dir / win_name
        diff_path = win_dir / "td3_g1_g3_diff.json"
        if not diff_path.exists():
            continue
        obj = json.loads(diff_path.read_text()).get("td3", {})
        s = obj.get("summary", {})
        pair = _window_metrics_pair(win_dir)
        rows.append(
            {
                "run_id": run_dir.name,
                "setup": setup,
                "intrinsic_w": intrinsic_w,
                "window": win_name,
                "diagnosis": str(obj.get("diagnosis", "")),
                "action_equal_ratio_mean": float(s.get("action_equal_ratio_mean", 0.0)),
                "eval_value_mae_mean": float(s.get("eval_value_mae_mean", 0.0)),
                "eval_reward_total_delta_mean": float(s.get("eval_reward_total_delta_mean", 0.0)),
                "intrinsic_mean_delta_mean": float(s.get("intrinsic_mean_delta_mean", 0.0)),
                "delta_sharpe": float(pair.get("delta_sharpe", float("nan"))),
                "delta_cr": float(pair.get("delta_cr", float("nan"))),
            }
        )
    return rows


def main() -> int:
    rows = []
    evidence_rows = []
    excluded = []
    for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        cfg = _read_yaml(run_dir / "config.yaml")
        if not _is_target_run(cfg):
            continue
        ok, missing, excluded_windows = _completeness_for_run(run_dir)
        base = {
            "run_id": run_dir.name,
            "window_setup": cfg.get("window_setup"),
            "intrinsic_w": float(cfg.get("intrinsic_w")),
            "complete": ok,
            "missing_root_files": ",".join(missing),
            "excluded_window_count": len(excluded_windows),
        }
        if ok:
            base.update(_load_aggregate_metrics(run_dir))
            base.update(_load_window_diagnosis(run_dir))
            rows.append(base)
            evidence_rows.extend(
                _load_window_evidence_rows(
                    run_dir,
                    str(base["window_setup"]),
                    float(base["intrinsic_w"]),
                )
            )
        else:
            excluded.append(base | {"excluded_windows": excluded_windows})

    rows = sorted(rows, key=lambda r: (r["window_setup"], r["intrinsic_w"], r["run_id"]))
    lines = [
        "# TD3 max_trade=500 Diagnostic Report",
        "",
        "## Complete Runs",
        "",
        "| run_id | setup | intrinsic_w | G1 Sharpe | G3 Sharpe | ΔSharpe(G3-G1) | G1 CR | G3 CR | ΔCR(G3-G1) | root_cause | diag_action_equal | diag_value_mae | diag_reward_total_delta | diag_intrinsic_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['window_setup']} | {r['intrinsic_w']:.1f} | "
            f"{r.get('G1_Sharpe', float('nan')):.6f} | {r.get('G3_Sharpe', float('nan')):.6f} | {r.get('Sharpe_delta_G3_minus_G1', float('nan')):.6f} | "
            f"{r.get('G1_CR', float('nan')):.6f} | {r.get('G3_CR', float('nan')):.6f} | {r.get('CR_delta_G3_minus_G1', float('nan')):.6f} | "
            f"{r.get('root_cause', 'NA')} | {r.get('diag_action_equal_ratio_mean', float('nan')):.6f} | {r.get('diag_eval_value_mae_mean', float('nan')):.6f} | "
            f"{r.get('diag_eval_reward_total_delta_mean', float('nan')):.6f} | {r.get('diag_intrinsic_mean_delta_mean', float('nan')):.6f} |"
        )

    lines += [
        "",
        "## Excluded Runs",
        "",
        "| run_id | setup | intrinsic_w | missing_root_files | excluded_window_count |",
        "|---|---|---:|---|---:|",
    ]
    for r in sorted(excluded, key=lambda x: x["run_id"]):
        lines.append(
            f"| {r['run_id']} | {r['window_setup']} | {r['intrinsic_w']:.1f} | {r['missing_root_files']} | {r['excluded_window_count']} |"
        )

    lines += [
        "",
        "## Evaluation Protocol Rationale (FinSABER-aligned)",
        "",
        "- Stage1 (`selected_4`): long-cycle recheck under common selective-evaluation setup (2y test, 1y step, prior<=3y).",
        "- Stage2 (`composite`): stricter dynamic-pool yearly re-evaluation (1y test, 1y step, prior<=2y).",
        "- This report keeps both setups to separate: (a) long-horizon reproducibility under legacy setting vs (b) robustness under dynamic selection and more frequent rebalancing.",
        "",
        "### Evidence Paths",
        "",
    ]
    for r in rows:
        run_id = r["run_id"]
        lines.append(f"- `{run_id}`: `runs/{run_id}/walk_forward_metrics_table.csv`, `runs/{run_id}/walk_forward_summary.json`, `runs/{run_id}/run_manifest.json`")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    ev_lines = [
        "# TD3 mt500 Evidence Matrix",
        "",
        "Window-level evidence for G1 vs G3 under long-window TD3 runs.",
        "",
        "| run_id | setup | intrinsic_w | window | diagnosis | action_equal_ratio | eval_value_mae | eval_reward_total_delta | intrinsic_delta | delta_sharpe | delta_cr |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(evidence_rows, key=lambda x: (x["setup"], x["intrinsic_w"], x["run_id"], x["window"])):
        ev_lines.append(
            f"| {r['run_id']} | {r['setup']} | {r['intrinsic_w']:.1f} | {r['window']} | {r['diagnosis']} | "
            f"{r['action_equal_ratio_mean']:.6f} | {r['eval_value_mae_mean']:.6f} | {r['eval_reward_total_delta_mean']:.6f} | "
            f"{r['intrinsic_mean_delta_mean']:.6f} | {r['delta_sharpe']:.6f} | {r['delta_cr']:.6f} |"
        )
    if excluded:
        ev_lines += [
            "",
            "## Excluded Runs",
            "",
            "| run_id | setup | intrinsic_w | missing_root_files | excluded_window_count |",
            "|---|---|---:|---|---:|",
        ]
        for r in sorted(excluded, key=lambda x: x["run_id"]):
            ev_lines.append(
                f"| {r['run_id']} | {r['window_setup']} | {r['intrinsic_w']:.1f} | {r['missing_root_files']} | {r['excluded_window_count']} |"
            )
    OUT_EVIDENCE_MD.write_text("\n".join(ev_lines))
    print(str(OUT_MD))
    print(str(OUT_EVIDENCE_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
