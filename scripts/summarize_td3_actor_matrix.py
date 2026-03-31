from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUT_MD = ROOT / "docs" / "steps" / "step_td3_diagnosis" / "td3_actor_matrix_summary.md"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _completion_ok(run_manifest_path: Path) -> bool:
    if not run_manifest_path.exists():
        return False
    data = json.loads(run_manifest_path.read_text())
    cc = data.get("completeness_check", {})
    return cc.get("status") == "complete" or cc.get("complete") is True


def _collect_complete_runs() -> dict[tuple[str, float, float], str]:
    key_to_run: dict[tuple[str, float, float], str] = {}
    for run_dir in sorted(RUNS_DIR.glob("20260211_*_demo")):
        cfg_path = run_dir / "config.yaml"
        manifest_path = run_dir / "run_manifest.json"
        table_path = run_dir / "walk_forward_metrics_table.csv"
        if not (cfg_path.exists() and manifest_path.exists() and table_path.exists()):
            continue
        cfg = _load_yaml(cfg_path)
        if str(cfg.get("algorithm", "")).lower() != "td3":
            continue
        if not _completion_ok(manifest_path):
            continue
        ws = str(cfg.get("window_setup", ""))
        iw = float(cfg.get("intrinsic_w", -1))
        td3 = cfg.get("td3") or {}
        actor = float(td3.get("actor_max_action", cfg.get("max_trade", -1)))
        key_to_run[(ws, iw, actor)] = run_dir.name
    return key_to_run


def _row_for_config(cfg_path: Path, key_to_run: dict[tuple[str, float, float], str]) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    ws = str(cfg.get("window_setup", ""))
    iw = float(cfg.get("intrinsic_w", -1))
    td3 = cfg.get("td3") or {}
    actor = float(td3.get("actor_max_action", cfg.get("max_trade", -1)))
    key = (ws, iw, actor)
    row: dict[str, Any] = {
        "config": cfg_path.name,
        "window_setup": ws,
        "intrinsic_w": iw,
        "actor_max_action": actor,
        "status": "missing",
        "run_id": "",
        "mean_abs_delta_sharpe": None,
        "mean_abs_delta_cr": None,
        "classification": "",
    }
    run_id = key_to_run.get(key)
    if not run_id:
        return row

    row["run_id"] = run_id
    table = pd.read_csv(RUNS_DIR / run_id / "walk_forward_metrics_table.csv")
    d = table[table["window_index"].astype(str) != "aggregate"]
    g1 = d[d["group"] == "G1_revise_only"][["window_index", "algorithm", "Sharpe_mean", "CR_mean"]].rename(
        columns={"Sharpe_mean": "sh1", "CR_mean": "cr1"}
    )
    g3 = d[d["group"] == "G3_revise_intrinsic"][["window_index", "algorithm", "Sharpe_mean", "CR_mean"]].rename(
        columns={"Sharpe_mean": "sh3", "CR_mean": "cr3"}
    )
    z = g1.merge(g3, on=["window_index", "algorithm"])
    if z.empty:
        row["status"] = "incomplete"
        return row

    d_sh = float((z["sh3"] - z["sh1"]).abs().mean())
    d_cr = float((z["cr3"] - z["cr1"]).abs().mean())
    row["mean_abs_delta_sharpe"] = d_sh
    row["mean_abs_delta_cr"] = d_cr
    row["status"] = "complete"
    row["classification"] = "separated" if (d_sh + d_cr) > 1e-12 else "not_separated"
    return row


def main() -> None:
    cfg_paths = sorted(ROOT.glob("configs/td3_*_actor*.yaml"))
    key_to_run = _collect_complete_runs()
    rows = [_row_for_config(p, key_to_run) for p in cfg_paths]
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["window_setup", "intrinsic_w", "actor_max_action"], kind="stable")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# TD3 Actor Matrix Summary")
    lines.append("")
    lines.append(f"- total_configs: {len(df)}")
    lines.append(f"- complete: {int((df['status'] == 'complete').sum())}")
    lines.append(f"- missing: {int((df['status'] == 'missing').sum())}")
    lines.append("")
    lines.append("| config | setup | intrinsic_w | actor_max_action | status | run_id | mean_abs_delta_sharpe | mean_abs_delta_cr | classification |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|---|")
    for _, r in df.iterrows():
        sh = "" if pd.isna(r["mean_abs_delta_sharpe"]) else f"{float(r['mean_abs_delta_sharpe']):.8f}"
        cr = "" if pd.isna(r["mean_abs_delta_cr"]) else f"{float(r['mean_abs_delta_cr']):.8f}"
        lines.append(
            f"| {r['config']} | {r['window_setup']} | {r['intrinsic_w']:.0f} | {r['actor_max_action']:.0f} | {r['status']} | {r['run_id']} | "
            f"{sh} | {cr} | {r['classification']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(str(OUT_MD.relative_to(ROOT)))


if __name__ == "__main__":
    main()
