from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.finsaber_data import load_finsaber_prices
from src.drl.metrics import bootstrap_mean_ci
from src.pipeline.demo import (
    DemoConfig,
    _build_completeness_check,
    _build_wf_candidate_fingerprint,
    _generate_windows_from_setup,
    _hash_payload,
    _is_confirmatory,
    _json_safe,
    _resolve_action_bound_penalty_cfg,
    _resolve_action_quantization_mode,
    _resolve_bootstrap_cfg,
    _resolve_decision_rule,
    _resolve_diagnostics_cfg,
    _resolve_experiment_cfg,
    _resolve_intrinsic_postprocess_cfg,
    _resolve_intrinsic_w_tuning_cfg,
    _stable_seed,
    run_demo,
)
from src.utils.hash import sha256_file
from src.utils.paths import ensure_dir, repo_root


def _load_cfg(run_dir: Path) -> DemoConfig:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return DemoConfig(**raw)


def _all_windows(cfg: DemoConfig) -> list[dict]:
    windows = (cfg.walk_forward or {}).get("windows", [])
    if windows:
        return [_json_safe(w) for w in windows]
    if cfg.data_source != "finsaber" or not cfg.finsaber_price_path:
        raise ValueError("auto window generation requires finsaber data_source and finsaber_price_path")
    root = repo_root()
    data = load_finsaber_prices(
        (root / cfg.finsaber_price_path).resolve(),
        None,
        cfg.start_date,
        cfg.end_date or cfg.start_date,
    )
    return _generate_windows_from_setup(data, cfg)


def _sub_required_files(sub_run_dir: Path, eval_algos: list[str]) -> list[Path]:
    required = [
        sub_run_dir / "metrics_table.csv",
        sub_run_dir / "run_manifest.json",
        sub_run_dir / "reward_trace.json",
        sub_run_dir / "policy_behavior_summary.json",
    ]
    algo_lower = [str(a).lower() for a in eval_algos]
    if "td3" in algo_lower:
        required.extend(
            [
                sub_run_dir / "td3_g1_g3_diff.json",
                sub_run_dir / "td3_action_saturation.json",
                sub_run_dir / "state_scale_summary.json",
            ]
        )
    if any(a != "td3" for a in algo_lower):
        required.append(sub_run_dir / "sb3_action_trace.json")
    return required


def _window_complete(sub_run_dir: Path, eval_algos: list[str]) -> tuple[bool, dict]:
    check = _build_completeness_check(_sub_required_files(sub_run_dir, eval_algos))
    return (not check["excluded_incomplete"]), check


def _stable_json_key(payload: object) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False)


def _resolve_parent_field_from_windows(
    values_by_window: dict[str, object],
    *,
    empty_value: object,
    mixed_value: object,
) -> object:
    non_empty = {
        name: _json_safe(value)
        for name, value in values_by_window.items()
        if value not in (None, "", {}, [])
    }
    if not non_empty:
        return empty_value
    unique = {_stable_json_key(value): value for value in non_empty.values()}
    if len(unique) == 1:
        return next(iter(unique.values()))
    return mixed_value


def _run_single_window(
    cfg: DemoConfig,
    run_dir: Path,
    window: dict,
    idx: int,
    fixed_candidate_path: str | None,
    disable_llm: bool,
) -> None:
    sub_cfg_dict = dict(cfg.__dict__)
    sub_cfg_dict["eval_protocol"] = "temporal_split"
    sub_cfg_dict["data_split"] = window
    sub_cfg_dict["walk_forward"] = {"enabled": False}
    if fixed_candidate_path:
        sub_cfg_dict["fixed_candidate_path"] = fixed_candidate_path
    if disable_llm:
        llm_cfg = dict(sub_cfg_dict.get("llm") or {})
        llm_cfg["enabled"] = False
        sub_cfg_dict["llm"] = llm_cfg
    sub_cfg = DemoConfig(**sub_cfg_dict)
    sub_run_dir = run_dir / f"wf_window_{idx:02d}"
    ensure_dir(sub_run_dir)
    run_demo(sub_cfg, run_dir=sub_run_dir, data_dir=repo_root() / "data")


def _rebuild_parent(cfg: DemoConfig, run_dir: Path, windows: list[dict], resumed: list[int], overrides: dict) -> None:
    root = repo_root()
    min_days = int((cfg.walk_forward or {}).get("min_days_per_split", 10))
    aggregate_mode = str((cfg.walk_forward or {}).get("aggregate", "mean_std"))
    bootstrap_cfg = _resolve_bootstrap_cfg(cfg.bootstrap)
    experiment_cfg = _resolve_experiment_cfg(cfg)
    config_fingerprint = _hash_payload(cfg.__dict__)
    eval_algos = cfg.eval_algorithms or [cfg.algorithm]

    window_tables: list[pd.DataFrame] = []
    window_infos: list[dict] = []
    excluded_windows: list[dict] = []
    scoring_objective_by_window: dict[str, object] = {}
    candidate_scoring_effective_by_window: dict[str, object] = {}
    best_candidate_by_algo_by_window: dict[str, object] = {}
    candidate_fingerprint_by_algo_by_window: dict[str, object] = {}
    scenario_profile_by_window: dict[str, object] = {}
    for idx, window in enumerate(windows):
        sub_run_dir = run_dir / f"wf_window_{idx:02d}"
        sub_manifest_path = sub_run_dir / "run_manifest.json"
        sub_table_path = sub_run_dir / "metrics_table.csv"
        sub_reward_trace_path = sub_run_dir / "reward_trace.json"
        sub_state_scale_path = sub_run_dir / "state_scale_summary.json"
        sub_policy_behavior_path = sub_run_dir / "policy_behavior_summary.json"
        sub_sb3_trace_path = sub_run_dir / "sb3_action_trace.json"
        sub_td3_diff_path = sub_run_dir / "td3_g1_g3_diff.json"
        sub_td3_sat_path = sub_run_dir / "td3_action_saturation.json"

        complete, sub_completeness = _window_complete(sub_run_dir, eval_algos)
        if not complete:
            excluded_windows.append(
                {
                    "window_index": idx,
                    "window_name": f"wf_window_{idx:02d}",
                    "missing_files": sub_completeness["missing_files"],
                }
            )
            continue

        sub_manifest = json.loads(sub_manifest_path.read_text(encoding="utf-8"))
        window_name = f"wf_window_{idx:02d}"
        split_meta = sub_manifest.get("split", {})
        for split_name in ["train", "val", "test"]:
            cur_days = int(split_meta.get(split_name, {}).get("days", 0))
            if cur_days < min_days:
                raise ValueError(
                    f"walk-forward window {idx} {split_name} has {cur_days} days, below min_days_per_split={min_days}"
                )

        table = pd.read_csv(sub_table_path)
        table.insert(0, "window_index", idx)
        table.insert(1, "window_name", window_name)
        table.insert(2, "window_train", f"{window['train']['start']}->{window['train']['end']}")
        table.insert(3, "window_val", f"{window['val']['start']}->{window['val']['end']}")
        table.insert(4, "window_test", f"{window['test']['start']}->{window['test']['end']}")
        window_tables.append(table)

        scoring_objective_by_window[window_name] = sub_manifest.get("scoring_objective")
        candidate_scoring_effective_by_window[window_name] = sub_manifest.get("candidate_scoring_effective")
        best_candidate_by_algo_by_window[window_name] = sub_manifest.get("best_candidate_by_algo")
        candidate_fingerprint_by_algo_by_window[window_name] = sub_manifest.get("candidate_fingerprint_by_algo")
        scenario_profile_by_window[window_name] = sub_manifest.get("scenario_profile")

        window_infos.append(
            {
                "window_index": idx,
                "window_name": window_name,
                "split": window,
                "run_dir": str(sub_run_dir.relative_to(root)),
                "metrics_table": str(sub_table_path.relative_to(root)),
                "reward_trace": str(sub_reward_trace_path.relative_to(root)),
                "run_manifest": str(sub_manifest_path.relative_to(root)),
                "td3_g1_g3_diff": str(sub_td3_diff_path.relative_to(root)) if sub_td3_diff_path.exists() else "",
                "td3_action_saturation": str(sub_td3_sat_path.relative_to(root)) if sub_td3_sat_path.exists() else "",
                "state_scale_summary": str(sub_state_scale_path.relative_to(root)) if sub_state_scale_path.exists() else "",
                "policy_behavior_summary": str(sub_policy_behavior_path.relative_to(root))
                if sub_policy_behavior_path.exists()
                else "",
                "sb3_action_trace": str(sub_sb3_trace_path.relative_to(root)) if sub_sb3_trace_path.exists() else "",
                "scoring_objective": sub_manifest.get("scoring_objective"),
                "candidate_scoring_effective": _json_safe(sub_manifest.get("candidate_scoring_effective")),
                "best_candidate_by_algo": _json_safe(sub_manifest.get("best_candidate_by_algo")),
                "candidate_fingerprint_by_algo": _json_safe(sub_manifest.get("candidate_fingerprint_by_algo")),
                "scenario_profile": _json_safe(sub_manifest.get("scenario_profile")),
                "completeness_check": sub_completeness,
                "hashes": {
                    "metrics_table": sha256_file(sub_table_path),
                    "reward_trace": sha256_file(sub_reward_trace_path),
                    "run_manifest": sha256_file(sub_manifest_path),
                    "td3_g1_g3_diff": sha256_file(sub_td3_diff_path) if sub_td3_diff_path.exists() else "",
                    "td3_action_saturation": sha256_file(sub_td3_sat_path) if sub_td3_sat_path.exists() else "",
                    "state_scale_summary": sha256_file(sub_state_scale_path) if sub_state_scale_path.exists() else "",
                    "policy_behavior_summary": sha256_file(sub_policy_behavior_path)
                    if sub_policy_behavior_path.exists()
                    else "",
                    "sb3_action_trace": sha256_file(sub_sb3_trace_path) if sub_sb3_trace_path.exists() else "",
                },
            }
        )

    wf_table = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    agg_rows: list[dict] = []
    if not wf_table.empty:
        for (algo, group), grp in wf_table.groupby(["algorithm", "group"]):
            row = {
                "window_index": "aggregate",
                "window_name": "aggregate",
                "window_train": "",
                "window_val": "",
                "window_test": "",
                "algorithm": algo,
                "group": group,
                "Sharpe_mean": float(grp["Sharpe_mean"].mean()),
                "Sharpe_std": float(grp["Sharpe_mean"].std()),
                "CR_mean": float(grp["CR_mean"].mean()),
                "CR_std": float(grp["CR_mean"].std()),
                "MDD_mean": float(grp["MDD_mean"].mean()),
                "MDD_std": float(grp["MDD_mean"].std()),
                "AV_mean": float(grp["AV_mean"].mean()),
                "AV_std": float(grp["AV_mean"].std()),
                "intrinsic_mean": float(grp["intrinsic_mean"].mean()),
                "intrinsic_std": float(grp["intrinsic_mean"].std()),
                "intrinsic_w_effective_mean": float(grp["intrinsic_w_effective_mean"].mean()),
                "Sharpe_ci_low": "",
                "Sharpe_ci_high": "",
                "CR_ci_low": "",
                "CR_ci_high": "",
            }
            if bootstrap_cfg["enabled"]:
                sharpe_bs = bootstrap_mean_ci(
                    grp["Sharpe_mean"].to_list(),
                    n_resamples=bootstrap_cfg["n_resamples"],
                    alpha=bootstrap_cfg["alpha"],
                    random_seed=_stable_seed(bootstrap_cfg["random_seed"], f"wf:{algo}:{group}:Sharpe"),
                )
                cr_bs = bootstrap_mean_ci(
                    grp["CR_mean"].to_list(),
                    n_resamples=bootstrap_cfg["n_resamples"],
                    alpha=bootstrap_cfg["alpha"],
                    random_seed=_stable_seed(bootstrap_cfg["random_seed"], f"wf:{algo}:{group}:CR"),
                )
                row["Sharpe_ci_low"] = sharpe_bs["ci_low"]
                row["Sharpe_ci_high"] = sharpe_bs["ci_high"]
                row["CR_ci_low"] = cr_bs["ci_low"]
                row["CR_ci_high"] = cr_bs["ci_high"]
            agg_rows.append(row)

    wf_table_out = pd.concat([wf_table, pd.DataFrame(agg_rows)], ignore_index=True) if not wf_table.empty else pd.DataFrame(agg_rows)
    wf_table_path = run_dir / "walk_forward_metrics_table.csv"
    wf_table_out.to_csv(wf_table_path, index=False)

    scoring_objective_parent = _resolve_parent_field_from_windows(
        scoring_objective_by_window,
        empty_value="Sharpe + CR",
        mixed_value="mixed; see scoring_objective_by_window",
    )
    candidate_scoring_effective_parent = _resolve_parent_field_from_windows(
        candidate_scoring_effective_by_window,
        empty_value={},
        mixed_value={"note": "varies by window; see candidate_scoring_effective_by_window"},
    )
    best_candidate_by_algo_parent = _resolve_parent_field_from_windows(
        best_candidate_by_algo_by_window,
        empty_value={},
        mixed_value={"note": "varies by window; see best_candidate_by_algo_by_window"},
    )
    candidate_fingerprint_by_algo_parent = _resolve_parent_field_from_windows(
        candidate_fingerprint_by_algo_by_window,
        empty_value={},
        mixed_value={"note": "varies by window; see candidate_fingerprint_by_algo_by_window"},
    )

    summary = {
        "mode": "walk_forward",
        "window_setup": cfg.window_setup,
        "aggregate": aggregate_mode,
        "window_count": len(window_infos),
        "bootstrap_ci": bootstrap_cfg,
        "windows": window_infos,
        "scoring_objective": scoring_objective_parent,
        "scoring_objective_by_window": _json_safe(scoring_objective_by_window),
        "candidate_scoring_effective": _json_safe(candidate_scoring_effective_parent),
        "candidate_scoring_effective_by_window": _json_safe(candidate_scoring_effective_by_window),
        "best_candidate_by_algo": _json_safe(best_candidate_by_algo_parent),
        "best_candidate_by_algo_by_window": _json_safe(best_candidate_by_algo_by_window),
        "candidate_fingerprint_by_algo": _json_safe(candidate_fingerprint_by_algo_parent),
        "candidate_fingerprint_by_algo_by_window": _json_safe(candidate_fingerprint_by_algo_by_window),
        "scenario_profile_by_window": _json_safe(scenario_profile_by_window),
        "resume": {"resumed_window_indices": resumed, "overrides": overrides},
    }
    summary_path = run_dir / "walk_forward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"walk_forward": summary}, indent=2), encoding="utf-8")

    run_manifest = {
        "protocol_version": "trading-lesr-v2-walk-forward",
        "eval_protocol": "temporal_split",
        "scoring_objective": scoring_objective_parent,
        "experiment_phase": experiment_cfg["phase"],
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
        "is_confirmatory": _is_confirmatory(experiment_cfg),
        "config_fingerprint": config_fingerprint,
        "window_setup": cfg.window_setup,
        "decision_ts_rule": _resolve_decision_rule(cfg),
        "action_quantization_mode": _resolve_action_quantization_mode(cfg),
        "action_bound_penalty_effective": _resolve_action_bound_penalty_cfg(cfg),
        "universe": cfg.universe or {"mode": "fixed"},
        "algorithm": cfg.algorithm,
        "eval_algorithms": eval_algos,
        "state_norm_effective": {"note": "window-level manifests contain exact values"},
        "intrinsic_timing_effective": cfg.intrinsic_timing,
        "intrinsic_postprocess_effective": _resolve_intrinsic_postprocess_cfg(cfg.intrinsic_postprocess),
        "intrinsic_w_tuning_effective": _resolve_intrinsic_w_tuning_cfg(cfg),
        "diagnostics_effective": _resolve_diagnostics_cfg(cfg.diagnostics),
        "max_trade_effective": int(cfg.max_trade),
        "warmup_ratio": cfg.warmup_ratio,
        "bootstrap_ci": bootstrap_cfg,
        "candidate_scoring_effective": _json_safe(candidate_scoring_effective_parent),
        "candidate_scoring_effective_by_window": _json_safe(candidate_scoring_effective_by_window),
        "best_candidate_by_algo": _json_safe(best_candidate_by_algo_parent),
        "best_candidate_by_algo_by_window": _json_safe(best_candidate_by_algo_by_window),
        "candidate_fingerprint_by_algo": _json_safe(candidate_fingerprint_by_algo_parent),
        "candidate_fingerprint_by_algo_by_window": _json_safe(candidate_fingerprint_by_algo_by_window),
        "scenario_profile_by_window": _json_safe(scenario_profile_by_window),
        "walk_forward": {
            "enabled": True,
            "aggregate": aggregate_mode,
            "min_days_per_split": min_days,
            "window_count": len(window_infos),
            "windows": window_infos,
            "excluded_windows": excluded_windows,
        },
        "resume": {"resumed_window_indices": resumed, "overrides": overrides},
        "stub": {
            "use_finagent_signal": cfg.use_finagent_signal,
            "finagent_weight": cfg.finagent_weight,
        },
    }
    run_manifest["candidate_fingerprint"] = _build_wf_candidate_fingerprint(root, window_infos)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    run_required = [run_manifest_path, summary_path, wf_table_path]
    wf_completeness = _build_completeness_check(run_required)
    if excluded_windows:
        wf_completeness["excluded_incomplete"] = True
        wf_completeness["status"] = "incomplete"
    wf_completeness["excluded_window_count"] = int(len(excluded_windows))
    wf_completeness["excluded_windows"] = excluded_windows
    run_manifest["completeness_check"] = wf_completeness
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    artifacts = {
        "metrics": str(metrics_path.relative_to(root)),
        "run_manifest": str(run_manifest_path.relative_to(root)),
        "walk_forward_summary": str(summary_path.relative_to(root)),
        "walk_forward_metrics_table": str(wf_table_path.relative_to(root)),
    }
    artifacts_path = run_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    hashes = {
        "metrics": sha256_file(metrics_path),
        "run_manifest": sha256_file(run_manifest_path),
        "walk_forward_summary": sha256_file(summary_path),
        "walk_forward_metrics_table": sha256_file(wf_table_path),
    }
    hashes_path = run_dir / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Existing walk-forward run directory.")
    parser.add_argument(
        "--fixed-candidate-path",
        default="",
        help="Optional fixed candidate file path used for resumed windows.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable llm during resumed windows.",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Rerun complete windows too (default false).",
    )
    parser.add_argument(
        "--window-indices",
        nargs="+",
        type=int,
        default=None,
        help="Specific window indices to rerun regardless of completeness, e.g. --window-indices 2 3.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_cfg(run_dir)
    windows = _all_windows(cfg)
    eval_algos = cfg.eval_algorithms or [cfg.algorithm]
    selected_indices: list[int] | None = None
    if args.window_indices:
        selected_indices = []
        for idx in args.window_indices:
            if idx < 0 or idx >= len(windows):
                raise ValueError(f"window index {idx} out of range for {len(windows)} windows")
            if idx not in selected_indices:
                selected_indices.append(idx)

    resumed: list[int] = []
    for idx, window in enumerate(windows):
        sub_run_dir = run_dir / f"wf_window_{idx:02d}"
        is_complete, _ = _window_complete(sub_run_dir, eval_algos)
        if selected_indices is not None:
            if idx not in selected_indices:
                continue
        elif is_complete and not args.include_complete:
            continue
        _run_single_window(
            cfg=cfg,
            run_dir=run_dir,
            window=window,
            idx=idx,
            fixed_candidate_path=args.fixed_candidate_path or None,
            disable_llm=bool(args.disable_llm),
        )
        resumed.append(idx)

    overrides = {
        "fixed_candidate_path": args.fixed_candidate_path or "",
        "disable_llm": bool(args.disable_llm),
        "include_complete": bool(args.include_complete),
        "window_indices": selected_indices or [],
    }
    _rebuild_parent(cfg=cfg, run_dir=run_dir, windows=windows, resumed=resumed, overrides=overrides)
    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "total_windows": len(windows),
                "resumed_windows": resumed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
