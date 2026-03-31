from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.finsaber_data import load_finsaber_prices
from src.drl.metrics import compute_metrics
from src.pipeline.demo import (
    DemoConfig,
    _filter_assets_align_dates,
    _generate_windows_from_setup,
    _json_safe,
    _split_df_by_date,
    run_demo,
)
from src.pipeline.regime_specialist import (
    REGIME_ORDER,
    build_causal_regime_labels,
    load_algo_seed_traces,
    route_seed_row_by_regime,
    summarize_window_regime_coverage,
)
from src.utils.paths import ensure_dir, repo_root


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_unique_run_dir(root: Path, target: str) -> Path:
    now = dt.datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{int(now.microsecond / 1000):03d}"
    nonce = uuid.uuid4().hex[:4]
    run_dir = root / "runs" / f"{timestamp}_{ms}_{nonce}_{target}"
    while run_dir.exists():
        nonce = uuid.uuid4().hex[:4]
        run_dir = root / "runs" / f"{timestamp}_{ms}_{nonce}_{target}"
    return run_dir


def _resolve_windows(cfg: DemoConfig, raw_prices: pd.DataFrame) -> list[dict]:
    windows = (cfg.walk_forward or {}).get("windows", [])
    if windows:
        return [_json_safe(window) for window in windows]
    return _generate_windows_from_setup(raw_prices, cfg)


def _build_effective_shared_cfg_dict(base_cfg_dict: dict, windows: list[dict], args: argparse.Namespace) -> dict:
    cfg_dict = copy.deepcopy(base_cfg_dict)
    cfg_dict["eval_algorithms"] = list(args.eval_algorithms)
    cfg_dict["algorithm"] = (
        cfg_dict.get("algorithm")
        if str(cfg_dict.get("algorithm", "")).lower() in {str(a).lower() for a in args.eval_algorithms}
        else list(args.eval_algorithms)[-1]
    )
    cfg_dict["seeds"] = [int(seed) for seed in args.seeds]
    cfg_dict["intrinsic_w"] = float(args.intrinsic_w)
    cfg_dict["intrinsic_w_schedule"] = [float(args.intrinsic_w)]
    if not cfg_dict.get("assets"):
        raise ValueError("regime-specialist oracle requires a non-empty fixed asset list in config.assets")

    llm_cfg = dict(cfg_dict.get("llm") or {})
    llm_cfg["enabled"] = bool(llm_cfg.get("enabled", True))
    llm_cfg["k"] = int(args.k)
    llm_cfg["iterations"] = int(args.iterations)
    llm_cfg["branch_parallel_workers"] = int(args.branch_parallel_workers)
    candidate_scoring = dict(llm_cfg.get("candidate_scoring") or {})
    candidate_scoring["selection_seed_count"] = int(args.selection_seed_count)
    llm_cfg["candidate_scoring"] = candidate_scoring
    cfg_dict["llm"] = llm_cfg

    universe_cfg = dict(cfg_dict.get("universe") or {})
    universe_cfg["mode"] = "fixed"
    universe_cfg["n_assets"] = int(len(cfg_dict["assets"]))
    universe_cfg["restrict_to_config_assets"] = True
    universe_cfg["allow_auto_asset_pool"] = False
    cfg_dict["universe"] = universe_cfg

    walk_forward_cfg = dict(cfg_dict.get("walk_forward") or {})
    walk_forward_cfg["enabled"] = True
    walk_forward_cfg["aggregate"] = str(walk_forward_cfg.get("aggregate", "mean_std"))
    walk_forward_cfg["min_days_per_split"] = int(walk_forward_cfg.get("min_days_per_split", 20))
    walk_forward_cfg["windows"] = windows
    cfg_dict["walk_forward"] = walk_forward_cfg
    return cfg_dict


def _window_split_dates(df: pd.DataFrame, window: dict, assets: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        split_df = _split_df_by_date(df, window[split_name]["start"], window[split_name]["end"])
        split_df = _filter_assets_align_dates(split_df, assets)
        out[split_name] = sorted(split_df["date"].unique().tolist())
    return out


def _subrun_required_files(run_dir: Path, eval_algorithms: list[str]) -> list[Path]:
    required = [
        run_dir / "run_manifest.json",
        run_dir / "metrics_table.csv",
        run_dir / "reward_trace.json",
        run_dir / "policy_behavior_summary.json",
    ]
    algo_lower = [str(algo).lower() for algo in eval_algorithms]
    if "td3" in algo_lower:
        required.append(run_dir / "td3_seed_trace.json")
    if any(algo != "td3" for algo in algo_lower):
        required.append(run_dir / "sb3_action_trace.json")
    return required


def _run_complete(run_dir: Path, eval_algorithms: list[str]) -> bool:
    return all(path.exists() for path in _subrun_required_files(run_dir, eval_algorithms))


def _walk_forward_run_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "run_manifest.json",
        run_dir / "walk_forward_metrics_table.csv",
        run_dir / "walk_forward_summary.json",
    ]
    return all(path.exists() for path in required)


def _run_specialist_subwindow(
    base_cfg_dict: dict,
    *,
    run_dir: Path,
    window: dict,
    train_dates: list[str],
    val_dates: list[str],
    eval_algorithms: list[str],
) -> None:
    if _run_complete(run_dir, eval_algorithms):
        return
    sub_cfg_dict = copy.deepcopy(base_cfg_dict)
    sub_cfg_dict["walk_forward"] = {"enabled": False}
    sub_cfg_dict["eval_protocol"] = "temporal_split"
    sub_cfg_dict["data_split"] = window
    sub_cfg_dict["split_date_filters"] = {
        "train": {"include_dates": list(train_dates)},
        "val": {"include_dates": list(val_dates)},
    }
    ensure_dir(run_dir)
    run_demo(DemoConfig(**sub_cfg_dict), run_dir=run_dir, data_dir=repo_root() / "data")


def _group_seed_rows(groups_payload: dict[str, list[dict]], group: str) -> dict[int, dict]:
    return {
        int(row.get("seed", -1)): row
        for row in groups_payload.get(group, []) or []
        if isinstance(row, dict) and int(row.get("seed", -1)) >= 0
    }


def _mean_metric(metrics_by_seed: dict[int, dict], key: str) -> float:
    vals = [float(payload.get(key, 0.0)) for payload in metrics_by_seed.values()]
    return float(np.mean(vals)) if vals else 0.0


def _append_returns(store: dict, variant: str, algo: str, group: str, seed: int, returns: list[float]) -> None:
    store.setdefault(variant, {}).setdefault(algo, {}).setdefault(group, {}).setdefault(int(seed), []).extend(
        float(x) for x in returns
    )


def _append_regime_returns(
    store: dict,
    variant: str,
    algo: str,
    group: str,
    seed: int,
    routing_trace: list[dict],
) -> None:
    target = store.setdefault(variant, {}).setdefault(algo, {}).setdefault(group, {}).setdefault(int(seed), {})
    for regime in REGIME_ORDER:
        target.setdefault(regime, [])
    for row in routing_trace:
        regime = str(row.get("label", "sideways"))
        if regime not in target:
            regime = "sideways"
        target[regime].append(float(row.get("daily_return", 0.0)))


def _returns_to_values(daily_returns: list[float], initial_value: float = 100000.0) -> list[float]:
    values = [float(initial_value)]
    for ret in daily_returns:
        values.append(float(values[-1] * (1.0 + float(ret))))
    return values


def _metrics_from_returns(daily_returns: list[float]) -> dict:
    return compute_metrics(np.asarray(_returns_to_values(daily_returns), dtype=float))


def _summarize_variant_group(
    returns_store: dict,
    regime_returns_store: dict,
    window_metric_records: list[dict],
    *,
    variant: str,
    algorithm: str,
    group: str,
    fallback_ratio: float,
) -> dict:
    seed_returns = returns_store.get(variant, {}).get(algorithm, {}).get(group, {})
    metrics_by_seed = {
        int(seed): _metrics_from_returns(path_returns)
        for seed, path_returns in seed_returns.items()
        if path_returns
    }
    sharpe = _mean_metric(metrics_by_seed, "Sharpe")
    cr = _mean_metric(metrics_by_seed, "CR")
    mdd = _mean_metric(metrics_by_seed, "MDD")
    av = _mean_metric(metrics_by_seed, "AV")

    window_rows = [
        row
        for row in window_metric_records
        if row["variant"] == variant and row["algorithm"] == algorithm and row["group"] == group
    ]
    positive_quarter_ratio = (
        float(np.mean([float(row.get("sharpe_delta_vs_g0", 0.0)) > 0.0 for row in window_rows]))
        if window_rows
        else 0.0
    )

    regime_metrics: dict[str, dict] = {}
    regime_seed_returns = regime_returns_store.get(variant, {}).get(algorithm, {}).get(group, {})
    for regime in REGIME_ORDER:
        per_seed = {
            int(seed): _metrics_from_returns(seed_payload.get(regime, []))
            for seed, seed_payload in regime_seed_returns.items()
            if seed_payload.get(regime)
        }
        regime_metrics[regime] = {
            "Sharpe": _mean_metric(per_seed, "Sharpe"),
            "CR": _mean_metric(per_seed, "CR"),
            "MDD": _mean_metric(per_seed, "MDD"),
        }

    return {
        "variant": variant,
        "algorithm": algorithm,
        "group": group,
        "Sharpe": sharpe,
        "CR": cr,
        "MDD": mdd,
        "AV": av,
        "positive_quarter_ratio": positive_quarter_ratio,
        "fallback_usage_ratio": float(fallback_ratio),
        "window_count": int(len(window_rows)),
        "seed_count": int(len(metrics_by_seed)),
        "regime_metrics": regime_metrics,
    }


def _load_context_rows(run_dir: Path, variant_name: str) -> list[dict]:
    table_path = run_dir / "walk_forward_metrics_table.csv"
    if not table_path.exists():
        return []
    frame = pd.read_csv(table_path)
    frame = frame[frame["window_name"] == "aggregate"].copy()
    rows: list[dict] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "variant": variant_name,
                "algorithm": row.get("algorithm", ""),
                "group": row.get("group", ""),
                "Sharpe": float(row.get("Sharpe_mean", 0.0)),
                "CR": float(row.get("CR_mean", 0.0)),
                "MDD": float(row.get("MDD_mean", 0.0)),
                "AV": float(row.get("AV_mean", 0.0)),
                "positive_quarter_ratio": np.nan,
                "fallback_usage_ratio": np.nan,
                "window_count": np.nan,
                "seed_count": np.nan,
                "regime_metrics": {},
                "comparable": False,
            }
        )
    return rows


def _write_summary(
    path: Path,
    rows: list[dict],
    coverage_summary: dict,
    *,
    baseline_dir: Path,
    specialist_dir: Path,
) -> None:
    frame = pd.DataFrame(rows)
    lines = [
        "# Regime-Specialist Oracle Summary",
        "",
        f"- shared_fixed_universe_short_window: `{baseline_dir}`",
        f"- regime_specialist_fixed_universe_short_window: `{specialist_dir}`",
        f"- window_count: {int(coverage_summary.get('window_count', 0))}",
        f"- fallback_ratio_overall: {float(coverage_summary.get('fallback_ratio_overall', 0.0)):.4f}",
        "",
        "## Acceptance Check",
        "",
    ]

    specialist_rows = frame[frame["variant"] == "regime_specialist_fixed_universe_short_window"].copy()
    shared_rows = frame[frame["variant"] == "shared_fixed_universe_short_window"].copy()
    shared_lookup = {
        (row["algorithm"], row["group"]): row
        for _, row in shared_rows.iterrows()
    }
    accepted = False
    for algo in sorted(set(specialist_rows["algorithm"].tolist())):
        lines.append(f"### {algo}")
        algo_rows = specialist_rows[specialist_rows["algorithm"] == algo]
        for target_group in ("G1_revise_only", "G3_revise_intrinsic"):
            row = algo_rows[algo_rows["group"] == target_group]
            if row.empty:
                continue
            payload = row.iloc[0].to_dict()
            shared_payload = shared_lookup.get((algo, target_group), {})
            sharpe_delta = float(payload.get("Sharpe", 0.0)) - float(shared_payload.get("Sharpe", 0.0))
            pqr_delta = float(payload.get("positive_quarter_ratio", 0.0)) - float(
                shared_payload.get("positive_quarter_ratio", 0.0)
            )
            fallback_ratio = float(payload.get("fallback_usage_ratio", 1.0))
            passes = sharpe_delta >= 0.10 and pqr_delta >= 0.10 and fallback_ratio < 0.25
            accepted = accepted or passes
            lines.append(
                f"- {target_group}: sharpe_delta_vs_shared={sharpe_delta:.4f}, "
                f"positive_quarter_ratio_delta={pqr_delta:.4f}, fallback_usage_ratio={fallback_ratio:.4f}, "
                f"passes={str(bool(passes)).lower()}"
            )
        lines.append("")

    if not accepted:
        lines.append("Conclusion: v1 acceptance criteria not met; hard regime splitting remains unproven for expansion.")
    else:
        lines.append("Conclusion: v1 acceptance criteria met; next step is expanding to SAC and then testing drift-triggered routing.")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the regime-specialist oracle experiment.")
    parser.add_argument("--base-config", required=True, help="Base demo config path relative to repo root.")
    parser.add_argument("--run-name", default="regime_specialist_oracle", help="Run directory suffix.")
    parser.add_argument("--label-start", default="2014-01-01")
    parser.add_argument("--label-end", default="2023-12-31")
    parser.add_argument("--eval-algorithms", nargs="+", default=["ppo", "td3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--branch-parallel-workers", type=int, default=2)
    parser.add_argument("--selection-seed-count", type=int, default=1)
    parser.add_argument("--intrinsic-w", type=float, default=0.02)
    parser.add_argument("--min-train-days", type=int, default=84)
    parser.add_argument("--min-val-days", type=int, default=21)
    parser.add_argument("--min-test-days", type=int, default=21)
    parser.add_argument("--context-long-run", default="", help="Optional existing long-window run dir relative to repo root.")
    parser.add_argument("--context-short-run", default="", help="Optional existing short-window prompt-only run dir relative to repo root.")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = root / args.base_config
    base_cfg_dict = load_config(cfg_path)
    cfg_preview = DemoConfig(**copy.deepcopy(base_cfg_dict))
    if cfg_preview.data_source != "finsaber" or not cfg_preview.finsaber_price_path:
        raise ValueError("regime-specialist oracle currently supports only finsaber configs.")

    raw_prices = load_finsaber_prices(
        (root / cfg_preview.finsaber_price_path).resolve(),
        None,
        cfg_preview.start_date,
        cfg_preview.end_date or cfg_preview.start_date,
    )
    windows = _resolve_windows(cfg_preview, raw_prices)
    if not windows:
        raise ValueError("No walk-forward windows resolved for regime-specialist oracle experiment.")

    shared_cfg_dict = _build_effective_shared_cfg_dict(base_cfg_dict, windows, args)
    shared_cfg = DemoConfig(**copy.deepcopy(shared_cfg_dict))

    run_dir = _build_unique_run_dir(root, args.run_name)
    ensure_dir(run_dir)
    (run_dir / "base_config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "effective_shared_config.yaml").write_text(
        yaml.safe_dump(shared_cfg_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    regime_labels = build_causal_regime_labels(
        raw_prices,
        label_start=args.label_start,
        label_end=args.label_end,
    )
    regime_labels_path = run_dir / "regime_labels.csv"
    regime_labels.to_csv(regime_labels_path, index=False)
    label_by_date = {
        str(row["date"]): str(row["final_label"])
        for _, row in regime_labels[["date", "final_label"]].iterrows()
    }

    shared_run_dir = run_dir / "shared_fixed_universe_short_window"
    if not _walk_forward_run_complete(shared_run_dir):
        ensure_dir(shared_run_dir)
        run_demo(shared_cfg, run_dir=shared_run_dir, data_dir=root / "data")

    specialist_root = run_dir / "regime_specialist_fixed_universe_short_window"
    ensure_dir(specialist_root)

    effective_assets = list(shared_cfg.assets)
    raw_prices_fixed = raw_prices[raw_prices["asset"].isin(effective_assets)].copy()
    coverage_windows: list[dict] = []
    window_metric_records: list[dict] = []
    full_returns: dict = {}
    regime_returns: dict = {}
    specialist_fallback_counts: dict[tuple[str, str], int] = {}
    specialist_total_days: dict[tuple[str, str], int] = {}

    for idx, window in enumerate(windows):
        window_name = f"wf_window_{idx:02d}"
        split_dates = _window_split_dates(raw_prices_fixed, window, effective_assets)
        coverage = summarize_window_regime_coverage(
            window_name=window_name,
            train_dates=split_dates["train"],
            val_dates=split_dates["val"],
            test_dates=split_dates["test"],
            labels_df=regime_labels,
            min_train_days=int(args.min_train_days),
            min_val_days=int(args.min_val_days),
            min_test_days=int(args.min_test_days),
        )
        coverage_windows.append(coverage)

        shared_window_dir = shared_run_dir / window_name
        eligible_regimes = {
            regime
            for regime, payload in coverage["per_regime"].items()
            if bool(payload.get("eligible", False))
        }
        regime_run_dirs: dict[str, Path] = {}
        for regime in sorted(eligible_regimes):
            regime_dir = specialist_root / window_name / f"regime_{regime}"
            regime_run_dirs[regime] = regime_dir
            _run_specialist_subwindow(
                shared_cfg_dict,
                run_dir=regime_dir,
                window=window,
                train_dates=[
                    date
                    for date in split_dates["train"]
                    if label_by_date.get(date, "sideways") == regime
                ],
                val_dates=[
                    date
                    for date in split_dates["val"]
                    if label_by_date.get(date, "sideways") == regime
                ],
                eval_algorithms=list(shared_cfg.eval_algorithms or [shared_cfg.algorithm]),
            )

        for variant in ("shared_fixed_universe_short_window", "regime_specialist_fixed_universe_short_window"):
            for algo in list(shared_cfg.eval_algorithms or [shared_cfg.algorithm]):
                shared_groups = load_algo_seed_traces(shared_window_dir, algo)
                specialist_groups_by_regime = {
                    regime: load_algo_seed_traces(run_path, algo)
                    for regime, run_path in regime_run_dirs.items()
                }
                for group in list(shared_cfg.groups or []):
                    shared_seed_rows = _group_seed_rows(shared_groups, group)
                    window_metrics_by_seed: dict[int, dict] = {}
                    for seed in [int(seed) for seed in shared_cfg.seeds]:
                        shared_seed_row = shared_seed_rows.get(seed)
                        if shared_seed_row is None:
                            continue

                        if variant == "shared_fixed_universe_short_window":
                            routed = route_seed_row_by_regime(
                                shared_seed_row=shared_seed_row,
                                specialist_seed_rows_by_regime={},
                                label_by_date=label_by_date,
                                eligible_regimes=set(),
                                test_dates=split_dates["test"],
                            )
                        else:
                            specialist_seed_rows = {}
                            for regime, groups_payload in specialist_groups_by_regime.items():
                                specialist_seed_row = _group_seed_rows(groups_payload, group).get(seed)
                                if specialist_seed_row is not None:
                                    specialist_seed_rows[regime] = specialist_seed_row
                            routed = route_seed_row_by_regime(
                                shared_seed_row=shared_seed_row,
                                specialist_seed_rows_by_regime=specialist_seed_rows,
                                label_by_date=label_by_date,
                                eligible_regimes=set(eligible_regimes),
                                test_dates=split_dates["test"],
                            )
                            specialist_fallback_counts[(algo, group)] = specialist_fallback_counts.get((algo, group), 0) + int(
                                routed["fallback_count"]
                            )
                            specialist_total_days[(algo, group)] = specialist_total_days.get((algo, group), 0) + int(
                                len(routed["daily_returns"])
                            )

                        _append_returns(full_returns, variant, algo, group, seed, routed["daily_returns"])
                        _append_regime_returns(regime_returns, variant, algo, group, seed, routed["routing_trace"])
                        window_metrics_by_seed[int(seed)] = compute_metrics(np.asarray(routed["values"], dtype=float))

                    window_metric_records.append(
                        {
                            "variant": variant,
                            "window_name": window_name,
                            "algorithm": algo,
                            "group": group,
                            "Sharpe": _mean_metric(window_metrics_by_seed, "Sharpe"),
                            "CR": _mean_metric(window_metrics_by_seed, "CR"),
                            "MDD": _mean_metric(window_metrics_by_seed, "MDD"),
                        }
                    )

    for variant in ("shared_fixed_universe_short_window", "regime_specialist_fixed_universe_short_window"):
        variant_rows = [row for row in window_metric_records if row["variant"] == variant]
        grouped_rows: dict[tuple[str, str], list[dict]] = {}
        for row in variant_rows:
            grouped_rows.setdefault((row["algorithm"], row["window_name"]), []).append(row)
        for (algo, window_name), rows in grouped_rows.items():
            base_sharpe = next((float(row["Sharpe"]) for row in rows if row["group"] == "G0_baseline"), 0.0)
            for row in rows:
                row["sharpe_delta_vs_g0"] = float(row["Sharpe"]) - base_sharpe

    result_rows: list[dict] = []
    comparable_rows: list[dict] = []
    for variant in ("shared_fixed_universe_short_window", "regime_specialist_fixed_universe_short_window"):
        for algo in list(shared_cfg.eval_algorithms or [shared_cfg.algorithm]):
            for group in list(shared_cfg.groups or []):
                if variant == "shared_fixed_universe_short_window":
                    fallback_ratio = 0.0
                else:
                    total_days = specialist_total_days.get((algo, group), 0)
                    fallback_ratio = (
                        float(specialist_fallback_counts.get((algo, group), 0) / total_days)
                        if total_days > 0
                        else 0.0
                    )
                row = _summarize_variant_group(
                    full_returns,
                    regime_returns,
                    window_metric_records,
                    variant=variant,
                    algorithm=algo,
                    group=group,
                    fallback_ratio=fallback_ratio,
                )
                row["comparable"] = True
                comparable_rows.append(row)

    shared_lookup = {
        (row["algorithm"], row["group"]): row
        for row in comparable_rows
        if row["variant"] == "shared_fixed_universe_short_window"
    }
    group_baseline_lookup = {
        (row["variant"], row["algorithm"]): row
        for row in comparable_rows
        if row["group"] == "G0_baseline"
    }
    for row in comparable_rows:
        base_row = group_baseline_lookup.get((row["variant"], row["algorithm"]), {})
        shared_row = shared_lookup.get((row["algorithm"], row["group"]), {})
        out = {
            "variant": row["variant"],
            "algorithm": row["algorithm"],
            "group": row["group"],
            "Sharpe": float(row["Sharpe"]),
            "CR": float(row["CR"]),
            "MDD": float(row["MDD"]),
            "AV": float(row["AV"]),
            "positive_quarter_ratio": float(row["positive_quarter_ratio"]),
            "fallback_usage_ratio": float(row["fallback_usage_ratio"]),
            "window_count": int(row["window_count"]),
            "seed_count": int(row["seed_count"]),
            "Sharpe_delta_vs_g0": float(row["Sharpe"]) - float(base_row.get("Sharpe", 0.0)),
            "CR_delta_vs_g0": float(row["CR"]) - float(base_row.get("CR", 0.0)),
            "MDD_delta_vs_g0": float(row["MDD"]) - float(base_row.get("MDD", 0.0)),
            "Sharpe_delta_vs_shared": float(row["Sharpe"]) - float(shared_row.get("Sharpe", 0.0)),
            "CR_delta_vs_shared": float(row["CR"]) - float(shared_row.get("CR", 0.0)),
            "MDD_delta_vs_shared": float(row["MDD"]) - float(shared_row.get("MDD", 0.0)),
        }
        for regime in REGIME_ORDER:
            regime_metrics = row["regime_metrics"].get(regime, {})
            shared_regime_metrics = shared_row.get("regime_metrics", {}).get(regime, {})
            out[f"{regime}_Sharpe"] = float(regime_metrics.get("Sharpe", 0.0))
            out[f"{regime}_CR"] = float(regime_metrics.get("CR", 0.0))
            out[f"{regime}_MDD"] = float(regime_metrics.get("MDD", 0.0))
            out[f"{regime}_Sharpe_delta_vs_shared"] = float(regime_metrics.get("Sharpe", 0.0)) - float(
                shared_regime_metrics.get("Sharpe", 0.0)
            )
        result_rows.append(out)

    if args.context_long_run:
        result_rows.extend(
            _load_context_rows((root / args.context_long_run).resolve(), "context_long_window_prompt_only")
        )
    if args.context_short_run:
        result_rows.extend(
            _load_context_rows((root / args.context_short_run).resolve(), "context_short_window_prompt_only")
        )

    metrics_path = run_dir / "specialist_vs_shared_metrics.csv"
    pd.DataFrame(result_rows).to_csv(metrics_path, index=False)

    coverage_summary = {
        "window_count": int(len(coverage_windows)),
        "fallback_ratio_overall": float(
            np.mean([float(row.get("fallback_ratio", 0.0)) for row in coverage_windows])
        )
        if coverage_windows
        else 0.0,
        "eligible_window_counts_by_regime": {
            regime: int(
                sum(
                    1
                    for window_payload in coverage_windows
                    if bool(window_payload.get("per_regime", {}).get(regime, {}).get("eligible", False))
                )
            )
            for regime in REGIME_ORDER
        },
    }
    coverage_payload = {
        "windows": coverage_windows,
        "summary": coverage_summary,
    }
    coverage_path = run_dir / "regime_window_coverage.json"
    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")

    summary_path = run_dir / "specialist_vs_shared_summary.md"
    _write_summary(
        summary_path,
        result_rows,
        coverage_summary,
        baseline_dir=shared_run_dir,
        specialist_dir=specialist_root,
    )

    artifacts = {
        "regime_labels": str(regime_labels_path.relative_to(root)),
        "shared_fixed_universe_short_window": str(shared_run_dir.relative_to(root)),
        "regime_specialist_fixed_universe_short_window": str(specialist_root.relative_to(root)),
        "regime_window_coverage": str(coverage_path.relative_to(root)),
        "specialist_vs_shared_metrics": str(metrics_path.relative_to(root)),
        "specialist_vs_shared_summary": str(summary_path.relative_to(root)),
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
