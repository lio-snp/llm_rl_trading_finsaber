from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.synth_data import SynthConfig, generate_synth_ohlcv, save_raw_data
from src.data.finsaber_data import load_finsaber_prices
from src.data.features import add_indicators
from src.env.state_schema import StateSchema
from src.env.trading_env import EnvConfig, TradingEnv
from src.drl.policy import HeuristicPolicy, PolicyConfig
from src.drl.state_norm import (
    build_td3_state_fn,
    matrix_stats,
    resolve_td3_state_norm_config,
)
from src.drl.td3_runner import TD3Config, train_td3
from src.drl.sb3_runner import SB3Config, train_sb3
from src.drl.metrics import bootstrap_mean_ci, compute_metrics
from src.drl.finsaber_compat_preprocessor import load_default_finrl_indicators as load_compat_finrl_indicators
from src.drl.finsaber_compat_runner import FinsaberCompatConfig, train_finsaber_compat
from src.drl.finsaber_native_runner import (
    FinsaberNativeConfig,
    load_default_finrl_indicators as load_native_finrl_indicators,
    format_raw_data_for_fe as format_native_raw_for_fe,
    preprocess_data as preprocess_native_data,
    train_finsaber_native,
)
from src.finsaber_native.state_contract import (
    build_finsaber_native_state_contract,
    collect_finsaber_native_reference_states,
    select_native_validation_states,
)
from src.lesr.prompt_templates import (
    build_initial_prompt,
    build_cot_prompt,
    build_next_iteration_prompt,
    build_system_prompt,
)
from src.lesr.revision_candidates import generate_candidate_codes
from src.lesr.llm_sampler import extract_lesr_code
from src.llm.deepseek_client import from_env as deepseek_from_env
from src.llm.finagent_stub import FinAgentStub, FinAgentStubConfig
from src.utils.code_loader import load_functions_from_code
from src.utils.paths import ensure_dir, repo_root
from src.utils.hash import sha256_file


@dataclass
class DemoConfig:
    data_source: str
    finsaber_price_path: str | None
    start_date: str
    end_date: str | None
    assets: List[str]
    indicators: List[str]
    global_features: List[str]
    task_description: str
    days: int
    seed: int
    initial_cash: float
    max_trade: int
    fee_rate: float
    intrinsic_w: float
    n_small: int
    n_full: int
    seeds: List[int]
    algorithm: str
    td3: dict
    llm: dict
    groups: List[str] | None = None
    use_finagent_signal: bool = False
    finagent_weight: float = 0.0
    eval_algorithms: List[str] | None = None
    sb3: dict | None = None
    eval_protocol: str = "temporal_split"
    data_split: dict | None = None
    split_date_filters: dict | None = None
    warmup_ratio: float | None = 0.15
    intrinsic_scale_mode: str = "bounded_100"
    intrinsic_timing: str = "pre_action_state"
    bootstrap: dict | None = None
    walk_forward: dict | None = None
    fixed_candidate_path: str | None = None
    window_setup: str = "custom"
    benchmark_range: dict | None = None
    prior_years_max: int | None = None
    universe: dict | None = None
    execution: dict | None = None
    evaluation: dict | None = None
    intrinsic_postprocess: dict | None = None
    intrinsic_w_schedule: List[float] | None = None
    diagnostics: dict | None = None
    experiment: dict | None = None
    intrinsic_w_tuning: dict | None = None
    algo_tuning: dict | None = None


def _prices_for_day(df: pd.DataFrame, date: str) -> Dict[str, float]:
    day = df[df["date"] == date]
    return {row.asset: float(row.close) for row in day.itertuples()}


def _split_df_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_series = pd.to_datetime(df["date"])
    return df[(date_series >= start) & (date_series <= end)].copy()


def _load_filter_dates_from_path(path_like: str) -> set[str]:
    path = Path(path_like)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"split_date_filters path not found: {path}")

    suffix = path.suffix.lower()
    values: list[object]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("dates", [])
        if not isinstance(payload, list):
            raise ValueError(f"split_date_filters json must contain a list of dates: {path}")
        values = payload
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise ValueError(f"split_date_filters csv requires a 'date' column: {path}")
        values = frame["date"].dropna().tolist()
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out: set[str] = set()
    for value in values:
        try:
            out.add(str(pd.to_datetime(value).date()))
        except Exception:
            continue
    return out


def _normalize_split_date_filter(raw_filter: object) -> set[str] | None:
    if raw_filter in (None, "", [], {}, ()):
        return None
    if isinstance(raw_filter, dict):
        if "include_dates" in raw_filter:
            return _normalize_split_date_filter(raw_filter.get("include_dates"))
        if "dates" in raw_filter:
            return _normalize_split_date_filter(raw_filter.get("dates"))
        if "path" in raw_filter:
            return _load_filter_dates_from_path(str(raw_filter.get("path", "")))
        if "include_dates_path" in raw_filter:
            return _load_filter_dates_from_path(str(raw_filter.get("include_dates_path", "")))
        return None
    if isinstance(raw_filter, str):
        return _load_filter_dates_from_path(raw_filter)
    if isinstance(raw_filter, (list, tuple, set)):
        out: set[str] = set()
        for value in raw_filter:
            try:
                out.add(str(pd.to_datetime(value).date()))
            except Exception:
                continue
        return out
    return None


def _apply_split_date_filter(df: pd.DataFrame, raw_filter: object) -> tuple[pd.DataFrame, dict]:
    allowed_dates = _normalize_split_date_filter(raw_filter)
    requested_dates = sorted(df["date"].unique().tolist()) if not df.empty else []
    requested_date_count = int(len(requested_dates))
    if not allowed_dates:
        return df.copy(), {
            "filtered": False,
            "requested_date_count": requested_date_count,
            "allowed_date_count": requested_date_count,
            "effective_date_count": requested_date_count,
            "dropped_date_count": 0,
        }

    filtered = df[df["date"].isin(allowed_dates)].copy()
    effective_date_count = int(filtered["date"].nunique()) if not filtered.empty else 0
    return filtered, {
        "filtered": True,
        "requested_date_count": requested_date_count,
        "allowed_date_count": int(len(allowed_dates)),
        "effective_date_count": effective_date_count,
        "dropped_date_count": int(max(0, requested_date_count - effective_date_count)),
    }


def _apply_split_date_filters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date_filters: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    filters = split_date_filters or {}
    if not isinstance(filters, dict):
        filters = {}
    train_out, train_summary = _apply_split_date_filter(train_df, filters.get("train"))
    val_out, val_summary = _apply_split_date_filter(val_df, filters.get("val"))
    test_out, test_summary = _apply_split_date_filter(test_df, filters.get("test"))
    return train_out, val_out, test_out, {
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }


def _split_meta_block_from_df(df: pd.DataFrame, original_block: dict | None) -> dict:
    block = dict(original_block or {})
    block["requested_start"] = str(block.get("start", ""))
    block["requested_end"] = str(block.get("end", ""))
    block["requested_days"] = int(block.get("days", 0) or 0)
    unique_dates = sorted(df["date"].unique().tolist()) if not df.empty else []
    if unique_dates:
        block["start"] = unique_dates[0]
        block["end"] = unique_dates[-1]
    block["days"] = int(len(unique_dates))
    return block


def _build_temporal_splits(df: pd.DataFrame, cfg: DemoConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    protocol = (cfg.eval_protocol or "temporal_split").lower()
    if protocol == "same_window":
        unique_dates = sorted(df["date"].unique().tolist())
        split_meta = {
            "protocol": "same_window",
            "train": {"start": unique_dates[0], "end": unique_dates[-1], "days": len(unique_dates)},
            "val": {"start": unique_dates[0], "end": unique_dates[-1], "days": len(unique_dates)},
            "test": {"start": unique_dates[0], "end": unique_dates[-1], "days": len(unique_dates)},
            "anti_leak_passed": True,
        }
        return df.copy(), df.copy(), df.copy(), split_meta

    unique_dates = sorted(df["date"].unique().tolist())
    if len(unique_dates) < 9:
        raise ValueError("Need at least 9 trading days for temporal split.")

    if cfg.data_split:
        train_cfg = cfg.data_split.get("train", {})
        val_cfg = cfg.data_split.get("val", {})
        test_cfg = cfg.data_split.get("test", {})
        for name, block in [("train", train_cfg), ("val", val_cfg), ("test", test_cfg)]:
            if "start" not in block or "end" not in block:
                raise ValueError(f"data_split.{name} requires start/end.")
        train_df = _split_df_by_date(df, train_cfg["start"], train_cfg["end"])
        val_df = _split_df_by_date(df, val_cfg["start"], val_cfg["end"])
        test_df = _split_df_by_date(df, test_cfg["start"], test_cfg["end"])
    else:
        n = len(unique_dates)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:]
        train_df = df[df["date"].isin(train_dates)].copy()
        val_df = df[df["date"].isin(val_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Temporal split produced empty train/val/test subset.")

    train_last = pd.to_datetime(train_df["date"]).max()
    val_first = pd.to_datetime(val_df["date"]).min()
    val_last = pd.to_datetime(val_df["date"]).max()
    test_first = pd.to_datetime(test_df["date"]).min()
    anti_leak_passed = bool(train_last < val_first and val_last < test_first)
    if not anti_leak_passed:
        raise ValueError("Temporal split overlap detected between train/val/test.")

    split_meta = {
        "protocol": "temporal_split",
        "train": {
            "start": sorted(train_df["date"].unique())[0],
            "end": sorted(train_df["date"].unique())[-1],
            "days": int(train_df["date"].nunique()),
        },
        "val": {
            "start": sorted(val_df["date"].unique())[0],
            "end": sorted(val_df["date"].unique())[-1],
            "days": int(val_df["date"].nunique()),
        },
        "test": {
            "start": sorted(test_df["date"].unique())[0],
            "end": sorted(test_df["date"].unique())[-1],
            "days": int(test_df["date"].nunique()),
        },
        "anti_leak_passed": anti_leak_passed,
    }
    return train_df, val_df, test_df, split_meta


def _effective_steps(max_steps: int, available_days: int) -> int:
    return int(max(1, min(max_steps, available_days)))


def _resolve_td3_cfg(
    base_cfg: TD3Config,
    max_steps: int,
    warmup_ratio: float | None,
    evaluation_cfg: dict | None = None,
) -> TD3Config:
    cfg = replace(base_cfg)
    if warmup_ratio is not None:
        ratio = float(np.clip(warmup_ratio, 0.0, 0.9))
        cfg.start_timesteps = int(max(1, round(max_steps * ratio)))
    cfg.start_timesteps = int(min(max(0, cfg.start_timesteps), max(0, max_steps - 1)))
    eval_cfg = evaluation_cfg or {}
    if str(eval_cfg.get("eval_freq_mode", "absolute")).lower() == "relative":
        eval_points = int(max(1, eval_cfg.get("eval_points_min", 5)))
        cfg.eval_freq = int(max(1, round(max_steps / eval_points)))
    cfg.eval_freq = int(max(1, cfg.eval_freq))
    return cfg


def _generate_windows_from_setup(df: pd.DataFrame, cfg: DemoConfig) -> list[dict]:
    setup = str(cfg.window_setup or "custom").lower()
    if setup not in {"selected_4", "composite"}:
        return []

    defaults = {
        "selected_4": {"test_years": 2, "prior_years_max": 3},
        "composite": {"test_years": 1, "prior_years_max": 2},
    }[setup]
    test_years = int(defaults["test_years"])
    prior_years_max = int(cfg.prior_years_max or defaults["prior_years_max"])

    dates = pd.to_datetime(df["date"])
    data_start = dates.min()
    data_end = dates.max()
    bench_cfg = cfg.benchmark_range or {}
    bench_start = pd.to_datetime(bench_cfg.get("start", str(data_start.date())))
    bench_end = pd.to_datetime(bench_cfg.get("end", str(data_end.date())))
    bench_start = max(bench_start, data_start)
    bench_end = min(bench_end, data_end)

    windows: list[dict] = []
    test_start = pd.Timestamp(year=bench_start.year, month=1, day=1)
    if test_start < bench_start:
        test_start = test_start + pd.DateOffset(years=1)
    while True:
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_end > bench_end:
            break

        val_start = test_start - pd.DateOffset(years=1)
        val_end = test_start - pd.DateOffset(days=1)
        train_start = test_start - pd.DateOffset(years=prior_years_max)
        train_end = val_start - pd.DateOffset(days=1)

        if train_end >= train_start and train_start >= data_start and test_end <= data_end:
            windows.append(
                {
                    "train": {"start": str(train_start.date()), "end": str(train_end.date())},
                    "val": {"start": str(val_start.date()), "end": str(val_end.date())},
                    "test": {"start": str(test_start.date()), "end": str(test_end.date())},
                }
            )

        test_start = test_start + pd.DateOffset(years=1)

    return windows


def _rank_assets_by_momentum_factor(
    hist: pd.DataFrame,
    candidates: set[str],
    n_assets: int,
    momentum_days: int,
    skip_days: int,
) -> tuple[list[str], dict]:
    if momentum_days <= 0:
        momentum_days = 100
    if skip_days < 0:
        skip_days = 0
    if momentum_days <= skip_days:
        momentum_days = skip_days + 1

    score_rows: list[dict] = []
    for asset in sorted(candidates):
        series = (
            hist[hist["asset"] == asset]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")["close"]
            .astype(float)
            .reset_index(drop=True)
        )
        required_len = max(momentum_days, skip_days) + 1
        if series.shape[0] < required_len:
            continue
        price_skip = float(series.iloc[-1 - skip_days])
        price_momentum = float(series.iloc[-1 - momentum_days])
        if not np.isfinite(price_skip) or not np.isfinite(price_momentum) or abs(price_momentum) <= 1e-12:
            continue
        score_rows.append(
            {
                "asset": asset,
                "momentum_score": float(price_skip / price_momentum - 1.0),
                "history_days": int(series.shape[0]),
            }
        )

    score_rows.sort(key=lambda row: (float(row["momentum_score"]), row["asset"]), reverse=True)
    selected = [row["asset"] for row in score_rows[:n_assets]]
    return selected, {
        "selection_strategy": "momentum_factor",
        "momentum_days": int(momentum_days),
        "skip_days": int(skip_days),
        "scored_asset_count": int(len(score_rows)),
        "top_scores": score_rows[: max(0, min(10, len(score_rows)))],
    }


def _select_assets_for_window(df: pd.DataFrame, cfg: DemoConfig, split_meta: dict) -> tuple[list[str], dict]:
    universe_cfg = cfg.universe or {}
    mode = str(universe_cfg.get("mode", "fixed")).lower()
    if mode != "historical_dynamic":
        fixed_assets = list(cfg.assets)
        if not fixed_assets:
            fixed_assets = sorted(df["asset"].dropna().unique().tolist())[: int(universe_cfg.get("n_assets", 3))]
        return fixed_assets, {
            "mode": "fixed",
            "include_delisted": False,
            "candidate_count": len(fixed_assets),
            "selected_assets": fixed_assets,
        }

    n_assets = int(universe_cfg.get("n_assets", max(1, len(cfg.assets) or 3)))
    include_delisted = bool(universe_cfg.get("include_delisted", True))
    anchor = pd.to_datetime(split_meta["train"]["start"])
    test_end = pd.to_datetime(split_meta["test"]["end"])

    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])
    if cfg.assets:
        pool = set(cfg.assets)
        if bool(universe_cfg.get("restrict_to_config_assets", False)):
            df_work = df_work[df_work["asset"].isin(pool)]

    grouped = (
        df_work.groupby("asset")
        .agg(min_date=("date", "min"), max_date=("date", "max"))
        .reset_index()
    )
    eligible = grouped[grouped["min_date"] <= anchor]
    eligible = eligible[eligible["max_date"] >= anchor]
    if not include_delisted:
        eligible = eligible[eligible["max_date"] >= test_end]
    candidates = set(eligible["asset"].tolist())
    full_window_candidates = set(eligible[eligible["max_date"] >= test_end]["asset"].tolist())

    hist = df_work[df_work["date"] <= anchor]
    hist = hist[hist["asset"].isin(candidates)]
    selection_strategy = str(universe_cfg.get("selection_strategy", "volume")).lower()
    selector_snapshot: dict = {"selection_strategy": selection_strategy}
    require_split_coverage = bool(
        universe_cfg.get(
            "require_split_coverage",
            selection_strategy in {"momentum", "momentum_factor"},
        )
    )
    min_split_coverage_ratio = float(universe_cfg.get("min_split_coverage_ratio", 1.0))
    min_split_coverage_ratio = float(np.clip(min_split_coverage_ratio, 0.0, 1.0))
    split_coverage_candidates = set(candidates)
    split_coverage_summary: dict[str, dict] = {}
    if require_split_coverage and candidates:
        for split_name in ["train", "val", "test"]:
            split_start = pd.to_datetime(split_meta[split_name]["start"])
            split_end = pd.to_datetime(split_meta[split_name]["end"])
            split_mask = (df_work["date"] >= split_start) & (df_work["date"] <= split_end)
            required_days = int(df_work.loc[split_mask, "date"].nunique())
            split_part = df_work.loc[split_mask & df_work["asset"].isin(candidates), ["asset", "date"]]
            split_counts = split_part.groupby("asset")["date"].nunique()
            qualified = {
                asset
                for asset in candidates
                if required_days <= 0
                or (float(split_counts.get(asset, 0)) / float(required_days)) >= min_split_coverage_ratio
            }
            split_coverage_candidates &= qualified
            split_coverage_summary[split_name] = {
                "required_days": required_days,
                "qualified_candidate_count": int(len(qualified)),
            }
    if selection_strategy in {"momentum", "momentum_factor"}:
        require_full_window_coverage = bool(universe_cfg.get("require_full_window_coverage", True))
        momentum_days = int(universe_cfg.get("momentum_days", universe_cfg.get("momentum_lookback_days", 100)))
        skip_days = int(universe_cfg.get("skip_days", universe_cfg.get("momentum_skip_days", 21)))
        rank_candidates = set(candidates)
        if require_full_window_coverage and full_window_candidates:
            rank_candidates &= full_window_candidates
        if require_split_coverage and split_coverage_candidates:
            rank_candidates &= split_coverage_candidates
        if not rank_candidates and require_split_coverage and split_coverage_candidates:
            rank_candidates = set(split_coverage_candidates)
        if not rank_candidates and require_full_window_coverage and full_window_candidates:
            rank_candidates = set(full_window_candidates)
        if not rank_candidates:
            rank_candidates = set(candidates)
        selected, selector_snapshot = _rank_assets_by_momentum_factor(
            hist=hist,
            candidates=rank_candidates,
            n_assets=n_assets,
            momentum_days=momentum_days,
            skip_days=skip_days,
        )
        fill_candidates = set(rank_candidates)
        selector_snapshot["require_full_window_coverage"] = bool(require_full_window_coverage)
        selector_snapshot["full_window_candidate_count"] = int(len(full_window_candidates))
        selector_snapshot["require_split_coverage"] = bool(require_split_coverage)
        selector_snapshot["min_split_coverage_ratio"] = float(min_split_coverage_ratio)
        selector_snapshot["split_coverage_candidate_count"] = int(len(split_coverage_candidates))
        selector_snapshot["split_coverage_summary"] = split_coverage_summary
    else:
        lookback_days = int(universe_cfg.get("volume_lookback_days", 60))
        lookback_days = max(1, lookback_days)
        lookback_start = anchor - pd.DateOffset(days=lookback_days)
        hist_vol = hist[hist["date"] >= lookback_start]
        vol_rank = (
            hist_vol.groupby("asset")["volume"].mean().sort_values(ascending=False).reset_index()
            if not hist_vol.empty
            else pd.DataFrame({"asset": sorted(candidates), "volume": 0.0})
        )
        selected = vol_rank["asset"].tolist()[:n_assets]
        selector_snapshot = {
            "selection_strategy": "volume",
            "volume_lookback_days": int(lookback_days),
            "scored_asset_count": int(vol_rank.shape[0]),
            "top_scores": [
                {
                    "asset": str(row.asset),
                    "volume_score": float(row.volume),
                }
                for row in vol_rank.head(max(0, min(10, len(vol_rank)))).itertuples()
            ],
        }
        fill_candidates = set(candidates)
        if require_split_coverage and split_coverage_candidates:
            fill_candidates &= split_coverage_candidates
        if not fill_candidates:
            fill_candidates = set(candidates)
        selector_snapshot["require_split_coverage"] = bool(require_split_coverage)
        selector_snapshot["min_split_coverage_ratio"] = float(min_split_coverage_ratio)
        selector_snapshot["split_coverage_candidate_count"] = int(len(split_coverage_candidates))
        selector_snapshot["split_coverage_summary"] = split_coverage_summary
    if len(selected) < n_assets:
        for asset in sorted(fill_candidates):
            if asset not in selected:
                selected.append(asset)
            if len(selected) >= n_assets:
                break
    if not selected and cfg.assets:
        selected = list(cfg.assets)[:n_assets]
    if not selected:
        raise ValueError("No tradable assets found for dynamic universe selection.")

    snapshot = {
        "mode": "historical_dynamic",
        "include_delisted": include_delisted,
        "n_assets": n_assets,
        "anchor_date": str(anchor.date()),
        "candidate_count": int(len(candidates)),
        "selected_assets": selected,
        "selector": selector_snapshot,
    }
    return selected, snapshot


def _resolve_decision_rule(cfg: DemoConfig) -> str:
    execution_cfg = cfg.execution or {}
    decision_price = str(execution_cfg.get("decision_price", "close_t")).lower()
    fill_price = str(execution_cfg.get("fill_price", "open_t1")).lower()
    if decision_price == "close_t" and fill_price == "open_t1":
        return "close_t_to_open_t1"
    return "close_t_to_close_t"


def _resolve_action_quantization_mode(cfg: DemoConfig) -> str:
    execution_cfg = cfg.execution or {}
    mode = str(execution_cfg.get("action_quantization", "integer")).lower()
    if mode in {"integer", "integer_shares", "int"}:
        return "integer"
    if mode in {"none", "continuous", "raw"}:
        return "none"
    return "integer"


def _resolve_drl_backend(cfg: DemoConfig) -> str:
    execution_cfg = cfg.execution or {}
    backend = str(execution_cfg.get("drl_backend", "current")).strip().lower()
    if backend in {"finsaber_compat", "finsaber-compat", "compat", "finsaber"}:
        return "finsaber_compat"
    if backend in {"finsaber_native", "finsaber-native", "native", "finsaber_original"}:
        return "finsaber_native"
    return "current"


def _resolve_discrete_action_levels(cfg: DemoConfig, algo: str | None = None) -> int:
    execution_cfg = cfg.execution or {}
    levels = execution_cfg.get("discrete_action_levels", 3)
    algo_key = str(algo or "").strip().lower()
    if algo_key:
        levels = (execution_cfg.get("discrete_action_levels_by_algo", {}) or {}).get(algo_key, levels)
    try:
        levels = int(levels)
    except Exception:
        levels = 3
    levels = max(3, levels)
    if levels % 2 == 0:
        raise ValueError(f"discrete_action_levels must be odd, got {levels}")
    return levels


def _resolve_action_bound_penalty_cfg(cfg: DemoConfig, algo: str | None = None) -> dict:
    execution_cfg = cfg.execution or {}
    penalty_cfg = dict(execution_cfg.get("action_bound_penalty", {}) or {})
    algo_key = str(algo or "").strip().lower()
    if algo_key in {"sac", "td3"}:
        penalty_cfg.update(dict(execution_cfg.get("continuous_action_bound_penalty", {}) or {}))
    penalty_cfg.update(dict((execution_cfg.get("action_bound_penalty_by_algo", {}) or {}).get(algo_key, {}) or {}))
    coef = float(penalty_cfg.get("coef", 0.0))
    threshold = float(penalty_cfg.get("threshold", 0.95))
    power = float(penalty_cfg.get("power", 2.0))
    if coef < 0:
        coef = 0.0
    threshold = float(np.clip(threshold, 0.0, 0.999999))
    if power < 1.0:
        power = 1.0
    return {
        "coef": coef,
        "threshold": threshold,
        "power": power,
        "enabled": bool(coef > 0),
    }


def _resolve_action_bound_penalty_reference_bound(cfg: DemoConfig, algo: str | None = None) -> float:
    algo_key = str(algo or "").strip().lower()
    if algo_key == "td3":
        td3_cfg = cfg.td3 or {}
        actor_max_action = td3_cfg.get("actor_max_action", cfg.max_trade)
        try:
            return max(1.0, float(actor_max_action))
        except Exception:
            return max(1.0, float(cfg.max_trade))
    return max(1.0, float(cfg.max_trade))


def _env_cfg_with_algo_penalty(cfg: DemoConfig, env_cfg: EnvConfig, algo: str | None) -> EnvConfig:
    penalty_cfg = _resolve_action_bound_penalty_cfg(cfg, algo)
    return replace(
        env_cfg,
        discrete_action_levels=_resolve_discrete_action_levels(cfg, algo),
        action_bound_penalty_coef=float(penalty_cfg["coef"]),
        action_bound_penalty_threshold=float(penalty_cfg["threshold"]),
        action_bound_penalty_power=float(penalty_cfg["power"]),
        action_bound_penalty_reference_bound=float(_resolve_action_bound_penalty_reference_bound(cfg, algo)),
    )


def _resolve_intrinsic_w_tuning_cfg(cfg: DemoConfig) -> dict:
    tuning_cfg = cfg.intrinsic_w_tuning or {}
    probe_seed_count = int(tuning_cfg.get("probe_seed_count", min(3, max(1, len(cfg.seeds or [])))))
    if probe_seed_count <= 0:
        probe_seed_count = 1
    tie_tolerance = float(tuning_cfg.get("tie_tolerance", 0.01))
    if tie_tolerance < 0:
        tie_tolerance = 0.0
    return {
        "probe_seed_count": probe_seed_count,
        "tie_tolerance": tie_tolerance,
        "prefer_smallest_w": bool(tuning_cfg.get("prefer_smallest_w", True)),
    }


def _filter_assets_align_dates(df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    out = df[df["asset"].isin(assets)].copy()
    if out.empty:
        return out
    counts = out.groupby("date")["asset"].nunique()
    valid_dates = counts[counts == len(assets)].index
    out = out[out["date"].isin(valid_dates)].copy()
    return out


def _volume_indices(schema: StateSchema) -> List[int]:
    base = len(schema.global_features)
    stride = 6 + len(schema.indicators)
    return [int(base + i * stride + 4) for i in range(len(schema.assets))]


def _collect_reference_states(train_df: pd.DataFrame, schema: StateSchema, initial_cash: float) -> np.ndarray:
    dates = sorted(train_df["date"].unique().tolist())
    holdings = {asset: 0.0 for asset in schema.assets}
    rows = []
    for dt in dates:
        day_df = train_df[train_df["date"] == dt]
        if day_df.empty:
            continue
        try:
            state = schema.build_state(day_df, holdings, initial_cash)
        except Exception:
            continue
        rows.append(np.asarray(state, dtype=np.float32))
    if not rows:
        rows = [np.zeros(schema.dim(), dtype=np.float32)]
    return np.stack(rows, axis=0)


def _scale_intrinsic_value(value: float, mode: str) -> float:
    mode = (mode or "raw").lower()
    if mode == "bounded_100":
        return float(np.clip(value, -100.0, 100.0))
    if mode == "normalized":
        return float(np.tanh(value) * 100.0)
    return float(value)


def _resolve_bootstrap_cfg(bootstrap_cfg: dict | None) -> dict:
    cfg = bootstrap_cfg or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "n_resamples": int(cfg.get("n_resamples", 5000)),
        "alpha": float(cfg.get("alpha", 0.05)),
        "random_seed": int(cfg.get("random_seed", 42)),
    }


def _stable_seed(base_seed: int, key: str) -> int:
    checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return int((base_seed + checksum) % (2**32 - 1))


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "isoformat") and not isinstance(obj, str):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_payload(payload) -> str:
    try:
        serialized = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False)
    except Exception:
        serialized = repr(payload)
    return _sha256_text(serialized)


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


def _resolve_experiment_cfg(cfg: DemoConfig) -> dict:
    exp = cfg.experiment or {}
    return {
        "phase": str(exp.get("phase", "")).upper(),
        "claim_id": str(exp.get("claim_id", "")),
        "hypothesis_id": str(exp.get("hypothesis_id", "")),
        "frozen": bool(exp.get("frozen", False)),
    }


def _is_confirmatory(exp_cfg: dict) -> bool:
    phase = str(exp_cfg.get("phase", "")).upper()
    return bool(exp_cfg.get("frozen", False) or phase.startswith("C"))


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _resolve_algo_tuning_cfg(cfg: dict | None) -> dict:
    raw = cfg or {}
    shared = raw.get("shared", {}) if isinstance(raw, dict) else {}
    if not isinstance(shared, dict):
        shared = {}
    out: dict = {}
    for algo in ["a2c", "ppo", "sac", "td3"]:
        merged = {}
        merged.update(shared)
        specific = raw.get(algo, {}) if isinstance(raw, dict) else {}
        if isinstance(specific, dict):
            merged.update(specific)
        out[algo] = merged
    return out


def _split_sb3_tuning(tuning: dict | None) -> tuple[dict, dict]:
    tuning = tuning or {}
    cfg_keys = {
        "total_timesteps",
        "learning_rate",
        "gamma",
        "batch_size",
        "ent_coef",
        "eval_episodes",
    }
    cfg_overrides = {}
    model_kwargs = {}
    for key, value in tuning.items():
        if key in cfg_keys:
            cfg_overrides[key] = value
        else:
            model_kwargs[key] = value
    return cfg_overrides, model_kwargs


def _apply_sb3_cfg_overrides(base_cfg: SB3Config, overrides: dict | None) -> SB3Config:
    payload = {
        "total_timesteps": base_cfg.total_timesteps,
        "learning_rate": base_cfg.learning_rate,
        "gamma": base_cfg.gamma,
        "batch_size": base_cfg.batch_size,
        "ent_coef": base_cfg.ent_coef,
        "eval_episodes": base_cfg.eval_episodes,
    }
    for key, value in (overrides or {}).items():
        if key in payload and value is not None:
            payload[key] = value
    return SB3Config(**payload)


def _split_td3_tuning(tuning: dict | None) -> tuple[dict, dict]:
    tuning = tuning or {}
    known_keys = set(TD3Config.__dataclass_fields__.keys()) - {"max_action"}
    known_overrides = {}
    ignored = {}
    for key, value in tuning.items():
        if key in known_keys:
            known_overrides[key] = value
        else:
            ignored[key] = value
    return known_overrides, ignored


def _resolve_td3_backend(td3_cfg: dict | None) -> str:
    backend = str((td3_cfg or {}).get("backend", "sb3")).lower()
    if backend in {"legacy", "custom"}:
        return "legacy"
    return "sb3"


def _td3_cfg_to_sb3_cfg(base_sb3_cfg: SB3Config, td3_cfg: TD3Config, total_timesteps: int) -> SB3Config:
    return replace(
        base_sb3_cfg,
        total_timesteps=int(total_timesteps),
        gamma=float(td3_cfg.discount),
        batch_size=int(td3_cfg.batch_size),
    )


def _td3_cfg_to_sb3_kwargs(td3_cfg: TD3Config) -> dict:
    return {
        "start_timesteps": int(td3_cfg.start_timesteps),
        "expl_noise": float(td3_cfg.expl_noise),
        "tau": float(td3_cfg.tau),
        "policy_noise": float(td3_cfg.policy_noise),
        "noise_clip": float(td3_cfg.noise_clip),
        "policy_freq": int(td3_cfg.policy_freq),
        "hidden_dim": int(td3_cfg.hidden_dim),
        "actor_max_action": (
            float(td3_cfg.actor_max_action)
            if td3_cfg.actor_max_action is not None
            else None
        ),
    }


def _td3_policy_action_bound(td3_cfg: TD3Config, fallback: float) -> float:
    if td3_cfg.actor_max_action is not None:
        return float(td3_cfg.actor_max_action)
    return float(fallback)


def _build_candidate_fingerprint(best_name: str, candidate_code_map: dict[str, str]) -> dict:
    rows = []
    for name in sorted(candidate_code_map.keys()):
        rows.append(
            {
                "name": name,
                "sha256": _sha256_text(str(candidate_code_map[name])),
            }
        )
    best_sha = next((r["sha256"] for r in rows if r["name"] == best_name), "")
    return {
        "best_candidate": best_name,
        "best_candidate_sha256": best_sha,
        "candidate_count": int(len(rows)),
        "candidate_set_sha256": _hash_payload(rows),
    }


def _candidate_origin_from_name(name: str) -> str:
    text = str(name or "")
    if "fallback_" in text:
        return "fallback"
    if "fixed_candidate" in text:
        return "fixed"
    if "llm_it" in text:
        return "llm"
    return "static"


def _extract_declared_feature_groups(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    groups: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != "FEATURE_GROUPS":
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for item in node.value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        val = str(item.value).strip().lower()
                        if val:
                            groups.append(val)
    deduped = []
    for item in groups:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _infer_candidate_feature_groups(code: str) -> list[str]:
    low = str(code or "").lower()
    groups: list[str] = []
    if any(tok in low for tok in ["holding", "cash", "exposure", "concentration", "entropy", "rebal", "weight"]):
        groups.append("portfolio_memory")
    if any(
        tok in low
        for tok in [
            "vol",
            "drawdown",
            "risk",
            "stress",
            "trend_strength",
            "vol_ratio",
            "mu_ann",
            "vol_ann",
        ]
    ):
        groups.append("regime")
    if any(
        tok in low
        for tok in [
            "dispersion",
            "spread",
            "correlation",
            "corr",
            "breadth",
            "rank",
            "winner",
            "loser",
            "momentum",
        ]
    ):
        groups.append("dispersion")
    if any(tok in low for tok in ["ret_ema", "ret_sq_ema", "drawdown_20", "turnover_ema", "running-risk", "running_risk"]):
        groups.append("running_risk_state")
    declared = _extract_declared_feature_groups(code)
    for item in declared:
        if item not in groups:
            groups.append(item)
    return groups


def _extract_function_source_blocks(code: str) -> dict[str, str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    blocks: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"revise_state", "intrinsic_reward"}:
            segment = ast.get_source_segment(code, node)
            if segment:
                blocks[node.name] = segment.strip()
    return blocks


def _extract_candidate_module_prelude(code: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return ["import numpy as np"], []
    imports: list[str] = []
    assigns: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            snippet = ast.get_source_segment(code, node) or ast.unparse(node)
            snippet = str(snippet).strip()
            if snippet and snippet not in imports:
                imports.append(snippet)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            snippet = ast.get_source_segment(code, node) or ast.unparse(node)
            snippet = str(snippet).strip()
            if snippet and snippet not in assigns:
                assigns.append(snippet)
    if not imports:
        imports = ["import numpy as np"]
    return imports, assigns


def _extract_candidate_component_hashes(code: str) -> dict[str, str]:
    blocks = _extract_function_source_blocks(code)
    revise_src = blocks.get("revise_state", "")
    intrinsic_src = blocks.get("intrinsic_reward", "")
    return {
        "revise_hash": _sha256_text(revise_src) if revise_src else "",
        "intrinsic_hash": _sha256_text(intrinsic_src) if intrinsic_src else "",
    }


def _build_combined_candidate_code(state_code: str, intrinsic_code: str) -> str:
    state_blocks = _extract_function_source_blocks(state_code)
    intrinsic_blocks = _extract_function_source_blocks(intrinsic_code)
    revise_src = state_blocks.get("revise_state", "")
    intrinsic_src = intrinsic_blocks.get("intrinsic_reward", "")
    if not revise_src or not intrinsic_src:
        return ""
    imports_a, assigns_a = _extract_candidate_module_prelude(state_code)
    imports_b, assigns_b = _extract_candidate_module_prelude(intrinsic_code)
    imports = []
    assigns = []
    for item in imports_a + imports_b:
        if item not in imports:
            imports.append(item)
    for item in assigns_a + assigns_b:
        if item not in assigns:
            assigns.append(item)
    if "import numpy as np" not in imports:
        imports.insert(0, "import numpy as np")
    chunks = imports + assigns + [revise_src, intrinsic_src]
    return "\n\n".join([chunk for chunk in chunks if str(chunk).strip()]).strip() + "\n"


def _resolve_consensus_promotion_cfg(llm_cfg: dict | None) -> dict:
    cfg = (llm_cfg or {}).get("consensus_promotion", {}) if isinstance(llm_cfg, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    llm_mode = str((llm_cfg or {}).get("iteration_mode", "")) if isinstance(llm_cfg, dict) else ""
    default_support_min = 1 if llm_mode == "per_algorithm_branches" else 2
    legacy_probe_floor = float(cfg.get("probe_floor", -0.05))
    legacy_pick_floor = float(cfg.get("pick_floor", 0.0))
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "top_k_per_algo": int(max(1, cfg.get("top_k_per_algo", 3))),
        "max_state_cores": int(max(1, cfg.get("max_state_cores", 4))),
        "max_intrinsic_cores": int(max(1, cfg.get("max_intrinsic_cores", 4))),
        "joint_top_pairs": int(max(1, cfg.get("joint_top_pairs", 6))),
        "probe_floor": legacy_probe_floor,
        "support_min": int(max(1, cfg.get("support_min", default_support_min))),
        "pick_floor": legacy_pick_floor,
        "candidate_reject_floor": float(cfg.get("candidate_reject_floor", min(legacy_probe_floor, -0.20))),
        "state_pick_floor": float(cfg.get("state_pick_floor", min(legacy_pick_floor, -0.15))),
        "intrinsic_pick_floor": float(cfg.get("intrinsic_pick_floor", -0.10)),
        "joint_pick_floor": float(cfg.get("joint_pick_floor", min(legacy_pick_floor, -0.15))),
        "td3_intrinsic_probe_floor": float(cfg.get("td3_intrinsic_probe_floor", -0.05)),
        "intrinsic_nontrivial_floor": float(cfg.get("intrinsic_nontrivial_floor", 0.02)),
    }


def _native_small_budget_algo_kwargs(algo: str, algo_kwargs: dict | None, total_timesteps: int) -> dict:
    out = dict(algo_kwargs or {})
    algo_key = str(algo or "").strip().lower()
    if algo_key not in {"sac", "td3"}:
        return out
    steps = int(max(1, total_timesteps))
    buffer_size = int(min(int(out.get("buffer_size", 1_000_000)), max(4096, steps * 8)))
    learning_starts_cap = max(32, min(buffer_size // 8, max(steps // 4, 32)))
    batch_size_cap = max(32, min(buffer_size // 8, 256))
    out["buffer_size"] = buffer_size
    if "learning_starts" in out:
        out["learning_starts"] = int(min(int(out.get("learning_starts", learning_starts_cap)), learning_starts_cap))
    elif algo_key == "sac":
        out["learning_starts"] = int(learning_starts_cap)
    if "batch_size" in out:
        out["batch_size"] = int(min(int(out.get("batch_size", batch_size_cap)), batch_size_cap))
    return out


def _static_candidate_codes_for_backend(schema: StateSchema, drl_backend: str) -> List[Tuple[str, str]]:
    candidates = generate_candidate_codes(schema)
    if str(drl_backend or "").strip().lower() != "finsaber_native":
        return candidates
    if not candidates:
        return []
    identity_name, identity_code = candidates[0]
    return [(str(identity_name), str(identity_code))]


def _validate_candidate_code_for_backend(code: str, *, drl_backend: str) -> None:
    if str(drl_backend or "").strip().lower() != "finsaber_native":
        return
    text = str(code or "")
    native_disallowed_patterns = [
        (r"\bOPEN_IDXS\b", "native_generic_open_indices_disallowed"),
        (r"\bHIGH_IDXS\b", "native_generic_high_indices_disallowed"),
        (r"\bLOW_IDXS\b", "native_generic_low_indices_disallowed"),
        (r"\bVOLUME_IDXS\b", "native_generic_volume_indices_disallowed"),
        (r"field_offset\s*=\s*\{[^}]*['\"]open['\"]", "native_generic_open_field_map_disallowed"),
        (r"field_offset\s*=\s*\{[^}]*['\"]high['\"]", "native_generic_high_field_map_disallowed"),
        (r"field_offset\s*=\s*\{[^}]*['\"]low['\"]", "native_generic_low_field_map_disallowed"),
        (r"field_offset\s*=\s*\{[^}]*['\"]volume['\"]", "native_generic_volume_field_map_disallowed"),
    ]
    for pattern, error_key in native_disallowed_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            raise ValueError(error_key)


def _build_wf_candidate_fingerprint(root: Path, window_infos: List[dict]) -> dict:
    rows = []
    for item in window_infos:
        sub_manifest_path = root / item.get("run_manifest", "")
        if not sub_manifest_path.exists():
            continue
        try:
            sub_manifest = json.loads(sub_manifest_path.read_text())
        except Exception:
            continue
        rows.append(
            {
                "window_index": int(item.get("window_index", 0)),
                "window_name": str(item.get("window_name", "")),
                "candidate_fingerprint": _json_safe(sub_manifest.get("candidate_fingerprint", {})),
            }
        )
    return {
        "mode": "walk_forward",
        "window_count": int(len(rows)),
        "candidate_set_sha256": _hash_payload(rows),
        "windows": rows,
    }


def _build_cross_window_distillation(root: Path, window_infos: List[dict]) -> dict:
    algo_payload: dict[str, dict] = {}
    official_payload: dict[str, Counter] = {
        "state_core_family_frequency": Counter(),
        "state_core_design_mode_frequency": Counter(),
        "intrinsic_core_family_frequency": Counter(),
        "intrinsic_core_design_mode_frequency": Counter(),
        "joint_pair_family_frequency": Counter(),
        "joint_pair_design_mode_frequency": Counter(),
    }
    official_windows: list[dict] = []
    for item in window_infos:
        sub_manifest_path = root / item.get("run_manifest", "")
        sub_table_path = root / item.get("metrics_table", "")
        if not sub_manifest_path.exists() or not sub_table_path.exists():
            continue
        try:
            sub_manifest = json.loads(sub_manifest_path.read_text())
            table = pd.read_csv(sub_table_path)
        except Exception:
            continue
        best_candidate_by_algo = sub_manifest.get("best_candidate_by_algo", {}) or {}
        candidate_fingerprint_by_algo = sub_manifest.get("candidate_fingerprint_by_algo", {}) or {}
        official_shared_cores = sub_manifest.get("official_shared_cores", {}) or {}
        iter_trace_path = sub_manifest_path.parent / "llm_iter_trace.json"
        iter_trace = []
        if iter_trace_path.exists():
            try:
                iter_trace = json.loads(iter_trace_path.read_text())
            except Exception:
                iter_trace = []

        candidate_meta = {}
        for row in iter_trace or []:
            algo_name = str(row.get("algorithm", ""))
            for cand in row.get("candidates", []) or []:
                if not isinstance(cand, dict):
                    continue
                candidate_meta[(algo_name, str(cand.get("name", "")))] = {
                    "family": str(cand.get("family", "")),
                    "design_mode": str(cand.get("design_mode", "")),
                }

        for algo, algo_best_name in best_candidate_by_algo.items():
            algo_rows = table[table["algorithm"] == algo].copy()
            if algo_rows.empty:
                continue
            row_g0 = algo_rows[algo_rows["group"] == "G0_baseline"]
            if row_g0.empty:
                continue
            g0_sharpe = float(row_g0.iloc[0]["Sharpe_mean"])
            g0_score = float(row_g0.iloc[0]["Sharpe_mean"]) + float(row_g0.iloc[0]["CR_mean"])
            payload = algo_payload.setdefault(
                str(algo),
                {
                    "positive_window_count": {"G1": 0, "G2": 0, "G3": 0},
                    "delta_sharpe": {"G1": [], "G2": [], "G3": []},
                    "delta_score": {"G1": [], "G2": [], "G3": []},
                    "best_candidate_family_frequency": Counter(),
                    "best_candidate_design_mode_frequency": Counter(),
                    "best_candidate_sha_frequency": Counter(),
                    "windows": [],
                },
            )
            meta = candidate_meta.get((str(algo), str(algo_best_name)), {})
            family = str(meta.get("family", ""))
            design_mode = str(meta.get("design_mode", ""))
            best_sha = str((candidate_fingerprint_by_algo.get(algo, {}) or {}).get("best_candidate_sha256", ""))
            if family:
                payload["best_candidate_family_frequency"][family] += 1
            if design_mode:
                payload["best_candidate_design_mode_frequency"][design_mode] += 1
            if best_sha:
                payload["best_candidate_sha_frequency"][best_sha] += 1

            window_row = {
                "window_index": int(item.get("window_index", 0)),
                "window_name": str(item.get("window_name", "")),
                "best_candidate": str(algo_best_name),
                "best_candidate_family": family,
                "best_candidate_design_mode": design_mode,
                "best_candidate_sha256": best_sha,
                "deltas": {},
            }
            for group_name, short_name in [
                ("G1_revise_only", "G1"),
                ("G2_intrinsic_only", "G2"),
                ("G3_revise_intrinsic", "G3"),
            ]:
                group_row = algo_rows[algo_rows["group"] == group_name]
                if group_row.empty:
                    delta_sharpe = 0.0
                    delta_score = 0.0
                else:
                    delta_sharpe = float(group_row.iloc[0]["Sharpe_mean"]) - g0_sharpe
                    delta_score = (
                        float(group_row.iloc[0]["Sharpe_mean"]) + float(group_row.iloc[0]["CR_mean"]) - g0_score
                    )
                payload["delta_sharpe"][short_name].append(delta_sharpe)
                payload["delta_score"][short_name].append(delta_score)
                if delta_sharpe > 0:
                    payload["positive_window_count"][short_name] += 1
                window_row["deltas"][short_name] = {
                    "delta_sharpe": delta_sharpe,
                    "delta_score": delta_score,
                }
            payload["windows"].append(window_row)

        if official_shared_cores:
            window_entry = {
                "window_index": int(item.get("window_index", 0)),
                "window_name": str(item.get("window_name", "")),
            }
            for core_name in ["state_core", "intrinsic_core", "joint_pair"]:
                core = official_shared_cores.get(core_name, {}) or {}
                family = str(core.get("family", ""))
                design_mode = str(core.get("design_mode", ""))
                if family:
                    official_payload[f"{core_name}_family_frequency"][family] += 1
                if design_mode:
                    official_payload[f"{core_name}_design_mode_frequency"][design_mode] += 1
                window_entry[core_name] = {
                    "name": str(core.get("name", "")),
                    "family": family,
                    "design_mode": design_mode,
                    "low_confidence": bool((official_shared_cores.get("low_confidence", {}) or {}).get(core_name, False)),
                }
            official_windows.append(window_entry)

    summary_algos = {}
    for algo, payload in algo_payload.items():
        summary_algos[algo] = {
            "positive_window_count": payload["positive_window_count"],
            "delta_sharpe": {
                g: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "values": [float(v) for v in vals],
                }
                for g, vals in payload["delta_sharpe"].items()
            },
            "delta_score": {
                g: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "values": [float(v) for v in vals],
                }
                for g, vals in payload["delta_score"].items()
            },
            "best_candidate_family_frequency": dict(payload["best_candidate_family_frequency"]),
            "best_candidate_design_mode_frequency": dict(payload["best_candidate_design_mode_frequency"]),
            "candidate_fingerprint_recurrence": {
                "unique_best_candidate_sha_count": int(len(payload["best_candidate_sha_frequency"])),
                "best_candidate_sha_frequency": dict(payload["best_candidate_sha_frequency"]),
            },
            "windows": payload["windows"],
        }
    return {
        "mode": "post_hoc_distillation",
        "selection_note": "Post-hoc cross-window stability summary only; not used as an in-window LESR selection gate.",
        "window_count": int(len(window_infos)),
        "algorithms": summary_algos,
        "official_shared_cores": {
            **{key: dict(counter) for key, counter in official_payload.items()},
            "windows": official_windows,
        },
    }


def _rollout(
    df: pd.DataFrame,
    schema: StateSchema,
    env_cfg: EnvConfig,
    policy: HeuristicPolicy,
    revise_state,
    intrinsic_reward,
    use_revised_for_policy: bool,
    use_intrinsic: bool,
    intrinsic_w: float,
    max_steps: int,
    rng: np.random.Generator,
    finagent: FinAgentStub | None = None,
    finagent_weight: float = 0.0,
    intrinsic_scale_mode: str = "raw",
) -> Tuple[List[float], List[float]]:
    env = TradingEnv(df, schema.assets, schema, env_cfg)
    state = env.reset()
    values = [env.last_value]
    rewards = []

    step = 0
    done = False
    while not done and step < max_steps:
        revised = revise_state(state) if revise_state is not None else state
        policy_state = revised if use_revised_for_policy else state
        action = policy.act(state, policy_state)
        if finagent is not None and finagent_weight > 0.0:
            prices = {row.asset: float(row.close) for row in env._day_df().itertuples()}
            fa_actions = finagent.step(prices)
            fa_vec = np.array([fa_actions[a] for a in schema.assets], dtype=float)
            action = action + finagent_weight * fa_vec * env_cfg.max_trade
        # small noise for seed variation
        noise = rng.normal(0, 0.05, size=action.shape)
        action = np.clip(action + noise, -env_cfg.max_trade, env_cfg.max_trade)

        next_state, reward_env, done, info = env.step(action)
        action_penalty = _sanitize_float(info.get("action_bound_penalty", 0.0))
        reward = float(reward_env) - float(action_penalty)
        if use_intrinsic and intrinsic_reward is not None:
            try:
                r_int = float(intrinsic_reward(revised))
            except Exception:
                r_int = 0.0
            if not np.isfinite(r_int):
                r_int = 0.0
            r_int = _scale_intrinsic_value(r_int, intrinsic_scale_mode)
            reward += intrinsic_w * float(r_int)
        rewards.append(float(reward))
        values.append(float(info.get("portfolio_value", values[-1])))
        state = next_state
        step += 1

    return values, rewards


def _score_from_metrics(metrics: dict) -> float:
    # performance objective
    return float(metrics.get("Sharpe", 0.0) + metrics.get("CR", 0.0))


def _candidate_scoring_objective(scoring_cfg: dict | None) -> str:
    mode = str((scoring_cfg or {}).get("performance_mode", "absolute")).lower()
    target_name = "Sharpe"
    if mode == "delta_to_g0":
        state_obj = f"state_core=delta_performance({target_name}_G1-{target_name}_G0)+behavior+turnover"
        intrinsic_obj = f"intrinsic_core=delta_performance({target_name}_G2-{target_name}_G0)+raw_nontriviality+behavior+turnover"
        joint_obj = f"joint_pair=delta_performance({target_name}_G3-{target_name}_G0)+lipschitz+behavior+turnover"
    else:
        state_obj = f"state_core=performance({target_name}_G1)+behavior+turnover"
        intrinsic_obj = f"intrinsic_core=performance({target_name}_G2)+raw_nontriviality+behavior+turnover"
        joint_obj = f"joint_pair=performance({target_name}_G3)+lipschitz+behavior+turnover"
    return "; ".join([state_obj, intrinsic_obj, joint_obj])


def _resolve_llm_iteration_mode(llm_cfg: dict | None) -> str:
    mode = str((llm_cfg or {}).get("iteration_mode", "single_branch")).strip().lower()
    if mode in {"per_algo", "per_algorithm", "per_algo_branches", "per_algorithm_branches"}:
        return "per_algorithm_branches"
    return "single_branch"


def _resolve_llm_branch_parallel_workers(llm_cfg: dict | None, branch_algos: List[str]) -> int:
    if not branch_algos:
        return 1
    raw_workers = (llm_cfg or {}).get("branch_parallel_workers", 1) if isinstance(llm_cfg, dict) else 1
    try:
        workers = int(raw_workers)
    except Exception:
        workers = 1
    if workers <= 0:
        workers = len(branch_algos)
    return int(max(1, min(len(branch_algos), workers)))


def _resolve_candidate_selection_seeds(cfg: DemoConfig, llm_cfg: dict | None) -> List[int]:
    base_seeds = [int(x) for x in (cfg.seeds or [cfg.seed])]
    if not base_seeds:
        return [int(cfg.seed)]

    scoring_cfg = (llm_cfg or {}).get("candidate_scoring", {}) if isinstance(llm_cfg, dict) else {}
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}

    explicit = scoring_cfg.get("selection_seeds")
    if isinstance(explicit, list) and explicit:
        selection_seeds: List[int] = []
        for item in explicit:
            try:
                selection_seeds.append(int(item))
            except Exception:
                continue
        if selection_seeds:
            return selection_seeds

    raw_count = scoring_cfg.get("selection_seed_count")
    if raw_count is None:
        return base_seeds
    try:
        count = int(raw_count)
    except Exception:
        return base_seeds
    count = int(max(1, min(len(base_seeds), count)))
    return base_seeds[:count]


def _candidate_metric_value(metrics: dict, scoring_cfg: dict | None) -> float:
    metric = str((scoring_cfg or {}).get("performance_metric", "score")).lower()
    # LESR branch evaluation and final promotion are unified on delta-Sharpe.
    # Keep legacy config aliases, but always interpret the performance metric as Sharpe.
    if metric in {"score", "sharpe", "sharpe_only", "sharpe-first", "sharpe_first"}:
        return float(_sanitize_float((metrics or {}).get("Sharpe", 0.0)))
    return float(_score_from_metrics(metrics))


def _candidate_performance_payload(metrics: dict, baseline_metrics: dict | None, scoring_cfg: dict | None) -> dict:
    absolute_score = float(_candidate_metric_value(metrics, scoring_cfg))
    baseline_score = float(_candidate_metric_value(baseline_metrics or {}, scoring_cfg))
    delta_score = float(absolute_score - baseline_score)
    mode = str((scoring_cfg or {}).get("performance_mode", "absolute")).lower()
    metric = "sharpe"
    effective_score = delta_score if mode == "delta_to_g0" else absolute_score
    return {
        "performance_mode": mode,
        "performance_metric": metric,
        "performance_score": float(effective_score),
        "performance_score_absolute": absolute_score,
        "performance_score_baseline": baseline_score,
        "performance_score_delta": delta_score,
    }


def _resolve_candidate_scoring_cfg(llm_cfg: dict | None) -> dict:
    scoring_cfg = (llm_cfg or {}).get("candidate_scoring", {}) if isinstance(llm_cfg, dict) else {}
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}
    perf_mode = "delta_to_g0"
    performance_metric = str(scoring_cfg.get("performance_metric", "sharpe")).lower()
    if performance_metric in {"score", "sharpe_only", "sharpe-first", "sharpe_first"}:
        performance_metric = "sharpe"
    if performance_metric != "sharpe":
        performance_metric = "sharpe"
    perf_w = float(scoring_cfg.get("performance_weight", 1.0))
    lip_w = float(scoring_cfg.get("lipschitz_weight", 0.2))
    beh_w = float(scoring_cfg.get("behavior_weight", 0.15))
    intrinsic_probe_w = float(scoring_cfg.get("intrinsic_probe_weight", 0.3))
    turnover_w = float(scoring_cfg.get("turnover_weight", 0.10))
    if perf_w < 0:
        perf_w = 0.0
    if lip_w < 0:
        lip_w = 0.0
    if beh_w < 0:
        beh_w = 0.0
    if intrinsic_probe_w < 0:
        intrinsic_probe_w = 0.0
    if turnover_w < 0:
        turnover_w = 0.0
    if perf_w == 0 and lip_w == 0 and beh_w == 0 and intrinsic_probe_w == 0 and turnover_w == 0:
        perf_w = 1.0
    lip_quantile = float(scoring_cfg.get("lipschitz_quantile", 0.9))
    lip_quantile = float(np.clip(lip_quantile, 0.5, 0.99))
    max_pairs = int(scoring_cfg.get("lipschitz_max_pairs", 256))
    max_pairs = int(max(16, max_pairs))
    algo_probe_steps_floor = dict(scoring_cfg.get("algo_probe_steps_floor", {}) or {})
    algo_probe_seed_floor = dict(scoring_cfg.get("algo_probe_seed_floor", {}) or {})
    return {
        "performance_mode": perf_mode,
        "performance_metric": performance_metric,
        "performance_weight": perf_w,
        "lipschitz_weight": lip_w,
        "behavior_weight": beh_w,
        "intrinsic_probe_weight": intrinsic_probe_w,
        "turnover_weight": turnover_w,
        "lipschitz_quantile": lip_quantile,
        "lipschitz_max_pairs": max_pairs,
        "algo_probe_steps_floor": {
            str(k): int(max(0, _sanitize_float(v) or 0.0))
            for k, v in algo_probe_steps_floor.items()
        }
        if algo_probe_steps_floor
        else {"sac": 120, "td3": 180},
        "algo_probe_seed_floor": {
            str(k): int(max(1, _sanitize_float(v) or 1.0))
            for k, v in algo_probe_seed_floor.items()
        }
        if algo_probe_seed_floor
        else {"sac": 2, "td3": 3},
    }


def _resolve_candidate_scoring_budget(
    *,
    cfg: DemoConfig,
    algo: str,
    candidate_scoring_cfg: dict,
    requested_seeds: List[int],
    train_df: pd.DataFrame,
) -> tuple[List[int], int]:
    algo_probe_seed_floor = int(
        max(
            1,
            _sanitize_float((candidate_scoring_cfg.get("algo_probe_seed_floor", {}) or {}).get(str(algo), 1.0))
            or 1.0,
        )
    )
    scoring_seed_count = int(max(1, len(requested_seeds or []), algo_probe_seed_floor))
    scoring_seed_pool: List[int] = []
    for raw_seed in list(requested_seeds or []) + list(cfg.seeds or []) + [int(cfg.seed)]:
        try:
            cur_seed = int(raw_seed)
        except Exception:
            continue
        if cur_seed not in scoring_seed_pool:
            scoring_seed_pool.append(cur_seed)
    scoring_seeds = list(scoring_seed_pool[:scoring_seed_count]) if scoring_seed_pool else [int(cfg.seed)]
    algo_probe_steps_floor = int(
        max(
            0,
            _sanitize_float((candidate_scoring_cfg.get("algo_probe_steps_floor", {}) or {}).get(str(algo), 0.0))
            or 0.0,
        )
    )
    steps_small = int(max(_effective_steps(cfg.n_small, int(train_df["date"].nunique())), algo_probe_steps_floor))
    return scoring_seeds, steps_small


def _identity_revise_state(state):
    return np.asarray(state, dtype=np.float32)


def _zero_intrinsic_reward(_state):
    return 0.0


def _validation_error_message(exc: Exception) -> str:
    msg = str(exc).strip()
    if isinstance(exc, IndexError):
        return f"index_out_of_bounds:{msg or 'IndexError'}"
    if isinstance(exc, ZeroDivisionError):
        return f"zero_division:{msg or 'ZeroDivisionError'}"
    low = msg.lower()
    if "out of bounds" in low or "index" in low and "size" in low:
        return f"index_out_of_bounds:{msg}"
    etype = type(exc).__name__
    return f"{etype}:{msg}" if msg else etype


def _validate_candidate_pair_for_schema(revise_state_fn, intrinsic_reward_fn, schema: StateSchema) -> tuple[int, float]:
    test_state = np.zeros(schema.dim(), dtype=np.float32)
    try:
        revised = revise_state_fn(test_state)
    except Exception as exc:
        raise ValueError(f"revise_state_exception:{_validation_error_message(exc)}") from exc
    if revised is None:
        raise ValueError("revise_state_returned_none")
    try:
        revised_arr = np.asarray(revised, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"revise_state_not_numeric:{_validation_error_message(exc)}") from exc
    if revised_arr.size == 0:
        raise ValueError("revised_state_empty")
    if revised_arr.shape[0] < schema.dim():
        raise ValueError(
            f"revised_state_dim_too_small:got={int(revised_arr.shape[0])},expected>={int(schema.dim())}"
        )
    if not np.isfinite(revised_arr).all():
        raise ValueError("revised_state_non_finite")
    try:
        intrinsic_raw_source = intrinsic_reward_fn(test_state)
    except Exception as exc:
        raise ValueError(f"intrinsic_exception:{_validation_error_message(exc)}") from exc
    if intrinsic_raw_source is None:
        raise ValueError("intrinsic_returned_none")
    try:
        intrinsic_val_source = float(intrinsic_raw_source)
    except Exception as exc:
        raise ValueError(f"intrinsic_not_scalar:{type(intrinsic_raw_source).__name__}") from exc
    if not np.isfinite(intrinsic_val_source):
        raise ValueError(f"intrinsic_non_finite_raw:{intrinsic_val_source}")
    if intrinsic_val_source < -100.0 or intrinsic_val_source > 100.0:
        raise ValueError(f"intrinsic_out_of_range_raw:{intrinsic_val_source}")
    try:
        intrinsic_raw_revised = intrinsic_reward_fn(revised_arr)
    except Exception as exc:
        raise ValueError(f"intrinsic_exception_revised:{_validation_error_message(exc)}") from exc
    if intrinsic_raw_revised is None:
        raise ValueError("intrinsic_returned_none_revised")
    try:
        intrinsic_val_revised = float(intrinsic_raw_revised)
    except Exception as exc:
        raise ValueError(f"intrinsic_not_scalar_revised:{type(intrinsic_raw_revised).__name__}") from exc
    if not np.isfinite(intrinsic_val_revised):
        raise ValueError(f"intrinsic_non_finite_revised:{intrinsic_val_revised}")
    if intrinsic_val_revised < -100.0 or intrinsic_val_revised > 100.0:
        raise ValueError(f"intrinsic_out_of_range_revised:{intrinsic_val_revised}")
    return int(revised_arr.shape[0]), float(intrinsic_val_revised)


def _validate_candidate_pair_for_native_states(
    revise_state_fn,
    intrinsic_reward_fn,
    validation_states: np.ndarray,
    raw_dim: int,
) -> tuple[int, float]:
    states = select_native_validation_states(validation_states, max_states=3)
    if not states:
        raise ValueError("native_validation_states_empty")
    revised_dim: int | None = None
    intrinsic_val_revised = 0.0
    for sample_idx, sample_state in enumerate(states):
        test_state = np.asarray(sample_state, dtype=np.float32).reshape(-1)
        if test_state.size != int(raw_dim):
            raise ValueError(f"native_state_dim_mismatch:got={int(test_state.size)},expected={int(raw_dim)}")
        try:
            revised = revise_state_fn(test_state)
        except Exception as exc:
            raise ValueError(
                f"revise_state_exception_native_sample_{sample_idx}:{_validation_error_message(exc)}"
            ) from exc
        if revised is None:
            raise ValueError("revise_state_returned_none")
        try:
            revised_arr = np.asarray(revised, dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ValueError(f"revise_state_not_numeric:{_validation_error_message(exc)}") from exc
        if revised_arr.size < int(raw_dim):
            raise ValueError(
                f"revised_state_dim_too_small:got={int(revised_arr.size)},expected>={int(raw_dim)}"
            )
        if not np.isfinite(revised_arr).all():
            raise ValueError("revised_state_non_finite")
        if not np.allclose(
            revised_arr[: int(raw_dim)],
            test_state.astype(np.float64),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("native_prefix_not_preserved")
        if revised_dim is None:
            revised_dim = int(revised_arr.size)
        elif revised_dim != int(revised_arr.size):
            raise ValueError(
                f"revised_state_dim_inconsistent:prev={int(revised_dim)},cur={int(revised_arr.size)}"
            )
        try:
            intrinsic_raw_source = intrinsic_reward_fn(test_state)
        except Exception as exc:
            raise ValueError(
                f"intrinsic_exception_native_raw_sample_{sample_idx}:{_validation_error_message(exc)}"
            ) from exc
        if intrinsic_raw_source is None:
            raise ValueError("intrinsic_returned_none")
        try:
            intrinsic_val_source = float(intrinsic_raw_source)
        except Exception as exc:
            raise ValueError(f"intrinsic_not_scalar:{type(intrinsic_raw_source).__name__}") from exc
        if not np.isfinite(intrinsic_val_source):
            raise ValueError(f"intrinsic_non_finite_raw:{intrinsic_val_source}")
        if intrinsic_val_source < -100.0 or intrinsic_val_source > 100.0:
            raise ValueError(f"intrinsic_out_of_range_raw:{intrinsic_val_source}")
        try:
            intrinsic_raw_revised = intrinsic_reward_fn(revised_arr)
        except Exception as exc:
            raise ValueError(
                f"intrinsic_exception_native_revised_sample_{sample_idx}:{_validation_error_message(exc)}"
            ) from exc
        if intrinsic_raw_revised is None:
            raise ValueError("intrinsic_returned_none_revised")
        try:
            intrinsic_val_revised = float(intrinsic_raw_revised)
        except Exception as exc:
            raise ValueError(f"intrinsic_not_scalar_revised:{type(intrinsic_raw_revised).__name__}") from exc
        if not np.isfinite(intrinsic_val_revised):
            raise ValueError(f"intrinsic_non_finite_revised:{intrinsic_val_revised}")
        if intrinsic_val_revised < -100.0 or intrinsic_val_revised > 100.0:
            raise ValueError(f"intrinsic_out_of_range_revised:{intrinsic_val_revised}")
    return int(revised_dim or raw_dim), float(intrinsic_val_revised)


def _validate_candidate_pair_for_backend(
    revise_state_fn,
    intrinsic_reward_fn,
    *,
    drl_backend: str,
    schema: StateSchema | None = None,
    native_validation_states: np.ndarray | None = None,
    native_raw_dim: int | None = None,
) -> tuple[int, float]:
    backend = str(drl_backend or "current").strip().lower()
    if backend == "finsaber_native":
        if native_validation_states is None or native_raw_dim is None:
            raise ValueError("native_candidate_validation_context_missing")
        return _validate_candidate_pair_for_native_states(
            revise_state_fn,
            intrinsic_reward_fn,
            native_validation_states,
            int(native_raw_dim),
        )
    if schema is None:
        raise ValueError("schema_required_for_generic_candidate_validation")
    return _validate_candidate_pair_for_schema(revise_state_fn, intrinsic_reward_fn, schema)


def _build_policy_state_fn_for_selection(
    base_state_fn,
    *,
    cfg: DemoConfig,
    schema: StateSchema,
    reference_states: np.ndarray,
    drl_backend: str = "current",
    native_raw_dim: int | None = None,
    algorithm: str | None = None,
):
    if str(drl_backend or "current").strip().lower() == "finsaber_native":
        if str(algorithm or "").strip().lower() == "td3":
            policy_state_fn, _ = build_td3_state_fn(
                base_state_fn=base_state_fn,
                reference_states=reference_states,
                raw_dim=int(native_raw_dim or schema.dim()),
                volume_indices=[],
                norm_cfg=resolve_td3_state_norm_config(cfg.td3),
            )
            return policy_state_fn
        def _policy_state_fn(state):
            out = base_state_fn(state)
            arr = np.asarray(out, dtype=np.float32).reshape(-1)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return _policy_state_fn
    policy_state_fn, _ = build_td3_state_fn(
        base_state_fn=base_state_fn,
        reference_states=reference_states,
        raw_dim=int(schema.dim()),
        volume_indices=_volume_indices(schema),
        norm_cfg=resolve_td3_state_norm_config(cfg.td3),
    )
    return policy_state_fn


def _prepare_intrinsic_for_selection(
    revise_state_fn,
    intrinsic_reward_fn,
    *,
    cfg: DemoConfig,
    reference_states: np.ndarray,
    input_mode: str = "revised",
):
    intrinsic_fn, _ = _build_intrinsic_postprocessed_fn(
        intrinsic_reward=intrinsic_reward_fn,
        revise_state=revise_state_fn,
        reference_states=reference_states,
        post_cfg=_resolve_intrinsic_postprocess_cfg(cfg.intrinsic_postprocess),
        input_mode=input_mode,
    )
    return intrinsic_fn or intrinsic_reward_fn


def _estimate_intrinsic_signal_stats(
    *,
    revise_state_fn,
    intrinsic_reward_fn,
    reference_states: np.ndarray,
    input_mode: str = "revised",
) -> dict:
    try:
        states_arr = np.asarray(reference_states, dtype=np.float64)
    except Exception:
        states_arr = np.asarray([], dtype=np.float64)
    if states_arr.ndim != 2 or states_arr.shape[0] == 0:
        return {
            "available": False,
            "count": 0,
            "std": 0.0,
            "span": 0.0,
            "mean_abs": 0.0,
            "nonzero_ratio": 0.0,
            "nontrivial": False,
        }

    values: list[float] = []
    for row in states_arr:
        try:
            state = np.asarray(row, dtype=np.float64).reshape(-1)
            intrinsic_input = state
            if str(input_mode).strip().lower() != "raw":
                intrinsic_input = np.asarray(revise_state_fn(state), dtype=np.float64).reshape(-1)
            intrinsic_val = float(intrinsic_reward_fn(intrinsic_input))
        except Exception:
            continue
        if np.isfinite(intrinsic_val):
            values.append(float(np.clip(intrinsic_val, -100.0, 100.0)))

    if len(values) < 4:
        return {
            "available": False,
            "count": int(len(values)),
            "std": 0.0,
            "span": 0.0,
            "mean_abs": 0.0,
            "nonzero_ratio": 0.0,
            "nontrivial": False,
        }

    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    span = float(np.max(arr) - np.min(arr))
    mean_abs = float(np.mean(np.abs(arr)))
    nonzero_ratio = float(np.mean(np.abs(arr) > 1e-4))
    nontrivial = bool(std >= 1e-3 and span >= 1e-3 and mean_abs >= 1e-3 and nonzero_ratio >= 0.2)
    return {
        "available": True,
        "count": int(arr.size),
        "std": std,
        "span": span,
        "mean_abs": mean_abs,
        "nonzero_ratio": nonzero_ratio,
        "nontrivial": nontrivial,
    }


def _resolve_final_selection_cfg(llm_cfg: dict | None) -> dict:
    scoring_cfg = (llm_cfg or {}).get("candidate_scoring", {}) if isinstance(llm_cfg, dict) else {}
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}
    mode = str(scoring_cfg.get("final_selection_mode", "auto")).strip().lower()
    if mode not in {"auto", "threadpool", "serial", "subprocess"}:
        mode = "auto"
    top_n = scoring_cfg.get("final_selection_top_n_per_algo", 12)
    try:
        top_n = int(top_n)
    except Exception:
        top_n = 12
    top_n = int(max(0, top_n))
    timeout_s = scoring_cfg.get("final_selection_timeout_s", 1800)
    try:
        timeout_s = int(timeout_s)
    except Exception:
        timeout_s = 1800
    timeout_s = int(max(60, timeout_s))
    poll_s = scoring_cfg.get("final_selection_poll_s", 5)
    try:
        poll_s = int(poll_s)
    except Exception:
        poll_s = 5
    poll_s = int(max(1, poll_s))
    heartbeat_timeout_s = scoring_cfg.get("final_selection_heartbeat_timeout_s", 600)
    try:
        heartbeat_timeout_s = int(heartbeat_timeout_s)
    except Exception:
        heartbeat_timeout_s = 600
    heartbeat_timeout_s = int(max(30, heartbeat_timeout_s))
    bootstrap_timeout_s = scoring_cfg.get(
        "final_selection_bootstrap_timeout_s",
        min(120, heartbeat_timeout_s),
    )
    try:
        bootstrap_timeout_s = int(bootstrap_timeout_s)
    except Exception:
        bootstrap_timeout_s = min(120, heartbeat_timeout_s)
    bootstrap_timeout_s = int(max(10, min(bootstrap_timeout_s, heartbeat_timeout_s)))
    return {
        "mode": mode,
        "top_n_per_algo": top_n,
        "timeout_s": timeout_s,
        "poll_s": poll_s,
        "heartbeat_timeout_s": heartbeat_timeout_s,
        "bootstrap_timeout_s": bootstrap_timeout_s,
    }


def _effective_final_selection_mode(
    llm_iteration_mode: str,
    branch_parallel_workers: int,
    eval_algos: List[str],
    final_selection_cfg: dict | None,
) -> str:
    cfg = final_selection_cfg or {}
    mode = str(cfg.get("mode", "auto")).strip().lower()
    if mode in {"serial", "threadpool", "subprocess"}:
        return mode
    if llm_iteration_mode != "per_algorithm_branches" or branch_parallel_workers <= 1 or len(eval_algos) <= 1:
        return "serial"
    if os.name == "nt":
        return "subprocess"
    return "threadpool"


def _resolve_branch_iteration_worker_cfg(llm_cfg: dict | None) -> dict:
    cfg = llm_cfg or {}
    mode = str(cfg.get("branch_iteration_mode", "auto")).strip().lower()
    if mode not in {"auto", "threadpool", "serial", "subprocess"}:
        mode = "auto"
    timeout_s = cfg.get("branch_iteration_timeout_s", 1800)
    try:
        timeout_s = int(timeout_s)
    except Exception:
        timeout_s = 1800
    timeout_s = int(max(60, timeout_s))
    poll_s = cfg.get("branch_iteration_poll_s", 5)
    try:
        poll_s = int(poll_s)
    except Exception:
        poll_s = 5
    poll_s = int(max(1, poll_s))
    heartbeat_timeout_s = cfg.get("branch_iteration_heartbeat_timeout_s", 600)
    try:
        heartbeat_timeout_s = int(heartbeat_timeout_s)
    except Exception:
        heartbeat_timeout_s = 600
    heartbeat_timeout_s = int(max(30, heartbeat_timeout_s))
    bootstrap_timeout_s = cfg.get("branch_iteration_bootstrap_timeout_s", min(120, heartbeat_timeout_s))
    try:
        bootstrap_timeout_s = int(bootstrap_timeout_s)
    except Exception:
        bootstrap_timeout_s = min(120, heartbeat_timeout_s)
    bootstrap_timeout_s = int(max(10, min(bootstrap_timeout_s, heartbeat_timeout_s)))
    return {
        "mode": mode,
        "timeout_s": timeout_s,
        "poll_s": poll_s,
        "heartbeat_timeout_s": heartbeat_timeout_s,
        "bootstrap_timeout_s": bootstrap_timeout_s,
    }


def _effective_branch_iteration_mode(
    llm_iteration_mode: str,
    branch_parallel_workers: int,
    llm_branch_algos: List[str],
    branch_iteration_cfg: dict | None,
) -> str:
    cfg = branch_iteration_cfg or {}
    mode = str(cfg.get("mode", "auto")).strip().lower()
    if mode in {"serial", "threadpool", "subprocess"}:
        return mode
    if llm_iteration_mode != "per_algorithm_branches" or branch_parallel_workers <= 1 or len(llm_branch_algos) <= 1:
        return "serial"
    if os.name == "nt":
        return "subprocess"
    return "threadpool"


def _combine_candidate_score(
    performance_score: float,
    lipschitz_raw: float | None,
    scoring_cfg: dict,
    behavior_score: float | None = None,
    intrinsic_probe_score: float | None = None,
    turnover_score: float | None = None,
) -> dict:
    perf = _sanitize_float(performance_score)
    lip_raw = _sanitize_float(lipschitz_raw) if lipschitz_raw is not None else None
    lip_raw = lip_raw if lip_raw is not None and np.isfinite(lip_raw) and lip_raw >= 0.0 else None
    lip_score = float(1.0 / (1.0 + lip_raw)) if lip_raw is not None else 0.0
    beh_score = _sanitize_float(behavior_score) if behavior_score is not None else 0.0
    beh_score = float(np.clip(beh_score, 0.0, 1.0))
    perf_w = float(scoring_cfg.get("performance_weight", 1.0))
    lip_w = float(scoring_cfg.get("lipschitz_weight", 0.2))
    beh_w = float(scoring_cfg.get("behavior_weight", 0.0))
    intrinsic_probe = _sanitize_float(intrinsic_probe_score) if intrinsic_probe_score is not None else 0.0
    intrinsic_probe_w = float(scoring_cfg.get("intrinsic_probe_weight", 0.0))
    turnover = _sanitize_float(turnover_score) if turnover_score is not None else 0.0
    turnover = float(np.clip(turnover, 0.0, 1.0))
    turnover_w = float(scoring_cfg.get("turnover_weight", 0.0))
    total = float(
        perf_w * perf
        + lip_w * lip_score
        + beh_w * beh_score
        + intrinsic_probe_w * intrinsic_probe
        + turnover_w * turnover
    )
    return {
        "score": total,
        "performance_score": perf,
        "lipschitz_raw": lip_raw,
        "lipschitz_score": lip_score,
        "behavior_score": beh_score,
        "intrinsic_probe_score": intrinsic_probe,
        "turnover_score": turnover,
        "weights": {
            "performance": perf_w,
            "lipschitz": lip_w,
            "behavior": beh_w,
            "intrinsic_probe": intrinsic_probe_w,
            "turnover": turnover_w,
        },
    }


def _candidate_probe_delta(row: dict) -> float:
    return float(
        _sanitize_float(
            row.get(
                "intrinsic_probe_delta_sharpe",
                row.get(
                    "intrinsic_probe_score_delta",
                    row.get("intrinsic_probe_score", 0.0),
                ),
            )
        )
    )


def _rank_candidate_rows(rows: List[dict], scoring_cfg: dict | None) -> List[dict]:
    valid_rows = [row for row in (rows or []) if isinstance(row, dict)]
    if not valid_rows:
        return []

    def _sort_key(row: dict) -> tuple[float, float, float, float, float, float]:
        perf_delta = float(
            _sanitize_float(
                row.get(
                    "performance_delta_sharpe",
                    row.get("performance_score_delta", row.get("performance_score", 0.0)),
                )
            )
        )
        state_delta = float(
            _sanitize_float(
                row.get(
                    "state_probe_delta_sharpe",
                    row.get("state_probe_score_delta", row.get("state_probe_score", 0.0)),
                )
            )
        )
        return (
            perf_delta,
            state_delta,
            float(_candidate_probe_delta(row)),
            float(_sanitize_float(row.get("score", 0.0))),
            float(_sanitize_float(row.get("behavior_score", 0.0))),
            float(_sanitize_float(row.get("turnover_score", 0.0))),
        )

    return sorted(valid_rows, key=_sort_key, reverse=True)


def _set_windows_safe_worker_limits() -> None:
    if os.name != "nt":
        return
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _prefilter_rows_from_candidate_meta(
    algo: str,
    candidate_funcs: List[Tuple[str, object, object, str]],
    candidate_meta_by_algo: Dict[str, Dict[str, dict]],
    scoring_cfg: dict,
) -> List[dict]:
    rows: List[dict] = []
    meta_map = candidate_meta_by_algo.get(algo, {})
    for name, _, _, _ in candidate_funcs:
        meta = dict(meta_map.get(name, {}) or {})
        rows.append(
            {
                "name": name,
                "origin": str(meta.get("origin", _candidate_origin_from_name(name))),
                "family": str(meta.get("family", "")),
                "design_mode": str(meta.get("design_mode", "")),
                "feature_groups": list(meta.get("feature_groups", [])),
                "revise_hash": str(meta.get("revise_hash", "")),
                "intrinsic_hash": str(meta.get("intrinsic_hash", "")),
                "score": float(_sanitize_float(meta.get("score", 0.0))),
                "performance_score": float(_sanitize_float(meta.get("performance_score", 0.0))),
                "performance_score_absolute": float(_sanitize_float(meta.get("performance_score_absolute", 0.0))),
                "performance_score_baseline": float(_sanitize_float(meta.get("performance_score_baseline", 0.0))),
                "performance_score_delta": float(_sanitize_float(meta.get("performance_score_delta", 0.0))),
                "performance_delta_sharpe": float(_sanitize_float(meta.get("performance_delta_sharpe", 0.0))),
                "state_probe_score": float(_sanitize_float(meta.get("state_probe_score", 0.0))),
                "state_probe_score_absolute": float(_sanitize_float(meta.get("state_probe_score_absolute", 0.0))),
                "state_probe_score_baseline": float(_sanitize_float(meta.get("state_probe_score_baseline", 0.0))),
                "state_probe_score_delta": float(_sanitize_float(meta.get("state_probe_score_delta", 0.0))),
                "state_probe_delta_sharpe": float(_sanitize_float(meta.get("state_probe_delta_sharpe", 0.0))),
                "intrinsic_probe_score": float(_sanitize_float(meta.get("intrinsic_probe_score", 0.0))),
                "intrinsic_probe_score_absolute": float(_sanitize_float(meta.get("intrinsic_probe_score_absolute", 0.0))),
                "intrinsic_probe_score_baseline": float(_sanitize_float(meta.get("intrinsic_probe_score_baseline", 0.0))),
                "intrinsic_probe_score_delta": float(_sanitize_float(meta.get("intrinsic_probe_score_delta", 0.0))),
                "intrinsic_probe_delta_sharpe": float(_sanitize_float(meta.get("intrinsic_probe_delta_sharpe", 0.0))),
                "intrinsic_signal_nontrivial_raw": bool(meta.get("intrinsic_signal_nontrivial_raw", False)),
                "behavior_score": float(_sanitize_float(meta.get("behavior_score", 0.0))),
                "turnover_score": float(_sanitize_float(meta.get("turnover_score", 0.0))),
                "lipschitz_raw": (
                    float(_sanitize_float(meta.get("lipschitz_raw")))
                    if meta.get("lipschitz_raw") is not None
                    else None
                ),
                "lipschitz_score": float(_sanitize_float(meta.get("lipschitz_score", 0.0))),
            }
        )
    return _rank_candidate_rows(rows, scoring_cfg)


def _apply_candidate_prefilter_for_algo(
    algo: str,
    candidate_funcs: List[Tuple[str, object, object, str]],
    candidate_meta_by_algo: Dict[str, Dict[str, dict]],
    scoring_cfg: dict,
    top_n: int,
) -> tuple[List[Tuple[str, object, object, str]], dict]:
    total_before = int(len(candidate_funcs))
    if top_n <= 0 or total_before <= top_n:
        return candidate_funcs, {
            "algorithm": algo,
            "applied": False,
            "top_n": int(top_n),
            "total_before": total_before,
            "total_after": total_before,
            "kept_names": [name for name, _, _, _ in candidate_funcs],
        }
    ranked_rows = _prefilter_rows_from_candidate_meta(algo, candidate_funcs, candidate_meta_by_algo, scoring_cfg)
    if not ranked_rows:
        return candidate_funcs, {
            "algorithm": algo,
            "applied": False,
            "top_n": int(top_n),
            "reason": "missing_prefilter_meta",
            "total_before": total_before,
            "total_after": total_before,
            "kept_names": [name for name, _, _, _ in candidate_funcs],
        }
    keep_names: List[str] = []
    if str(algo).lower() == "td3":
        td3_intrinsic_rows = [
            row
            for row in ranked_rows
            if str(row.get("design_mode", "")) in {"intrinsic_first", "balanced"}
            and bool(row.get("intrinsic_signal_nontrivial_raw", False))
            and (
                float(
                    _sanitize_float(
                        row.get("intrinsic_probe_delta_sharpe", row.get("intrinsic_probe_score_delta", 0.0))
                    )
                )
                > 0.0
                or float(_sanitize_float(row.get("intrinsic_probe_score_delta", 0.0))) > 0.0
            )
        ]
        if td3_intrinsic_rows:
            seed_name = str(td3_intrinsic_rows[0].get("name", ""))
            if seed_name:
                keep_names.append(seed_name)
    for row in ranked_rows:
        name = str(row.get("name", ""))
        if name and name not in keep_names:
            keep_names.append(name)
        if len(keep_names) >= top_n:
            break
    keep_set = set(keep_names)
    filtered = [entry for entry in candidate_funcs if entry[0] in keep_set]
    if not filtered:
        filtered = list(candidate_funcs[:top_n])
        keep_names = [name for name, _, _, _ in filtered]
        keep_set = set(keep_names)
    dropped_names = [name for name, _, _, _ in candidate_funcs if name not in keep_set]
    return filtered, {
        "algorithm": algo,
        "applied": True,
        "top_n": int(top_n),
        "total_before": total_before,
        "total_after": int(len(filtered)),
        "kept_names": keep_names,
        "dropped_count": int(len(dropped_names)),
        "dropped_names": dropped_names,
    }


def _write_final_selection_progress(progress_path: Path, payload: dict) -> None:
    progress = dict(payload)
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_path.write_text(json.dumps(_json_safe(progress), indent=2), encoding="utf-8")


def _consensus_sharpe_score(delta_sharpes: list[float]) -> dict:
    vals = np.array([_sanitize_float(v) for v in delta_sharpes], dtype=float)
    if vals.size == 0:
        return {
            "consensus_score": 0.0,
            "mean_delta_sharpe": 0.0,
            "min_delta_sharpe": 0.0,
            "std_delta_sharpe": 0.0,
        }
    return {
        "consensus_score": float(np.min(vals) + 0.5 * np.mean(vals) - 0.5 * np.std(vals)),
        "mean_delta_sharpe": float(np.mean(vals)),
        "min_delta_sharpe": float(np.min(vals)),
        "std_delta_sharpe": float(np.std(vals)),
    }


def _compute_baseline_metrics_for_algo(
    *,
    cfg: DemoConfig,
    algo: str,
    runtime: dict,
    seeds: List[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: StateSchema,
    env_cfg: EnvConfig,
    state_fn_raw,
    finagent: FinAgentStub | None,
    drl_backend: str = "current",
    steps_small_override: int | None = None,
) -> Dict[int, dict]:
    algo_env_cfg = _env_cfg_with_algo_penalty(cfg, env_cfg, algo)
    metrics_by_seed: Dict[int, dict] = {}
    steps_small = int(steps_small_override) if steps_small_override is not None else _effective_steps(
        cfg.n_small,
        int(train_df["date"].nunique()),
    )
    native_backend = str(drl_backend or "current").strip().lower() == "finsaber_native"
    native_cfg = runtime.get("native_cfg") if isinstance(runtime, dict) else None
    native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {}) if isinstance(runtime, dict) else {}
    eval_history_df = train_df.sort_values(["date", "asset"]).reset_index(drop=True)
    for sd in seeds:
        if native_backend:
            if native_cfg is None:
                raise ValueError("finsaber_native runtime missing native_cfg")
            native_cfg_small = replace(native_cfg, total_timesteps=int(steps_small))
            native_algo_kwargs_small = _native_small_budget_algo_kwargs(algo, native_algo_kwargs, int(steps_small))
            result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                eval_history_df=eval_history_df,
                cfg=native_cfg_small,
                seed=int(sd),
                algo_kwargs=native_algo_kwargs_small,
                revise_state=_identity_revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                policy_state_fn=state_fn_raw,
                use_revised=False,
                use_intrinsic=False,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                intrinsic_input_mode="raw",
            )
            metrics, _ = _sb3_metrics_from_eval(result)
        elif algo == "td3" and runtime["is_td3_legacy"]:
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
            eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
            result = train_td3(
                env=train_env,
                state_dim=state_fn_raw(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                action_dim=len(schema.assets),
                cfg=td3_cfg_small,
                max_steps=steps_small,
                state_fn=state_fn_raw,
                revise_state=_identity_revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                use_intrinsic=False,
                intrinsic_timing=cfg.intrinsic_timing,
                finagent=finagent,
                finagent_weight=cfg.finagent_weight,
                seed=sd,
                eval_env=eval_env,
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
            )
            metrics = compute_metrics(np.array(result.eval_values_final))
        elif not native_backend and algo == "td3":
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            sb3_cfg_small = _td3_cfg_to_sb3_cfg(runtime["sb3_algo_base_cfg"], td3_cfg_small, steps_small)
            sb3_kwargs_small = dict(runtime["sb3_algo_kwargs"])
            sb3_kwargs_small.update(_td3_cfg_to_sb3_kwargs(td3_cfg_small))
            result = train_sb3(
                algo="td3",
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type="continuous",
                policy_action_bound=runtime["td3_policy_action_bound"],
                revise_state=_identity_revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=False,
                use_intrinsic=False,
                policy_state_fn=state_fn_raw,
                seed=sd,
                algo_kwargs=sb3_kwargs_small,
            )
            metrics, _ = _sb3_metrics_from_eval(result)
        elif not native_backend:
            sb3_cfg_small = replace(runtime["sb3_algo_base_cfg"], total_timesteps=int(steps_small))
            result = train_sb3(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type=_action_space_type(algo),
                policy_action_bound=None,
                revise_state=_identity_revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=False,
                use_intrinsic=False,
                policy_state_fn=state_fn_raw,
                seed=sd,
                algo_kwargs=runtime["sb3_algo_kwargs"],
            )
            metrics, _ = _sb3_metrics_from_eval(result)
        metrics_by_seed[int(sd)] = {
            "Sharpe": float(metrics.get("Sharpe", 0.0)),
            "CR": float(metrics.get("CR", 0.0)),
        }
    return metrics_by_seed


def _score_candidate_payload_for_algo_external(
    *,
    cfg: DemoConfig,
    algo: str,
    runtime: dict,
    revise_state,
    intrinsic_reward_eval,
    intrinsic_reward_probe_eval=None,
    policy_state_fn_candidate,
    seeds: List[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: StateSchema,
    env_cfg: EnvConfig,
    state_fn_raw,
    finagent: FinAgentStub | None,
    candidate_scoring_cfg: dict,
    reference_states: np.ndarray,
    drl_backend: str = "current",
) -> dict:
    if intrinsic_reward_probe_eval is None:
        intrinsic_reward_probe_eval = intrinsic_reward_eval
    scoring_seeds, steps_small = _resolve_candidate_scoring_budget(
        cfg=cfg,
        algo=algo,
        candidate_scoring_cfg=candidate_scoring_cfg,
        requested_seeds=seeds,
        train_df=train_df,
    )
    algo_env_cfg = _env_cfg_with_algo_penalty(cfg, env_cfg, algo)
    baseline_metrics_by_seed = _compute_baseline_metrics_for_algo(
        cfg=cfg,
        algo=algo,
        runtime=runtime,
        seeds=scoring_seeds,
        train_df=train_df,
        val_df=val_df,
        schema=schema,
        env_cfg=algo_env_cfg,
        state_fn_raw=state_fn_raw,
        finagent=finagent,
        drl_backend=drl_backend,
        steps_small_override=steps_small,
    )
    seed_scores: List[float] = []
    seed_metric_rows: List[dict] = []
    seed_perf_rows: List[dict] = []
    seed_state_perf_rows: List[dict] = []
    seed_probe_perf_rows: List[dict] = []
    seed_behavior_rows: List[dict] = []
    corrs_accum: List[np.ndarray] = []
    native_backend = str(drl_backend or "current").strip().lower() == "finsaber_native"
    native_cfg = runtime.get("native_cfg") if isinstance(runtime, dict) else None
    native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {}) if isinstance(runtime, dict) else {}
    eval_history_df = train_df.sort_values(["date", "asset"]).reset_index(drop=True)
    for sd in scoring_seeds:
        if native_backend:
            if native_cfg is None:
                raise ValueError("finsaber_native runtime missing native_cfg")
            native_cfg_small = replace(native_cfg, total_timesteps=int(steps_small))
            native_algo_kwargs_small = _native_small_budget_algo_kwargs(algo, native_algo_kwargs, int(steps_small))
            result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                eval_history_df=eval_history_df,
                cfg=native_cfg_small,
                seed=int(sd),
                algo_kwargs=native_algo_kwargs_small,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_eval,
                policy_state_fn=policy_state_fn_candidate,
                use_revised=True,
                use_intrinsic=True,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                intrinsic_input_mode="revised",
            )
            metrics, _ = _sb3_metrics_from_eval(result)
            eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
            eval_trace_rows = result.get("eval_trace", []) or []
            eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
            eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
            probe_result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                eval_history_df=eval_history_df,
                cfg=native_cfg_small,
                seed=int(sd),
                algo_kwargs=native_algo_kwargs_small,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_probe_eval,
                policy_state_fn=state_fn_raw,
                use_revised=False,
                use_intrinsic=True,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                intrinsic_input_mode="raw",
            )
            probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
            state_probe_result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                eval_history_df=eval_history_df,
                cfg=native_cfg_small,
                seed=int(sd),
                algo_kwargs=native_algo_kwargs_small,
                revise_state=revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                policy_state_fn=policy_state_fn_candidate,
                use_revised=True,
                use_intrinsic=False,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                intrinsic_input_mode="revised",
            )
            state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
        elif algo == "td3" and runtime["is_td3_legacy"]:
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
            eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
            result = train_td3(
                env=train_env,
                state_dim=policy_state_fn_candidate(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                action_dim=len(schema.assets),
                cfg=td3_cfg_small,
                max_steps=steps_small,
                state_fn=policy_state_fn_candidate,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                use_intrinsic=True,
                intrinsic_timing=cfg.intrinsic_timing,
                finagent=finagent,
                finagent_weight=cfg.finagent_weight,
                seed=sd,
                eval_env=eval_env,
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_input_mode="revised",
            )
            metrics = compute_metrics(np.array(result.eval_values_final))
            corrs_accum.append(np.abs(np.array(result.corrs, dtype=np.float64)))
            eval_actions = result.eval_actions_final or []
            eval_trace_rows = result.eval_trace_final or []
            eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
            eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
            probe_train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
            probe_eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
            probe_result = train_td3(
                env=probe_train_env,
                state_dim=state_fn_raw(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                action_dim=len(schema.assets),
                cfg=td3_cfg_small,
                max_steps=steps_small,
                state_fn=state_fn_raw,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_probe_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                use_intrinsic=True,
                intrinsic_timing=cfg.intrinsic_timing,
                finagent=finagent,
                finagent_weight=cfg.finagent_weight,
                seed=sd,
                eval_env=probe_eval_env,
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_input_mode="raw",
            )
            probe_metrics = compute_metrics(np.array(probe_result.eval_values_final))
            state_probe_train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
            state_probe_eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
            state_probe_result = train_td3(
                env=state_probe_train_env,
                state_dim=policy_state_fn_candidate(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                action_dim=len(schema.assets),
                cfg=td3_cfg_small,
                max_steps=steps_small,
                state_fn=policy_state_fn_candidate,
                revise_state=revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                use_intrinsic=False,
                intrinsic_timing=cfg.intrinsic_timing,
                finagent=finagent,
                finagent_weight=cfg.finagent_weight,
                seed=sd,
                eval_env=state_probe_eval_env,
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_input_mode="revised",
            )
            state_probe_metrics = compute_metrics(np.array(state_probe_result.eval_values_final))
        elif algo == "td3":
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            sb3_cfg_small = _td3_cfg_to_sb3_cfg(runtime["sb3_algo_base_cfg"], td3_cfg_small, steps_small)
            sb3_kwargs_small = dict(runtime["sb3_algo_kwargs"])
            sb3_kwargs_small.update(_td3_cfg_to_sb3_kwargs(td3_cfg_small))
            result = train_sb3(
                algo="td3",
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type="continuous",
                policy_action_bound=runtime["td3_policy_action_bound"],
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=True,
                use_intrinsic=True,
                intrinsic_input_mode="revised",
                policy_state_fn=policy_state_fn_candidate,
                seed=sd,
                algo_kwargs=sb3_kwargs_small,
            )
            metrics, _ = _sb3_metrics_from_eval(result)
            eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
            eval_trace_rows = result.get("eval_trace", []) or []
            eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
            eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
            probe_result = train_sb3(
                algo="td3",
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type="continuous",
                policy_action_bound=runtime["td3_policy_action_bound"],
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_probe_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=False,
                use_intrinsic=True,
                intrinsic_input_mode="raw",
                policy_state_fn=state_fn_raw,
                seed=sd,
                algo_kwargs=sb3_kwargs_small,
            )
            probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
            state_probe_result = train_sb3(
                algo="td3",
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type="continuous",
                policy_action_bound=runtime["td3_policy_action_bound"],
                revise_state=revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=True,
                use_intrinsic=False,
                intrinsic_input_mode="revised",
                policy_state_fn=policy_state_fn_candidate,
                seed=sd,
                algo_kwargs=sb3_kwargs_small,
            )
            state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
        else:
            sb3_cfg_small = replace(runtime["sb3_algo_base_cfg"], total_timesteps=int(steps_small))
            result = train_sb3(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type=_action_space_type(algo),
                policy_action_bound=None,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=True,
                use_intrinsic=True,
                intrinsic_input_mode="revised",
                policy_state_fn=policy_state_fn_candidate,
                seed=sd,
                algo_kwargs=runtime["sb3_algo_kwargs"],
            )
            metrics, _ = _sb3_metrics_from_eval(result)
            eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
            eval_trace_rows = result.get("eval_trace", []) or []
            eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
            eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
            probe_result = train_sb3(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type=_action_space_type(algo),
                policy_action_bound=None,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward_probe_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=False,
                use_intrinsic=True,
                intrinsic_input_mode="raw",
                policy_state_fn=state_fn_raw,
                seed=sd,
                algo_kwargs=runtime["sb3_algo_kwargs"],
            )
            probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
            state_probe_result = train_sb3(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type=_action_space_type(algo),
                policy_action_bound=None,
                revise_state=revise_state,
                intrinsic_reward=_zero_intrinsic_reward,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=True,
                use_intrinsic=False,
                intrinsic_input_mode="revised",
                policy_state_fn=policy_state_fn_candidate,
                seed=sd,
                algo_kwargs=runtime["sb3_algo_kwargs"],
            )
            state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
        perf_payload = _candidate_performance_payload(
            metrics=metrics,
            baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
            scoring_cfg=candidate_scoring_cfg,
        )
        state_perf_payload = _candidate_performance_payload(
            metrics=state_probe_metrics,
            baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
            scoring_cfg=candidate_scoring_cfg,
        )
        probe_perf_payload = _candidate_performance_payload(
            metrics=probe_metrics,
            baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
            scoring_cfg=candidate_scoring_cfg,
        )
        action_bound = (
            float(runtime["td3_policy_action_bound"])
            if (algo == "td3" and runtime.get("td3_policy_action_bound") is not None)
            else (
                float(runtime.get("native_action_bound", 0.0))
                if native_backend and runtime.get("native_action_bound") is not None
                else float(env_cfg.max_trade)
            )
        )
        seed_behavior_rows.append(
            {
                "seed": int(sd),
                **_candidate_behavior_payload(
                    eval_actions,
                    action_bound=action_bound,
                    portfolio_weights=eval_weight_rows,
                    portfolio_weight_changes=eval_weight_changes,
                ),
            }
        )
        perf = float(perf_payload["performance_score"])
        seed_scores.append(perf)
        seed_perf_rows.append(perf_payload)
        seed_state_perf_rows.append(state_perf_payload)
        seed_probe_perf_rows.append(probe_perf_payload)
        seed_metric_rows.append(
            {
                "seed": int(sd),
                "Sharpe": float(metrics.get("Sharpe", 0.0)),
                "CR": float(metrics.get("CR", 0.0)),
                "performance_mode": str(perf_payload["performance_mode"]),
                "performance_score": perf,
                "performance_score_absolute": float(perf_payload["performance_score_absolute"]),
                "performance_score_baseline": float(perf_payload["performance_score_baseline"]),
                "performance_score_delta": float(perf_payload["performance_score_delta"]),
                "performance_delta_sharpe": float(
                    metrics.get("Sharpe", 0.0) - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)
                ),
                "state_probe_score": float(state_perf_payload["performance_score"]),
                "state_probe_score_absolute": float(state_perf_payload["performance_score_absolute"]),
                "state_probe_score_baseline": float(state_perf_payload["performance_score_baseline"]),
                "state_probe_score_delta": float(state_perf_payload["performance_score_delta"]),
                "state_probe_delta_sharpe": float(
                    state_probe_metrics.get("Sharpe", 0.0)
                    - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)
                ),
                "intrinsic_probe_score": float(probe_perf_payload["performance_score"]),
                "intrinsic_probe_score_absolute": float(probe_perf_payload["performance_score_absolute"]),
                "intrinsic_probe_score_baseline": float(probe_perf_payload["performance_score_baseline"]),
                "intrinsic_probe_score_delta": float(probe_perf_payload["performance_score_delta"]),
                "intrinsic_probe_delta_sharpe": float(
                    probe_metrics.get("Sharpe", 0.0) - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)
                ),
            }
        )
    perf_mean = float(np.mean(seed_scores)) if seed_scores else 0.0
    perf_abs_mean = float(np.mean([row["performance_score_absolute"] for row in seed_perf_rows])) if seed_perf_rows else 0.0
    perf_base_mean = float(np.mean([row["performance_score_baseline"] for row in seed_perf_rows])) if seed_perf_rows else 0.0
    perf_delta_mean = float(np.mean([row["performance_score_delta"] for row in seed_perf_rows])) if seed_perf_rows else 0.0
    perf_delta_sharpe_mean = (
        float(np.mean([float(seed_row.get("performance_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
        if seed_metric_rows
        else 0.0
    )
    state_perf_mean = (
        float(np.mean([row["performance_score"] for row in seed_state_perf_rows])) if seed_state_perf_rows else 0.0
    )
    state_abs_mean = (
        float(np.mean([row["performance_score_absolute"] for row in seed_state_perf_rows]))
        if seed_state_perf_rows
        else 0.0
    )
    state_base_mean = (
        float(np.mean([row["performance_score_baseline"] for row in seed_state_perf_rows]))
        if seed_state_perf_rows
        else 0.0
    )
    state_delta_mean = (
        float(np.mean([row["performance_score_delta"] for row in seed_state_perf_rows]))
        if seed_state_perf_rows
        else 0.0
    )
    state_delta_sharpe_mean = (
        float(np.mean([float(seed_row.get("state_probe_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
        if seed_metric_rows
        else 0.0
    )
    probe_perf_mean = float(np.mean([row["performance_score"] for row in seed_probe_perf_rows])) if seed_probe_perf_rows else 0.0
    probe_abs_mean = float(np.mean([row["performance_score_absolute"] for row in seed_probe_perf_rows])) if seed_probe_perf_rows else 0.0
    probe_base_mean = float(np.mean([row["performance_score_baseline"] for row in seed_probe_perf_rows])) if seed_probe_perf_rows else 0.0
    probe_delta_mean = float(np.mean([row["performance_score_delta"] for row in seed_probe_perf_rows])) if seed_probe_perf_rows else 0.0
    probe_delta_sharpe_mean = (
        float(np.mean([float(seed_row.get("intrinsic_probe_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
        if seed_metric_rows
        else 0.0
    )
    avg_corrs: List[float] = []
    lip_raw_model = None
    if corrs_accum:
        avg_corrs_arr = np.mean(np.stack(corrs_accum, axis=0), axis=0)
        avg_corrs = list(avg_corrs_arr)
        lip_raw_model = float(np.mean(avg_corrs_arr))
    lip_payload = _estimate_intrinsic_lipschitz(
        reference_states=reference_states,
        revise_state_fn=revise_state,
        intrinsic_reward_fn=intrinsic_reward_eval,
        max_pairs=int(candidate_scoring_cfg["lipschitz_max_pairs"]),
        quantile=float(candidate_scoring_cfg["lipschitz_quantile"]),
    )
    intrinsic_signal_stats_raw = _estimate_intrinsic_signal_stats(
        revise_state_fn=revise_state,
        intrinsic_reward_fn=intrinsic_reward_probe_eval,
        reference_states=reference_states,
        input_mode="raw",
    )
    lip_raw = lip_raw_model if lip_raw_model is not None else lip_payload.get("raw")
    behavior_payload = _aggregate_candidate_behavior(seed_behavior_rows)
    score_payload = _combine_candidate_score(
        perf_mean,
        lip_raw,
        candidate_scoring_cfg,
        behavior_score=float(behavior_payload["behavior_score"]),
        intrinsic_probe_score=probe_perf_mean,
        turnover_score=float(behavior_payload["turnover_stability_score"]),
    )
    return {
        "score": float(score_payload["score"]),
        "performance_mode": str(candidate_scoring_cfg.get("performance_mode", "absolute")),
        "performance_score": float(score_payload["performance_score"]),
        "performance_score_absolute": perf_abs_mean,
        "performance_score_baseline": perf_base_mean,
        "performance_score_delta": perf_delta_mean,
        "performance_delta_sharpe": perf_delta_sharpe_mean,
        "state_probe_score": state_perf_mean,
        "state_probe_score_absolute": state_abs_mean,
        "state_probe_score_baseline": state_base_mean,
        "state_probe_score_delta": state_delta_mean,
        "state_probe_delta_sharpe": state_delta_sharpe_mean,
        "lipschitz_raw": float(score_payload["lipschitz_raw"]) if score_payload["lipschitz_raw"] is not None else None,
        "lipschitz_score": float(score_payload["lipschitz_score"]),
        "behavior_score": float(score_payload["behavior_score"]),
        "intrinsic_probe_score": float(score_payload["intrinsic_probe_score"]),
        "intrinsic_probe_score_absolute": probe_abs_mean,
        "intrinsic_probe_score_baseline": probe_base_mean,
        "intrinsic_probe_score_delta": probe_delta_mean,
        "intrinsic_probe_delta_sharpe": probe_delta_sharpe_mean,
        "intrinsic_signal_stats_raw": intrinsic_signal_stats_raw,
        "intrinsic_signal_nontrivial_raw": bool(intrinsic_signal_stats_raw.get("nontrivial", False)),
        "behavior": behavior_payload,
        "turnover_score": float(score_payload["turnover_score"]),
        "seed_behavior": seed_behavior_rows,
        "seed_metrics": seed_metric_rows,
        "corrs": avg_corrs,
    }


def _final_selection_worker_entry(payload: dict, output_path_str: str, progress_path_str: str) -> None:
    output_path = Path(output_path_str)
    progress_path = Path(progress_path_str)
    algo = str(payload.get("algo", "unknown"))
    try:
        _set_windows_safe_worker_limits()
        cfg = payload["cfg"]
        if isinstance(cfg, dict):
            cfg = DemoConfig(**cfg)
        train_df = payload["train_df"]
        val_df = payload["val_df"]
        schema = payload["schema"]
        env_cfg = payload["env_cfg"]
        algo_env_cfg = _env_cfg_with_algo_penalty(cfg, env_cfg, algo)
        runtime = payload["runtime"]
        drl_backend = str(payload.get("drl_backend", "current"))
        native_validation_states = payload.get("native_validation_states")
        native_raw_dim = payload.get("native_raw_dim")
        candidate_scoring_cfg = payload["candidate_scoring_cfg"]
        reference_states = np.asarray(payload["reference_states"], dtype=np.float32)
        selection_seeds = [int(x) for x in payload["selection_seeds"]]
        state_fn_raw = _build_policy_state_fn_for_selection(
            _identity_revise_state,
            cfg=cfg,
            schema=schema,
            reference_states=reference_states,
            drl_backend=drl_backend,
            native_raw_dim=native_raw_dim,
            algorithm=algo,
        )
        finagent = FinAgentStub(FinAgentStubConfig()) if cfg.use_finagent_signal else None
        candidate_items = list(payload.get("candidates", []))
        rows: List[dict] = []
        worker_errors: List[dict] = []
        _write_final_selection_progress(
            progress_path,
            {
                "status": "running",
                "algorithm": algo,
                "candidate_count": int(len(candidate_items)),
                "completed_candidates": 0,
            },
        )
        for idx, item in enumerate(candidate_items, start=1):
            name = str(item.get("name", ""))
            code = str(item.get("code", ""))
            meta = dict(item.get("meta", {}) or {})
            try:
                _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
                revise_state, intrinsic_reward = load_functions_from_code(code)
                _validate_candidate_pair_for_backend(
                    revise_state,
                    intrinsic_reward,
                    drl_backend=drl_backend,
                    schema=schema,
                    native_validation_states=native_validation_states,
                    native_raw_dim=native_raw_dim,
                )
                intrinsic_reward_eval = _prepare_intrinsic_for_selection(
                    revise_state,
                    intrinsic_reward,
                    cfg=cfg,
                    reference_states=reference_states,
                    input_mode="revised",
                )
                intrinsic_reward_probe_eval = _prepare_intrinsic_for_selection(
                    revise_state,
                    intrinsic_reward,
                    cfg=cfg,
                    reference_states=reference_states,
                    input_mode="raw",
                )
                policy_state_fn_candidate = _build_policy_state_fn_for_selection(
                    revise_state,
                    cfg=cfg,
                    schema=schema,
                    reference_states=reference_states,
                    drl_backend=drl_backend,
                    native_raw_dim=native_raw_dim,
                    algorithm=algo,
                )
                score_payload = _score_candidate_payload_for_algo_external(
                    cfg=cfg,
                    algo=algo,
                    runtime=runtime,
                    revise_state=revise_state,
                    intrinsic_reward_eval=intrinsic_reward_eval,
                    intrinsic_reward_probe_eval=intrinsic_reward_probe_eval,
                    policy_state_fn_candidate=policy_state_fn_candidate,
                    seeds=selection_seeds,
                    train_df=train_df,
                    val_df=val_df,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    state_fn_raw=state_fn_raw,
                    finagent=finagent,
                    candidate_scoring_cfg=candidate_scoring_cfg,
                    reference_states=reference_states,
                    drl_backend=drl_backend,
                )
                rows.append(
                    {
                        "name": name,
                        "origin": str(meta.get("origin", _candidate_origin_from_name(name))),
                        "family": str(meta.get("family", "")),
                        "design_mode": str(meta.get("design_mode", "")),
                        "feature_groups": list(meta.get("feature_groups", [])),
                        "revise_hash": str(meta.get("revise_hash", "")),
                        "intrinsic_hash": str(meta.get("intrinsic_hash", "")),
                        **score_payload,
                    }
                )
            except Exception as exc:
                worker_errors.append(
                    {
                        "candidate": name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_final_selection_progress(
                progress_path,
                {
                    "status": "running",
                    "algorithm": algo,
                    "candidate_count": int(len(candidate_items)),
                    "completed_candidates": int(idx),
                    "last_candidate": name,
                    "error_count": int(len(worker_errors)),
                },
            )
        rows = _rank_candidate_rows(rows, candidate_scoring_cfg)
        output_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "algorithm": algo,
                    "rows": _json_safe(rows),
                    "worker_errors": _json_safe(worker_errors),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_final_selection_progress(
            progress_path,
            {
                "status": "done",
                "algorithm": algo,
                "candidate_count": int(len(candidate_items)),
                "completed_candidates": int(len(candidate_items)),
                "ranked_rows": int(len(rows)),
                "error_count": int(len(worker_errors)),
            },
        )
    except Exception:
        output_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "algorithm": algo,
                    "error": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_final_selection_progress(
            progress_path,
            {
                "status": "failed",
                "algorithm": algo,
                "error": traceback.format_exc(),
            },
        )


def _score_candidates_by_algo_subprocess(
    *,
    run_dir: Path,
    cfg: DemoConfig,
    eval_algos: List[str],
    candidate_funcs_by_algo: Dict[str, List[Tuple[str, object, object, str]]],
    candidate_meta_by_algo: Dict[str, Dict[str, dict]],
    algo_runtime_cache: Dict[str, dict],
    candidate_scoring_cfg: dict,
    selection_seeds: List[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: StateSchema,
    env_cfg: EnvConfig,
    reference_states: np.ndarray,
    drl_backend: str,
    native_validation_states: np.ndarray | None,
    native_raw_dim: int | None,
    timeout_s: int,
    poll_s: int,
    heartbeat_timeout_s: int,
    bootstrap_timeout_s: int,
) -> tuple[Dict[str, List[dict]], Dict[str, object]]:
    ctx = get_context("spawn")
    state_dir = run_dir / "final_selection_state"
    ensure_dir(state_dir)
    processes: Dict[str, object] = {}
    artifacts: Dict[str, object] = {"mode": "subprocess", "state_dir": str(state_dir), "workers": {}}
    for algo in eval_algos:
        candidate_items = []
        meta_map = candidate_meta_by_algo.get(algo, {})
        for name, _, _, code in candidate_funcs_by_algo[algo]:
            candidate_items.append({"name": name, "code": code, "meta": dict(meta_map.get(name, {}) or {})})
        output_path = state_dir / f"candidate_scores_{algo}.json"
        progress_path = state_dir / f"progress_{algo}.json"
        if output_path.exists():
            output_path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        payload = {
            "cfg": cfg.__dict__,
            "algo": algo,
            "runtime": algo_runtime_cache[algo],
            "selection_seeds": list(selection_seeds),
            "candidate_scoring_cfg": dict(candidate_scoring_cfg),
            "train_df": train_df,
            "val_df": val_df,
            "schema": schema,
            "env_cfg": env_cfg,
            "reference_states": np.asarray(reference_states, dtype=np.float32),
            "drl_backend": str(drl_backend),
            "native_validation_states": None
            if native_validation_states is None
            else np.asarray(native_validation_states, dtype=np.float32),
            "native_raw_dim": None if native_raw_dim is None else int(native_raw_dim),
            "candidates": candidate_items,
        }
        proc = ctx.Process(
            target=_final_selection_worker_entry,
            args=(payload, str(output_path), str(progress_path)),
            name=f"lesr-final-{algo}",
        )
        proc.daemon = False
        proc.start()
        processes[algo] = {
            "process": proc,
            "output_path": output_path,
            "progress_path": progress_path,
            "started_at": time.time(),
        }
        artifacts["workers"][algo] = {
            "pid": int(proc.pid or 0),
            "output_path": str(output_path),
            "progress_path": str(progress_path),
            "candidate_count": int(len(candidate_items)),
        }
    candidate_scores_by_algo: Dict[str, List[dict]] = {}
    worker_errors: Dict[str, object] = {}
    pending = set(eval_algos)
    while pending:
        finished_any = False
        for algo in list(pending):
            info = processes[algo]
            proc = info["process"]
            progress_path = info["progress_path"]
            progress_exists = progress_path.exists()
            if progress_exists:
                try:
                    info["last_progress_at"] = float(progress_path.stat().st_mtime)
                except Exception:
                    info["last_progress_at"] = max(float(info.get("last_progress_at", 0.0)), time.time())
            proc.join(timeout=0)
            if proc.is_alive():
                now = time.time()
                elapsed = now - float(info["started_at"])
                last_progress_at = float(info.get("last_progress_at", info["started_at"]))
                if not progress_exists and elapsed > float(bootstrap_timeout_s):
                    termination = _terminate_child_process(proc)
                    worker_errors[algo] = {
                        "type": "bootstrap_timeout",
                        "message": f"final selection worker emitted no heartbeat within {bootstrap_timeout_s}s",
                        "progress_path": str(progress_path),
                        "progress_tail": _read_text_tail(progress_path),
                        **termination,
                    }
                    pending.remove(algo)
                    finished_any = True
                    continue
                if progress_exists and (now - last_progress_at) > float(heartbeat_timeout_s):
                    termination = _terminate_child_process(proc)
                    worker_errors[algo] = {
                        "type": "heartbeat_timeout",
                        "message": f"final selection worker heartbeat stale for more than {heartbeat_timeout_s}s",
                        "progress_path": str(progress_path),
                        "progress_tail": _read_text_tail(progress_path),
                        **termination,
                    }
                    pending.remove(algo)
                    finished_any = True
                    continue
                if elapsed > float(timeout_s):
                    termination = _terminate_child_process(proc)
                    worker_errors[algo] = {
                        "type": "timeout",
                        "message": f"final selection worker timed out after {timeout_s}s",
                        "progress_path": str(info["progress_path"]),
                        "progress_tail": _read_text_tail(progress_path),
                        **termination,
                    }
                    pending.remove(algo)
                    finished_any = True
                continue
            finished_any = True
            pending.remove(algo)
            artifacts["workers"][algo]["returncode"] = int(proc.exitcode if proc.exitcode is not None else -9)
            artifacts["workers"][algo]["elapsed_s"] = float(time.time() - float(info["started_at"]))
            output_path = info["output_path"]
            if not output_path.exists():
                worker_errors[algo] = {
                    "type": "missing_output",
                    "message": f"worker exited without writing {output_path.name}",
                    "progress_path": str(progress_path),
                    "progress_tail": _read_text_tail(progress_path),
                    "returncode": int(proc.exitcode if proc.exitcode is not None else -9),
                }
                continue
            result, read_error = _safe_read_json_payload(output_path)
            if read_error is not None:
                worker_errors[algo] = {
                    **read_error,
                    "output_path": str(output_path),
                    "progress_path": str(progress_path),
                    "progress_tail": _read_text_tail(progress_path),
                    "returncode": int(proc.exitcode if proc.exitcode is not None else -9),
                }
                continue
            if not bool(result.get("ok", False)):
                worker_errors[algo] = {
                    "type": "worker_failed",
                    "message": str(result.get("error", "")),
                    "progress_path": str(progress_path),
                    "progress_tail": _read_text_tail(progress_path),
                    "returncode": int(proc.exitcode if proc.exitcode is not None else -9),
                }
                continue
            candidate_scores_by_algo[algo] = list(result.get("rows", []))
            if result.get("worker_errors"):
                worker_errors[f"{algo}__candidate_errors"] = result.get("worker_errors")
        if pending and not finished_any:
            time.sleep(float(poll_s))
    for info in processes.values():
        proc = info["process"]
        try:
            if hasattr(proc, "close"):
                proc.close()
        except Exception:
            pass
    artifacts["worker_errors"] = _json_safe(worker_errors)
    return candidate_scores_by_algo, artifacts


def _build_branch_timeout_result(payload: dict, error_type: str, message: str) -> dict:
    branch_state = dict(payload["branch_state"])
    iteration = int(payload["iteration"])
    branch_algo = str(payload["branch_algo"])
    dialogs = list(branch_state["dialogs"])
    iter_log = {
        "algorithm": branch_algo,
        "iteration": iteration,
        "prompt": dialogs[-1]["content"] if dialogs else "",
        "candidates": [],
        "feedback": None,
        "llm_iteration_mode": str(payload["llm_iteration_mode"]),
        "sample_attempts": 0,
        "sample_valid_count": 0,
        "sample_failed_calls": 1,
        "sample_stop_by_failure_limit": True,
    }
    return {
        "algorithm": branch_algo,
        "candidate_entries": [],
        "iter_log": iter_log,
        "llm_responses": [],
        "llm_errors": [
            {
                "algorithm": branch_algo,
                "iteration": iteration,
                "phase": "branch_worker",
                "attempt": 1,
                "error_type": error_type,
                "message": message,
            }
        ],
        "dialog_text": "",
        "next_dialogs": dialogs,
        "all_it_func_results": list(branch_state["all_it_func_results"]),
        "all_it_cot_suggestions": list(branch_state["all_it_cot_suggestions"]),
        "seen_candidate_hashes": list(branch_state["seen_candidate_hashes"]),
    }


def _safe_close_handle(handle) -> None:
    try:
        if handle is not None:
            handle.close()
    except Exception:
        pass


def _read_text_tail(path: Path | None, max_chars: int = 4000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(errors="replace")
        except Exception:
            return ""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _safe_read_json_payload(path: Path | None, max_chars: int = 4000) -> tuple[object | None, dict | None]:
    if path is None:
        return None, {"type": "missing_output", "message": "no output path provided"}
    if not path.exists():
        return None, {"type": "missing_output", "message": f"{path.name} does not exist"}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, {
            "type": "output_read_failed",
            "message": f"{type(exc).__name__}: {exc}",
            "raw_tail": _read_text_tail(path, max_chars=max_chars),
        }
    try:
        return json.loads(text), None
    except Exception as exc:
        raw_tail = text.strip()
        if max_chars > 0 and len(raw_tail) > max_chars:
            raw_tail = raw_tail[-max_chars:]
        return None, {
            "type": "invalid_output_json",
            "message": f"{type(exc).__name__}: {exc}",
            "raw_tail": raw_tail,
        }


def _terminate_child_process(proc, wait_timeout: float = 5.0) -> dict:
    timed_out = False
    still_alive = False
    returncode = None
    try:
        proc.kill()
    except Exception:
        pass
    try:
        if hasattr(proc, "wait"):
            try:
                proc.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        elif hasattr(proc, "join"):
            proc.join(timeout=wait_timeout)
            if hasattr(proc, "is_alive"):
                still_alive = bool(proc.is_alive())
                timed_out = timed_out or still_alive
    except Exception:
        pass
    try:
        if hasattr(proc, "is_alive"):
            still_alive = bool(proc.is_alive())
    except Exception:
        still_alive = False
    if still_alive and hasattr(proc, "terminate"):
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            if hasattr(proc, "wait"):
                proc.wait(timeout=1)
            elif hasattr(proc, "join"):
                proc.join(timeout=1)
                if hasattr(proc, "is_alive"):
                    still_alive = bool(proc.is_alive())
        except Exception:
            pass
    for attr in ("poll",):
        if hasattr(proc, attr):
            try:
                value = getattr(proc, attr)()
            except Exception:
                value = None
            if value is not None:
                returncode = int(value)
                break
    if returncode is None:
        for attr in ("returncode", "exitcode"):
            if hasattr(proc, attr):
                try:
                    value = getattr(proc, attr)
                except Exception:
                    value = None
                if value is not None:
                    returncode = int(value)
                    break
    if returncode is None:
        returncode = -9
    return {
        "returncode": int(returncode),
        "timed_out_wait": bool(timed_out),
        "still_alive": bool(still_alive),
    }


def _run_branch_iterations_by_subprocess(
    *,
    run_dir: Path,
    cfg: DemoConfig,
    llm_cfg: dict,
    branch_iteration_cfg: dict,
    it: int,
    llm_branch_algos: List[str],
    branch_state_by_algo: Dict[str, dict],
    system_prompt: str,
    state_desc: List[str],
    state_contract_note: str,
    scenario_profile: dict,
    scenario_enabled: bool,
    scenario_priority: List[str],
    candidates_per_family: int,
    candidate_scoring_cfg: dict,
    llm_iteration_mode: str,
    generation_target: str,
    max_iterations: int,
    selection_seeds: List[int],
    algo_runtime_cache: Dict[str, dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: StateSchema,
    env_cfg: EnvConfig,
    reference_states: np.ndarray,
    drl_backend: str = "current",
    native_validation_states: np.ndarray | None = None,
    native_raw_dim: int | None = None,
) -> tuple[Dict[str, dict], dict]:
    iter_dir = run_dir / "branch_iteration_state" / f"it{it:02d}"
    ensure_dir(iter_dir)
    worker_script = repo_root() / "src" / "pipeline" / "branch_iteration_worker.py"
    procs: Dict[str, dict] = {}
    artifacts: Dict[str, object] = {"mode": "subprocess", "iteration": int(it), "workers": {}, "worker_errors": {}}
    for branch_algo in llm_branch_algos:
        payload_path = iter_dir / f"payload_{branch_algo}.pkl"
        output_path = iter_dir / f"branch_{branch_algo}.json"
        progress_path = iter_dir / f"progress_{branch_algo}.json"
        stdout_path = iter_dir / f"worker_{branch_algo}.stdout.log"
        stderr_path = iter_dir / f"worker_{branch_algo}.stderr.log"
        payload = {
            "cfg": cfg.__dict__,
            "llm_cfg": dict(llm_cfg),
            "branch_algo": branch_algo,
            "iteration": int(it),
            "branch_state": _json_safe(branch_state_by_algo[branch_algo]),
            "system_prompt": system_prompt,
            "state_desc": state_desc,
            "state_contract_note": state_contract_note,
            "drl_backend": str(drl_backend),
            "native_validation_states": None
            if native_validation_states is None
            else np.asarray(native_validation_states, dtype=np.float32),
            "native_raw_dim": None if native_raw_dim is None else int(native_raw_dim),
            "scenario_profile": _json_safe(scenario_profile),
            "scenario_enabled": bool(scenario_enabled),
            "scenario_priority": list(scenario_priority),
            "candidates_per_family": int(candidates_per_family),
            "candidate_scoring_cfg": dict(candidate_scoring_cfg),
            "llm_iteration_mode": llm_iteration_mode,
            "generation_target": generation_target,
            "max_iterations": int(max_iterations),
            "selection_seeds": list(selection_seeds),
            "runtime": algo_runtime_cache[branch_algo],
            "train_df": train_df,
            "val_df": val_df,
            "schema": schema,
            "env_cfg": env_cfg,
            "reference_states": np.asarray(reference_states, dtype=np.float32),
        }
        with payload_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        if output_path.exists():
            output_path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        stdout_fh = stdout_path.open("w", encoding="utf-8", errors="replace")
        stderr_fh = stderr_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(worker_script),
                "--payload",
                str(payload_path),
                "--output",
                str(output_path),
                "--progress",
                str(progress_path),
            ],
            cwd=str(repo_root()),
            stdout=stdout_fh,
            stderr=stderr_fh,
        )
        procs[branch_algo] = {
            "process": proc,
            "payload": payload,
            "payload_path": payload_path,
            "output_path": output_path,
            "progress_path": progress_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_fh": stdout_fh,
            "stderr_fh": stderr_fh,
            "started_at": time.time(),
        }
        artifacts["workers"][branch_algo] = {
            "pid": int(proc.pid or 0),
            "payload_path": str(payload_path),
            "output_path": str(output_path),
            "progress_path": str(progress_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    timeout_s = int(branch_iteration_cfg.get("timeout_s", 1800))
    poll_s = int(branch_iteration_cfg.get("poll_s", 5))
    heartbeat_timeout_s = int(branch_iteration_cfg.get("heartbeat_timeout_s", 600))
    bootstrap_timeout_s = int(branch_iteration_cfg.get("bootstrap_timeout_s", min(120, heartbeat_timeout_s)))
    branch_results_by_algo: Dict[str, dict] = {}
    pending = set(llm_branch_algos)
    while pending:
        finished_any = False
        for branch_algo in list(pending):
            info = procs[branch_algo]
            proc = info["process"]
            progress_path = info["progress_path"]
            progress_exists = progress_path.exists()
            if progress_exists:
                try:
                    info["last_progress_at"] = float(progress_path.stat().st_mtime)
                except Exception:
                    info["last_progress_at"] = max(float(info.get("last_progress_at", 0.0)), time.time())
            retcode = proc.poll()
            if retcode is None:
                now = time.time()
                elapsed = now - float(info["started_at"])
                last_progress_at = float(info.get("last_progress_at", info["started_at"]))
                if not progress_exists and elapsed > float(bootstrap_timeout_s):
                    termination = _terminate_child_process(proc)
                    _safe_close_handle(info.get("stdout_fh"))
                    _safe_close_handle(info.get("stderr_fh"))
                    artifacts["worker_errors"][branch_algo] = {
                        "type": "bootstrap_timeout",
                        "message": f"branch iteration worker emitted no heartbeat within {bootstrap_timeout_s}s",
                        "progress_path": str(progress_path),
                        **termination,
                        "stderr_tail": _read_text_tail(info.get("stderr_path")),
                    }
                    branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                        info["payload"],
                        "branch_worker_bootstrap_timeout",
                        f"branch iteration worker emitted no heartbeat within {bootstrap_timeout_s}s",
                    )
                    pending.remove(branch_algo)
                    finished_any = True
                    continue
                if progress_exists and (now - last_progress_at) > float(heartbeat_timeout_s):
                    termination = _terminate_child_process(proc)
                    _safe_close_handle(info.get("stdout_fh"))
                    _safe_close_handle(info.get("stderr_fh"))
                    artifacts["worker_errors"][branch_algo] = {
                        "type": "heartbeat_timeout",
                        "message": f"branch iteration worker heartbeat stale for more than {heartbeat_timeout_s}s",
                        "progress_path": str(progress_path),
                        **termination,
                        "stderr_tail": _read_text_tail(info.get("stderr_path")),
                    }
                    branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                        info["payload"],
                        "branch_worker_heartbeat_timeout",
                        f"branch iteration worker heartbeat stale for more than {heartbeat_timeout_s}s",
                    )
                    pending.remove(branch_algo)
                    finished_any = True
                    continue
                if elapsed > float(timeout_s):
                    termination = _terminate_child_process(proc)
                    _safe_close_handle(info.get("stdout_fh"))
                    _safe_close_handle(info.get("stderr_fh"))
                    artifacts["worker_errors"][branch_algo] = {
                        "type": "timeout",
                        "message": f"branch iteration worker timed out after {timeout_s}s",
                        "progress_path": str(info["progress_path"]),
                        **termination,
                        "stderr_tail": _read_text_tail(info.get("stderr_path")),
                    }
                    branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                        info["payload"],
                        "branch_worker_timeout",
                        f"branch iteration worker timed out after {timeout_s}s",
                    )
                    pending.remove(branch_algo)
                    finished_any = True
                continue
            finished_any = True
            pending.remove(branch_algo)
            _safe_close_handle(info.get("stdout_fh"))
            _safe_close_handle(info.get("stderr_fh"))
            artifacts["workers"][branch_algo]["returncode"] = int(retcode)
            artifacts["workers"][branch_algo]["elapsed_s"] = float(time.time() - float(info["started_at"]))
            output_path = info["output_path"]
            if not output_path.exists():
                artifacts["worker_errors"][branch_algo] = {
                    "type": "missing_output",
                    "message": f"worker exited without writing {output_path.name}",
                    "returncode": int(retcode),
                    "stdout_tail": _read_text_tail(info.get("stdout_path")),
                    "stderr_tail": _read_text_tail(info.get("stderr_path")),
                }
                branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                    info["payload"],
                    "branch_worker_missing_output",
                    f"worker exited without writing {output_path.name}",
                )
                continue
            result, read_error = _safe_read_json_payload(output_path)
            if read_error is not None:
                artifacts["worker_errors"][branch_algo] = {
                    **read_error,
                    "returncode": int(retcode),
                    "stdout_tail": _read_text_tail(info.get("stdout_path")),
                    "stderr_tail": _read_text_tail(info.get("stderr_path")),
                }
                branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                    info["payload"],
                    "branch_worker_invalid_output_json",
                    str(read_error.get("message", "invalid output json")),
                )
                continue
            if not bool(result.get("ok", False)):
                artifacts["worker_errors"][branch_algo] = {
                    "type": "worker_failed",
                    "message": str(result.get("error", "")),
                    "returncode": int(retcode),
                    "stdout_tail": _read_text_tail(info.get("stdout_path")),
                    "stderr_tail": _read_text_tail(info.get("stderr_path")),
                }
                branch_results_by_algo[branch_algo] = _build_branch_timeout_result(
                    info["payload"],
                    "branch_worker_failed",
                    str(result.get("error", "")),
                )
                continue
            branch_results_by_algo[branch_algo] = dict(result.get("result", {}))
        if pending and not finished_any:
            time.sleep(float(poll_s))
    artifacts["worker_errors"] = _json_safe(artifacts["worker_errors"])
    return branch_results_by_algo, artifacts


def _estimate_intrinsic_lipschitz(
    reference_states: np.ndarray,
    revise_state_fn,
    intrinsic_reward_fn,
    max_pairs: int = 256,
    quantile: float = 0.9,
) -> dict:
    if reference_states is None:
        return {"raw": None, "score": 0.0, "pair_count": 0}
    arr = np.asarray(reference_states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return {"raw": None, "score": 0.0, "pair_count": 0}

    n_states = int(arr.shape[0])
    max_pairs = int(max(16, max_pairs))
    pair_count = int(min(max_pairs, n_states - 1))
    if pair_count <= 0:
        return {"raw": None, "score": 0.0, "pair_count": 0}

    idx = np.linspace(0, n_states - 1, num=pair_count + 1, dtype=int)
    ratios: List[float] = []
    valid_pairs = 0
    for i in range(1, len(idx)):
        s_prev = arr[idx[i - 1]]
        s_cur = arr[idx[i]]
        try:
            r_prev = np.asarray(revise_state_fn(s_prev), dtype=np.float64).reshape(-1)
            r_cur = np.asarray(revise_state_fn(s_cur), dtype=np.float64).reshape(-1)
            if r_prev.size == 0 or r_cur.size == 0:
                continue
            n = int(min(r_prev.shape[0], r_cur.shape[0]))
            if n <= 0:
                continue
            x_prev = r_prev[:n]
            x_cur = r_cur[:n]
            if not np.isfinite(x_prev).all() or not np.isfinite(x_cur).all():
                continue
            y_prev = _sanitize_float(intrinsic_reward_fn(x_prev))
            y_cur = _sanitize_float(intrinsic_reward_fn(x_cur))
            if not np.isfinite(y_prev) or not np.isfinite(y_cur):
                continue
            dx = float(np.linalg.norm(x_cur - x_prev))
            dy = abs(float(y_cur - y_prev))
            ratio = dy / (dx + 1e-3)
            if np.isfinite(ratio):
                ratios.append(float(max(0.0, ratio)))
                valid_pairs += 1
        except Exception:
            continue

    if not ratios:
        return {"raw": None, "score": 0.0, "pair_count": int(valid_pairs)}
    q = float(np.clip(quantile, 0.5, 0.99))
    lip_raw = float(np.quantile(np.array(ratios, dtype=np.float64), q))
    lip_score = float(1.0 / (1.0 + lip_raw))
    return {"raw": lip_raw, "score": lip_score, "pair_count": int(valid_pairs)}


def _window_returns_profile(train_df: pd.DataFrame, val_df: pd.DataFrame, assets: List[str]) -> dict:
    df = pd.concat([train_df, val_df], ignore_index=True)
    if df.empty:
        return {"mu_ann": 0.0, "vol_ann": 0.0, "max_dd": 0.0, "n_days": 0}
    close_df = (
        df[df["asset"].isin(assets)][["date", "asset", "close"]]
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    if close_df.empty:
        return {"mu_ann": 0.0, "vol_ann": 0.0, "max_dd": 0.0, "n_days": 0}
    market = close_df.mean(axis=1)
    ret = market.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return {"mu_ann": 0.0, "vol_ann": 0.0, "max_dd": 0.0, "n_days": 0}
    mu_ann = float(ret.mean() * 252.0)
    vol_ann = float(ret.std(ddof=0) * np.sqrt(252.0))
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = (1.0 - equity / (peak + 1e-12)).fillna(0.0)
    max_dd = float(dd.max()) if not dd.empty else 0.0
    return {"mu_ann": mu_ann, "vol_ann": vol_ann, "max_dd": max_dd, "n_days": int(ret.shape[0])}


def _percentile_rank(value: float, samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    return float((np.sum(arr <= float(value)) + 0.5) / (arr.size + 1.0))


def _infer_scenario_family(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    assets: List[str],
    router_cfg: dict | None = None,
) -> dict:
    router_cfg = router_cfg or {}
    profile = _window_returns_profile(train_df, val_df, assets)
    mu_ann = float(profile["mu_ann"])
    vol_ann = float(profile["vol_ann"])
    max_dd = float(profile["max_dd"])

    df = pd.concat([train_df, val_df], ignore_index=True)
    close_df = (
        df[df["asset"].isin(assets)][["date", "asset", "close"]]
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    ret = pd.Series(dtype=float)
    if not close_df.empty:
        market = close_df.mean(axis=1)
        ret = market.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    asset_ret = pd.DataFrame()
    if not close_df.empty:
        asset_ret = close_df.pct_change().replace([np.inf, -np.inf], np.nan)

    lookback = int(router_cfg.get("lookback_days", 63))
    lookback = int(max(20, min(lookback, 126)))
    if ret.shape[0] < lookback + 5:
        lookback = int(max(10, ret.shape[0] // 2))

    mu_hist = np.array([mu_ann], dtype=np.float64)
    vol_hist = np.array([vol_ann], dtype=np.float64)
    dd_hist = np.array([max_dd], dtype=np.float64)
    vol_ratio_hist = np.array([1.0], dtype=np.float64)
    if ret.shape[0] >= max(15, lookback):
        roll_mu = (ret.rolling(lookback).mean() * 252.0).dropna()
        roll_vol = (ret.rolling(lookback).std(ddof=0) * np.sqrt(252.0)).dropna()
        vol_short_hist = (ret.rolling(20).std(ddof=0) * np.sqrt(252.0)).dropna()
        vol_long_hist = (ret.rolling(60).std(ddof=0) * np.sqrt(252.0)).dropna()
        if not roll_mu.empty:
            mu_hist = roll_mu.to_numpy(dtype=np.float64)
        if not roll_vol.empty:
            vol_hist = roll_vol.to_numpy(dtype=np.float64)
        if not vol_short_hist.empty and not vol_long_hist.empty:
            joined_hist = pd.concat([vol_short_hist.rename("short"), vol_long_hist.rename("long")], axis=1).dropna()
            if not joined_hist.empty:
                vol_ratio_hist = (
                    joined_hist["short"] / joined_hist["long"].clip(lower=1e-9)
                ).to_numpy(dtype=np.float64)
        if ret.shape[0] > lookback:
            dd_vals = []
            ret_vals = ret.to_numpy(dtype=np.float64)
            for i in range(lookback, ret_vals.shape[0] + 1):
                seg = ret_vals[i - lookback : i]
                eq = np.cumprod(1.0 + seg)
                peak = np.maximum.accumulate(eq)
                dd = 1.0 - (eq / (peak + 1e-12))
                dd_vals.append(float(np.max(dd)))
            if dd_vals:
                dd_hist = np.asarray(dd_vals, dtype=np.float64)

    mu_rank = _percentile_rank(mu_ann, mu_hist)
    vol_rank = _percentile_rank(vol_ann, vol_hist)
    dd_rank = _percentile_rank(max_dd, dd_hist)

    trend_score = 1.35 * mu_rank + 0.35 * (1.0 - vol_rank) + 0.45 * (1.0 - dd_rank)
    mean_revert_score = (
        1.0 - min(1.0, abs(mu_rank - 0.5) * 2.2) + 0.35 * (1.0 - vol_rank) + 0.25 * (1.0 - dd_rank)
    )
    risk_shield_score = 0.95 * vol_rank + 1.05 * dd_rank + 0.2 * (1.0 - mu_rank)

    family_scores = {
        "trend_follow": float(trend_score),
        "mean_revert": float(mean_revert_score),
        "risk_shield": float(risk_shield_score),
    }
    family = max(family_scores.items(), key=lambda kv: kv[1])[0]

    short_window = int(max(5, min(20, ret.shape[0]))) if ret.shape[0] > 0 else 0
    long_window = int(max(10, min(60, ret.shape[0]))) if ret.shape[0] > 0 else 0
    short_seg = ret.tail(short_window) if short_window > 0 else pd.Series(dtype=float)
    long_seg = ret.tail(long_window) if long_window > 0 else pd.Series(dtype=float)
    vol_short_ann = float(short_seg.std(ddof=0) * np.sqrt(252.0)) if not short_seg.empty else 0.0
    vol_long_ann = float(long_seg.std(ddof=0) * np.sqrt(252.0)) if not long_seg.empty else 0.0
    vol_ratio_20_60 = float(vol_short_ann / max(vol_long_ann, 1e-9)) if vol_long_ann > 0 else 0.0
    trend_strength_20 = float(short_seg.mean() * 252.0) if not short_seg.empty else 0.0
    asset_ret_tail = asset_ret.tail(short_window) if short_window > 0 else pd.DataFrame()
    if not asset_ret_tail.empty:
        dispersion_series = asset_ret_tail.std(axis=1, ddof=0).dropna()
        dispersion_20 = float(dispersion_series.mean()) if not dispersion_series.empty else 0.0
    else:
        dispersion_20 = 0.0
    vol_ratio_rank = _percentile_rank(vol_ratio_20_60, vol_ratio_hist)
    vol_long_rank = _percentile_rank(vol_long_ann, vol_hist)
    market_stress_score = float(
        np.clip(0.4 * vol_ratio_rank + 0.35 * vol_long_rank + 0.25 * dd_rank, 0.0, 1.0)
    )

    return {
        "family": family,
        "mu_ann": mu_ann,
        "vol_ann": vol_ann,
        "max_dd": max_dd,
        "vol_short_ann": vol_short_ann,
        "vol_long_ann": vol_long_ann,
        "vol_ratio_20_60": vol_ratio_20_60,
        "trend_strength_20": trend_strength_20,
        "dispersion_20": dispersion_20,
        "market_stress_score": market_stress_score,
        "mu_rank": float(mu_rank),
        "vol_rank": float(vol_rank),
        "max_dd_rank": float(dd_rank),
        "vol_ratio_rank": float(vol_ratio_rank),
        "vol_long_rank": float(vol_long_rank),
        "family_scores": family_scores,
        "router_mode": "quantile_score",
        "router_lookback_days": int(lookback),
        "n_days": int(profile["n_days"]),
    }


def _llm_chat_with_retries(client, llm_cfg: dict, messages: list[dict], llm_errors: list, iteration: int, phase: str):
    max_retries = int(max(1, llm_cfg.get("max_retries", 3)))
    retry_base_sleep_s = float(max(0.2, llm_cfg.get("retry_base_sleep_s", 1.5)))
    retry_max_sleep_s = float(max(retry_base_sleep_s, llm_cfg.get("retry_max_sleep_s", 12.0)))
    for attempt in range(1, max_retries + 1):
        try:
            content = client.chat(
                model=llm_cfg["model"],
                messages=messages,
                temperature=float(llm_cfg.get("temperature", 0.2)),
                max_tokens=int(llm_cfg.get("max_tokens", 800)),
            )
            return content
        except Exception as e:
            llm_errors.append(
                {
                    "iteration": iteration,
                    "phase": phase,
                    "attempt": attempt,
                    "error_type": type(e).__name__,
                    "message": str(e),
                }
            )
            if attempt < max_retries:
                sleep_s = min(retry_base_sleep_s * (2 ** (attempt - 1)), retry_max_sleep_s)
                time.sleep(sleep_s)
    return None


def _sb3_metrics_from_eval(result: dict) -> tuple[dict, str]:
    episodes = result.get("values_episodes") or []
    if episodes:
        episode_metrics = [
            compute_metrics(np.array(v, dtype=float))
            for v in episodes
            if len(v) > 1
        ]
        if episode_metrics:
            keys = episode_metrics[0].keys()
            return (
                {k: float(np.mean([m[k] for m in episode_metrics])) for k in keys},
                "sb3.eval_values_episode_agg",
            )
    return compute_metrics(np.array(result.get("values", []), dtype=float)), "sb3.eval_values_path"


def _action_space_type(algo: str, drl_backend: str = "current") -> str:
    if str(drl_backend or "current").strip().lower() == "finsaber_native":
        return "continuous"
    algo = algo.lower()
    if algo in {"a2c", "ppo"}:
        return "discrete"
    return "continuous"


def _resolve_finsaber_algo_kwargs(cfg: DemoConfig, algo: str) -> dict:
    algo_key = str(algo).lower()
    payload = dict(((cfg.algo_tuning or {}).get(algo_key, {}) or {}))
    if algo_key == "td3":
        if "policy_noise" in payload and "target_policy_noise" not in payload:
            payload["target_policy_noise"] = payload.pop("policy_noise")
        if "noise_clip" in payload and "target_noise_clip" not in payload:
            payload["target_noise_clip"] = payload.pop("noise_clip")
        if "policy_freq" in payload and "policy_delay" not in payload:
            payload["policy_delay"] = payload.pop("policy_freq")
        payload.pop("actor_max_action", None)
    return payload


def _aggregate_metric_summary(metric_rows: list[dict], *, bootstrap_cfg: dict, algo: str, group_name: str) -> dict:
    if not metric_rows:
        zero = {"mean": 0.0, "std": 0.0}
        return {name: dict(zero) for name in ["AR", "CR", "AV", "MDD", "Sharpe", "Sortino"]}
    summary: dict[str, dict] = {}
    for metric_name in metric_rows[0].keys():
        vals = [float(row.get(metric_name, 0.0)) for row in metric_rows]
        summary[metric_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        if bootstrap_cfg["enabled"] and metric_name in {"Sharpe", "CR"}:
            bs = bootstrap_mean_ci(
                vals,
                n_resamples=bootstrap_cfg["n_resamples"],
                alpha=bootstrap_cfg["alpha"],
                random_seed=_stable_seed(bootstrap_cfg["random_seed"], f"{algo}:{group_name}:{metric_name}"),
            )
            summary[metric_name]["bootstrap"] = {
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
                "n_resamples": bs["n_resamples"],
                "alpha": bs["alpha"],
            }
    return summary


def _write_compat_run_summary(run_dir: Path, rows: list[str]) -> str:
    path = run_dir / "run_summary.md"
    path.write_text("\n".join(rows).strip() + "\n", encoding="utf-8")
    return str(path)


def _resolve_finsaber_compat_cfg(cfg: DemoConfig, total_timesteps: int) -> tuple[FinsaberCompatConfig, dict]:
    execution_cfg = cfg.execution or {}
    compat_raw = dict(execution_cfg.get("finsaber_compat", {}) or {})
    env_raw = dict(compat_raw.get("env", {}) or {})
    model_kwargs_by_algo = {
        str(algo).lower(): dict(payload or {})
        for algo, payload in dict(compat_raw.get("model_kwargs_by_algo", {}) or {}).items()
    }
    indicators = compat_raw.get("tech_indicator_list", env_raw.get("tech_indicator_list"))
    indicator_list = list(indicators) if isinstance(indicators, list) else None
    compat_cfg = FinsaberCompatConfig(
        total_timesteps=int(total_timesteps),
        initial_amount=float(env_raw.get("initial_amount", compat_raw.get("initial_amount", cfg.initial_cash))),
        hmax=int(env_raw.get("hmax", compat_raw.get("hmax", cfg.max_trade))),
        buy_cost_pct=float(env_raw.get("buy_cost_pct", compat_raw.get("buy_cost_pct", cfg.fee_rate))),
        sell_cost_pct=float(env_raw.get("sell_cost_pct", compat_raw.get("sell_cost_pct", cfg.fee_rate))),
        reward_scaling=float(env_raw.get("reward_scaling", compat_raw.get("reward_scaling", 1e-4))),
        tech_indicator_list=indicator_list,
        use_turbulence=_coerce_bool(compat_raw.get("use_turbulence", True), True),
        use_vix=_coerce_bool(compat_raw.get("use_vix", False), False),
        user_defined_feature=_coerce_bool(compat_raw.get("user_defined_feature", False), False),
        deterministic_eval=_coerce_bool(compat_raw.get("deterministic_eval", True), True),
        eval_episodes=int(max(1, compat_raw.get("eval_episodes", 1))),
    )
    compat_summary = {
        "total_timesteps": int(compat_cfg.total_timesteps),
        "initial_amount": float(compat_cfg.initial_amount),
        "hmax": int(compat_cfg.hmax),
        "buy_cost_pct": float(compat_cfg.buy_cost_pct),
        "sell_cost_pct": float(compat_cfg.sell_cost_pct),
        "reward_scaling": float(compat_cfg.reward_scaling),
        "tech_indicator_list": list(compat_cfg.tech_indicator_list or load_compat_finrl_indicators()),
        "use_turbulence": bool(compat_cfg.use_turbulence),
        "use_vix": bool(compat_cfg.use_vix),
        "user_defined_feature": bool(compat_cfg.user_defined_feature),
        "deterministic_eval": bool(compat_cfg.deterministic_eval),
        "eval_episodes": int(compat_cfg.eval_episodes),
        "model_kwargs_by_algo": _json_safe(model_kwargs_by_algo),
    }
    return compat_cfg, compat_summary


def _resolve_finsaber_native_cfg(cfg: DemoConfig, total_timesteps: int) -> tuple[FinsaberNativeConfig, dict]:
    execution_cfg = cfg.execution or {}
    native_raw = dict(execution_cfg.get("finsaber_native", {}) or {})
    env_raw = dict(native_raw.get("env", {}) or {})
    model_kwargs_by_algo = {
        str(algo).lower(): dict(payload or {})
        for algo, payload in dict(native_raw.get("model_kwargs_by_algo", {}) or {}).items()
    }
    indicators = native_raw.get("tech_indicator_list", env_raw.get("tech_indicator_list"))
    indicator_list = list(indicators) if isinstance(indicators, list) else None
    native_cfg = FinsaberNativeConfig(
        total_timesteps=int(total_timesteps),
        initial_amount=float(env_raw.get("initial_amount", native_raw.get("initial_amount", cfg.initial_cash))),
        hmax=int(env_raw.get("hmax", native_raw.get("hmax", cfg.max_trade))),
        buy_cost_pct=float(env_raw.get("buy_cost_pct", native_raw.get("buy_cost_pct", 0.0049))),
        sell_cost_pct=float(env_raw.get("sell_cost_pct", native_raw.get("sell_cost_pct", 0.0049))),
        reward_scaling=float(env_raw.get("reward_scaling", native_raw.get("reward_scaling", 1e-4))),
        tech_indicator_list=indicator_list,
        use_turbulence=_coerce_bool(native_raw.get("use_turbulence", True), True),
        use_vix=_coerce_bool(native_raw.get("use_vix", False), False),
        user_defined_feature=_coerce_bool(native_raw.get("user_defined_feature", False), False),
        deterministic_eval=_coerce_bool(native_raw.get("deterministic_eval", True), True),
        eval_episodes=int(max(1, native_raw.get("eval_episodes", 1))),
        print_verbosity=int(max(1, native_raw.get("print_verbosity", 10))),
    )
    native_summary = {
        "total_timesteps": int(native_cfg.total_timesteps),
        "initial_amount": float(native_cfg.initial_amount),
        "hmax": int(native_cfg.hmax),
        "buy_cost_pct": float(native_cfg.buy_cost_pct),
        "sell_cost_pct": float(native_cfg.sell_cost_pct),
        "reward_scaling": float(native_cfg.reward_scaling),
        "tech_indicator_list": list(native_cfg.tech_indicator_list or load_native_finrl_indicators()),
        "use_turbulence": bool(native_cfg.use_turbulence),
        "use_vix": bool(native_cfg.use_vix),
        "user_defined_feature": bool(native_cfg.user_defined_feature),
        "deterministic_eval": bool(native_cfg.deterministic_eval),
        "eval_episodes": int(native_cfg.eval_episodes),
        "print_verbosity": int(native_cfg.print_verbosity),
        "model_kwargs_by_algo": _json_safe(model_kwargs_by_algo),
        "migration_style": "finsaber_original_classes_with_minimal_patches",
        "original_class_names": {
            "FeatureEngineer": "FeatureEngineer",
            "DRLAgent": "DRLAgent",
            "StockTradingEnv": "StockTradingEnv",
            "FinRLStrategy": "FinRLStrategy",
        },
    }
    return native_cfg, native_summary


def _resolve_finsaber_native_state_contract(
    cfg: DemoConfig,
    assets: list[str],
    reference_df: pd.DataFrame | None = None,
):
    execution_cfg = cfg.execution or {}
    native_raw = dict(execution_cfg.get("finsaber_native", {}) or {})
    env_raw = dict(native_raw.get("env", {}) or {})
    indicators = native_raw.get("tech_indicator_list", env_raw.get("tech_indicator_list"))
    indicator_list = (
        [str(item) for item in indicators]
        if isinstance(indicators, list)
        else list(load_native_finrl_indicators())
    )
    ordered_assets = [str(asset) for asset in assets]
    if reference_df is not None and not reference_df.empty:
        asset_col = "asset" if "asset" in reference_df.columns else ("tic" if "tic" in reference_df.columns else None)
        if asset_col is not None:
            ref = reference_df.copy()
            if "date" in ref.columns:
                ref["date"] = pd.to_datetime(ref["date"])
                ref = ref.sort_values(["date"]).reset_index(drop=True)
                first_date = ref["date"].iloc[0]
                first_frame = ref[ref["date"] == first_date]
            else:
                first_frame = ref
            if not first_frame.empty:
                seen: set[str] = set()
                ordered_assets = []
                for item in first_frame[asset_col].tolist():
                    asset_name = str(item)
                    if asset_name in seen:
                        continue
                    seen.add(asset_name)
                    ordered_assets.append(asset_name)
            else:
                ordered_assets = [str(item) for item in reference_df[asset_col].dropna().unique().tolist()]
    contract = build_finsaber_native_state_contract(ordered_assets, indicator_list)
    return contract, contract.summary()


def _run_finsaber_native_demo(
    *,
    cfg: DemoConfig,
    run_dir: Path,
    raw_path: Path,
    processed_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_meta: dict,
    split_date_filter_summary: dict,
    selected_assets: list[str],
    universe_snapshot: dict,
    scenario_profile_path: Path,
) -> dict:
    eval_algos = cfg.eval_algorithms or [cfg.algorithm]
    if cfg.groups and set(cfg.groups) != {"G0_baseline"}:
        raise ValueError("finsaber_native phase-1 currently supports only groups: ['G0_baseline']")

    ensure_dir(run_dir)
    cand_dir = run_dir / "revision_candidates"
    ensure_dir(cand_dir)
    llm_errors: list[dict] = []
    iter_trace: list[dict] = []
    (run_dir / "llm_errors.json").write_text(json.dumps(llm_errors, indent=2))

    experiment_cfg = _resolve_experiment_cfg(cfg)
    bootstrap_cfg = _resolve_bootstrap_cfg(cfg.bootstrap)
    config_fingerprint = _hash_payload(cfg.__dict__)
    decision_ts_rule = _resolve_decision_rule(cfg)
    scenario_profile = json.loads(scenario_profile_path.read_text(encoding="utf-8"))
    eval_history_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    native_timesteps = _effective_steps(cfg.n_full, int(train_df["date"].nunique()))
    native_cfg, native_summary = _resolve_finsaber_native_cfg(cfg, native_timesteps)
    processed_train_for_contract, _ = preprocess_native_data(
        format_native_raw_for_fe(train_df),
        tech_indicator_list=list(native_cfg.tech_indicator_list or load_native_finrl_indicators()),
        use_vix=bool(native_cfg.use_vix),
        use_turbulence=bool(native_cfg.use_turbulence),
        user_defined_feature=bool(native_cfg.user_defined_feature),
    )
    native_contract, native_contract_summary = _resolve_finsaber_native_state_contract(
        cfg,
        selected_assets,
        reference_df=processed_train_for_contract,
    )
    native_reference_states = collect_finsaber_native_reference_states(
        processed_train_for_contract,
        contract=native_contract,
        initial_cash=float(native_cfg.initial_amount),
        max_samples=None,
    )
    model_kwargs_by_algo = dict(native_summary.get("model_kwargs_by_algo", {}) or {})
    candidate_scoring_effective = {"enabled": False, "reason": "finsaber_native_g0_only"}
    best_candidate_by_algo = {str(algo): "" for algo in eval_algos}
    candidate_fingerprint_by_algo = {str(algo): {} for algo in eval_algos}

    groups = {"G0_baseline": dict(use_revised=False, use_intrinsic=False)}
    reward_trace: Dict[str, Dict[str, List[dict]]] = {}
    td3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    sb3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    metrics_source_map: Dict[str, Dict[str, str]] = {}
    algo_results: Dict[str, dict] = {}

    for algo in eval_algos:
        reward_trace[algo] = {"G0_baseline": []}
        metrics_source_map[algo] = {}
        if str(algo).lower() == "td3":
            td3_seed_trace[algo] = {"G0_baseline": []}
        else:
            sb3_seed_trace[algo] = {"G0_baseline": []}

        per_seed = []
        metrics_source = "unknown"
        algo_kwargs = model_kwargs_by_algo.get(str(algo).lower(), {})
        for sd in cfg.seeds:
            result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=test_df,
                eval_history_df=eval_history_df,
                cfg=native_cfg,
                seed=int(sd),
                algo_kwargs=algo_kwargs,
            )
            metrics, metrics_source = _sb3_metrics_from_eval(result)
            intrinsic_vals = result.get("intrinsic", []) or []
            reward_env_vals = result.get("reward_env", []) or []
            action_penalty_vals = result.get("action_penalty", []) or []
            reward_total_vals = result.get("reward_total", []) or []
            reward_trace[algo]["G0_baseline"].append(
                {
                    "seed": int(sd),
                    "reward_env": _reward_stats(reward_env_vals),
                    "action_penalty": _reward_stats(action_penalty_vals),
                    "reward_total": _reward_stats(reward_total_vals),
                    "intrinsic": _reward_stats(intrinsic_vals),
                    "intrinsic_w_effective": 0.0,
                    "intrinsic_effect_ratio": _reward_stats([0.0 for _ in intrinsic_vals]),
                    "intrinsic_effect_ratio_robust": _reward_stats([0.0 for _ in intrinsic_vals]),
                    "env_near_zero_ratio": 0.0,
                    "reward_total_minus_env": _reward_stats(
                        [
                            float(rt) - float(re)
                            for rt, re in zip(reward_total_vals[: len(reward_env_vals)], reward_env_vals)
                        ]
                    ),
                }
            )
            per_seed.append(
                {
                    "seed": int(sd),
                    "metrics": metrics,
                    "intrinsic": {
                        "mean": 0.0,
                        "std": 0.0,
                        "count": int(len(intrinsic_vals)),
                    },
                    "intrinsic_w_effective": 0.0,
                    "metrics_source": metrics_source,
                }
            )
            if str(algo).lower() == "td3":
                td3_seed_trace[algo]["G0_baseline"].append(_td3_seed_trace_from_sb3_result(result, int(sd)))
            else:
                sb3_seed_trace[algo]["G0_baseline"].append(_sb3_seed_trace_from_result(result, int(sd)))

        agg = {}
        for metric_name in per_seed[0]["metrics"].keys():
            vals = [float(row["metrics"][metric_name]) for row in per_seed]
            agg[metric_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            if bootstrap_cfg["enabled"] and metric_name in {"Sharpe", "CR"}:
                bs = bootstrap_mean_ci(
                    vals,
                    n_resamples=bootstrap_cfg["n_resamples"],
                    alpha=bootstrap_cfg["alpha"],
                    random_seed=_stable_seed(bootstrap_cfg["random_seed"], f"native:{algo}:G0_baseline:{metric_name}"),
                )
                agg[metric_name]["bootstrap"] = {
                    "ci_low": bs["ci_low"],
                    "ci_high": bs["ci_high"],
                    "n_resamples": bs["n_resamples"],
                    "alpha": bs["alpha"],
                }
        algo_results[algo] = {"groups": {"G0_baseline": {"per_seed": per_seed, "summary": agg}}, "metrics_source": {"G0_baseline": metrics_source}}
        metrics_source_map[algo]["G0_baseline"] = metrics_source

    td3_g1_g3_diff = {}
    td3_diff_summary = {}
    for algo, algo_trace in td3_seed_trace.items():
        diff_payload = _build_td3_g1_g3_diff(algo_trace)
        td3_g1_g3_diff[algo] = diff_payload
        td3_diff_summary[algo] = {
            "diagnosis": diff_payload.get("diagnosis"),
            "seed_count": diff_payload.get("summary", {}).get("seed_count", 0),
            "action_equal_ratio_mean": diff_payload.get("summary", {}).get("action_equal_ratio_mean", 0.0),
            "eval_value_mae_mean": diff_payload.get("summary", {}).get("eval_value_mae_mean", 0.0),
            "eval_reward_total_delta_mean": diff_payload.get("summary", {}).get("eval_reward_total_delta_mean", 0.0),
        }

    td3_action_saturation_summary = _build_td3_action_saturation_summary(
        td3_seed_trace,
        action_bound=float(native_cfg.hmax),
        collapse_threshold=0.95,
    )
    policy_behavior_summary = _build_policy_behavior_summary(
        td3_seed_trace=td3_seed_trace,
        sb3_seed_trace=sb3_seed_trace,
        action_bound=float(native_cfg.hmax),
        td3_action_bound=float(native_cfg.hmax),
        collapse_threshold=0.95,
    )
    actor_collapse_detected = bool(
        td3_action_saturation_summary.get("td3", {}).get("_overall", {}).get("actor_collapse_detected", False)
    )

    state_scale_summary = {
        "backend": "finsaber_native",
        "selected_assets": list(selected_assets),
        "tech_indicator_list": list(native_summary.get("tech_indicator_list", [])),
        "state_space_semantics": "finsaber_original",
        "action_space_semantics": "finsaber_original_continuous_hmax_integer_shares",
        "reward_semantics": "finsaber_original_delta_asset_scaled",
        "reference_sample_count": int(native_reference_states.shape[0]),
        "original_class_names": native_summary.get("original_class_names", {}),
        "state_contract_summary": native_contract_summary,
    }

    results = {
        "algorithms": algo_results,
        "protocol": {
            "eval_protocol": split_meta["protocol"],
            "split": split_meta,
            "selected_assets": list(selected_assets),
            "window_setup": cfg.window_setup,
            "universe_snapshot": universe_snapshot,
            "decision_ts_rule": decision_ts_rule,
            "action_quantization_mode": "finsaber_original_continuous_hmax",
            "discrete_action_levels": 0,
            "drl_backend": "finsaber_native",
            "groups": list(groups.keys()),
            "metrics_source": metrics_source_map,
            "scoring_objective": "g0_baseline_only",
            "candidate_scoring_effective": candidate_scoring_effective,
            "best_candidate_by_algo": best_candidate_by_algo,
            "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
            "scenario_profile": scenario_profile,
            "finsaber_native_effective": native_summary,
            "finsaber_native_state_contract": native_contract_summary,
            "action_space_type_by_algo": {algo: "continuous" for algo in eval_algos},
            "policy_behavior_summary": {
                algo: payload.get("_overall", {})
                for algo, payload in policy_behavior_summary.items()
            },
            "td3_diff_summary": td3_diff_summary,
            "td3_action_saturation_summary": td3_action_saturation_summary,
            "actor_collapse_detected": actor_collapse_detected,
            "evaluation": cfg.evaluation or {},
            "walk_forward": cfg.walk_forward or {"enabled": False},
            "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
            "split_date_filter_summary": _json_safe(split_date_filter_summary),
            "experiment_phase": experiment_cfg["phase"],
            "experiment_frozen": bool(experiment_cfg["frozen"]),
            "claim_id": experiment_cfg["claim_id"],
            "hypothesis_id": experiment_cfg["hypothesis_id"],
        },
    }
    if len(eval_algos) == 1:
        results["groups"] = algo_results[eval_algos[0]]["groups"]

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2))
    state_scale_path = run_dir / "state_scale_summary.json"
    state_scale_path.write_text(json.dumps(state_scale_summary, indent=2))
    td3_trace_path = run_dir / "td3_seed_trace.json"
    sb3_trace_path = run_dir / "sb3_action_trace.json"
    td3_diff_path = run_dir / "td3_g1_g3_diff.json"
    td3_sat_path = run_dir / "td3_action_saturation.json"
    policy_behavior_path = run_dir / "policy_behavior_summary.json"
    td3_trace_path.write_text(json.dumps(td3_seed_trace, indent=2))
    sb3_trace_path.write_text(json.dumps(sb3_seed_trace, indent=2))
    td3_diff_path.write_text(json.dumps(td3_g1_g3_diff, indent=2))
    td3_sat_path.write_text(json.dumps(td3_action_saturation_summary, indent=2))
    policy_behavior_path.write_text(json.dumps(policy_behavior_summary, indent=2))

    run_manifest = {
        "protocol_version": "trading-lesr-v2",
        "eval_protocol": split_meta["protocol"],
        "split": split_meta,
        "algorithm": cfg.algorithm,
        "eval_algorithms": eval_algos,
        "groups": list(groups.keys()),
        "config_fingerprint": config_fingerprint,
        "experiment_phase": experiment_cfg["phase"],
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
        "is_confirmatory": _is_confirmatory(experiment_cfg),
        "experiment_frozen": bool(experiment_cfg["frozen"]),
        "selected_assets": list(selected_assets),
        "window_setup": cfg.window_setup,
        "universe_snapshot": universe_snapshot,
        "decision_ts_rule": decision_ts_rule,
        "action_quantization_mode": "finsaber_original_continuous_hmax",
        "discrete_action_levels": 0,
        "drl_backend": "finsaber_native",
        "metrics_source": metrics_source_map,
        "scoring_objective": "g0_baseline_only",
        "candidate_scoring_effective": candidate_scoring_effective,
        "best_candidate_by_algo": best_candidate_by_algo,
        "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
        "scenario_profile": scenario_profile,
        "finsaber_native_effective": native_summary,
        "finsaber_native_state_contract": native_contract_summary,
        "action_space_type_by_algo": {algo: "continuous" for algo in eval_algos},
        "td3_diff_summary": td3_diff_summary,
        "td3_action_saturation_summary": td3_action_saturation_summary,
        "policy_behavior_summary": {
            algo: payload.get("_overall", {})
            for algo, payload in policy_behavior_summary.items()
        },
        "actor_collapse_detected": actor_collapse_detected,
        "evaluation": cfg.evaluation or {},
        "walk_forward": cfg.walk_forward or {"enabled": False},
        "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
        "split_date_filter_summary": _json_safe(split_date_filter_summary),
    }
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    metrics_table_path = _write_metrics_table(run_dir, results)
    summary_path = _write_run_summary(run_dir, cfg, results, iter_trace, llm_errors, split_meta)

    required_files = [
        Path(metrics_table_path),
        run_manifest_path,
        run_dir / "reward_trace.json",
        state_scale_path,
        policy_behavior_path,
        scenario_profile_path,
    ]
    if "td3" in [str(a).lower() for a in eval_algos]:
        required_files.append(td3_diff_path)
        required_files.append(td3_sat_path)
    if any(str(a).lower() != "td3" for a in eval_algos):
        required_files.append(sb3_trace_path)
    run_manifest["completeness_check"] = _build_completeness_check(required_files)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    root = repo_root()
    artifacts = {
        "raw_data": str(raw_path.relative_to(root)),
        "processed_data": str(processed_path.relative_to(root)),
        "system_prompt": str((run_dir / "system_prompt.txt").relative_to(root)),
        "prompt": str((run_dir / "prompt.txt").relative_to(root)),
        "metrics": str((run_dir / "metrics.json").relative_to(root)),
        "reward_trace": str((run_dir / "reward_trace.json").relative_to(root)),
        "state_scale_summary": str(state_scale_path.relative_to(root)),
        "td3_seed_trace": str(td3_trace_path.relative_to(root)),
        "sb3_action_trace": str(sb3_trace_path.relative_to(root)),
        "td3_g1_g3_diff": str(td3_diff_path.relative_to(root)),
        "td3_action_saturation": str(td3_sat_path.relative_to(root)),
        "policy_behavior_summary": str(policy_behavior_path.relative_to(root)),
        "scenario_profile": str(scenario_profile_path.relative_to(root)),
        "revision_candidates": str(cand_dir.relative_to(root)),
        "metrics_table": str(Path(metrics_table_path).relative_to(root)),
        "run_summary": str(Path(summary_path).relative_to(root)),
        "llm_errors": str((run_dir / "llm_errors.json").relative_to(root)),
        "run_manifest": str(run_manifest_path.relative_to(root)),
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

    hashes = {
        "raw_data": sha256_file(raw_path),
        "processed_data": sha256_file(processed_path),
        "system_prompt": sha256_file(run_dir / "system_prompt.txt"),
        "prompt": sha256_file(run_dir / "prompt.txt"),
        "metrics": sha256_file(run_dir / "metrics.json"),
        "reward_trace": sha256_file(run_dir / "reward_trace.json"),
        "state_scale_summary": sha256_file(state_scale_path),
        "td3_seed_trace": sha256_file(td3_trace_path),
        "sb3_action_trace": sha256_file(sb3_trace_path),
        "td3_g1_g3_diff": sha256_file(td3_diff_path),
        "td3_action_saturation": sha256_file(td3_sat_path),
        "policy_behavior_summary": sha256_file(policy_behavior_path),
        "scenario_profile": sha256_file(scenario_profile_path),
        "metrics_table": sha256_file(Path(metrics_table_path)),
        "run_summary": sha256_file(Path(summary_path)),
        "llm_errors": sha256_file(run_dir / "llm_errors.json"),
        "run_manifest": sha256_file(run_manifest_path),
    }
    (run_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))
    return results


def _run_finsaber_compat_demo(
    *,
    cfg: DemoConfig,
    run_dir: Path,
    raw_path: Path,
    processed_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_meta: dict,
    split_date_filter_summary: dict,
    selected_assets: list[str],
    universe_snapshot: dict,
    scenario_profile_path: Path,
) -> dict:
    eval_algos = cfg.eval_algorithms or [cfg.algorithm]
    if cfg.groups and set(cfg.groups) != {"G0_baseline"}:
        raise ValueError("finsaber_compat phase-1 currently supports only groups: ['G0_baseline']")

    ensure_dir(run_dir)
    cand_dir = run_dir / "revision_candidates"
    ensure_dir(cand_dir)
    llm_errors: list[dict] = []
    iter_trace: list[dict] = []
    (run_dir / "llm_errors.json").write_text(json.dumps(llm_errors, indent=2))

    experiment_cfg = _resolve_experiment_cfg(cfg)
    bootstrap_cfg = _resolve_bootstrap_cfg(cfg.bootstrap)
    config_fingerprint = _hash_payload(cfg.__dict__)
    decision_ts_rule = _resolve_decision_rule(cfg)
    scenario_profile = json.loads(scenario_profile_path.read_text(encoding="utf-8"))
    eval_history_df = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    compat_timesteps = _effective_steps(cfg.n_full, int(train_df["date"].nunique()))
    compat_cfg, compat_summary = _resolve_finsaber_compat_cfg(cfg, compat_timesteps)
    model_kwargs_by_algo = dict(compat_summary.get("model_kwargs_by_algo", {}) or {})
    candidate_scoring_effective = {"enabled": False, "reason": "finsaber_compat_g0_only"}
    best_candidate_by_algo = {str(algo): "" for algo in eval_algos}
    candidate_fingerprint_by_algo = {str(algo): {} for algo in eval_algos}

    groups = {"G0_baseline": dict(use_revised=False, use_intrinsic=False)}
    reward_trace: Dict[str, Dict[str, List[dict]]] = {}
    td3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    sb3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    metrics_source_map: Dict[str, Dict[str, str]] = {}
    algo_results: Dict[str, dict] = {}

    for algo in eval_algos:
        reward_trace[algo] = {"G0_baseline": []}
        metrics_source_map[algo] = {}
        if str(algo).lower() == "td3":
            td3_seed_trace[algo] = {"G0_baseline": []}
        else:
            sb3_seed_trace[algo] = {"G0_baseline": []}

        per_seed = []
        metrics_source = "unknown"
        algo_kwargs = model_kwargs_by_algo.get(str(algo).lower(), {})
        for sd in cfg.seeds:
            result = train_finsaber_compat(
                algo=algo,
                train_df=train_df,
                eval_df=test_df,
                eval_history_df=eval_history_df,
                cfg=compat_cfg,
                seed=int(sd),
                algo_kwargs=algo_kwargs,
            )
            metrics, metrics_source = _sb3_metrics_from_eval(result)
            intrinsic_vals = result.get("intrinsic", []) or []
            reward_env_vals = result.get("reward_env", []) or []
            action_penalty_vals = result.get("action_penalty", []) or []
            reward_total_vals = result.get("reward_total", []) or []
            reward_trace[algo]["G0_baseline"].append(
                {
                    "seed": int(sd),
                    "reward_env": _reward_stats(reward_env_vals),
                    "action_penalty": _reward_stats(action_penalty_vals),
                    "reward_total": _reward_stats(reward_total_vals),
                    "intrinsic": _reward_stats(intrinsic_vals),
                    "intrinsic_w_effective": 0.0,
                    "intrinsic_effect_ratio": _reward_stats([0.0 for _ in intrinsic_vals]),
                    "intrinsic_effect_ratio_robust": _reward_stats([0.0 for _ in intrinsic_vals]),
                    "env_near_zero_ratio": 0.0,
                    "reward_total_minus_env": _reward_stats(
                        [
                            float(rt) - float(re)
                            for rt, re in zip(reward_total_vals[: len(reward_env_vals)], reward_env_vals)
                        ]
                    ),
                }
            )
            per_seed.append(
                {
                    "seed": int(sd),
                    "metrics": metrics,
                    "intrinsic": {
                        "mean": 0.0,
                        "std": 0.0,
                        "count": int(len(intrinsic_vals)),
                    },
                    "intrinsic_w_effective": 0.0,
                    "metrics_source": metrics_source,
                }
            )
            if str(algo).lower() == "td3":
                td3_seed_trace[algo]["G0_baseline"].append(_td3_seed_trace_from_sb3_result(result, int(sd)))
            else:
                sb3_seed_trace[algo]["G0_baseline"].append(_sb3_seed_trace_from_result(result, int(sd)))

        agg = {}
        for metric_name in per_seed[0]["metrics"].keys():
            vals = [float(row["metrics"][metric_name]) for row in per_seed]
            agg[metric_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            if bootstrap_cfg["enabled"] and metric_name in {"Sharpe", "CR"}:
                bs = bootstrap_mean_ci(
                    vals,
                    n_resamples=bootstrap_cfg["n_resamples"],
                    alpha=bootstrap_cfg["alpha"],
                    random_seed=_stable_seed(bootstrap_cfg["random_seed"], f"compat:{algo}:G0_baseline:{metric_name}"),
                )
                agg[metric_name]["bootstrap"] = {
                    "ci_low": bs["ci_low"],
                    "ci_high": bs["ci_high"],
                    "n_resamples": bs["n_resamples"],
                    "alpha": bs["alpha"],
                }
        algo_results[algo] = {"groups": {"G0_baseline": {"per_seed": per_seed, "summary": agg}}, "metrics_source": {"G0_baseline": metrics_source}}
        metrics_source_map[algo]["G0_baseline"] = metrics_source

    td3_g1_g3_diff = {}
    td3_diff_summary = {}
    for algo, algo_trace in td3_seed_trace.items():
        diff_payload = _build_td3_g1_g3_diff(algo_trace)
        td3_g1_g3_diff[algo] = diff_payload
        td3_diff_summary[algo] = {
            "diagnosis": diff_payload.get("diagnosis"),
            "seed_count": diff_payload.get("summary", {}).get("seed_count", 0),
            "action_equal_ratio_mean": diff_payload.get("summary", {}).get("action_equal_ratio_mean", 0.0),
            "eval_value_mae_mean": diff_payload.get("summary", {}).get("eval_value_mae_mean", 0.0),
            "eval_reward_total_delta_mean": diff_payload.get("summary", {}).get("eval_reward_total_delta_mean", 0.0),
        }

    td3_action_saturation_summary = _build_td3_action_saturation_summary(
        td3_seed_trace,
        action_bound=float(compat_cfg.hmax),
        collapse_threshold=0.95,
    )
    policy_behavior_summary = _build_policy_behavior_summary(
        td3_seed_trace=td3_seed_trace,
        sb3_seed_trace=sb3_seed_trace,
        action_bound=float(compat_cfg.hmax),
        td3_action_bound=float(compat_cfg.hmax),
        collapse_threshold=0.95,
    )
    actor_collapse_detected = bool(
        td3_action_saturation_summary.get("td3", {}).get("_overall", {}).get("actor_collapse_detected", False)
    )

    state_scale_summary = {
        "backend": "finsaber_compat",
        "selected_assets": list(selected_assets),
        "tech_indicator_list": list(compat_summary.get("tech_indicator_list", [])),
        "state_space_semantics": "finrl_style",
        "action_space_semantics": "continuous_box_hmax_to_integer_shares",
        "reward_semantics": "delta_asset_scaled",
        "reference_sample_count": int(train_df["date"].nunique()),
    }

    results = {
        "algorithms": algo_results,
        "protocol": {
            "eval_protocol": split_meta["protocol"],
            "split": split_meta,
            "selected_assets": list(selected_assets),
            "window_setup": cfg.window_setup,
            "universe_snapshot": universe_snapshot,
            "decision_ts_rule": decision_ts_rule,
            "action_quantization_mode": "finsaber_continuous_hmax",
            "discrete_action_levels": 0,
            "drl_backend": "finsaber_compat",
            "groups": list(groups.keys()),
            "metrics_source": metrics_source_map,
            "scoring_objective": "g0_baseline_only",
            "candidate_scoring_effective": candidate_scoring_effective,
            "best_candidate_by_algo": best_candidate_by_algo,
            "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
            "scenario_profile": scenario_profile,
            "finsaber_compat_effective": compat_summary,
            "action_space_type_by_algo": {algo: "continuous" for algo in eval_algos},
            "policy_behavior_summary": {
                algo: payload.get("_overall", {})
                for algo, payload in policy_behavior_summary.items()
            },
            "td3_diff_summary": td3_diff_summary,
            "td3_action_saturation_summary": td3_action_saturation_summary,
            "actor_collapse_detected": actor_collapse_detected,
            "evaluation": cfg.evaluation or {},
            "walk_forward": cfg.walk_forward or {"enabled": False},
            "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
            "split_date_filter_summary": _json_safe(split_date_filter_summary),
            "experiment_phase": experiment_cfg["phase"],
            "experiment_frozen": bool(experiment_cfg["frozen"]),
            "claim_id": experiment_cfg["claim_id"],
            "hypothesis_id": experiment_cfg["hypothesis_id"],
        },
    }
    if len(eval_algos) == 1:
        results["groups"] = algo_results[eval_algos[0]]["groups"]

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2))
    state_scale_path = run_dir / "state_scale_summary.json"
    state_scale_path.write_text(json.dumps(state_scale_summary, indent=2))
    td3_trace_path = run_dir / "td3_seed_trace.json"
    sb3_trace_path = run_dir / "sb3_action_trace.json"
    td3_diff_path = run_dir / "td3_g1_g3_diff.json"
    td3_sat_path = run_dir / "td3_action_saturation.json"
    policy_behavior_path = run_dir / "policy_behavior_summary.json"
    td3_trace_path.write_text(json.dumps(td3_seed_trace, indent=2))
    sb3_trace_path.write_text(json.dumps(sb3_seed_trace, indent=2))
    td3_diff_path.write_text(json.dumps(td3_g1_g3_diff, indent=2))
    td3_sat_path.write_text(json.dumps(td3_action_saturation_summary, indent=2))
    policy_behavior_path.write_text(json.dumps(policy_behavior_summary, indent=2))

    drl_backend = _resolve_drl_backend(cfg)
    if drl_backend == "finsaber_compat":
        walk_forward_action_quantization_mode = "finsaber_continuous_hmax"
        walk_forward_discrete_action_levels = 0
    elif drl_backend == "finsaber_native":
        walk_forward_action_quantization_mode = "finsaber_original_continuous_hmax"
        walk_forward_discrete_action_levels = 0
    else:
        walk_forward_action_quantization_mode = _resolve_action_quantization_mode(cfg)
        walk_forward_discrete_action_levels = _resolve_discrete_action_levels(cfg)

    run_manifest = {
        "protocol_version": "trading-lesr-v2",
        "eval_protocol": split_meta["protocol"],
        "split": split_meta,
        "algorithm": cfg.algorithm,
        "eval_algorithms": eval_algos,
        "groups": list(groups.keys()),
        "config_fingerprint": config_fingerprint,
        "experiment_phase": experiment_cfg["phase"],
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
        "is_confirmatory": _is_confirmatory(experiment_cfg),
        "experiment_frozen": bool(experiment_cfg["frozen"]),
        "selected_assets": list(selected_assets),
        "window_setup": cfg.window_setup,
        "universe_snapshot": universe_snapshot,
        "decision_ts_rule": decision_ts_rule,
        "action_quantization_mode": "finsaber_continuous_hmax",
        "discrete_action_levels": 0,
        "drl_backend": "finsaber_compat",
        "metrics_source": metrics_source_map,
        "scoring_objective": "g0_baseline_only",
        "candidate_scoring_effective": candidate_scoring_effective,
        "best_candidate_by_algo": best_candidate_by_algo,
        "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
        "scenario_profile": scenario_profile,
        "finsaber_compat_effective": compat_summary,
        "action_space_type_by_algo": {algo: "continuous" for algo in eval_algos},
        "td3_diff_summary": td3_diff_summary,
        "td3_action_saturation_summary": td3_action_saturation_summary,
        "policy_behavior_summary": {
            algo: payload.get("_overall", {})
            for algo, payload in policy_behavior_summary.items()
        },
        "actor_collapse_detected": actor_collapse_detected,
        "evaluation": cfg.evaluation or {},
        "walk_forward": cfg.walk_forward or {"enabled": False},
        "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
        "split_date_filter_summary": _json_safe(split_date_filter_summary),
    }
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    metrics_table_path = _write_metrics_table(run_dir, results)
    summary_path = _write_run_summary(run_dir, cfg, results, iter_trace, llm_errors, split_meta)

    required_files = [
        Path(metrics_table_path),
        run_manifest_path,
        run_dir / "reward_trace.json",
        state_scale_path,
        policy_behavior_path,
        scenario_profile_path,
    ]
    if "td3" in [str(a).lower() for a in eval_algos]:
        required_files.append(td3_diff_path)
        required_files.append(td3_sat_path)
    if any(str(a).lower() != "td3" for a in eval_algos):
        required_files.append(sb3_trace_path)
    run_manifest["completeness_check"] = _build_completeness_check(required_files)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    root = repo_root()
    artifacts = {
        "raw_data": str(raw_path.relative_to(root)),
        "processed_data": str(processed_path.relative_to(root)),
        "system_prompt": str((run_dir / "system_prompt.txt").relative_to(root)),
        "prompt": str((run_dir / "prompt.txt").relative_to(root)),
        "metrics": str((run_dir / "metrics.json").relative_to(root)),
        "reward_trace": str((run_dir / "reward_trace.json").relative_to(root)),
        "state_scale_summary": str(state_scale_path.relative_to(root)),
        "td3_seed_trace": str(td3_trace_path.relative_to(root)),
        "sb3_action_trace": str(sb3_trace_path.relative_to(root)),
        "td3_g1_g3_diff": str(td3_diff_path.relative_to(root)),
        "td3_action_saturation": str(td3_sat_path.relative_to(root)),
        "policy_behavior_summary": str(policy_behavior_path.relative_to(root)),
        "scenario_profile": str(scenario_profile_path.relative_to(root)),
        "revision_candidates": str(cand_dir.relative_to(root)),
        "metrics_table": str(Path(metrics_table_path).relative_to(root)),
        "run_summary": str(Path(summary_path).relative_to(root)),
        "llm_errors": str((run_dir / "llm_errors.json").relative_to(root)),
        "run_manifest": str(run_manifest_path.relative_to(root)),
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

    hashes = {
        "raw_data": sha256_file(raw_path),
        "processed_data": sha256_file(processed_path),
        "system_prompt": sha256_file(run_dir / "system_prompt.txt"),
        "prompt": sha256_file(run_dir / "prompt.txt"),
        "metrics": sha256_file(run_dir / "metrics.json"),
        "reward_trace": sha256_file(run_dir / "reward_trace.json"),
        "state_scale_summary": sha256_file(state_scale_path),
        "td3_seed_trace": sha256_file(td3_trace_path),
        "sb3_action_trace": sha256_file(sb3_trace_path),
        "td3_g1_g3_diff": sha256_file(td3_diff_path),
        "td3_action_saturation": sha256_file(td3_sat_path),
        "policy_behavior_summary": sha256_file(policy_behavior_path),
        "scenario_profile": sha256_file(scenario_profile_path),
        "metrics_table": sha256_file(Path(metrics_table_path)),
        "run_summary": sha256_file(Path(summary_path)),
        "llm_errors": sha256_file(run_dir / "llm_errors.json"),
        "run_manifest": sha256_file(run_manifest_path),
    }
    (run_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))
    return results


def _behavior_score_from_stats(stats: dict) -> float:
    near_bound = float(np.clip(_sanitize_float(stats.get("near_bound_ratio", 0.0)), 0.0, 1.0))
    entropy = max(0.0, _sanitize_float(stats.get("action_entropy", 0.0)))
    entropy_score = float(entropy / (1.0 + entropy))
    unique_count = max(0.0, _sanitize_float(stats.get("unique_action_count", 0.0)))
    diversity_score = float(np.clip(unique_count / 8.0, 0.0, 1.0))
    return float(0.65 * (1.0 - near_bound) + 0.2 * entropy_score + 0.15 * diversity_score)


def _continuous_behavior_guard(row: dict, algo: str) -> dict:
    algo = str(algo or "").lower()
    if algo not in {"sac", "td3"}:
        return {
            "enabled": False,
            "hard_reject": False,
            "collapse_all": False,
            "near_bound": 0.0,
            "entropy": 0.0,
            "unique": 0.0,
            "bound_margin": 1.0,
            "collapse_penalty": 0.0,
        }
    behavior = dict(row.get("behavior", {}) or {})
    seed_behavior = list(row.get("seed_behavior", []) or [])
    near_bound = float(np.clip(_sanitize_float(behavior.get("near_bound_ratio_mean", 0.0)), 0.0, 1.0))
    entropy = max(0.0, float(_sanitize_float(behavior.get("action_entropy_mean", 0.0))))
    unique = max(0.0, float(_sanitize_float(behavior.get("unique_action_count_mean", 0.0))))
    collapse_all = bool(seed_behavior) and all(bool(seed.get("actor_collapse_detected", False)) for seed in seed_behavior)
    hard_reject = bool(
        collapse_all
        and (
            near_bound >= 0.997
            or (near_bound >= 0.995 and entropy <= 0.08 and unique <= 3.0)
        )
    )
    collapse_penalty = float(max(0.0, near_bound - 0.95))
    return {
        "enabled": True,
        "hard_reject": hard_reject,
        "collapse_all": collapse_all,
        "near_bound": near_bound,
        "entropy": entropy,
        "unique": unique,
        "bound_margin": float(1.0 - near_bound),
        "collapse_penalty": collapse_penalty,
    }


def _turnover_stats_from_weight_rows(weight_rows, portfolio_weight_changes=None) -> dict:
    changes = []
    for item in portfolio_weight_changes or []:
        try:
            val = float(item)
        except Exception:
            continue
        if np.isfinite(val):
            changes.append(val)
    if changes:
        return {
            "portfolio_weight_observation_count": int(len(changes)),
            "avg_daily_portfolio_weight_change": float(np.mean(changes)),
            "max_daily_portfolio_weight_change": float(np.max(changes)),
            "turnover_stability_score": float(1.0 / (1.0 + float(np.mean(changes)))),
        }
    rows = []
    for item in weight_rows or []:
        try:
            arr = np.asarray(item, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        rows.append(arr)
    if len(rows) < 2:
        return {
            "portfolio_weight_observation_count": int(len(rows)),
            "avg_daily_portfolio_weight_change": 0.0,
            "max_daily_portfolio_weight_change": 0.0,
            "turnover_stability_score": 0.0,
        }
    changes = []
    for idx in range(1, len(rows)):
        n = int(min(rows[idx - 1].shape[0], rows[idx].shape[0]))
        if n <= 0:
            continue
        changes.append(float(np.sum(np.abs(rows[idx][:n] - rows[idx - 1][:n]))))
    if not changes:
        return {
            "portfolio_weight_observation_count": int(len(rows)),
            "avg_daily_portfolio_weight_change": 0.0,
            "max_daily_portfolio_weight_change": 0.0,
            "turnover_stability_score": 0.0,
        }
    avg_change = float(np.mean(changes))
    max_change = float(np.max(changes))
    return {
        "portfolio_weight_observation_count": int(len(rows)),
        "avg_daily_portfolio_weight_change": avg_change,
        "max_daily_portfolio_weight_change": max_change,
        "turnover_stability_score": float(1.0 / (1.0 + avg_change)),
    }


def _candidate_behavior_payload(
    actions,
    action_bound: float,
    collapse_threshold: float = 0.95,
    portfolio_weights=None,
    portfolio_weight_changes=None,
) -> dict:
    arr = np.array(actions, dtype=float) if actions is not None else np.zeros((0, 1), dtype=float)
    stats = _action_behavior_stats(
        arr,
        action_bound=float(action_bound),
        collapse_threshold=float(collapse_threshold),
    )
    turnover_stats = _turnover_stats_from_weight_rows(portfolio_weights, portfolio_weight_changes)
    return {
        **stats,
        **turnover_stats,
        "behavior_score": _behavior_score_from_stats(stats),
    }


def _aggregate_candidate_behavior(seed_payloads: List[dict]) -> dict:
    if not seed_payloads:
        return {
            "seed_count": 0,
            "near_bound_ratio_mean": 0.0,
            "action_entropy_mean": 0.0,
            "unique_action_count_mean": 0.0,
            "avg_daily_portfolio_weight_change_mean": 0.0,
            "max_daily_portfolio_weight_change_mean": 0.0,
            "turnover_stability_score": 0.0,
            "behavior_score": 0.0,
        }

    def _mean(key: str) -> float:
        return float(np.mean([_sanitize_float(row.get(key, 0.0)) for row in seed_payloads]))

    return {
        "seed_count": int(len(seed_payloads)),
        "near_bound_ratio_mean": _mean("near_bound_ratio"),
        "action_entropy_mean": _mean("action_entropy"),
        "unique_action_count_mean": _mean("unique_action_count"),
        "avg_daily_portfolio_weight_change_mean": _mean("avg_daily_portfolio_weight_change"),
        "max_daily_portfolio_weight_change_mean": _mean("max_daily_portfolio_weight_change"),
        "turnover_stability_score": _mean("turnover_stability_score"),
        "behavior_score": _mean("behavior_score"),
    }


def _write_metrics_table(run_dir: Path, results: dict) -> str:
    path = run_dir / "metrics_table.csv"
    algos = results.get("algorithms")
    if algos is None:
        groups_payload = results.get("groups", {})
        metrics_source = results.get("protocol", {}).get("metrics_source", {})
        if isinstance(metrics_source, dict) and metrics_source:
            # Single-algo runs keep only `groups`; infer the algorithm label from protocol metadata.
            if len(metrics_source) == 1:
                algo_name = next(iter(metrics_source.keys()))
                source_map = metrics_source.get(algo_name, {})
                algos = {
                    algo_name: {
                        "groups": groups_payload,
                        "metrics_source": source_map if isinstance(source_map, dict) else {},
                    }
                }
            else:
                algos = {}
                for algo_name, source_map in metrics_source.items():
                    algos[algo_name] = {
                        "groups": groups_payload,
                        "metrics_source": source_map if isinstance(source_map, dict) else {},
                    }
        else:
            algos = {"td3": {"groups": groups_payload, "metrics_source": {}}}
    rows = []
    for algo, info in algos.items():
        groups = info.get("groups", {})
        for gname, ginfo in groups.items():
            summary = ginfo["summary"]
            per_seed = ginfo.get("per_seed", [])
            source_map = info.get("metrics_source", {})
            metrics_source = source_map.get(gname, "")
            intrinsic_means = [p.get("intrinsic", {}).get("mean", 0.0) for p in per_seed]
            intrinsic_mean = float(np.mean(intrinsic_means)) if intrinsic_means else 0.0
            intrinsic_std = float(np.std(intrinsic_means)) if intrinsic_means else 0.0
            intrinsic_w_vals = [p.get("intrinsic_w_effective", 0.0) for p in per_seed]
            intrinsic_w_mean = float(np.mean(intrinsic_w_vals)) if intrinsic_w_vals else 0.0
            rows.append(
                {
                    "algorithm": algo,
                    "group": gname,
                    "metrics_source": metrics_source,
                    "Sharpe_mean": summary["Sharpe"]["mean"],
                    "Sharpe_std": summary["Sharpe"]["std"],
                    "Sharpe_ci_low": summary["Sharpe"].get("bootstrap", {}).get("ci_low", ""),
                    "Sharpe_ci_high": summary["Sharpe"].get("bootstrap", {}).get("ci_high", ""),
                    "CR_mean": summary["CR"]["mean"],
                    "CR_std": summary["CR"]["std"],
                    "CR_ci_low": summary["CR"].get("bootstrap", {}).get("ci_low", ""),
                    "CR_ci_high": summary["CR"].get("bootstrap", {}).get("ci_high", ""),
                    "MDD_mean": summary["MDD"]["mean"],
                    "MDD_std": summary["MDD"]["std"],
                    "AV_mean": summary["AV"]["mean"],
                    "AV_std": summary["AV"]["std"],
                    "intrinsic_mean": intrinsic_mean,
                    "intrinsic_std": intrinsic_std,
                    "intrinsic_w_effective_mean": intrinsic_w_mean,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "group",
                "metrics_source",
                "Sharpe_mean",
                "Sharpe_std",
                "Sharpe_ci_low",
                "Sharpe_ci_high",
                "CR_mean",
                "CR_std",
                "CR_ci_low",
                "CR_ci_high",
                "MDD_mean",
                "MDD_std",
                "AV_mean",
                "AV_std",
                "intrinsic_mean",
                "intrinsic_std",
                "intrinsic_w_effective_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def _reward_stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "count": int(arr.size)}


def _sanitize_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _resolve_intrinsic_postprocess_cfg(cfg: dict | None) -> dict:
    cfg = cfg or {}
    mode = str(cfg.get("mode", "raw")).lower()
    if mode not in {"raw", "centered", "zscore_tanh"}:
        mode = "raw"
    eps = float(cfg.get("eps", 1e-6))
    if eps <= 0:
        eps = 1e-6
    return {"mode": mode, "eps": eps}


def _resolve_diagnostics_cfg(cfg: dict | None) -> dict:
    cfg = cfg or {}
    floor = float(cfg.get("robust_ratio_floor", 1.0))
    if floor <= 0:
        floor = 1.0
    return {
        "dump_sb3_action_trace": bool(cfg.get("dump_sb3_action_trace", True)),
        "robust_ratio_floor": floor,
    }


def _robust_intrinsic_ratio_vals(
    intrinsic_vals: List[float],
    reward_env_vals: List[float],
    action_penalty_vals: List[float] | None,
    intrinsic_w: float,
    floor: float,
) -> tuple[List[float], float]:
    n = min(len(intrinsic_vals), len(reward_env_vals))
    if action_penalty_vals is not None:
        n = min(n, len(action_penalty_vals))
    if n == 0:
        return [], 0.0
    ratios: List[float] = []
    near_zero = 0
    for i in range(n):
        intr = _sanitize_float(intrinsic_vals[i])
        env = _sanitize_float(reward_env_vals[i])
        penalty = _sanitize_float(action_penalty_vals[i]) if action_penalty_vals is not None else 0.0
        den = max(abs(env) + abs(penalty), floor)
        if abs(env) + abs(penalty) <= floor:
            near_zero += 1
        ratios.append(abs(float(intrinsic_w) * intr) / den)
    near_zero_ratio = float(near_zero / n)
    return ratios, near_zero_ratio


def _build_intrinsic_postprocessed_fn(
    intrinsic_reward,
    revise_state,
    reference_states: np.ndarray,
    post_cfg: dict,
    input_mode: str = "revised",
):
    if intrinsic_reward is None:
        return None, {"mode": post_cfg.get("mode", "raw"), "available": False}
    mode = str(post_cfg.get("mode", "raw")).lower()
    eps = float(post_cfg.get("eps", 1e-6))
    if mode == "raw":
        return intrinsic_reward, {
            "mode": mode,
            "available": True,
            "mean": 0.0,
            "std": 1.0,
            "reference_count": int(reference_states.shape[0]) if reference_states is not None else 0,
        }

    raw_vals: List[float] = []
    if reference_states is not None and len(reference_states) > 0:
        for state in reference_states:
            try:
                intrinsic_input = (
                    state
                    if str(input_mode or "revised").lower() == "raw"
                    else (revise_state(state) if revise_state is not None else state)
                )
                raw_vals.append(_sanitize_float(intrinsic_reward(intrinsic_input)))
            except Exception:
                raw_vals.append(0.0)
    center = float(np.mean(raw_vals)) if raw_vals else 0.0
    scale = float(np.std(raw_vals)) if raw_vals else 1.0
    if scale < eps:
        scale = eps

    def _wrapped(x):
        raw = _sanitize_float(intrinsic_reward(x))
        if mode == "centered":
            return raw - center
        if mode == "zscore_tanh":
            return float(np.tanh((raw - center) / scale) * 100.0)
        return raw

    return _wrapped, {
        "mode": mode,
        "available": True,
        "mean": center,
        "std": scale,
        "reference_count": int(reference_states.shape[0]) if reference_states is not None else 0,
    }


def _build_completeness_check(required_files: List[Path]) -> dict:
    missing = [str(p.name) for p in required_files if not p.exists()]
    return {
        "excluded_incomplete": bool(missing),
        "status": "incomplete" if missing else "complete",
        "required_files": [str(p.name) for p in required_files],
        "missing_files": missing,
    }


def _to_float_list(vals) -> List[float]:
    if vals is None:
        return []
    return [float(v) for v in vals]


def _td3_seed_trace_from_result(result, seed: int) -> dict:
    return {
        "seed": int(seed),
        "eval_values_final": _to_float_list(result.eval_values_final),
        "eval_reward_env": _to_float_list(result.eval_reward_env),
        "eval_action_penalties": _to_float_list(result.eval_action_penalties),
        "eval_reward_total": _to_float_list(result.eval_reward_total),
        "eval_intrinsic_values": _to_float_list(result.eval_intrinsic_values),
        "eval_intrinsic_ratio": _to_float_list(result.eval_intrinsic_ratio),
        "eval_actions_final": [[float(x) for x in step] for step in (result.eval_actions_final or [])],
        "eval_portfolio_weights": _extract_portfolio_weights_from_trace(result.eval_trace_final or []),
        "eval_portfolio_weight_changes": _extract_portfolio_weight_changes_from_trace(result.eval_trace_final or []),
        "eval_states_final": [str(x) for x in (result.eval_states_final or [])],
        "eval_q1_final": _to_float_list(result.eval_q1_final),
        "eval_trace_final": result.eval_trace_final or [],
    }


def _td3_seed_trace_from_sb3_result(result: dict, seed: int) -> dict:
    trace_rows = result.get("eval_trace", []) or []
    actions_executed = result.get("eval_actions_executed", []) or []
    states = [
        str(row.get("state_signature", ""))
        for row in trace_rows
        if isinstance(row, dict)
    ]
    q1_vals = [
        _sanitize_float(row.get("q1", 0.0))
        for row in trace_rows
        if isinstance(row, dict)
    ]
    if not q1_vals:
        q1_vals = [0.0 for _ in range(len(actions_executed))]
    return {
        "seed": int(seed),
        "eval_values_final": _to_float_list(result.get("values", [])),
        "eval_reward_env": _to_float_list(result.get("reward_env", [])),
        "eval_action_penalties": _to_float_list(result.get("action_penalty", [])),
        "eval_reward_total": _to_float_list(result.get("reward_total", [])),
        "eval_intrinsic_values": _to_float_list(result.get("intrinsic", [])),
        "eval_intrinsic_ratio": _to_float_list(result.get("intrinsic_ratio", [])),
        "eval_actions_final": [[float(x) for x in step] for step in actions_executed],
        "eval_portfolio_weights": _extract_portfolio_weights_from_trace(trace_rows),
        "eval_portfolio_weight_changes": _extract_portfolio_weight_changes_from_trace(trace_rows),
        "eval_states_final": states,
        "eval_q1_final": _to_float_list(q1_vals),
        "eval_trace_final": trace_rows,
    }


def _sb3_seed_trace_from_result(result: dict, seed: int) -> dict:
    trace_rows = result.get("eval_trace", []) or []
    actions_executed = result.get("eval_actions_executed", []) or []
    actions_policy = result.get("eval_actions_policy", []) or []
    return {
        "seed": int(seed),
        "eval_values": _to_float_list(result.get("values", [])),
        "eval_reward_env": _to_float_list(result.get("reward_env", [])),
        "eval_action_penalties": _to_float_list(result.get("action_penalty", [])),
        "eval_reward_total": _to_float_list(result.get("reward_total", [])),
        "eval_intrinsic_values": _to_float_list(result.get("intrinsic", [])),
        "eval_actions_final": [[float(x) for x in step] for step in actions_executed],
        "eval_portfolio_weights": _extract_portfolio_weights_from_trace(trace_rows),
        "eval_portfolio_weight_changes": _extract_portfolio_weight_changes_from_trace(trace_rows),
        "eval_actions_policy": [[float(x) for x in step] for step in actions_policy],
        "eval_trace": trace_rows,
    }


def _mean_delta(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    arr_a = np.array(a[:n], dtype=float)
    arr_b = np.array(b[:n], dtype=float)
    return float(np.mean(arr_b - arr_a))


def _build_td3_g1_g3_diff(td3_algo_trace: dict) -> dict:
    g1_rows = td3_algo_trace.get("G1_revise_only", [])
    g3_rows = td3_algo_trace.get("G3_revise_intrinsic", [])
    g1_by_seed = {int(row.get("seed", -1)): row for row in g1_rows}
    g3_by_seed = {int(row.get("seed", -1)): row for row in g3_rows}
    shared_seeds = sorted(set(g1_by_seed.keys()) & set(g3_by_seed.keys()))
    seed_diffs: List[dict] = []
    for sd in shared_seeds:
        row_g1 = g1_by_seed[sd]
        row_g3 = g3_by_seed[sd]
        actions_g1 = row_g1.get("eval_actions_final", []) or []
        actions_g3 = row_g3.get("eval_actions_final", []) or []
        n_action = min(len(actions_g1), len(actions_g3))
        action_equal = 0
        for i in range(n_action):
            a1 = np.array(actions_g1[i], dtype=float)
            a3 = np.array(actions_g3[i], dtype=float)
            if a1.shape == a3.shape and np.allclose(a1, a3, atol=1e-12, rtol=0.0):
                action_equal += 1
        action_equal_ratio = float(action_equal / n_action) if n_action > 0 else 0.0

        values_g1 = _to_float_list(row_g1.get("eval_values_final", []))
        values_g3 = _to_float_list(row_g3.get("eval_values_final", []))
        n_value = min(len(values_g1), len(values_g3))
        eval_value_mae = (
            float(np.mean(np.abs(np.array(values_g3[:n_value], dtype=float) - np.array(values_g1[:n_value], dtype=float))))
            if n_value > 0
            else 0.0
        )

        reward_env_delta_mean = _mean_delta(
            _to_float_list(row_g1.get("eval_reward_env", [])),
            _to_float_list(row_g3.get("eval_reward_env", [])),
        )
        reward_total_delta_mean = _mean_delta(
            _to_float_list(row_g1.get("eval_reward_total", [])),
            _to_float_list(row_g3.get("eval_reward_total", [])),
        )
        intrinsic_mean_delta = _mean_delta(
            _to_float_list(row_g1.get("eval_intrinsic_values", [])),
            _to_float_list(row_g3.get("eval_intrinsic_values", [])),
        )
        q1_delta_mean = _mean_delta(
            _to_float_list(row_g1.get("eval_q1_final", [])),
            _to_float_list(row_g3.get("eval_q1_final", [])),
        )
        seed_diffs.append(
            {
                "seed": int(sd),
                "action_equal_ratio": action_equal_ratio,
                "eval_value_mae": eval_value_mae,
                "eval_reward_env_delta_mean": reward_env_delta_mean,
                "eval_reward_total_delta_mean": reward_total_delta_mean,
                "intrinsic_mean_delta": intrinsic_mean_delta,
                "q1_delta_mean": q1_delta_mean,
                "len_actions_g1": int(len(actions_g1)),
                "len_actions_g3": int(len(actions_g3)),
                "len_values_g1": int(len(values_g1)),
                "len_values_g3": int(len(values_g3)),
            }
        )

    summary = {
        "seed_count": int(len(seed_diffs)),
        "action_equal_ratio_mean": float(np.mean([x["action_equal_ratio"] for x in seed_diffs])) if seed_diffs else 0.0,
        "eval_value_mae_mean": float(np.mean([x["eval_value_mae"] for x in seed_diffs])) if seed_diffs else 0.0,
        "eval_reward_total_delta_mean": float(np.mean([x["eval_reward_total_delta_mean"] for x in seed_diffs])) if seed_diffs else 0.0,
        "intrinsic_mean_delta_mean": float(np.mean([x["intrinsic_mean_delta"] for x in seed_diffs])) if seed_diffs else 0.0,
        "q1_delta_mean": float(np.mean([x["q1_delta_mean"] for x in seed_diffs])) if seed_diffs else 0.0,
    }
    if seed_diffs:
        if summary["action_equal_ratio_mean"] >= 0.999 and abs(summary["eval_reward_total_delta_mean"]) > 1e-9:
            diagnosis = "intrinsic_changed_reward_total_but_policy_actions_remained_identical"
        elif summary["action_equal_ratio_mean"] < 0.999 and summary["eval_value_mae_mean"] <= 1e-9:
            diagnosis = "policy_actions_changed_but_eval_values_not_sensitive"
        elif summary["action_equal_ratio_mean"] >= 0.999 and abs(summary["eval_reward_total_delta_mean"]) <= 1e-9:
            diagnosis = "intrinsic_has_no_effect_on_reward_total_and_policy"
        else:
            diagnosis = "mixed_or_partial_separation"
    else:
        diagnosis = "insufficient_seed_overlap"

    return {
        "available": bool(seed_diffs),
        "diagnosis": diagnosis,
        "shared_seeds": shared_seeds,
        "summary": summary,
        "seed_diffs": seed_diffs,
    }


def _build_td3_action_saturation_summary(
    td3_seed_trace: dict, action_bound: float, collapse_threshold: float = 0.95
) -> dict:
    threshold = 0.999 * abs(float(action_bound))
    out: dict = {}
    for algo, algo_trace in td3_seed_trace.items():
        out[algo] = {}
        algo_near_ratios: List[float] = []
        algo_unique_counts: List[float] = []
        algo_flip_rates: List[float] = []
        for gname, rows in (algo_trace or {}).items():
            seed_stats = []
            for row in rows or []:
                actions = row.get("eval_actions_final", []) or []
                if actions:
                    arr = np.array(actions, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                else:
                    arr = np.zeros((0, 1), dtype=float)

                flat = np.abs(arr).reshape(-1)
                n = int(flat.size)
                near = int(np.sum(flat >= threshold))
                near_ratio = float(near / n) if n > 0 else 0.0
                if arr.shape[0] > 0:
                    rounded = np.round(arr, 6)
                    unique_action_count = int(np.unique(rounded, axis=0).shape[0])
                else:
                    unique_action_count = 0
                if arr.shape[0] > 1:
                    sign = np.sign(arr)
                    flips = int(np.sum(sign[1:] != sign[:-1]))
                    flip_rate = float(flips / max(1, (arr.shape[0] - 1) * arr.shape[1]))
                else:
                    flip_rate = 0.0
                seed_stats.append(
                    {
                        "seed": int(row.get("seed", -1)),
                        "action_count": int(n),
                        "near_actor_count": int(near),
                        "near_actor_ratio": near_ratio,
                        "unique_action_count": unique_action_count,
                        "sign_flip_rate": flip_rate,
                    }
                )
            near_ratios = [float(s["near_actor_ratio"]) for s in seed_stats]
            unique_counts = [float(s["unique_action_count"]) for s in seed_stats]
            flip_rates = [float(s["sign_flip_rate"]) for s in seed_stats]
            group_summary = {
                "seed_count": int(len(seed_stats)),
                "near_actor_ratio_mean": float(np.mean(near_ratios)) if near_ratios else 0.0,
                "near_actor_ratio_min": float(np.min(near_ratios)) if near_ratios else 0.0,
                "near_actor_ratio_max": float(np.max(near_ratios)) if near_ratios else 0.0,
                "unique_action_count_mean": float(np.mean(unique_counts)) if unique_counts else 0.0,
                "unique_action_count_min": float(np.min(unique_counts)) if unique_counts else 0.0,
                "unique_action_count_max": float(np.max(unique_counts)) if unique_counts else 0.0,
                "sign_flip_rate_mean": float(np.mean(flip_rates)) if flip_rates else 0.0,
                "sign_flip_rate_min": float(np.min(flip_rates)) if flip_rates else 0.0,
                "sign_flip_rate_max": float(np.max(flip_rates)) if flip_rates else 0.0,
                "threshold_abs_action": float(threshold),
                "collapse_threshold": float(collapse_threshold),
                "actor_collapse_detected": bool(
                    (float(np.mean(near_ratios)) if near_ratios else 0.0) >= float(collapse_threshold)
                ),
            }
            out[algo][gname] = {"summary": group_summary, "seeds": seed_stats}
            algo_near_ratios.extend(near_ratios)
            algo_unique_counts.extend(unique_counts)
            algo_flip_rates.extend(flip_rates)
        out[algo]["_overall"] = {
            "near_actor_ratio_mean": float(np.mean(algo_near_ratios)) if algo_near_ratios else 0.0,
            "near_actor_ratio_min": float(np.min(algo_near_ratios)) if algo_near_ratios else 0.0,
            "near_actor_ratio_max": float(np.max(algo_near_ratios)) if algo_near_ratios else 0.0,
            "unique_action_count_mean": float(np.mean(algo_unique_counts)) if algo_unique_counts else 0.0,
            "sign_flip_rate_mean": float(np.mean(algo_flip_rates)) if algo_flip_rates else 0.0,
            "threshold_abs_action": float(threshold),
            "collapse_threshold": float(collapse_threshold),
            "actor_collapse_detected": bool(
                (float(np.mean(algo_near_ratios)) if algo_near_ratios else 0.0) >= float(collapse_threshold)
            ),
        }
    return out


def _action_behavior_stats(actions: np.ndarray, action_bound: float, collapse_threshold: float) -> dict:
    if actions.size == 0:
        return {
            "action_count": 0,
            "near_bound_ratio": 0.0,
            "unique_action_count": 0,
            "sign_flip_rate": 0.0,
            "action_entropy": 0.0,
            "threshold_abs_action": float(0.999 * abs(action_bound)),
            "collapse_threshold": float(collapse_threshold),
            "actor_collapse_detected": False,
        }

    arr = np.array(actions, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    flat = np.abs(arr).reshape(-1)
    threshold = 0.999 * abs(float(action_bound))
    near = int(np.sum(flat >= threshold))
    near_ratio = float(near / max(1, flat.size))

    rounded = np.round(arr, 6)
    unique_action_count = int(np.unique(rounded, axis=0).shape[0]) if arr.shape[0] > 0 else 0

    if arr.shape[0] > 1:
        sign = np.sign(arr)
        flips = int(np.sum(sign[1:] != sign[:-1]))
        sign_flip_rate = float(flips / max(1, (arr.shape[0] - 1) * arr.shape[1]))
    else:
        sign_flip_rate = 0.0

    tuples = [tuple(x) for x in rounded.tolist()]
    cnt = Counter(tuples)
    probs = np.array([v / max(1, len(tuples)) for v in cnt.values()], dtype=float)
    action_entropy = float(-np.sum(probs * np.log(probs + 1e-12))) if probs.size > 0 else 0.0

    return {
        "action_count": int(arr.shape[0] * arr.shape[1]),
        "near_bound_ratio": near_ratio,
        "unique_action_count": unique_action_count,
        "sign_flip_rate": sign_flip_rate,
        "action_entropy": action_entropy,
        "threshold_abs_action": float(threshold),
        "collapse_threshold": float(collapse_threshold),
        "actor_collapse_detected": bool(near_ratio >= float(collapse_threshold)),
    }


def _extract_portfolio_weights_from_trace(trace_rows: List[dict]) -> List[List[float]]:
    weights = []
    for step in trace_rows or []:
        if not isinstance(step, dict):
            continue
        row = step.get("portfolio_weights", [])
        try:
            arr = np.asarray(row, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        weights.append(arr.tolist())
    return weights


def _extract_portfolio_weight_changes_from_trace(trace_rows: List[dict]) -> List[float]:
    changes = []
    for step in trace_rows or []:
        if not isinstance(step, dict):
            continue
        try:
            val = float(step.get("portfolio_weight_change", 0.0))
        except Exception:
            continue
        if np.isfinite(val):
            changes.append(val)
    return changes


def _behavior_summary_from_seed_rows(seed_rows: List[dict], action_bound: float, collapse_threshold: float) -> dict:
    seed_stats = []
    for row in seed_rows or []:
        actions = row.get("eval_actions_final")
        portfolio_weights = row.get("eval_portfolio_weights")
        portfolio_weight_changes = row.get("eval_portfolio_weight_changes")
        if actions is None:
            trace = row.get("eval_trace", []) or []
            actions = []
            for step in trace:
                if isinstance(step, dict):
                    if "action_executed" in step:
                            actions.append(step.get("action_executed") or [])
                    elif "action" in step:
                        actions.append(step.get("action") or [])
            portfolio_weights = _extract_portfolio_weights_from_trace(trace)
            portfolio_weight_changes = _extract_portfolio_weight_changes_from_trace(trace)
        arr = np.array(actions, dtype=float) if actions is not None else np.zeros((0, 1), dtype=float)
        stats = _action_behavior_stats(arr, action_bound=action_bound, collapse_threshold=collapse_threshold)
        stats.update(_turnover_stats_from_weight_rows(portfolio_weights, portfolio_weight_changes))
        stats["seed"] = int(row.get("seed", -1))
        seed_stats.append(stats)

    if not seed_stats:
        empty_summary = _action_behavior_stats(np.zeros((0, 1), dtype=float), action_bound, collapse_threshold)
        empty_summary.update(_turnover_stats_from_weight_rows([]))
        return {"summary": empty_summary, "seeds": []}

    def _mean(key: str) -> float:
        return float(np.mean([float(x.get(key, 0.0)) for x in seed_stats]))

    def _min(key: str) -> float:
        return float(np.min([float(x.get(key, 0.0)) for x in seed_stats]))

    def _max(key: str) -> float:
        return float(np.max([float(x.get(key, 0.0)) for x in seed_stats]))

    summary = {
        "seed_count": int(len(seed_stats)),
        "near_bound_ratio_mean": _mean("near_bound_ratio"),
        "near_bound_ratio_min": _min("near_bound_ratio"),
        "near_bound_ratio_max": _max("near_bound_ratio"),
        "unique_action_count_mean": _mean("unique_action_count"),
        "unique_action_count_min": _min("unique_action_count"),
        "unique_action_count_max": _max("unique_action_count"),
        "sign_flip_rate_mean": _mean("sign_flip_rate"),
        "sign_flip_rate_min": _min("sign_flip_rate"),
        "sign_flip_rate_max": _max("sign_flip_rate"),
        "action_entropy_mean": _mean("action_entropy"),
        "action_entropy_min": _min("action_entropy"),
        "action_entropy_max": _max("action_entropy"),
        "avg_daily_portfolio_weight_change_mean": _mean("avg_daily_portfolio_weight_change"),
        "avg_daily_portfolio_weight_change_min": _min("avg_daily_portfolio_weight_change"),
        "avg_daily_portfolio_weight_change_max": _max("avg_daily_portfolio_weight_change"),
        "turnover_stability_score_mean": _mean("turnover_stability_score"),
        "threshold_abs_action": float(seed_stats[0].get("threshold_abs_action", 0.0)),
        "collapse_threshold": float(seed_stats[0].get("collapse_threshold", collapse_threshold)),
        "actor_collapse_detected": bool(_mean("near_bound_ratio") >= float(collapse_threshold)),
    }
    return {"summary": summary, "seeds": seed_stats}


def _build_policy_behavior_summary(
    td3_seed_trace: dict,
    sb3_seed_trace: dict,
    action_bound: float,
    td3_action_bound: float | None = None,
    collapse_threshold: float = 0.95,
) -> dict:
    out: dict = {}
    merged = {}
    for algo, trace in (td3_seed_trace or {}).items():
        merged[algo] = trace
    for algo, trace in (sb3_seed_trace or {}).items():
        merged[algo] = trace

    for algo, algo_trace in merged.items():
        out[algo] = {}
        all_near: List[float] = []
        all_entropy: List[float] = []
        all_turnover: List[float] = []
        cur_action_bound = (
            float(td3_action_bound)
            if (str(algo).lower() == "td3" and td3_action_bound is not None)
            else float(action_bound)
        )
        for group_name, rows in (algo_trace or {}).items():
            group_payload = _behavior_summary_from_seed_rows(
                rows or [],
                cur_action_bound,
                collapse_threshold,
            )
            out[algo][group_name] = group_payload
            all_near.append(float(group_payload["summary"].get("near_bound_ratio_mean", 0.0)))
            all_entropy.append(float(group_payload["summary"].get("action_entropy_mean", 0.0)))
            all_turnover.append(float(group_payload["summary"].get("avg_daily_portfolio_weight_change_mean", 0.0)))

        g1 = out[algo].get("G1_revise_only", {}).get("summary", {})
        g3 = out[algo].get("G3_revise_intrinsic", {}).get("summary", {})
        out[algo]["_overall"] = {
            "near_bound_ratio_mean": float(np.mean(all_near)) if all_near else 0.0,
            "action_entropy_mean": float(np.mean(all_entropy)) if all_entropy else 0.0,
            "avg_daily_portfolio_weight_change_mean": float(np.mean(all_turnover)) if all_turnover else 0.0,
            "actor_collapse_detected": bool((float(np.mean(all_near)) if all_near else 0.0) >= float(collapse_threshold)),
            "g1_g3_near_bound_delta": float(g3.get("near_bound_ratio_mean", 0.0) - g1.get("near_bound_ratio_mean", 0.0)),
            "g1_g3_entropy_delta": float(g3.get("action_entropy_mean", 0.0) - g1.get("action_entropy_mean", 0.0)),
            "g1_g3_turnover_delta": float(
                g3.get("avg_daily_portfolio_weight_change_mean", 0.0)
                - g1.get("avg_daily_portfolio_weight_change_mean", 0.0)
            ),
        }
    return out


def _write_run_summary(
    run_dir: Path,
    cfg: DemoConfig,
    results: dict,
    iter_trace: list,
    llm_errors: list,
    split_meta: dict,
) -> str:
    path = run_dir / "run_summary.md"
    selected_assets = results.get("protocol", {}).get("selected_assets") or cfg.assets
    algos = results.get("algorithms")
    algo_list = list(algos.keys()) if algos else [cfg.algorithm]
    total_candidates = sum(len(it.get("candidates", [])) for it in iter_trace)
    valid_candidates = sum(
        1 for it in iter_trace for c in it.get("candidates", []) if c.get("valid")
    )
    invalid_candidates = total_candidates - valid_candidates
    llm_cfg = cfg.llm or {}
    lines = [
        "# Run Summary",
        "",
        f"- run_id: {run_dir.name}",
        f"- data_source: {cfg.data_source}",
        f"- date_range: {cfg.start_date} -> {cfg.end_date}",
        f"- assets: {', '.join(selected_assets)}",
        f"- algorithms: {', '.join(algo_list)}",
        f"- groups: {', '.join(cfg.groups or [])}",
        f"- scoring_objective: {results.get('protocol', {}).get('scoring_objective', 'performance(Sharpe+CR)+lipschitz')}",
        f"- eval_protocol: {split_meta.get('protocol')}",
        f"- split_train: {split_meta.get('train', {}).get('start')} -> {split_meta.get('train', {}).get('end')} ({split_meta.get('train', {}).get('days')} days)",
        f"- split_val: {split_meta.get('val', {}).get('start')} -> {split_meta.get('val', {}).get('end')} ({split_meta.get('val', {}).get('days')} days)",
        f"- split_test: {split_meta.get('test', {}).get('start')} -> {split_meta.get('test', {}).get('end')} ({split_meta.get('test', {}).get('days')} days)",
        f"- anti_leak_passed: {split_meta.get('anti_leak_passed')}",
        f"- decision_ts_rule: {results.get('protocol', {}).get('decision_ts_rule')}",
        f"- action_quantization_mode: {results.get('protocol', {}).get('action_quantization_mode')}",
        f"- action_bound_penalty: {results.get('protocol', {}).get('action_bound_penalty_effective')}",
        f"- window_setup: {results.get('protocol', {}).get('window_setup')}",
        f"- intrinsic_scale_mode: {cfg.intrinsic_scale_mode}",
        f"- intrinsic_timing: {cfg.intrinsic_timing}",
        f"- intrinsic_postprocess: {cfg.intrinsic_postprocess or {'mode': 'raw'}}",
        f"- intrinsic_w_schedule: {cfg.intrinsic_w_schedule or [cfg.intrinsic_w]}",
        f"- intrinsic_w_tuning: {results.get('protocol', {}).get('intrinsic_w_tuning_effective')}",
        f"- algo_tuning: {cfg.algo_tuning or {}}",
        f"- td3.actor_max_action: {(cfg.td3 or {}).get('actor_max_action', cfg.max_trade)}",
        f"- td3.state_norm: {(cfg.td3 or {}).get('state_norm', {'mode': 'none', 'eps': 1e-6, 'log_volume': True})}",
        f"- warmup_ratio: {cfg.warmup_ratio}",
        f"- finagent_stub_enabled: {cfg.use_finagent_signal} (weight={cfg.finagent_weight})",
        f"- LLM: model={llm_cfg.get('model')} k={llm_cfg.get('k')} iterations={llm_cfg.get('iterations')} max_tokens={llm_cfg.get('max_tokens')} temperature={llm_cfg.get('temperature')}",
        f"- llm_iteration_mode: {results.get('protocol', {}).get('llm_iteration_mode', _resolve_llm_iteration_mode(llm_cfg))}",
        f"- llm_branch_parallel_workers: {results.get('protocol', {}).get('llm_branch_parallel_workers', _resolve_llm_branch_parallel_workers(llm_cfg, algo_list))}",
        f"- candidate_selection_seeds: {results.get('protocol', {}).get('candidate_selection_seeds', _resolve_candidate_selection_seeds(cfg, llm_cfg))}",
        f"- candidates: total={total_candidates} valid={valid_candidates} invalid={invalid_candidates}",
        f"- llm_errors: {len(llm_errors)}",
        f"- outputs: metrics.json, metrics_table.csv, reward_trace.json, state_scale_summary.json, run_manifest.json, td3_seed_trace.json, td3_g1_g3_diff.json, td3_action_saturation.json, sb3_action_trace.json, policy_behavior_summary.json, llm_iter_trace.json, dialogs_it*.txt, dialogs_*_it*.txt",
        "",
    ]
    path.write_text("\n".join(lines))
    return str(path)


def _run_walk_forward(cfg: DemoConfig, run_dir: Path, data_dir: Path) -> dict:
    root = repo_root()
    wf_cfg = cfg.walk_forward or {}
    windows = wf_cfg.get("windows", [])
    if not windows:
        if cfg.data_source != "finsaber" or cfg.finsaber_price_path is None:
            raise ValueError("walk_forward auto window generation requires finsaber price data.")
        finsaber_path = (root / cfg.finsaber_price_path).resolve()
        if cfg.assets and not bool((cfg.universe or {}).get("allow_auto_asset_pool", True)):
            df_for_windows = load_finsaber_prices(finsaber_path, cfg.assets, cfg.start_date, cfg.end_date or cfg.start_date)
        else:
            df_for_windows = load_finsaber_prices(finsaber_path, None, cfg.start_date, cfg.end_date or cfg.start_date)
        windows = _generate_windows_from_setup(df_for_windows, cfg)
    if not windows:
        raise ValueError("walk_forward.enabled=true requires non-empty walk_forward.windows")

    min_days = int(wf_cfg.get("min_days_per_split", 10))
    aggregate_mode = str(wf_cfg.get("aggregate", "mean_std"))
    bootstrap_cfg = _resolve_bootstrap_cfg(cfg.bootstrap)
    experiment_cfg = _resolve_experiment_cfg(cfg)
    config_fingerprint = _hash_payload(cfg.__dict__)
    candidate_scoring_cfg = _resolve_candidate_scoring_cfg(cfg.llm or {})
    scoring_objective_default = _candidate_scoring_objective(candidate_scoring_cfg)
    selection_seeds = _resolve_candidate_selection_seeds(cfg, cfg.llm or {})
    llm_iteration_mode = _resolve_llm_iteration_mode(cfg.llm or {})
    llm_branch_algos = list(cfg.eval_algorithms or [cfg.algorithm]) if llm_iteration_mode == "per_algorithm_branches" else [cfg.algorithm]
    branch_parallel_workers = _resolve_llm_branch_parallel_workers(cfg.llm or {}, llm_branch_algos)
    ensure_dir(run_dir)

    window_tables: List[pd.DataFrame] = []
    window_infos: List[dict] = []
    excluded_windows: List[dict] = []
    scoring_objective_by_window: dict[str, object] = {}
    candidate_scoring_effective_by_window: dict[str, object] = {}
    best_candidate_by_algo_by_window: dict[str, object] = {}
    candidate_fingerprint_by_algo_by_window: dict[str, object] = {}
    scenario_profile_by_window: dict[str, object] = {}
    for idx, window_raw in enumerate(windows):
        window = _json_safe(window_raw)
        for split_name in ["train", "val", "test"]:
            if split_name not in window:
                raise ValueError(f"walk_forward.windows[{idx}] missing {split_name}")
            if "start" not in window[split_name] or "end" not in window[split_name]:
                raise ValueError(f"walk_forward.windows[{idx}].{split_name} requires start/end")

        sub_cfg_dict = dict(cfg.__dict__)
        sub_cfg_dict["eval_protocol"] = "temporal_split"
        sub_cfg_dict["data_split"] = window
        sub_cfg_dict["walk_forward"] = {"enabled": False}
        sub_cfg = DemoConfig(**sub_cfg_dict)

        sub_run_dir = run_dir / f"wf_window_{idx:02d}"
        ensure_dir(sub_run_dir)
        run_demo(sub_cfg, run_dir=sub_run_dir, data_dir=data_dir)

        sub_manifest_path = sub_run_dir / "run_manifest.json"
        sub_table_path = sub_run_dir / "metrics_table.csv"
        sub_reward_trace_path = sub_run_dir / "reward_trace.json"
        sub_state_scale_path = sub_run_dir / "state_scale_summary.json"
        sub_policy_behavior_path = sub_run_dir / "policy_behavior_summary.json"
        sub_sb3_trace_path = sub_run_dir / "sb3_action_trace.json"
        sub_td3_diff_path = sub_run_dir / "td3_g1_g3_diff.json"
        sub_td3_sat_path = sub_run_dir / "td3_action_saturation.json"
        sub_required = [sub_table_path, sub_manifest_path, sub_reward_trace_path, sub_policy_behavior_path]
        sub_eval_algos = cfg.eval_algorithms or [cfg.algorithm]
        if "td3" in [str(a).lower() for a in sub_eval_algos]:
            sub_required.append(sub_td3_diff_path)
            sub_required.append(sub_td3_sat_path)
            sub_required.append(sub_state_scale_path)
        if any(str(a).lower() != "td3" for a in sub_eval_algos):
            sub_required.append(sub_sb3_trace_path)
        sub_completeness = _build_completeness_check(sub_required)
        if sub_completeness["excluded_incomplete"]:
            excluded_windows.append(
                {
                    "window_index": idx,
                    "window_name": f"wf_window_{idx:02d}",
                    "missing_files": sub_completeness["missing_files"],
                }
            )
        sub_manifest = json.loads(sub_manifest_path.read_text())
        window_name = f"wf_window_{idx:02d}"
        split_meta = sub_manifest.get("split", {})
        scoring_objective_by_window[window_name] = sub_manifest.get("scoring_objective")
        candidate_scoring_effective_by_window[window_name] = sub_manifest.get("candidate_scoring_effective")
        best_candidate_by_algo_by_window[window_name] = sub_manifest.get("best_candidate_by_algo")
        candidate_fingerprint_by_algo_by_window[window_name] = sub_manifest.get("candidate_fingerprint_by_algo")
        scenario_profile_by_window[window_name] = sub_manifest.get("scenario_profile")

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
                "policy_behavior_summary": str(sub_policy_behavior_path.relative_to(root)) if sub_policy_behavior_path.exists() else "",
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
                    "policy_behavior_summary": sha256_file(sub_policy_behavior_path) if sub_policy_behavior_path.exists() else "",
                    "sb3_action_trace": sha256_file(sub_sb3_trace_path) if sub_sb3_trace_path.exists() else "",
                },
            }
        )

    wf_table = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    agg_rows: List[dict] = []
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
        empty_value=scoring_objective_default,
        mixed_value="mixed; see scoring_objective_by_window",
    )
    candidate_scoring_effective_parent = _resolve_parent_field_from_windows(
        candidate_scoring_effective_by_window,
        empty_value=candidate_scoring_cfg,
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
    }
    summary_path = run_dir / "walk_forward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    cross_window_distillation = _build_cross_window_distillation(root, window_infos)
    cross_window_distillation_path = run_dir / "cross_window_distillation.json"
    cross_window_distillation_path.write_text(json.dumps(cross_window_distillation, indent=2))

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "walk_forward": summary,
                "cross_window_distillation": cross_window_distillation,
            },
            indent=2,
        )
    )

    drl_backend = _resolve_drl_backend(cfg)
    if drl_backend == "finsaber_compat":
        walk_forward_action_quantization_mode = "finsaber_continuous_hmax"
        walk_forward_discrete_action_levels = 0
    elif drl_backend == "finsaber_native":
        walk_forward_action_quantization_mode = "finsaber_original_continuous_hmax"
        walk_forward_discrete_action_levels = 0
    else:
        walk_forward_action_quantization_mode = _resolve_action_quantization_mode(cfg)
        walk_forward_discrete_action_levels = _resolve_discrete_action_levels(cfg)

    run_manifest = {
        "protocol_version": "trading-lesr-v2-walk-forward",
        "eval_protocol": "temporal_split",
        "scoring_objective": scoring_objective_parent,
        "llm_iteration_mode": llm_iteration_mode,
        "llm_branch_algorithms": llm_branch_algos,
        "llm_branch_parallel_workers": int(branch_parallel_workers),
        "candidate_selection_seeds": list(selection_seeds),
        "candidate_selection_seed_count": int(len(selection_seeds)),
        "experiment_phase": experiment_cfg["phase"],
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
        "is_confirmatory": _is_confirmatory(experiment_cfg),
        "experiment_frozen": bool(experiment_cfg["frozen"]),
        "config_fingerprint": config_fingerprint,
        "window_setup": cfg.window_setup,
        "decision_ts_rule": _resolve_decision_rule(cfg),
        "action_quantization_mode": walk_forward_action_quantization_mode,
        "discrete_action_levels": walk_forward_discrete_action_levels,
        "drl_backend": drl_backend,
        "action_bound_penalty_effective": _resolve_action_bound_penalty_cfg(cfg),
        "action_bound_penalty_effective_by_algo": {
            algo: {
                **_resolve_action_bound_penalty_cfg(cfg, algo),
                "reference_bound": float(_resolve_action_bound_penalty_reference_bound(cfg, algo)),
            }
            for algo in (cfg.eval_algorithms or [cfg.algorithm])
        },
        "universe": cfg.universe or {"mode": "fixed"},
        "algorithm": cfg.algorithm,
        "eval_algorithms": cfg.eval_algorithms or [cfg.algorithm],
        "state_norm_effective": _json_safe(resolve_td3_state_norm_config(cfg.td3).__dict__),
        "intrinsic_timing_effective": cfg.intrinsic_timing,
        "intrinsic_postprocess_effective": _resolve_intrinsic_postprocess_cfg(cfg.intrinsic_postprocess),
        "algo_tuning_effective": _resolve_algo_tuning_cfg(cfg.algo_tuning),
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
        "cross_window_distillation": cross_window_distillation,
        "stub": {
            "use_finagent_signal": cfg.use_finagent_signal,
            "finagent_weight": cfg.finagent_weight,
        },
    }
    run_manifest["candidate_fingerprint"] = _build_wf_candidate_fingerprint(root, window_infos)
    td3_window_collapse = []
    for item in window_infos:
        sub_manifest_path = root / item["run_manifest"]
        if not sub_manifest_path.exists():
            continue
        try:
            sub_manifest = json.loads(sub_manifest_path.read_text())
        except Exception:
            continue
        td3_window_collapse.append(
            {
                "window_index": item["window_index"],
                "window_name": item["window_name"],
                "actor_collapse_detected": bool(sub_manifest.get("actor_collapse_detected", False)),
            }
        )
    run_manifest["td3_action_saturation_summary"] = {
        "windows": td3_window_collapse,
        "collapsed_window_count": int(sum(1 for r in td3_window_collapse if r["actor_collapse_detected"])),
    }
    run_manifest["actor_collapse_detected"] = bool(
        any(r.get("actor_collapse_detected", False) for r in td3_window_collapse)
    )
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
    run_required = [run_manifest_path, summary_path, wf_table_path]
    wf_completeness = _build_completeness_check(run_required)
    if excluded_windows:
        wf_completeness["excluded_incomplete"] = True
        wf_completeness["status"] = "incomplete"
    wf_completeness["excluded_window_count"] = int(len(excluded_windows))
    wf_completeness["excluded_windows"] = excluded_windows
    run_manifest["completeness_check"] = wf_completeness
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    artifacts = {
        "metrics": str(metrics_path.relative_to(root)),
        "run_manifest": str(run_manifest_path.relative_to(root)),
        "walk_forward_summary": str(summary_path.relative_to(root)),
        "walk_forward_metrics_table": str(wf_table_path.relative_to(root)),
        "cross_window_distillation": str(cross_window_distillation_path.relative_to(root)),
    }
    artifacts_path = run_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2))

    hashes = {
        "metrics": sha256_file(metrics_path),
        "run_manifest": sha256_file(run_manifest_path),
        "walk_forward_summary": sha256_file(summary_path),
        "walk_forward_metrics_table": sha256_file(wf_table_path),
        "cross_window_distillation": sha256_file(cross_window_distillation_path),
    }
    hashes_path = run_dir / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2))

    return {"walk_forward": summary}


def run_demo(cfg: DemoConfig, run_dir: Path, data_dir: Path) -> dict:
    walk_forward_cfg = cfg.walk_forward or {}
    if bool(walk_forward_cfg.get("enabled", False)):
        return _run_walk_forward(cfg, run_dir, data_dir)

    _set_windows_safe_worker_limits()

    if cfg.data_source == "finsaber":
        if cfg.finsaber_price_path is None:
            raise ValueError("finsaber_price_path is required for data_source=finsaber")
        finsaber_path = (repo_root() / cfg.finsaber_price_path).resolve()
        universe_cfg = cfg.universe or {}
        load_assets = cfg.assets
        if str(universe_cfg.get("mode", "fixed")).lower() == "historical_dynamic":
            load_assets = None
        df_raw = load_finsaber_prices(finsaber_path, load_assets, cfg.start_date, cfg.end_date or cfg.start_date)
        raw_path = data_dir / "raw" / "finsaber_subset.csv"
        save_raw_data(df_raw, raw_path)
    else:
        raw_path = data_dir / "raw" / "demo_prices.csv"
        if not raw_path.exists():
            df_raw = generate_synth_ohlcv(
                SynthConfig(
                    assets=cfg.assets,
                    start_date=cfg.start_date,
                    days=cfg.days,
                    seed=cfg.seed,
                )
            )
            save_raw_data(df_raw, raw_path)
        else:
            df_raw = pd.read_csv(raw_path)

    drl_backend = _resolve_drl_backend(cfg)
    if drl_backend in {"finsaber_compat", "finsaber_native"}:
        df_feat = df_raw.copy()
    else:
        df_feat = add_indicators(df_raw, cfg.indicators)
    processed_path = data_dir / "processed" / "demo_features.csv"
    ensure_dir(processed_path.parent)
    df_feat.to_csv(processed_path, index=False)
    train_df_all, val_df_all, test_df_all, split_meta = _build_temporal_splits(df_feat, cfg)
    selected_assets, universe_snapshot = _select_assets_for_window(df_feat, cfg, split_meta)
    train_df = _filter_assets_align_dates(train_df_all, selected_assets)
    val_df = _filter_assets_align_dates(val_df_all, selected_assets)
    test_df = _filter_assets_align_dates(test_df_all, selected_assets)
    train_df, val_df, test_df, split_date_filter_summary = _apply_split_date_filters(
        train_df,
        val_df,
        test_df,
        cfg.split_date_filters,
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Asset filtering produced empty train/val/test subset.")

    split_meta = {
        **split_meta,
        "train": _split_meta_block_from_df(train_df, split_meta.get("train")),
        "val": _split_meta_block_from_df(val_df, split_meta.get("val")),
        "test": _split_meta_block_from_df(test_df, split_meta.get("test")),
    }

    if drl_backend == "finsaber_compat":
        ensure_dir(run_dir)
        scenario_profile = _infer_scenario_family(train_df, val_df, selected_assets)
        scenario_profile_path = run_dir / "scenario_profile.json"
        scenario_profile_path.write_text(json.dumps(scenario_profile, indent=2))
        (run_dir / "system_prompt.txt").write_text(
            "FINSABER-compatible DRL baseline backend. Phase-1 baseline only; LESR disabled."
        )
        (run_dir / "prompt.txt").write_text(
            "\n".join(
                [
                    "FINSABER-compatible DRL baseline evaluation.",
                    "Groups: G0_baseline only.",
                    f"Selected assets: {', '.join(selected_assets)}",
                    f"Train: {split_meta.get('train', {}).get('start')} -> {split_meta.get('train', {}).get('end')}",
                    f"Val: {split_meta.get('val', {}).get('start')} -> {split_meta.get('val', {}).get('end')}",
                    f"Test: {split_meta.get('test', {}).get('start')} -> {split_meta.get('test', {}).get('end')}",
                    f"Scenario family hint: {scenario_profile.get('family', '')}",
                ]
            )
        )
        return _run_finsaber_compat_demo(
            cfg=cfg,
            run_dir=run_dir,
            raw_path=raw_path,
            processed_path=processed_path,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            split_meta=split_meta,
            split_date_filter_summary=split_date_filter_summary,
            selected_assets=selected_assets,
            universe_snapshot=universe_snapshot,
            scenario_profile_path=scenario_profile_path,
        )

    schema = StateSchema(assets=selected_assets, indicators=cfg.indicators, global_features=cfg.global_features)
    native_cfg = None
    native_summary = None
    native_contract = None
    native_contract_summary = None
    native_selection_history_df = None
    native_eval_history_df = None
    if drl_backend == "finsaber_native":
        native_timesteps = _effective_steps(cfg.n_full, int(train_df["date"].nunique()))
        native_cfg, native_summary = _resolve_finsaber_native_cfg(cfg, native_timesteps)
        processed_train_for_contract, _ = preprocess_native_data(
            format_native_raw_for_fe(train_df),
            tech_indicator_list=list(native_cfg.tech_indicator_list or load_native_finrl_indicators()),
            use_vix=bool(native_cfg.use_vix),
            use_turbulence=bool(native_cfg.use_turbulence),
            user_defined_feature=bool(native_cfg.user_defined_feature),
        )
        native_contract, native_contract_summary = _resolve_finsaber_native_state_contract(
            cfg,
            selected_assets,
            reference_df=processed_train_for_contract,
        )
        raw_state_dim = int(native_contract.state_dim)
        volume_indices = []
        reference_states = collect_finsaber_native_reference_states(
            processed_train_for_contract,
            contract=native_contract,
            initial_cash=float(native_cfg.initial_amount),
            max_samples=None,
        )
        state_desc = native_contract.describe_compact()
        state_contract_note = native_contract.prompt_note()
        native_validation_states = reference_states
        native_raw_dim = int(native_contract.state_dim)
        native_selection_history_df = train_df.sort_values(["date", "asset"]).reset_index(drop=True)
        native_eval_history_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    else:
        raw_state_dim = int(schema.dim())
        volume_indices = _volume_indices(schema)
        reference_states = _collect_reference_states(train_df, schema, cfg.initial_cash)
        state_desc = schema.describe()
        state_contract_note = ""
        native_selection_history_df = None
        native_validation_states = None
        native_raw_dim = None
    llm_cfg = cfg.llm or {}
    candidate_scoring_cfg = _resolve_candidate_scoring_cfg(llm_cfg)
    candidate_scoring_objective = _candidate_scoring_objective(candidate_scoring_cfg)
    system_prompt = build_system_prompt(llm_cfg)
    prompt_base = build_initial_prompt(
        cfg.task_description,
        state_desc,
        state_contract_note=state_contract_note,
    )
    prompt = prompt_base

    decision_ts_rule = _resolve_decision_rule(cfg)
    action_quantization_mode = _resolve_action_quantization_mode(cfg)
    action_bound_penalty_cfg = _resolve_action_bound_penalty_cfg(cfg)
    env_cfg = EnvConfig(
        initial_cash=cfg.initial_cash,
        max_trade=cfg.max_trade,
        fee_rate=cfg.fee_rate,
        decision_ts_rule=decision_ts_rule,
        action_quantization_mode=action_quantization_mode,
        discrete_action_levels=_resolve_discrete_action_levels(cfg),
        action_bound_penalty_coef=float(action_bound_penalty_cfg["coef"]),
        action_bound_penalty_threshold=float(action_bound_penalty_cfg["threshold"]),
        action_bound_penalty_power=float(action_bound_penalty_cfg["power"]),
        action_bound_penalty_reference_bound=float(_resolve_action_bound_penalty_reference_bound(cfg)),
    )
    policy = HeuristicPolicy(schema, PolicyConfig(max_trade=cfg.max_trade))
    finagent = FinAgentStub(FinAgentStubConfig()) if cfg.use_finagent_signal else None
    td3_dict = dict(cfg.td3 or {})
    td3_backend = _resolve_td3_backend(td3_dict)
    state_norm_cfg = resolve_td3_state_norm_config(td3_dict)
    td3_dict.pop("backend", None)
    td3_dict.pop("state_norm", None)
    sb3_cfg = SB3Config(**(cfg.sb3 or {}))
    algo_tuning_cfg = _resolve_algo_tuning_cfg(cfg.algo_tuning)
    td3_tuning_overrides, td3_tuning_ignored = _split_td3_tuning(algo_tuning_cfg.get("td3", {}))
    td3_sb3_cfg_overrides, td3_sb3_model_kwargs = _split_sb3_tuning(td3_tuning_ignored)
    td3_effective_dict = dict(td3_dict)
    td3_effective_dict.update(td3_tuning_overrides)
    td3_base_cfg = TD3Config(max_action=cfg.max_trade, **td3_effective_dict)
    td3_sb3_base_cfg = _apply_sb3_cfg_overrides(
        replace(
            sb3_cfg,
            gamma=float(td3_base_cfg.discount),
            batch_size=int(td3_base_cfg.batch_size),
        ),
        td3_sb3_cfg_overrides,
    )
    td3_sb3_base_kwargs = _td3_cfg_to_sb3_kwargs(td3_base_cfg)
    td3_sb3_base_kwargs.update(td3_sb3_model_kwargs)
    td3_default_policy_action_bound = _td3_policy_action_bound(td3_base_cfg, cfg.max_trade)
    state_norm_effective = _json_safe(state_norm_cfg.__dict__)
    intrinsic_postprocess_cfg = _resolve_intrinsic_postprocess_cfg(cfg.intrinsic_postprocess)
    diagnostics_cfg = _resolve_diagnostics_cfg(cfg.diagnostics)
    intrinsic_w_tuning_cfg = _resolve_intrinsic_w_tuning_cfg(cfg)
    experiment_cfg = _resolve_experiment_cfg(cfg)
    eval_algos = cfg.eval_algorithms or [cfg.algorithm]
    config_fingerprint = _hash_payload(cfg.__dict__)
    state_norm_policy_spaces: Dict[str, dict] = {}

    def _build_policy_state_fn(base_state_fn, summary_key: str | None = None, *, algorithm: str | None = None):
        if drl_backend == "finsaber_native":
            if str(algorithm or "").strip().lower() == "td3":
                policy_state_fn, summary = build_td3_state_fn(
                    base_state_fn=base_state_fn,
                    reference_states=reference_states,
                    raw_dim=raw_state_dim,
                    volume_indices=volume_indices,
                    norm_cfg=state_norm_cfg,
                )
                if summary_key:
                    native_summary = dict(summary)
                    native_summary["backend"] = "finsaber_native"
                    native_summary["algorithm"] = "td3"
                    native_summary["mode"] = f"native_td3_{native_summary.get('mode', 'identity')}"
                    state_norm_policy_spaces[summary_key] = native_summary
                return policy_state_fn
            def _native_policy_state_fn(state):
                out = base_state_fn(state)
                arr = np.asarray(out, dtype=np.float32).reshape(-1)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if summary_key:
                state_norm_policy_spaces[summary_key] = {
                    "backend": "finsaber_native",
                    "mode": "identity_backend_policy_state",
                    "algorithm": str(algorithm or ""),
                    "raw_dim": int(native_raw_dim or raw_state_dim),
                    "policy_dim": int(native_raw_dim or raw_state_dim),
                }
            return _native_policy_state_fn
        policy_state_fn, summary = build_td3_state_fn(
            base_state_fn=base_state_fn,
            reference_states=reference_states,
            raw_dim=raw_state_dim,
            volume_indices=volume_indices,
            norm_cfg=state_norm_cfg,
        )
        if summary_key:
            state_norm_policy_spaces[summary_key] = summary
        return policy_state_fn

    def _prepare_intrinsic(revise_state_fn, intrinsic_reward_fn):
        intrinsic_fn, _ = _build_intrinsic_postprocessed_fn(
            intrinsic_reward=intrinsic_reward_fn,
            revise_state=revise_state_fn,
            reference_states=reference_states,
            post_cfg=intrinsic_postprocess_cfg,
            input_mode="revised",
        )
        return intrinsic_fn or intrinsic_reward_fn

    def _validation_error_message(exc: Exception) -> str:
        return globals()["_validation_error_message"](exc)

    def _validate_candidate_pair(revise_state_fn, intrinsic_reward_fn) -> tuple[int, float]:
        return _validate_candidate_pair_for_backend(
            revise_state_fn,
            intrinsic_reward_fn,
            drl_backend=drl_backend,
            schema=schema,
            native_validation_states=native_validation_states,
            native_raw_dim=native_raw_dim,
        )

    def _identity_revise_state(state):
        return globals()["_identity_revise_state"](state)

    def _zero_intrinsic_reward(_state):
        return globals()["_zero_intrinsic_reward"](_state)

    state_fn_raw = _build_policy_state_fn(_identity_revise_state, summary_key="raw_identity")
    algo_state_fn_raw_cache: Dict[str, object] = {}

    def _state_fn_raw_for_algo(algo: str):
        algo_key = str(algo).strip().lower()
        if algo_key in algo_state_fn_raw_cache:
            return algo_state_fn_raw_cache[algo_key]
        if drl_backend == "finsaber_native" and algo_key == "td3":
            fn = _build_policy_state_fn(
                _identity_revise_state,
                summary_key="raw_identity_td3",
                algorithm=algo_key,
            )
        else:
            fn = state_fn_raw
        algo_state_fn_raw_cache[algo_key] = fn
        return fn

    generation_target = str(llm_cfg.get("generation_target", "global_best")).strip().lower()
    if generation_target not in {"global_best", "scenario_family"}:
        generation_target = "global_best"
    scenario_family_cfg = llm_cfg.get("scenario_family", {}) if isinstance(llm_cfg, dict) else {}
    if not isinstance(scenario_family_cfg, dict):
        scenario_family_cfg = {}
    default_families = ["trend_follow", "mean_revert", "risk_shield"]
    raw_families = scenario_family_cfg.get("families", default_families)
    if isinstance(raw_families, list):
        scenario_families = [str(x).strip() for x in raw_families if str(x).strip()]
    else:
        scenario_families = list(default_families)
    if not scenario_families:
        scenario_families = list(default_families)
    scenario_enabled = bool(scenario_family_cfg.get("enabled", False)) and generation_target == "scenario_family"
    candidates_per_family = int(max(1, scenario_family_cfg.get("candidates_per_family_per_iter", 1)))
    router_cfg = scenario_family_cfg.get("router", {})
    if not isinstance(router_cfg, dict):
        router_cfg = {}
    scenario_profile = _infer_scenario_family(train_df, val_df, selected_assets, router_cfg=router_cfg)
    scenario_profile_path = run_dir / "scenario_profile.json"
    scenario_profile_path.write_text(json.dumps(scenario_profile, indent=2))

    def _append_prompt_sections(base_prompt: str, sections: List[str]) -> str:
        non_empty = [str(section).strip() for section in sections if str(section).strip()]
        if not non_empty:
            return base_prompt
        return f"{base_prompt}\n\n" + "\n\n".join(non_empty)

    def _build_common_search_context() -> str:
        selected_assets_preview = ", ".join(selected_assets[:6])
        action_space_map = ", ".join(
            f"{algo}:{_action_space_type(algo, drl_backend)}" for algo in eval_algos
        )
        return "\n".join(
            [
                "Structured LESR search context:",
                f"- selected_assets={selected_assets_preview}",
                f"- universe_mode={universe_snapshot.get('mode', 'fixed')}",
                f"- eval_protocol={split_meta.get('protocol')}",
                f"- drl_backend={drl_backend}",
                f"- scenario_family={scenario_profile.get('family')}",
                f"- scenario_mu_ann={float(scenario_profile.get('mu_ann', 0.0)):.4f}",
                f"- scenario_vol_ann={float(scenario_profile.get('vol_ann', 0.0)):.4f}",
                f"- scenario_max_dd={float(scenario_profile.get('max_dd', 0.0)):.4f}",
                f"- regime_vol_short_ann={float(scenario_profile.get('vol_short_ann', 0.0)):.4f}",
                f"- regime_vol_long_ann={float(scenario_profile.get('vol_long_ann', 0.0)):.4f}",
                f"- regime_vol_ratio_20_60={float(scenario_profile.get('vol_ratio_20_60', 0.0)):.4f}",
                f"- regime_trend_strength_20={float(scenario_profile.get('trend_strength_20', 0.0)):.4f}",
                f"- regime_dispersion_20={float(scenario_profile.get('dispersion_20', 0.0)):.6f}",
                f"- regime_market_stress_score={float(scenario_profile.get('market_stress_score', 0.0)):.4f}",
                f"- action_spaces={action_space_map}",
                "- cross_algorithm_goal=prefer structures that stay helpful across discrete and continuous control, not branch-only hacks",
                "- mechanism_hint=revise_state should expose trend/rank/confidence together with portfolio-memory terms such as cash_ratio, concentration, entropy, exposure, and rebalancing pressure",
                "- mechanism_hint=intrinsic_reward should be smooth, bounded, confidence-gated, risk-adjusted, and should not dominate portfolio behavior",
                "- mechanism_hint=prefer candidates whose portfolio weights change smoothly instead of forcing unstable day-to-day reallocations",
                "- feature_group_hint=portfolio_memory means holdings/cash/exposure/concentration/entropy/rebalancing pressure",
                "- feature_group_hint=regime means volatility level, volatility ratio, drawdown, market-stress, trend-strength regime context",
                "- feature_group_hint=dispersion means spread/rank/breadth/cross-asset disagreement/winner-minus-loser structure",
                "- feature_group_hint=running_risk_state means ret_ema, ret_sq_ema, drawdown_ema-like, or turnover_ema-like running portfolio state",
                "- mechanism_hint=selection now rewards candidates whose intrinsic path remains useful under raw-policy control; avoid near-zero intrinsic channels",
                "- candidate_supply_hint=each iteration intentionally mixes intrinsic_first, balanced, and state_first candidate designs",
                "- trace_hint=if possible, declare FEATURE_GROUPS = [...] in the returned code to expose which semantic feature groups the candidate uses",
            ]
        )

    prompt = _append_prompt_sections(prompt_base, [_build_common_search_context()])

    # save prompt
    ensure_dir(run_dir)
    (run_dir / "system_prompt.txt").write_text(system_prompt)
    (run_dir / "prompt.txt").write_text(prompt)
    llm_iteration_mode = _resolve_llm_iteration_mode(llm_cfg)
    primary_algo = cfg.algorithm if cfg.algorithm in eval_algos else eval_algos[0]
    llm_branch_algos = list(eval_algos) if llm_iteration_mode == "per_algorithm_branches" else [primary_algo]

    def _resolve_algo_runtime(algo: str) -> dict:
        if drl_backend == "finsaber_native":
            native_algo_kwargs = _resolve_finsaber_algo_kwargs(cfg, algo)
            return {
                "is_td3_legacy": False,
                "td3_algo_base_cfg": None,
                "sb3_algo_base_cfg": None,
                "sb3_algo_kwargs": {},
                "td3_policy_action_bound": None,
                "native_cfg": native_cfg,
                "native_algo_kwargs": native_algo_kwargs,
                "native_action_bound": float(native_cfg.hmax) if native_cfg is not None else None,
                "tuning_effective": {
                    "backend": "finsaber_native",
                    "model_kwargs": native_algo_kwargs,
                },
            }
        algo_tuning = dict(algo_tuning_cfg.get(algo, {}))
        is_td3_legacy = bool(algo == "td3" and td3_backend == "legacy")
        td3_policy_action_bound = td3_default_policy_action_bound if algo == "td3" else None
        if is_td3_legacy:
            return {
                "is_td3_legacy": True,
                "td3_algo_base_cfg": td3_base_cfg,
                "sb3_algo_base_cfg": None,
                "sb3_algo_kwargs": {},
                "td3_policy_action_bound": td3_policy_action_bound,
                "tuning_effective": {
                    "backend": "legacy",
                    "config_overrides": td3_tuning_overrides,
                    "ignored_keys": td3_tuning_ignored,
                },
            }
        if algo == "td3":
            td3_algo_overrides, td3_algo_ignored = _split_td3_tuning(algo_tuning)
            td3_algo_base_cfg = replace(td3_base_cfg, **td3_algo_overrides) if td3_algo_overrides else td3_base_cfg
            td3_sb3_cfg_overrides_algo, td3_sb3_model_kwargs_algo = _split_sb3_tuning(td3_algo_ignored)
            sb3_algo_base_cfg = _apply_sb3_cfg_overrides(
                replace(
                    sb3_cfg,
                    gamma=float(td3_algo_base_cfg.discount),
                    batch_size=int(td3_algo_base_cfg.batch_size),
                ),
                td3_sb3_cfg_overrides_algo,
            )
            sb3_algo_kwargs = _td3_cfg_to_sb3_kwargs(td3_algo_base_cfg)
            sb3_algo_kwargs.update(td3_sb3_model_kwargs_algo)
            td3_policy_action_bound = _td3_policy_action_bound(td3_algo_base_cfg, cfg.max_trade)
            return {
                "is_td3_legacy": False,
                "td3_algo_base_cfg": td3_algo_base_cfg,
                "sb3_algo_base_cfg": sb3_algo_base_cfg,
                "sb3_algo_kwargs": sb3_algo_kwargs,
                "td3_policy_action_bound": td3_policy_action_bound,
                "tuning_effective": {
                    "backend": "sb3",
                    "td3_config_overrides": td3_algo_overrides,
                    "sb3_config_overrides": td3_sb3_cfg_overrides_algo,
                    "model_kwargs": sb3_algo_kwargs,
                },
            }

        sb3_cfg_overrides, sb3_algo_kwargs = _split_sb3_tuning(algo_tuning)
        sb3_algo_base_cfg = _apply_sb3_cfg_overrides(sb3_cfg, sb3_cfg_overrides)
        return {
            "is_td3_legacy": False,
            "td3_algo_base_cfg": None,
            "sb3_algo_base_cfg": sb3_algo_base_cfg,
            "sb3_algo_kwargs": sb3_algo_kwargs,
            "td3_policy_action_bound": None,
            "tuning_effective": {
                "config_overrides": sb3_cfg_overrides,
                "model_kwargs": sb3_algo_kwargs,
            },
        }

    algo_runtime_cache = {algo: _resolve_algo_runtime(algo) for algo in eval_algos}
    algo_tuning_effective: Dict[str, dict] = {
        algo: _json_safe(payload.get("tuning_effective", {}))
        for algo, payload in algo_runtime_cache.items()
    }
    selection_seeds = _resolve_candidate_selection_seeds(cfg, llm_cfg)
    branch_parallel_workers = _resolve_llm_branch_parallel_workers(llm_cfg, llm_branch_algos)
    branch_iteration_cfg = _resolve_branch_iteration_worker_cfg(llm_cfg)
    branch_iteration_mode = _effective_branch_iteration_mode(
        llm_iteration_mode,
        branch_parallel_workers,
        llm_branch_algos,
        branch_iteration_cfg,
    )
    final_selection_cfg = _resolve_final_selection_cfg(llm_cfg)
    final_selection_mode = _effective_final_selection_mode(
        llm_iteration_mode,
        branch_parallel_workers,
        eval_algos,
        final_selection_cfg,
    )
    selection_baseline_metrics_cache: Dict[tuple[str, int, tuple[int, ...]], Dict[int, dict]] = {}

    def _env_cfg_for_algo(algo: str) -> EnvConfig:
        return _env_cfg_with_algo_penalty(cfg, env_cfg, algo)

    def _baseline_metrics_for_algo(
        algo: str,
        runtime: dict,
        seeds: List[int],
        *,
        steps_small_override: int | None = None,
    ) -> Dict[int, dict]:
        algo_state_fn_raw = _state_fn_raw_for_algo(algo)
        steps_small = int(steps_small_override) if steps_small_override is not None else _effective_steps(
            cfg.n_small,
            int(train_df["date"].nunique()),
        )
        cache_key = (str(algo), int(steps_small), tuple(int(sd) for sd in seeds))
        if cache_key in selection_baseline_metrics_cache:
            return selection_baseline_metrics_cache[cache_key]
        algo_env_cfg = _env_cfg_for_algo(algo)
        metrics_by_seed: Dict[int, dict] = {}
        native_backend = drl_backend == "finsaber_native"
        runtime_native_cfg = runtime.get("native_cfg")
        runtime_native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {})
        for sd in seeds:
            if native_backend:
                runtime_native_cfg_small = replace(runtime_native_cfg, total_timesteps=int(steps_small))
                runtime_native_algo_kwargs_small = _native_small_budget_algo_kwargs(
                    algo,
                    runtime_native_algo_kwargs,
                    int(steps_small),
                )
                result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=runtime_native_cfg_small,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_small,
                    revise_state=_identity_revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    policy_state_fn=algo_state_fn_raw,
                    use_revised=False,
                    use_intrinsic=False,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode="raw",
                )
                metrics, _ = _sb3_metrics_from_eval(result)
            elif algo == "td3" and runtime["is_td3_legacy"]:
                td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
                train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
                eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
                result = train_td3(
                    env=train_env,
                    state_dim=algo_state_fn_raw(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                    action_dim=len(schema.assets),
                    cfg=td3_cfg_small,
                    max_steps=steps_small,
                    state_fn=algo_state_fn_raw,
                    revise_state=_identity_revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    use_intrinsic=False,
                    intrinsic_timing=cfg.intrinsic_timing,
                    finagent=finagent,
                    finagent_weight=cfg.finagent_weight,
                    seed=sd,
                    eval_env=eval_env,
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_input_mode="revised",
                )
                metrics = compute_metrics(np.array(result.eval_values_final))
            elif algo == "td3":
                td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
                sb3_cfg_small = _td3_cfg_to_sb3_cfg(runtime["sb3_algo_base_cfg"], td3_cfg_small, steps_small)
                sb3_kwargs_small = dict(runtime["sb3_algo_kwargs"])
                sb3_kwargs_small.update(_td3_cfg_to_sb3_kwargs(td3_cfg_small))
                result = train_sb3(
                    algo="td3",
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type="continuous",
                    policy_action_bound=runtime["td3_policy_action_bound"],
                    revise_state=_identity_revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=False,
                    use_intrinsic=False,
                    policy_state_fn=algo_state_fn_raw,
                    seed=sd,
                    algo_kwargs=sb3_kwargs_small,
                )
                metrics, _ = _sb3_metrics_from_eval(result)
            else:
                sb3_cfg_small = replace(runtime["sb3_algo_base_cfg"], total_timesteps=int(steps_small))
                result = train_sb3(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type=_action_space_type(algo),
                    policy_action_bound=None,
                    revise_state=_identity_revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=False,
                    use_intrinsic=False,
                    policy_state_fn=algo_state_fn_raw,
                    seed=sd,
                    algo_kwargs=runtime["sb3_algo_kwargs"],
                )
                metrics, _ = _sb3_metrics_from_eval(result)
            metrics_by_seed[int(sd)] = {
                "Sharpe": float(metrics.get("Sharpe", 0.0)),
                "CR": float(metrics.get("CR", 0.0)),
            }
        selection_baseline_metrics_cache[cache_key] = metrics_by_seed
        return metrics_by_seed

    def _score_candidate_payload_for_algo(
        algo: str,
        runtime: dict,
        revise_state,
        intrinsic_reward_eval,
        intrinsic_reward_probe_eval,
        policy_state_fn_candidate,
        seeds: List[int],
    ) -> dict:
        algo_state_fn_raw = _state_fn_raw_for_algo(algo)
        algo_env_cfg = _env_cfg_for_algo(algo)
        scoring_seeds, steps_small = _resolve_candidate_scoring_budget(
            cfg=cfg,
            algo=algo,
            candidate_scoring_cfg=candidate_scoring_cfg,
            requested_seeds=seeds,
            train_df=train_df,
        )
        baseline_metrics_by_seed = _baseline_metrics_for_algo(
            algo,
            runtime,
            scoring_seeds,
            steps_small_override=steps_small,
        )
        seed_scores: List[float] = []
        seed_metric_rows: List[dict] = []
        seed_perf_rows: List[dict] = []
        seed_state_perf_rows: List[dict] = []
        seed_probe_perf_rows: List[dict] = []
        seed_behavior_rows: List[dict] = []
        corrs_accum: List[np.ndarray] = []
        native_backend = drl_backend == "finsaber_native"
        runtime_native_cfg = runtime.get("native_cfg")
        runtime_native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {})
        for sd in scoring_seeds:
            if native_backend:
                runtime_native_cfg_small = replace(runtime_native_cfg, total_timesteps=int(steps_small))
                runtime_native_algo_kwargs_small = _native_small_budget_algo_kwargs(
                    algo,
                    runtime_native_algo_kwargs,
                    int(steps_small),
                )
                result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=runtime_native_cfg_small,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_small,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_eval,
                    policy_state_fn=policy_state_fn_candidate,
                    use_revised=True,
                    use_intrinsic=True,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode="revised",
                )
                metrics, _ = _sb3_metrics_from_eval(result)
                eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
                eval_trace_rows = result.get("eval_trace", []) or []
                eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
                eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
                probe_result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=runtime_native_cfg_small,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_small,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_probe_eval,
                    policy_state_fn=algo_state_fn_raw,
                    use_revised=False,
                    use_intrinsic=True,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode="raw",
                )
                probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
                state_probe_result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=runtime_native_cfg_small,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_small,
                    revise_state=revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    policy_state_fn=policy_state_fn_candidate,
                    use_revised=True,
                    use_intrinsic=False,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode="revised",
                )
                state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
            elif algo == "td3" and runtime["is_td3_legacy"]:
                td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
                train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
                eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
                result = train_td3(
                    env=train_env,
                    state_dim=policy_state_fn_candidate(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                    action_dim=len(schema.assets),
                    cfg=td3_cfg_small,
                    max_steps=steps_small,
                    state_fn=policy_state_fn_candidate,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    use_intrinsic=True,
                    intrinsic_timing=cfg.intrinsic_timing,
                    finagent=finagent,
                    finagent_weight=cfg.finagent_weight,
                    seed=sd,
                    eval_env=eval_env,
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                )
                metrics = compute_metrics(np.array(result.eval_values_final))
                corrs_accum.append(np.abs(np.array(result.corrs, dtype=np.float64)))
                eval_actions = result.eval_actions_final or []
                eval_trace_rows = result.eval_trace_final or []
                eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
                eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
                probe_train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
                probe_eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
                probe_result = train_td3(
                    env=probe_train_env,
                    state_dim=algo_state_fn_raw(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                    action_dim=len(schema.assets),
                    cfg=td3_cfg_small,
                    max_steps=steps_small,
                    state_fn=algo_state_fn_raw,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_probe_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    use_intrinsic=True,
                    intrinsic_timing=cfg.intrinsic_timing,
                    finagent=finagent,
                    finagent_weight=cfg.finagent_weight,
                    seed=sd,
                    eval_env=probe_eval_env,
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_input_mode="raw",
                )
                probe_metrics = compute_metrics(np.array(probe_result.eval_values_final))
                state_probe_train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
                state_probe_eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
                state_probe_result = train_td3(
                    env=state_probe_train_env,
                    state_dim=policy_state_fn_candidate(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                    action_dim=len(schema.assets),
                    cfg=td3_cfg_small,
                    max_steps=steps_small,
                    state_fn=policy_state_fn_candidate,
                    revise_state=revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    use_intrinsic=False,
                    intrinsic_timing=cfg.intrinsic_timing,
                    finagent=finagent,
                    finagent_weight=cfg.finagent_weight,
                    seed=sd,
                    eval_env=state_probe_eval_env,
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_input_mode="revised",
                )
                state_probe_metrics = compute_metrics(np.array(state_probe_result.eval_values_final))
            elif algo == "td3":
                td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
                sb3_cfg_small = _td3_cfg_to_sb3_cfg(runtime["sb3_algo_base_cfg"], td3_cfg_small, steps_small)
                sb3_kwargs_small = dict(runtime["sb3_algo_kwargs"])
                sb3_kwargs_small.update(_td3_cfg_to_sb3_kwargs(td3_cfg_small))
                result = train_sb3(
                    algo="td3",
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type="continuous",
                    policy_action_bound=runtime["td3_policy_action_bound"],
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_probe_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=True,
                    use_intrinsic=True,
                    intrinsic_input_mode="revised",
                    policy_state_fn=policy_state_fn_candidate,
                    seed=sd,
                    algo_kwargs=sb3_kwargs_small,
                )
                metrics, _ = _sb3_metrics_from_eval(result)
                eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
                eval_trace_rows = result.get("eval_trace", []) or []
                eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
                eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
                probe_result = train_sb3(
                    algo="td3",
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type="continuous",
                    policy_action_bound=runtime["td3_policy_action_bound"],
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_probe_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=False,
                    use_intrinsic=True,
                    intrinsic_input_mode="raw",
                    policy_state_fn=algo_state_fn_raw,
                    seed=sd,
                    algo_kwargs=sb3_kwargs_small,
                )
                probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
                state_probe_result = train_sb3(
                    algo="td3",
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type="continuous",
                    policy_action_bound=runtime["td3_policy_action_bound"],
                    revise_state=revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=True,
                    use_intrinsic=False,
                    intrinsic_input_mode="revised",
                    policy_state_fn=policy_state_fn_candidate,
                    seed=sd,
                    algo_kwargs=sb3_kwargs_small,
                )
                state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
            else:
                sb3_cfg_small = replace(runtime["sb3_algo_base_cfg"], total_timesteps=int(steps_small))
                result = train_sb3(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type=_action_space_type(algo),
                    policy_action_bound=None,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=True,
                    use_intrinsic=True,
                    intrinsic_input_mode="revised",
                    policy_state_fn=policy_state_fn_candidate,
                    seed=sd,
                    algo_kwargs=runtime["sb3_algo_kwargs"],
                )
                metrics, _ = _sb3_metrics_from_eval(result)
                eval_actions = result.get("eval_actions_executed", []) or result.get("eval_actions_policy", []) or []
                eval_trace_rows = result.get("eval_trace", []) or []
                eval_weight_rows = _extract_portfolio_weights_from_trace(eval_trace_rows)
                eval_weight_changes = _extract_portfolio_weight_changes_from_trace(eval_trace_rows)
                probe_result = train_sb3(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type=_action_space_type(algo),
                    policy_action_bound=None,
                    revise_state=revise_state,
                    intrinsic_reward=intrinsic_reward_eval,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=False,
                    use_intrinsic=True,
                    intrinsic_input_mode="raw",
                    policy_state_fn=algo_state_fn_raw,
                    seed=sd,
                    algo_kwargs=runtime["sb3_algo_kwargs"],
                )
                probe_metrics, _ = _sb3_metrics_from_eval(probe_result)
                state_probe_result = train_sb3(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    assets=schema.assets,
                    schema=schema,
                    env_cfg=algo_env_cfg,
                    cfg=sb3_cfg_small,
                    action_space_type=_action_space_type(algo),
                    policy_action_bound=None,
                    revise_state=revise_state,
                    intrinsic_reward=_zero_intrinsic_reward,
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    use_revised=True,
                    use_intrinsic=False,
                    intrinsic_input_mode="revised",
                    policy_state_fn=policy_state_fn_candidate,
                    seed=sd,
                    algo_kwargs=runtime["sb3_algo_kwargs"],
                )
                state_probe_metrics, _ = _sb3_metrics_from_eval(state_probe_result)
            perf_payload = _candidate_performance_payload(
                metrics=metrics,
                baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
                scoring_cfg=candidate_scoring_cfg,
            )
            state_perf_payload = _candidate_performance_payload(
                metrics=state_probe_metrics,
                baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
                scoring_cfg=candidate_scoring_cfg,
            )
            probe_perf_payload = _candidate_performance_payload(
                metrics=probe_metrics,
                baseline_metrics=baseline_metrics_by_seed.get(int(sd)),
                scoring_cfg=candidate_scoring_cfg,
            )
            action_bound = (
                float(runtime["td3_policy_action_bound"])
                if (algo == "td3" and runtime.get("td3_policy_action_bound") is not None)
                else (
                    float(runtime.get("native_action_bound", 0.0))
                    if native_backend and runtime.get("native_action_bound") is not None
                    else float(env_cfg.max_trade)
                )
            )
            seed_behavior_rows.append(
                {
                    "seed": int(sd),
                    **_candidate_behavior_payload(
                        eval_actions,
                        action_bound=action_bound,
                        portfolio_weights=eval_weight_rows,
                        portfolio_weight_changes=eval_weight_changes,
                    ),
                }
            )
            perf = float(perf_payload["performance_score"])
            seed_scores.append(perf)
            seed_perf_rows.append(perf_payload)
            seed_state_perf_rows.append(state_perf_payload)
            seed_probe_perf_rows.append(probe_perf_payload)
            seed_metric_rows.append(
                {
                    "seed": int(sd),
                    "Sharpe": float(metrics.get("Sharpe", 0.0)),
                    "CR": float(metrics.get("CR", 0.0)),
                    "performance_mode": str(perf_payload["performance_mode"]),
                    "performance_score": perf,
                    "performance_score_absolute": float(perf_payload["performance_score_absolute"]),
                    "performance_score_baseline": float(perf_payload["performance_score_baseline"]),
                    "performance_score_delta": float(perf_payload["performance_score_delta"]),
                    "performance_delta_sharpe": float(metrics.get("Sharpe", 0.0) - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)),
                    "state_probe_score": float(state_perf_payload["performance_score"]),
                    "state_probe_score_absolute": float(state_perf_payload["performance_score_absolute"]),
                    "state_probe_score_baseline": float(state_perf_payload["performance_score_baseline"]),
                    "state_probe_score_delta": float(state_perf_payload["performance_score_delta"]),
                    "state_probe_delta_sharpe": float(state_probe_metrics.get("Sharpe", 0.0) - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)),
                    "intrinsic_probe_score": float(probe_perf_payload["performance_score"]),
                    "intrinsic_probe_score_absolute": float(probe_perf_payload["performance_score_absolute"]),
                    "intrinsic_probe_score_baseline": float(probe_perf_payload["performance_score_baseline"]),
                    "intrinsic_probe_score_delta": float(probe_perf_payload["performance_score_delta"]),
                    "intrinsic_probe_delta_sharpe": float(probe_metrics.get("Sharpe", 0.0) - baseline_metrics_by_seed.get(int(sd), {}).get("Sharpe", 0.0)),
                }
            )

        perf_mean = float(np.mean(seed_scores)) if seed_scores else 0.0
        perf_abs_mean = (
            float(np.mean([row["performance_score_absolute"] for row in seed_perf_rows]))
            if seed_perf_rows
            else 0.0
        )
        perf_base_mean = (
            float(np.mean([row["performance_score_baseline"] for row in seed_perf_rows]))
            if seed_perf_rows
            else 0.0
        )
        perf_delta_mean = (
            float(np.mean([row["performance_score_delta"] for row in seed_perf_rows]))
            if seed_perf_rows
            else 0.0
        )
        perf_delta_sharpe_mean = (
            float(np.mean([float(seed_row.get("performance_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
            if seed_metric_rows
            else 0.0
        )
        state_perf_mean = (
            float(np.mean([row["performance_score"] for row in seed_state_perf_rows]))
            if seed_state_perf_rows
            else 0.0
        )
        state_abs_mean = (
            float(np.mean([row["performance_score_absolute"] for row in seed_state_perf_rows]))
            if seed_state_perf_rows
            else 0.0
        )
        state_base_mean = (
            float(np.mean([row["performance_score_baseline"] for row in seed_state_perf_rows]))
            if seed_state_perf_rows
            else 0.0
        )
        state_delta_mean = (
            float(np.mean([row["performance_score_delta"] for row in seed_state_perf_rows]))
            if seed_state_perf_rows
            else 0.0
        )
        state_delta_sharpe_mean = (
            float(np.mean([float(seed_row.get("state_probe_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
            if seed_metric_rows
            else 0.0
        )
        probe_perf_mean = (
            float(np.mean([row["performance_score"] for row in seed_probe_perf_rows]))
            if seed_probe_perf_rows
            else 0.0
        )
        probe_abs_mean = (
            float(np.mean([row["performance_score_absolute"] for row in seed_probe_perf_rows]))
            if seed_probe_perf_rows
            else 0.0
        )
        probe_base_mean = (
            float(np.mean([row["performance_score_baseline"] for row in seed_probe_perf_rows]))
            if seed_probe_perf_rows
            else 0.0
        )
        probe_delta_mean = (
            float(np.mean([row["performance_score_delta"] for row in seed_probe_perf_rows]))
            if seed_probe_perf_rows
            else 0.0
        )
        probe_delta_sharpe_mean = (
            float(np.mean([float(seed_row.get("intrinsic_probe_delta_sharpe", 0.0)) for seed_row in seed_metric_rows]))
            if seed_metric_rows
            else 0.0
        )
        avg_corrs: List[float] = []
        lip_raw_model = None
        if corrs_accum:
            avg_corrs_arr = np.mean(np.stack(corrs_accum, axis=0), axis=0)
            avg_corrs = list(avg_corrs_arr)
            lip_raw_model = float(np.mean(avg_corrs_arr))
        lip_payload = _estimate_intrinsic_lipschitz(
            reference_states=reference_states,
            revise_state_fn=revise_state,
            intrinsic_reward_fn=intrinsic_reward_eval,
            max_pairs=int(candidate_scoring_cfg["lipschitz_max_pairs"]),
            quantile=float(candidate_scoring_cfg["lipschitz_quantile"]),
        )
        intrinsic_signal_stats_raw = _estimate_intrinsic_signal_stats(
            revise_state_fn=revise_state,
            intrinsic_reward_fn=intrinsic_reward_probe_eval,
            reference_states=reference_states,
            input_mode="raw",
        )
        lip_raw = lip_raw_model if lip_raw_model is not None else lip_payload.get("raw")
        behavior_payload = _aggregate_candidate_behavior(seed_behavior_rows)
        score_payload = _combine_candidate_score(
            perf_mean,
            lip_raw,
            candidate_scoring_cfg,
            behavior_score=float(behavior_payload["behavior_score"]),
            intrinsic_probe_score=probe_perf_mean,
            turnover_score=float(behavior_payload["turnover_stability_score"]),
        )
        return {
            "score": float(score_payload["score"]),
            "performance_mode": str(candidate_scoring_cfg.get("performance_mode", "absolute")),
            "performance_score": float(score_payload["performance_score"]),
            "performance_score_absolute": perf_abs_mean,
            "performance_score_baseline": perf_base_mean,
            "performance_score_delta": perf_delta_mean,
            "performance_delta_sharpe": perf_delta_sharpe_mean,
            "state_probe_score": state_perf_mean,
            "state_probe_score_absolute": state_abs_mean,
            "state_probe_score_baseline": state_base_mean,
            "state_probe_score_delta": state_delta_mean,
            "state_probe_delta_sharpe": state_delta_sharpe_mean,
            "lipschitz_raw": (
                float(score_payload["lipschitz_raw"]) if score_payload["lipschitz_raw"] is not None else None
            ),
            "lipschitz_score": float(score_payload["lipschitz_score"]),
            "behavior_score": float(score_payload["behavior_score"]),
            "intrinsic_probe_score": float(score_payload["intrinsic_probe_score"]),
            "intrinsic_probe_score_absolute": probe_abs_mean,
            "intrinsic_probe_score_baseline": probe_base_mean,
            "intrinsic_probe_score_delta": probe_delta_mean,
            "intrinsic_probe_delta_sharpe": probe_delta_sharpe_mean,
            "intrinsic_signal_stats_raw": intrinsic_signal_stats_raw,
            "intrinsic_signal_nontrivial_raw": bool(intrinsic_signal_stats_raw.get("nontrivial", False)),
            "behavior": behavior_payload,
            "turnover_score": float(score_payload["turnover_score"]),
            "seed_behavior": seed_behavior_rows,
            "seed_metrics": seed_metric_rows,
            "corrs": avg_corrs,
        }

    primary_baseline_metrics_by_algo: Dict[str, Dict[int, dict]] = {}
    if llm_cfg.get("enabled", False) and candidate_scoring_cfg.get("performance_mode") == "delta_to_g0":
        use_parallel_baselines = bool(
            llm_iteration_mode == "per_algorithm_branches"
            and branch_parallel_workers > 1
            and len(llm_branch_algos) > 1
            and os.name != "nt"
        )
        if use_parallel_baselines:
            with ThreadPoolExecutor(
                max_workers=branch_parallel_workers,
                thread_name_prefix="lesr-baseline",
            ) as executor:
                future_to_algo = {
                    executor.submit(
                        _baseline_metrics_for_algo,
                        algo,
                        algo_runtime_cache[algo],
                        selection_seeds,
                    ): algo
                    for algo in llm_branch_algos
                }
                for future in as_completed(future_to_algo):
                    algo = future_to_algo[future]
                    primary_baseline_metrics_by_algo[algo] = future.result()
        else:
            for algo in llm_branch_algos:
                primary_baseline_metrics_by_algo[algo] = _baseline_metrics_for_algo(
                    algo,
                    algo_runtime_cache[algo],
                    selection_seeds,
                )

    # prepare candidates
    candidate_entries_by_algo: Dict[str, List[Tuple[str, str]]] = {algo: [] for algo in eval_algos}
    llm_responses = []
    llm_errors = []
    iter_trace = []
    branch_iteration_artifacts: List[dict] = []
    if llm_cfg.get("enabled", False):
        branch_client_by_algo = {
            algo: deepseek_from_env(
                llm_cfg["base_url"],
                timeout_s=int(llm_cfg.get("request_timeout_s", 60)),
                use_env_proxy=_coerce_bool(llm_cfg.get("use_env_proxy"), default=False),
            )
            for algo in llm_branch_algos
        }

        def _algo_branch_instruction(branch_algo: str) -> str:
            if llm_iteration_mode != "per_algorithm_branches":
                return ""
            action_space = _action_space_type(branch_algo, drl_backend)
            mechanism_hint = {
                "a2c": (
                    "Continuous-control bias under finsaber_native: prefer smooth, bounded revise_state features that "
                    "improve ranking, separability, and regime clarity without pushing actions to constant extremes."
                    if drl_backend == "finsaber_native"
                    else "Discrete-control bias: prioritize revise_state features that improve ranking, separability, "
                    "trend-confidence, and regime clarity. Keep intrinsic smooth and secondary."
                ),
                "ppo": (
                    "Continuous-control bias under finsaber_native: favor robust revise_state signal shaping first; "
                    "intrinsic reward should stabilize confidence without collapsing action diversity."
                    if drl_backend == "finsaber_native"
                    else "Discrete-control bias: favor robust revise_state signal shaping first; intrinsic reward should "
                    "stabilize confidence rather than dominate behavior."
                ),
                "sac": (
                    "Continuous-control bias: prefer balanced revise+intrinsic designs with confidence-gated risk "
                    "penalties. Avoid aggressive monotonic suppression that can flatten useful exploration or drive "
                    "near-bound action saturation; preserve action diversity and non-collapsed behavior."
                ),
                "td3": (
                    "Continuous-control bias: prefer smooth, bounded, action-sensitive intrinsic terms. Avoid "
                    "reward shaping that changes totals but leaves policy behavior or action diversity unchanged. "
                    "Explicitly avoid candidates that push actions to the boundary most of the time."
                ),
            }.get(
                branch_algo,
                "Prefer smooth, bounded, action-sensitive candidates with explicit confidence or risk gating.",
            )
            return (
                "Current LESR search branch target RL algorithm: "
                f"`{branch_algo}` (action_space=`{action_space}`).\n"
                + (
                    "This branch is using backend-specific state semantics; follow the authoritative state contract note "
                    "instead of assuming the generic LESR state layout. "
                    if state_contract_note
                    else "Use the same generic LESR mechanism as every other branch. "
                )
                + "Do not rely on algorithm-specific hacks; prefer robust, action-sensitive, smooth candidates that "
                + "score well under this branch's short-train feedback.\n"
                + f"{mechanism_hint}\n"
                + "The candidate scorer now checks whether intrinsic reward still helps under raw-policy control; "
                + "avoid designs where revise_state carries all useful signal and intrinsic_reward becomes negligible.\n"
                + "Shared objective: search for structures that could remain plausible across other branches and "
                + "market windows, not only this branch."
            )

        def _branch_prompt(base_prompt: str, branch_algo: str) -> str:
            extra = _algo_branch_instruction(branch_algo)
            if not extra:
                return base_prompt
            return f"{base_prompt}\n\n{extra}"

        def _summarize_error_types(errors: List[dict], top_k: int = 4) -> str:
            sample_errors = [str(row.get("error_type", "unknown")) for row in errors or [] if row.get("phase") == "sample"]
            if not sample_errors:
                return ""
            counts = Counter(sample_errors)
            top_rows = [f"{name}={counts[name]}" for name, _ in counts.most_common(top_k)]
            return "Recent sample failure mix: " + ", ".join(top_rows)

        def _build_branch_iteration_feedback(branch_algo: str, candidate_stats: List[dict], errors: List[dict]) -> str:
            lines: List[str] = []
            if candidate_stats:
                ranked = _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)
                best = ranked[0] if ranked else max(candidate_stats, key=lambda row: float(row.get("score", 0.0)))
                behavior = best.get("behavior", {}) or {}
                lines.append(
                    "Latest branch best: "
                    f"name={best.get('name')}, family={best.get('family')}, design_mode={best.get('design_mode')}, "
                    f"perf_delta={float(best.get('performance_score_delta', 0.0)):.4f}, "
                    f"intrinsic_probe_delta={float(best.get('intrinsic_probe_score_delta', 0.0)):.4f}, "
                    f"lipschitz={float(best.get('lipschitz_raw') or 0.0):.4f}, "
                    f"behavior_score={float(best.get('behavior_score', 0.0)):.4f}, "
                    f"near_bound={float(behavior.get('near_bound_ratio_mean', 0.0)):.4f}, "
                    f"avg_weight_change={float(behavior.get('avg_daily_portfolio_weight_change_mean', 0.0)):.4f}, "
                    f"turnover_score={float(best.get('turnover_score', 0.0)):.4f}"
                )
            error_summary = _summarize_error_types(errors)
            if error_summary:
                lines.append(error_summary)
            if not lines:
                return ""
            return "Structured branch feedback:\n" + "\n".join(f"- {line}" for line in lines)

        def _build_cross_algo_iteration_feedback(branch_results: Dict[str, dict]) -> str:
            if llm_iteration_mode != "per_algorithm_branches":
                return ""
            lines = [
                "Cross-algorithm search snapshot from the latest iteration:",
            ]
            for algo_name in llm_branch_algos:
                iter_log_payload = branch_results.get(algo_name, {}).get("iter_log", {})
                candidates = [row for row in iter_log_payload.get("candidates", []) if row.get("valid")]
                if not candidates:
                    lines.append(f"- {algo_name}: no valid candidates in the latest iteration; avoid repeating recent failure patterns.")
                    continue
                ranked = _rank_candidate_rows(candidates, candidate_scoring_cfg)
                best = ranked[0] if ranked else max(candidates, key=lambda row: float(row.get("score", 0.0)))
                lines.append(
                    f"- {algo_name}: best_family={best.get('family')}, design_mode={best.get('design_mode')}, "
                    f"perf_delta={float(best.get('performance_score_delta', 0.0)):.4f}, "
                    f"intrinsic_probe_delta={float(best.get('intrinsic_probe_score_delta', 0.0)):.4f}, "
                    f"behavior_score={float(best.get('behavior_score', 0.0)):.4f}, "
                    f"lipschitz={float(best.get('lipschitz_raw') or 0.0):.4f}, "
                    f"avg_weight_change={float((best.get('behavior') or {}).get('avg_daily_portfolio_weight_change_mean', 0.0)):.4f}"
                )
            lines.append(
                "- Synthesis target: preserve branch gains without causing near-bound action collapse, avoid intrinsic channels that are numerically non-zero but irrelevant under raw-policy control, and keep portfolio-weight changes smooth."
            )
            return "\n".join(lines)

        scenario_priority: List[str] = list(scenario_families)
        active_family = str(scenario_profile.get("family", "")).strip()
        if scenario_enabled and active_family in scenario_priority:
            scenario_priority = [active_family] + [f for f in scenario_priority if f != active_family]

        def _build_design_schedule(count: int) -> List[str]:
            if count <= 0:
                return []
            base = ["intrinsic_first", "balanced", "state_first"]
            return [base[i % len(base)] for i in range(count)]

        def _design_mode_instruction(design_mode: str, branch_algo: str) -> str:
            if design_mode == "intrinsic_first":
                return (
                    "Design mode: intrinsic_first.\n"
                    f"Target RL branch: `{branch_algo}`.\n"
                    "Build a candidate where intrinsic_reward is a primary signal even when use_revised=False.\n"
                    "intrinsic_reward must depend directly on original holdings/price/risk exposure terms, not only on revised proxy dims.\n"
                    "Use revised dims mainly as confidence gates or context scalars; avoid making revise_state carry the whole edge.\n"
                    "Prefer using global running-risk dims such as return EMA, return-squared EMA, drawdown, and turnover EMA when available.\n"
                    "Target a genuinely useful G2 path: avoid intrinsic functions that are mostly one-sided negative offsets with near-zero standalone Sharpe impact.\n"
                    "Prefer a bounded opportunity-minus-risk structure whose standalone effect can improve at least some raw-policy branches."
                )
            if design_mode == "state_first":
                return (
                    "Design mode: state_first.\n"
                    f"Target RL branch: `{branch_algo}`.\n"
                    "Build a candidate where revise_state does most of the work, while intrinsic_reward is a secondary stabilizer.\n"
                    "Keep intrinsic smooth and bounded; do not let it dominate behavior.\n"
                    "Portfolio-memory and running-risk dims may be used only as stabilizers, not as the sole edge."
                )
            return (
                "Design mode: balanced.\n"
                f"Target RL branch: `{branch_algo}`.\n"
                "Build a candidate where revise_state and intrinsic_reward both contribute meaningfully.\n"
                "Avoid candidates where either path is numerically non-trivial but behaviorally irrelevant.\n"
                "Running-risk and portfolio-memory dims may be used, but neither path should collapse into a pure turnover suppressor."
            )

        def _sampling_instruction(family: str, branch_algo: str, design_mode: str) -> str:
            family_block = ""
            if family:
                family_block = (
                    "Target family:\n"
                    f"- family={family}\n"
                    "Family semantics:\n"
                    "- trend_follow: action-sensitive trend/rank positive term + smooth risk budget.\n"
                    "- mean_revert: deviation-repair/position-gap repair with bounded control.\n"
                    "- risk_shield: volatility/drawdown-aware guard that avoids over-clipping.\n"
                )
            return (
                "Generate ONE candidate code pair.\n"
                f"{family_block}"
                "Current window profile:\n"
                f"- inferred_family={scenario_profile.get('family')}\n"
                f"- mu_ann={float(scenario_profile.get('mu_ann', 0.0)):.4f}\n"
                f"- vol_ann={float(scenario_profile.get('vol_ann', 0.0)):.4f}\n"
                f"- max_dd={float(scenario_profile.get('max_dd', 0.0)):.4f}\n"
                f"- vol_short_ann={float(scenario_profile.get('vol_short_ann', 0.0)):.4f}\n"
                f"- vol_long_ann={float(scenario_profile.get('vol_long_ann', 0.0)):.4f}\n"
                f"- vol_ratio_20_60={float(scenario_profile.get('vol_ratio_20_60', 0.0)):.4f}\n"
                f"- trend_strength_20={float(scenario_profile.get('trend_strength_20', 0.0)):.4f}\n"
                f"- dispersion_20={float(scenario_profile.get('dispersion_20', 0.0)):.6f}\n"
                f"- market_stress_score={float(scenario_profile.get('market_stress_score', 0.0)):.4f}\n"
                f"{_design_mode_instruction(design_mode, branch_algo)}\n"
                "Mechanism targeting rule:\n"
                "- Use the window profile above explicitly. If inferred_family is trend_follow, prefer persistent trend/rank/confidence structures. If mean_revert, prefer deviation-repair / position-gap repair / snapback logic. If risk_shield, prefer volatility, drawdown, and concentration-aware protection.\n"
                "- The candidate must be materially different from recent history. Do not submit the same cash_ratio + concentration + trend_strength + dispersion scaffold with only renamed variables or coefficient tweaks.\n"
                "- Change at least one of: core feature family, normalization rule, gating logic, or intrinsic opportunity/risk decomposition.\n"
                "Numeric safety rule:\n"
                "- Every ratio or normalization must be division-safe: use denominator floors and explicit fallback branches for tiny denominators.\n"
                "- Do not use raw mean/std/volume/position ratios without a safe branch when the scale is near zero.\n"
                "If possible, declare FEATURE_GROUPS = [...] using any of: portfolio_memory, regime, dispersion, running_risk_state.\n"
                "Prefer revise_state features that include portfolio-memory terms such as cash ratio, concentration, entropy, exposure ratio, or rebalancing pressure.\n"
                "Keep constraints valid and keep intrinsic_reward informative under raw-policy control while avoiding unstable portfolio-weight jumps."
            )

        branch_state_by_algo = {
            algo: {
                "dialogs": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _branch_prompt(prompt, algo)},
                ],
                "all_it_func_results": [],
                "all_it_cot_suggestions": [],
                "seen_candidate_hashes": set(),
            }
            for algo in llm_branch_algos
        }
        shared_candidates: List[Tuple[str, str]] = []
        max_iterations = int(llm_cfg.get("iterations", 1))

        def _run_branch_iteration(branch_algo: str, it: int) -> dict:
            branch_state = branch_state_by_algo[branch_algo]
            dialogs = list(branch_state["dialogs"])
            all_it_func_results = list(branch_state["all_it_func_results"])
            all_it_cot_suggestions = list(branch_state["all_it_cot_suggestions"])
            global_seen_candidate_hashes = set(branch_state["seen_candidate_hashes"])
            iteration_seen_candidate_hashes: Set[str] = set()
            local_llm_responses: List[dict] = []
            local_llm_errors: List[dict] = []
            family_schedule: List[str] = []
            target_valid_count = int(max(1, llm_cfg.get("k", 2)))
            if scenario_enabled:
                for fam in scenario_priority:
                    for _ in range(candidates_per_family):
                        family_schedule.append(fam)
                if family_schedule:
                    target_valid_count = len(family_schedule)
            design_schedule: List[str] = _build_design_schedule(target_valid_count)
            valid_sample_count = 0
            revise_code_buffer: List[str] = []
            revise_dim_buffer: List[int] = []
            assistant_reply_buffer: List[str] = []
            candidate_hash_map: Dict[str, str] = {}
            trying_count = 0
            failed_sample_calls = 0
            max_failed_calls = int(max(1, llm_cfg.get("max_failed_calls", 10)))
            max_empty_response_calls = int(max(1, llm_cfg.get("max_empty_response_calls", max_failed_calls * 2)))
            max_invalid_code_calls = int(max(1, llm_cfg.get("max_invalid_code_calls", max_failed_calls * 2)))
            max_duplicate_calls = int(max(max_failed_calls, llm_cfg.get("max_duplicate_calls", max_failed_calls * 4)))
            max_validation_failed_calls = int(
                max(max_failed_calls, llm_cfg.get("max_validation_failed_calls", max_failed_calls * 3))
            )
            sample_failure_counters = {
                "hard": 0,
                "empty_response": 0,
                "invalid_code": 0,
                "duplicate_candidate": 0,
                "validation_failed": 0,
            }
            stop_sampling = False
            candidates_it: List[Tuple[str, str]] = []
            candidate_family_map: Dict[str, str] = {}
            candidate_design_map: Dict[str, str] = {}
            iter_log = {
                "algorithm": branch_algo,
                "iteration": it,
                "prompt": dialogs[-1]["content"],
                "prompt_length": int(len(dialogs[-1]["content"])),
                "candidates": [],
                "feedback": None,
                "generation_target": generation_target,
                "scenario_enabled": bool(scenario_enabled),
                "scenario_profile": scenario_profile,
                "llm_iteration_mode": llm_iteration_mode,
            }
            if scenario_enabled:
                iter_log["family_schedule"] = list(family_schedule)
            iter_log["design_schedule"] = list(design_schedule)

            def _truncate_text(text: str, max_chars: int) -> str:
                text = str(text or "").strip()
                if max_chars <= 0 or len(text) <= max_chars:
                    return text
                head = max(0, int(max_chars * 0.65))
                tail = max(0, max_chars - head - len("\n...\n[truncated]\n...\n"))
                if tail <= 0:
                    return text[:max_chars]
                return f"{text[:head]}\n...\n[truncated]\n...\n{text[-tail:]}"

            def _fresh_iteration_prompt(extra_sections: List[str] | None = None) -> str:
                if all_it_func_results or all_it_cot_suggestions:
                    base_prompt = build_next_iteration_prompt(
                        cfg.task_description,
                        state_desc,
                        all_it_func_results,
                        all_it_cot_suggestions,
                        state_contract_note=state_contract_note,
                    )
                else:
                    base_prompt = build_initial_prompt(
                        cfg.task_description,
                        state_desc,
                        state_contract_note=state_contract_note,
                    )
                return _append_prompt_sections(base_prompt, extra_sections or [])

            def _record_sample_failure(error_type: str, message: str, attempt: int, extra: dict | None = None) -> bool:
                nonlocal failed_sample_calls
                counter_key = "hard"
                failure_limit = max_failed_calls
                limit_error_type = "max_failed_calls_reached"
                if error_type == "sample_empty_response":
                    counter_key = "empty_response"
                    failure_limit = max_empty_response_calls
                    limit_error_type = "max_empty_response_calls_reached"
                elif error_type == "invalid_code":
                    counter_key = "invalid_code"
                    failure_limit = max_invalid_code_calls
                    limit_error_type = "max_invalid_code_calls_reached"
                elif error_type == "duplicate_candidate":
                    counter_key = "duplicate_candidate"
                    failure_limit = max_duplicate_calls
                    limit_error_type = "max_duplicate_calls_reached"
                elif error_type == "validation_failed":
                    counter_key = "validation_failed"
                    failure_limit = max_validation_failed_calls
                    limit_error_type = "max_validation_failed_calls_reached"
                sample_failure_counters[counter_key] += 1
                failed_sample_calls = int(sum(sample_failure_counters.values()))
                payload = {
                    "algorithm": branch_algo,
                    "iteration": it,
                    "phase": "sample",
                    "attempt": attempt,
                    "error_type": error_type,
                    "message": message,
                    "failure_counter_key": counter_key,
                    "failure_counter_value": int(sample_failure_counters[counter_key]),
                    "failure_counter_limit": int(failure_limit),
                }
                if extra:
                    payload.update(extra)
                local_llm_errors.append(payload)
                if sample_failure_counters[counter_key] >= failure_limit:
                    local_llm_errors.append(
                        {
                            "algorithm": branch_algo,
                            "iteration": it,
                            "phase": "sample",
                            "attempt": attempt,
                            "error_type": limit_error_type,
                            "message": f"{counter_key} reached {failure_limit}, fallback to static candidates",
                            "failure_counter_key": counter_key,
                            "failure_counter_value": int(sample_failure_counters[counter_key]),
                            "failure_counter_limit": int(failure_limit),
                        }
                    )
                    return True
                return False

            def _compress_iteration_memory(candidate_rows: List[dict], errors: List[dict]) -> str:
                if not candidate_rows and not errors:
                    return ""
                lines: List[str] = []
                if candidate_rows:
                    ranked_rows = _rank_candidate_rows(candidate_rows, candidate_scoring_cfg)
                    best_row = ranked_rows[0] if ranked_rows else max(candidate_rows, key=lambda row: float(row.get("score", 0.0)))
                    behavior = best_row.get("behavior", {}) or {}
                    lines.append(
                        "best="
                        f"{best_row.get('name')} family={best_row.get('family')} design={best_row.get('design_mode')} "
                        f"score={float(best_row.get('score', 0.0)):.4f} "
                        f"perf_delta={float(best_row.get('performance_score_delta', 0.0)):.4f} "
                        f"intrinsic_delta={float(best_row.get('intrinsic_probe_score_delta', 0.0)):.4f} "
                        f"lipschitz={float(best_row.get('lipschitz_raw') or 0.0):.4f} "
                        f"behavior={float(best_row.get('behavior_score', 0.0)):.4f} "
                        f"near_bound={float(behavior.get('near_bound_ratio_mean', 0.0)):.4f}"
                    )
                error_summary = _summarize_error_types(errors)
                if error_summary:
                    lines.append(error_summary)
                return "\n".join(lines)

            def _add_fallback_candidates(start_slot: int) -> None:
                fallback_candidates = _static_candidate_codes_for_backend(schema, drl_backend)
                if not fallback_candidates:
                    return
                fallback_schedule = family_schedule if scenario_enabled and family_schedule else ["global_best"]
                for slot_idx in range(start_slot, int(target_valid_count)):
                    fam = fallback_schedule[slot_idx] if slot_idx < len(fallback_schedule) else fallback_schedule[-1]
                    design_mode = design_schedule[slot_idx] if slot_idx < len(design_schedule) else "balanced"
                    added = False
                    for offset in range(len(fallback_candidates)):
                        fallback_base_name, fallback_code = fallback_candidates[(slot_idx + offset) % len(fallback_candidates)]
                        fallback_hash = _sha256_text(fallback_code)
                        if fallback_hash in iteration_seen_candidate_hashes or fallback_hash in global_seen_candidate_hashes:
                            continue
                        fallback_prefix = f"{branch_algo}_" if llm_iteration_mode == "per_algorithm_branches" else ""
                        fallback_name = f"{fallback_prefix}fallback_it{it}_{fam}_{fallback_base_name}_{slot_idx}"
                        try:
                            revise_state_fb, intrinsic_reward_fb = load_functions_from_code(fallback_code)
                            fallback_dim, _ = _validate_candidate_pair(revise_state_fb, intrinsic_reward_fb)
                            iteration_seen_candidate_hashes.add(fallback_hash)
                            candidate_hash_map[fallback_name] = fallback_hash
                            candidates_it.append((fallback_name, fallback_code))
                            candidate_family_map[fallback_name] = fam
                            candidate_design_map[fallback_name] = design_mode
                            revise_code_buffer.append(fallback_code)
                            revise_dim_buffer.append(fallback_dim)
                            assistant_reply_buffer.append(fallback_code)
                            local_llm_errors.append(
                                {
                                    "algorithm": branch_algo,
                                    "iteration": it,
                                    "phase": "sample",
                                    "attempt": trying_count,
                                    "error_type": "fallback_static_candidate",
                                    "message": "use_static_candidate_to_fill_missing_slots",
                                    "fallback_name": fallback_name,
                                    "slot": int(slot_idx),
                                }
                            )
                            added = True
                            break
                        except Exception as exc:
                            local_llm_errors.append(
                                {
                                    "algorithm": branch_algo,
                                    "iteration": it,
                                    "phase": "sample",
                                    "attempt": trying_count,
                                    "error_type": "fallback_static_candidate_failed",
                                    "message": str(exc),
                                    "fallback_name": fallback_name,
                                    "slot": int(slot_idx),
                                }
                            )
                    if not added:
                        local_llm_errors.append(
                            {
                                "algorithm": branch_algo,
                                "iteration": it,
                                "phase": "sample",
                                "attempt": trying_count,
                                "error_type": "fallback_static_candidate_exhausted",
                                "message": "no additional unseen fallback candidate available",
                                "slot": int(slot_idx),
                            }
                        )
                        break

            while valid_sample_count < target_valid_count:
                trying_count += 1
                if trying_count > 50:
                    break
                requested_family = (
                    family_schedule[valid_sample_count]
                    if scenario_enabled and valid_sample_count < len(family_schedule)
                    else ""
                )
                requested_design_mode = (
                    design_schedule[valid_sample_count]
                    if valid_sample_count < len(design_schedule)
                    else "balanced"
                )
                sample_messages = dialogs + [
                    {
                        "role": "user",
                        "content": _sampling_instruction(
                            requested_family,
                            branch_algo,
                            requested_design_mode,
                        ),
                    }
                ]
                content = _llm_chat_with_retries(
                    client=branch_client_by_algo[branch_algo],
                    llm_cfg=llm_cfg,
                    messages=sample_messages,
                    llm_errors=local_llm_errors,
                    iteration=it,
                    phase="sample",
                )
                if content is None:
                    stop_sampling = _record_sample_failure(
                        error_type="sample_empty_response",
                        message="llm_response_empty_or_failed_after_retries",
                        attempt=trying_count,
                    )
                    if stop_sampling:
                        break
                    continue
                local_llm_responses.append(
                    {
                        "algorithm": branch_algo,
                        "iteration": it,
                        "attempt": trying_count,
                        "index": valid_sample_count,
                        "family": requested_family or "global_best",
                        "design_mode": requested_design_mode,
                        "content": content,
                    }
                )
                code = extract_lesr_code(content)
                if "def revise_state" not in code or "def intrinsic_reward" not in code:
                    has_revise = bool("def revise_state" in code)
                    has_intrinsic = bool("def intrinsic_reward" in code)
                    stop_sampling = _record_sample_failure(
                        error_type="invalid_code",
                        message=(
                            "missing function definitions:"
                            f"revise_state={has_revise},intrinsic_reward={has_intrinsic}"
                        ),
                        attempt=trying_count,
                        extra={"code_len": int(len(code)), "response_len": int(len(content))},
                    )
                    if stop_sampling:
                        break
                    continue
                try:
                    _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
                    revise_state, intrinsic_reward = load_functions_from_code(code)
                    revised_dim, _ = _validate_candidate_pair(revise_state, intrinsic_reward)
                except Exception as exc:
                    stop_sampling = _record_sample_failure(
                        error_type="validation_failed",
                        message=str(exc),
                        attempt=trying_count,
                    )
                    if stop_sampling:
                        break
                    continue

                code_hash = _sha256_text(code)
                if code_hash in iteration_seen_candidate_hashes or code_hash in global_seen_candidate_hashes:
                    stop_sampling = _record_sample_failure(
                        error_type="duplicate_candidate",
                        message="duplicate candidate code hash within this algorithm branch",
                        attempt=trying_count,
                    )
                    if stop_sampling:
                        break
                    continue
                iteration_seen_candidate_hashes.add(code_hash)

                prefix = f"{branch_algo}_" if llm_iteration_mode == "per_algorithm_branches" else ""
                if requested_family:
                    name = f"{prefix}llm_it{it}_{requested_family}_k{valid_sample_count}"
                    candidate_family_map[name] = requested_family
                else:
                    name = f"{prefix}llm_it{it}_k{valid_sample_count}"
                    candidate_family_map[name] = "global_best"
                candidate_design_map[name] = requested_design_mode
                candidate_hash_map[name] = code_hash
                candidates_it.append((name, code))
                revise_code_buffer.append(code)
                revise_dim_buffer.append(revised_dim)
                assistant_reply_buffer.append(content)
                valid_sample_count += 1

            if len(candidates_it) < int(target_valid_count):
                _add_fallback_candidates(len(candidates_it))

            candidate_stats = []
            every_score = []
            results_corr = []
            runtime = algo_runtime_cache[branch_algo]
            for idx, (name, code) in enumerate(candidates_it):
                feature_groups = _infer_candidate_feature_groups(code)
                component_hashes = _extract_candidate_component_hashes(code)
                cand_entry = {
                    "algorithm": branch_algo,
                    "name": name,
                    "family": candidate_family_map.get(name, "global_best"),
                    "design_mode": candidate_design_map.get(name, "balanced"),
                    "origin": _candidate_origin_from_name(name),
                    "feature_groups": feature_groups,
                    "revise_hash": component_hashes.get("revise_hash", ""),
                    "intrinsic_hash": component_hashes.get("intrinsic_hash", ""),
                    "code": code,
                    "valid": False,
                    "score": None,
                    "corrs": None,
                    "error": None,
                    "revised_dim": revise_dim_buffer[idx] if idx < len(revise_dim_buffer) else None,
                }
                try:
                    _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
                    revise_state, intrinsic_reward = load_functions_from_code(code)
                    checked_dim, _ = _validate_candidate_pair(revise_state, intrinsic_reward)
                    cand_entry["revised_dim"] = checked_dim
                except Exception as exc:
                    cand_entry["error"] = str(exc)
                    iter_log["candidates"].append(cand_entry)
                    continue
                intrinsic_reward_eval = _prepare_intrinsic(revise_state, intrinsic_reward)
                intrinsic_reward_probe_eval = _prepare_intrinsic_for_selection(
                    revise_state,
                    intrinsic_reward,
                    cfg=cfg,
                    reference_states=reference_states,
                    input_mode="raw",
                )
                policy_state_fn_candidate = _build_policy_state_fn(revise_state)
                score_payload = _score_candidate_payload_for_algo(
                    algo=branch_algo,
                    runtime=runtime,
                    revise_state=revise_state,
                    intrinsic_reward_eval=intrinsic_reward_eval,
                    intrinsic_reward_probe_eval=intrinsic_reward_probe_eval,
                    policy_state_fn_candidate=policy_state_fn_candidate,
                    seeds=selection_seeds,
                )
                candidate_stats.append(
                    {
                        "name": name,
                        "family": candidate_family_map.get(name, "global_best"),
                        "design_mode": candidate_design_map.get(name, "balanced"),
                        "origin": _candidate_origin_from_name(name),
                        "feature_groups": feature_groups,
                        "revise_hash": component_hashes.get("revise_hash", ""),
                        "intrinsic_hash": component_hashes.get("intrinsic_hash", ""),
                        "score": float(score_payload["score"]),
                        "performance_mode": str(score_payload["performance_mode"]),
                        "performance_score": float(score_payload["performance_score"]),
                        "performance_score_absolute": float(score_payload["performance_score_absolute"]),
                        "performance_score_baseline": float(score_payload["performance_score_baseline"]),
                        "performance_score_delta": float(score_payload["performance_score_delta"]),
                        "performance_delta_sharpe": float(score_payload["performance_delta_sharpe"]),
                        "state_probe_score": float(score_payload["state_probe_score"]),
                        "state_probe_score_absolute": float(score_payload["state_probe_score_absolute"]),
                        "state_probe_score_baseline": float(score_payload["state_probe_score_baseline"]),
                        "state_probe_score_delta": float(score_payload["state_probe_score_delta"]),
                        "state_probe_delta_sharpe": float(score_payload["state_probe_delta_sharpe"]),
                        "intrinsic_probe_score": float(score_payload["intrinsic_probe_score"]),
                        "intrinsic_probe_score_absolute": float(score_payload["intrinsic_probe_score_absolute"]),
                        "intrinsic_probe_score_baseline": float(score_payload["intrinsic_probe_score_baseline"]),
                        "intrinsic_probe_score_delta": float(score_payload["intrinsic_probe_score_delta"]),
                        "intrinsic_probe_delta_sharpe": float(score_payload["intrinsic_probe_delta_sharpe"]),
                        "intrinsic_signal_stats_raw": dict(score_payload.get("intrinsic_signal_stats_raw", {}) or {}),
                        "intrinsic_signal_nontrivial_raw": bool(
                            score_payload.get("intrinsic_signal_nontrivial_raw", False)
                        ),
                        "lipschitz_raw": score_payload["lipschitz_raw"],
                        "lipschitz_score": float(score_payload["lipschitz_score"]),
                        "behavior_score": float(score_payload["behavior_score"]),
                        "turnover_score": float(score_payload["turnover_score"]),
                        "behavior": score_payload["behavior"],
                        "corrs": score_payload["corrs"],
                    }
                )
                every_score.append(float(score_payload["score"]))
                results_corr.append(score_payload["corrs"])

                cand_entry["valid"] = True
                cand_entry["score"] = float(score_payload["score"])
                cand_entry["performance_mode"] = str(score_payload["performance_mode"])
                cand_entry["performance_score"] = float(score_payload["performance_score"])
                cand_entry["performance_score_absolute"] = float(score_payload["performance_score_absolute"])
                cand_entry["performance_score_baseline"] = float(score_payload["performance_score_baseline"])
                cand_entry["performance_score_delta"] = float(score_payload["performance_score_delta"])
                cand_entry["performance_delta_sharpe"] = float(score_payload["performance_delta_sharpe"])
                cand_entry["state_probe_score"] = float(score_payload["state_probe_score"])
                cand_entry["state_probe_score_absolute"] = float(score_payload["state_probe_score_absolute"])
                cand_entry["state_probe_score_baseline"] = float(score_payload["state_probe_score_baseline"])
                cand_entry["state_probe_score_delta"] = float(score_payload["state_probe_score_delta"])
                cand_entry["state_probe_delta_sharpe"] = float(score_payload["state_probe_delta_sharpe"])
                cand_entry["intrinsic_probe_score"] = float(score_payload["intrinsic_probe_score"])
                cand_entry["intrinsic_probe_score_absolute"] = float(score_payload["intrinsic_probe_score_absolute"])
                cand_entry["intrinsic_probe_score_baseline"] = float(score_payload["intrinsic_probe_score_baseline"])
                cand_entry["intrinsic_probe_score_delta"] = float(score_payload["intrinsic_probe_score_delta"])
                cand_entry["intrinsic_probe_delta_sharpe"] = float(score_payload["intrinsic_probe_delta_sharpe"])
                cand_entry["lipschitz_raw"] = score_payload["lipschitz_raw"]
                cand_entry["lipschitz_score"] = float(score_payload["lipschitz_score"])
                cand_entry["behavior_score"] = float(score_payload["behavior_score"])
                cand_entry["turnover_score"] = float(score_payload["turnover_score"])
                cand_entry["behavior"] = score_payload["behavior"]
                cand_entry["seed_behavior"] = score_payload["seed_behavior"]
                cand_entry["corrs"] = score_payload["corrs"]
                cand_entry["seed_metrics"] = score_payload["seed_metrics"]
                cand_entry["revised_dim_delta"] = (
                    int(cand_entry["revised_dim"] - schema.dim())
                    if cand_entry["revised_dim"] is not None
                    else None
                )
                iter_log["candidates"].append(cand_entry)

            if iter_log["candidates"]:
                ranked_names = [s["name"] for s in _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)]
                rank_map = {name: idx + 1 for idx, name in enumerate(ranked_names)}
                for c in iter_log["candidates"]:
                    c["rank"] = rank_map.get(c["name"])
                if scenario_enabled:
                    iter_log["family_counts"] = dict(
                        Counter(
                            str(c.get("family", "unknown"))
                            for c in iter_log["candidates"]
                            if bool(c.get("valid"))
                        )
                    )

            next_dialogs = dialogs
            dialog_text = ""
            if candidate_stats:
                ranked_candidates = _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)
                promoted_count = int(max(1, llm_cfg.get("global_seen_promote_top_n", min(2, max(1, target_valid_count)))))
                for promoted_row in ranked_candidates[:promoted_count]:
                    promoted_hash = candidate_hash_map.get(str(promoted_row.get("name", "")))
                    if promoted_hash:
                        global_seen_candidate_hashes.add(promoted_hash)
                best_candidate_name = (
                    str(ranked_candidates[0]["name"])
                    if ranked_candidates
                    else str(candidate_stats[int(np.argmax(np.array(every_score)))]["name"])
                )
                max_id = next(
                    (idx for idx, (cand_name, _code) in enumerate(candidates_it) if cand_name == best_candidate_name),
                    int(np.argmax(np.array(every_score))),
                )
                cot_prompt, cur_it_func_results = build_cot_prompt(
                    revise_code_buffer,
                    every_score,
                    max_id,
                    results_corr,
                    revise_dim_buffer,
                    schema.dim(),
                    task_name=f"{branch_algo.upper()} short-train score",
                )
                all_it_func_results.append(_compress_iteration_memory(candidate_stats, local_llm_errors) or cur_it_func_results)
                cot_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": assistant_reply_buffer[max_id]},
                    {"role": "user", "content": cot_prompt},
                ]
                cot_suggestion = _llm_chat_with_retries(
                    client=branch_client_by_algo[branch_algo],
                    llm_cfg=llm_cfg,
                    messages=cot_messages,
                    llm_errors=local_llm_errors,
                    iteration=it,
                    phase="cot",
                ) or ""
                all_it_cot_suggestions.append(
                    _truncate_text(cot_suggestion, int(max(300, llm_cfg.get("history_suggestion_max_chars", 700))))
                )
                iter_log["feedback"] = cot_suggestion
                iter_log["feedback_summary"] = cot_suggestion[:500]
                iter_log["cot_prompt"] = cot_prompt

                dialogs_log = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": iter_log["prompt"]},
                    {"role": "assistant", "content": assistant_reply_buffer[max_id]},
                    {"role": "user", "content": cot_prompt},
                    {"role": "assistant", "content": cot_suggestion},
                ]
                for dialog in dialogs_log:
                    cur_role, cur_content = dialog["role"], dialog["content"]
                    dialog_text += "*" * 50 + "\n"
                    dialog_text += "*" * 20 + f"role:{cur_role}" + "*" * 20 + "\n"
                    dialog_text += "*" * 50 + "\n"
                    dialog_text += f"{cur_content}\n\n"

                if it < max_iterations - 1:
                    branch_feedback = _build_branch_iteration_feedback(
                        branch_algo,
                        candidate_stats,
                        local_llm_errors,
                    )
                    next_prompt = _fresh_iteration_prompt([branch_feedback])
                    iter_log["next_prompt_length"] = int(len(next_prompt))
                    next_dialogs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _branch_prompt(next_prompt, branch_algo)},
                    ]
            elif it < max_iterations - 1:
                retry_feedback = _truncate_text(
                    _build_branch_iteration_feedback(branch_algo, candidate_stats, local_llm_errors),
                    int(max(400, llm_cfg.get("retry_feedback_max_chars", 2000))),
                )
                retry_prompt = _fresh_iteration_prompt(
                    [
                        "Retry guidance:",
                        "The previous iteration produced no valid candidate. Avoid repeating duplicate, invalid, or numerically unstable patterns.",
                        retry_feedback,
                    ]
                )
                iter_log["next_prompt_length"] = int(len(retry_prompt))
                next_dialogs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _branch_prompt(retry_prompt, branch_algo)},
                ]

            iter_log["sample_attempts"] = int(trying_count)
            iter_log["sample_valid_count"] = int(len(candidates_it))
            iter_log["sample_failed_calls"] = int(failed_sample_calls)
            iter_log["sample_failure_counters"] = {k: int(v) for k, v in sample_failure_counters.items()}
            iter_log["sample_failure_limits"] = {
                "hard": int(max_failed_calls),
                "empty_response": int(max_empty_response_calls),
                "invalid_code": int(max_invalid_code_calls),
                "duplicate_candidate": int(max_duplicate_calls),
                "validation_failed": int(max_validation_failed_calls),
            }
            iter_log["sample_stop_by_failure_limit"] = bool(stop_sampling)
            return {
                "algorithm": branch_algo,
                "candidate_entries": list(candidates_it),
                "iter_log": iter_log,
                "llm_responses": local_llm_responses,
                "llm_errors": local_llm_errors,
                "dialog_text": dialog_text,
                "next_dialogs": next_dialogs,
                "all_it_func_results": all_it_func_results,
                "all_it_cot_suggestions": all_it_cot_suggestions,
                "seen_candidate_hashes": global_seen_candidate_hashes,
            }

        for it in range(max_iterations):
            branch_results_by_algo: Dict[str, dict] = {}
            use_parallel_branches = bool(branch_iteration_mode == "threadpool")
            if branch_iteration_mode == "subprocess":
                branch_results_by_algo, iter_artifacts = _run_branch_iterations_by_subprocess(
                    run_dir=run_dir,
                    cfg=cfg,
                    llm_cfg=llm_cfg,
                    branch_iteration_cfg=branch_iteration_cfg,
                    it=it,
                    llm_branch_algos=llm_branch_algos,
                    branch_state_by_algo=branch_state_by_algo,
                    system_prompt=system_prompt,
                    state_desc=state_desc,
                    state_contract_note=state_contract_note,
                    scenario_profile=scenario_profile,
                    scenario_enabled=scenario_enabled,
                    scenario_priority=scenario_priority,
                    candidates_per_family=candidates_per_family,
                    candidate_scoring_cfg=candidate_scoring_cfg,
                    llm_iteration_mode=llm_iteration_mode,
                    generation_target=generation_target,
                    max_iterations=max_iterations,
                    selection_seeds=selection_seeds,
                    algo_runtime_cache=algo_runtime_cache,
                    train_df=train_df,
                    val_df=val_df,
                    schema=schema,
                    env_cfg=env_cfg,
                    reference_states=reference_states,
                    drl_backend=drl_backend,
                    native_validation_states=native_validation_states,
                    native_raw_dim=native_raw_dim,
                )
                branch_iteration_artifacts.append(_json_safe(iter_artifacts))
            elif use_parallel_branches:
                with ThreadPoolExecutor(
                    max_workers=branch_parallel_workers,
                    thread_name_prefix="lesr-branch",
                ) as executor:
                    future_to_algo = {
                        executor.submit(_run_branch_iteration, branch_algo, it): branch_algo
                        for branch_algo in llm_branch_algos
                    }
                    for future in as_completed(future_to_algo):
                        branch_algo = future_to_algo[future]
                        branch_results_by_algo[branch_algo] = future.result()
            else:
                for branch_algo in llm_branch_algos:
                    branch_results_by_algo[branch_algo] = _run_branch_iteration(branch_algo, it)

            shared_iteration_feedback = ""
            if it < max_iterations - 1:
                shared_iteration_feedback = _build_cross_algo_iteration_feedback(branch_results_by_algo)

            for branch_algo in llm_branch_algos:
                branch_result = branch_results_by_algo[branch_algo]
                branch_state = branch_state_by_algo[branch_algo]
                if (
                    shared_iteration_feedback
                    and len(branch_result["next_dialogs"]) >= 2
                    and branch_result["next_dialogs"][1].get("role") == "user"
                ):
                    branch_result["next_dialogs"][1]["content"] = _append_prompt_sections(
                        branch_result["next_dialogs"][1]["content"],
                        [shared_iteration_feedback],
                    )
                branch_state["dialogs"] = branch_result["next_dialogs"]
                branch_state["all_it_func_results"] = branch_result["all_it_func_results"]
                branch_state["all_it_cot_suggestions"] = branch_result["all_it_cot_suggestions"]
                branch_state["seen_candidate_hashes"] = branch_result["seen_candidate_hashes"]
                llm_responses.extend(branch_result["llm_responses"])
                llm_errors.extend(branch_result["llm_errors"])
                iter_trace.append(branch_result["iter_log"])
                if llm_iteration_mode == "single_branch":
                    shared_candidates.extend(branch_result["candidate_entries"])
                else:
                    candidate_entries_by_algo[branch_algo].extend(branch_result["candidate_entries"])
                if branch_result["dialog_text"]:
                    if llm_iteration_mode == "per_algorithm_branches":
                        dialog_path = run_dir / f"dialogs_{branch_algo}_it{it}.txt"
                        dialog_path.write_text(branch_result["dialog_text"])
                        if branch_algo == primary_algo:
                            (run_dir / f"dialogs_it{it}.txt").write_text(branch_result["dialog_text"])
                    else:
                        (run_dir / f"dialogs_it{it}.txt").write_text(branch_result["dialog_text"])

        if llm_iteration_mode == "single_branch":
            for algo in eval_algos:
                candidate_entries_by_algo[algo] = list(shared_candidates)

    if cfg.fixed_candidate_path:
        fixed_path = Path(cfg.fixed_candidate_path)
        if not fixed_path.is_absolute():
            fixed_path = (repo_root() / fixed_path).resolve()
        if fixed_path.exists():
            fixed_code = fixed_path.read_text()
            for algo in eval_algos:
                fixed_name = "fixed_candidate" if llm_iteration_mode == "single_branch" else f"{algo}_fixed_candidate"
                candidate_entries_by_algo[algo] = [(fixed_name, fixed_code)]
        else:
            llm_errors.append(
                {
                    "iteration": -1,
                    "phase": "fixed_candidate",
                    "attempt": 1,
                    "error_type": "file_not_found",
                    "message": str(fixed_path),
                }
            )

    if not any(candidate_entries_by_algo.values()):
        static_candidates = _static_candidate_codes_for_backend(schema, drl_backend)
        for algo in eval_algos:
            if llm_iteration_mode == "single_branch":
                candidate_entries_by_algo[algo] = list(static_candidates)
            else:
                candidate_entries_by_algo[algo] = [(f"{algo}_{name}", code) for name, code in static_candidates]
    elif llm_iteration_mode == "per_algorithm_branches":
        static_candidates = _static_candidate_codes_for_backend(schema, drl_backend)
        for algo in eval_algos:
            if not candidate_entries_by_algo.get(algo):
                candidate_entries_by_algo[algo] = [(f"{algo}_{name}", code) for name, code in static_candidates]
    cand_dir = run_dir / "revision_candidates"
    ensure_dir(cand_dir)

    if llm_responses:
        (run_dir / "llm_responses.json").write_text(json.dumps(llm_responses, indent=2))
    if iter_trace:
        (run_dir / "llm_iter_trace.json").write_text(json.dumps(iter_trace, indent=2))

    candidate_meta_by_algo: Dict[str, Dict[str, dict]] = {algo: {} for algo in eval_algos}
    for row in iter_trace or []:
        algo_name = str(row.get("algorithm", ""))
        if algo_name not in candidate_meta_by_algo:
            continue
        for cand in row.get("candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            name = str(cand.get("name", ""))
            if not name:
                continue
            candidate_meta_by_algo[algo_name][name] = {
                "algorithm": algo_name,
                "iteration": int(row.get("iteration", -1)),
                "family": str(cand.get("family", "")),
                "design_mode": str(cand.get("design_mode", "")),
                "origin": str(cand.get("origin", _candidate_origin_from_name(name))),
                "feature_groups": list(cand.get("feature_groups", [])),
                "revise_hash": str(cand.get("revise_hash", "")),
                "intrinsic_hash": str(cand.get("intrinsic_hash", "")),
                "valid": bool(cand.get("valid", False)),
                "score": float(_sanitize_float(cand.get("score", 0.0))),
                "performance_mode": str(cand.get("performance_mode", candidate_scoring_cfg.get("performance_mode", "absolute"))),
                "performance_score": float(_sanitize_float(cand.get("performance_score", 0.0))),
                "performance_score_absolute": float(_sanitize_float(cand.get("performance_score_absolute", 0.0))),
                "performance_score_baseline": float(_sanitize_float(cand.get("performance_score_baseline", 0.0))),
                "performance_score_delta": float(_sanitize_float(cand.get("performance_score_delta", 0.0))),
                "intrinsic_probe_score": float(_sanitize_float(cand.get("intrinsic_probe_score", 0.0))),
                "intrinsic_probe_score_absolute": float(_sanitize_float(cand.get("intrinsic_probe_score_absolute", 0.0))),
                "intrinsic_probe_score_baseline": float(_sanitize_float(cand.get("intrinsic_probe_score_baseline", 0.0))),
                "intrinsic_probe_score_delta": float(_sanitize_float(cand.get("intrinsic_probe_score_delta", 0.0))),
                "behavior_score": float(_sanitize_float(cand.get("behavior_score", 0.0))),
                "turnover_score": float(_sanitize_float(cand.get("turnover_score", 0.0))),
                "lipschitz_raw": (
                    float(_sanitize_float(cand.get("lipschitz_raw")))
                    if cand.get("lipschitz_raw") is not None
                    else None
                ),
                "lipschitz_score": float(_sanitize_float(cand.get("lipschitz_score", 0.0))),
            }

    candidate_funcs_by_algo: Dict[str, List[Tuple[str, object, object, str]]] = {algo: [] for algo in eval_algos}
    candidate_code_map_by_algo: Dict[str, Dict[str, str]] = {algo: {} for algo in eval_algos}
    cand_map_by_algo: Dict[str, Dict[str, Tuple[object, object]]] = {algo: {} for algo in eval_algos}
    written_candidate_names: set[str] = set()
    static_candidates_cache = _static_candidate_codes_for_backend(schema, drl_backend)

    def _append_candidate_for_algo(algo: str, name: str, code: str, phase: str) -> bool:
        if name not in written_candidate_names:
            (cand_dir / f"{name}.py").write_text(code)
            written_candidate_names.add(name)
        try:
            _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
            revise_state, intrinsic_reward = load_functions_from_code(code)
            _validate_candidate_pair(revise_state, intrinsic_reward)
            candidate_funcs_by_algo[algo].append((name, revise_state, intrinsic_reward, code))
            candidate_code_map_by_algo[algo][name] = code
            cand_map_by_algo[algo][name] = (revise_state, intrinsic_reward)
            candidate_meta_by_algo.setdefault(algo, {}).setdefault(
                name,
                {
                    "algorithm": algo,
                    "iteration": -1,
                    "family": "",
                    "design_mode": "",
                    "origin": _candidate_origin_from_name(name),
                    "feature_groups": _infer_candidate_feature_groups(code),
                    **_extract_candidate_component_hashes(code),
                    "valid": True,
                    "score": 0.0,
                    "performance_mode": candidate_scoring_cfg.get("performance_mode", "absolute"),
                    "performance_score": 0.0,
                    "performance_score_absolute": 0.0,
                    "performance_score_baseline": 0.0,
                    "performance_score_delta": 0.0,
                    "intrinsic_probe_score": 0.0,
                    "intrinsic_probe_score_absolute": 0.0,
                    "intrinsic_probe_score_baseline": 0.0,
                    "intrinsic_probe_score_delta": 0.0,
                    "behavior_score": 0.0,
                    "turnover_score": 0.0,
                    "lipschitz_raw": None,
                    "lipschitz_score": 0.0,
                },
            )
            return True
        except Exception as exc:
            llm_errors.append(
                {
                    "iteration": -1,
                    "phase": phase,
                    "attempt": 1,
                    "error_type": "validation_failed",
                    "algorithm": algo,
                    "candidate": name,
                    "message": str(exc),
                }
            )
            return False

    for algo in eval_algos:
        seen_hashes: set[str] = set()
        for name, code in candidate_entries_by_algo.get(algo, []):
            code_hash = _sha256_text(code)
            if code_hash in seen_hashes:
                llm_errors.append(
                    {
                        "iteration": -1,
                        "phase": "final_filter",
                        "attempt": 1,
                        "error_type": "duplicate_candidate",
                        "algorithm": algo,
                        "candidate": name,
                        "message": "duplicate candidate code hash inside final algorithm pool",
                    }
                )
                continue
            seen_hashes.add(code_hash)
            _append_candidate_for_algo(algo, name, code, phase="final_filter")

        if candidate_funcs_by_algo[algo]:
            continue

        fallback_entries = (
            list(static_candidates_cache)
            if llm_iteration_mode == "single_branch"
            else [(f"{algo}_{name}", code) for name, code in static_candidates_cache]
        )
        for name, code in fallback_entries:
            code_hash = _sha256_text(code)
            if code_hash in seen_hashes:
                continue
            seen_hashes.add(code_hash)
            _append_candidate_for_algo(algo, name, code, phase="final_filter_static")

    (run_dir / "llm_errors.json").write_text(json.dumps(llm_errors, indent=2))

    if not any(candidate_funcs_by_algo.values()):
        raise ValueError("No valid candidate functions available after filtering.")

    candidate_pool_size_input_by_algo = {algo: int(len(candidate_funcs_by_algo[algo])) for algo in eval_algos}
    candidate_prefilter_summary_by_algo: Dict[str, dict] = {}
    prefilter_top_n = int(final_selection_cfg.get("top_n_per_algo", 0))
    for algo in eval_algos:
        filtered_candidates, prefilter_summary = _apply_candidate_prefilter_for_algo(
            algo,
            candidate_funcs_by_algo[algo],
            candidate_meta_by_algo,
            candidate_scoring_cfg,
            prefilter_top_n,
        )
        candidate_funcs_by_algo[algo] = filtered_candidates
        candidate_prefilter_summary_by_algo[algo] = prefilter_summary
    (run_dir / "final_selection_prefilter.json").write_text(
        json.dumps(
            {
                "mode": final_selection_mode,
                "cfg": _json_safe(final_selection_cfg),
                "candidate_pool_size_input_by_algo": candidate_pool_size_input_by_algo,
                "candidate_pool_size_after_prefilter_by_algo": {
                    algo: int(len(candidate_funcs_by_algo[algo]))
                    for algo in eval_algos
                },
                "by_algorithm": _json_safe(candidate_prefilter_summary_by_algo),
            },
            indent=2,
        )
    )

    candidate_cache = {}
    for algo in eval_algos:
        for name, revise_state, intrinsic_reward, _ in candidate_funcs_by_algo[algo]:
            if name in candidate_cache:
                continue
            intrinsic_reward_eval = _prepare_intrinsic(revise_state, intrinsic_reward)
            intrinsic_reward_probe_eval = _prepare_intrinsic_for_selection(
                revise_state,
                intrinsic_reward,
                cfg=cfg,
                reference_states=reference_states,
                input_mode="raw",
            )
            meta = candidate_meta_by_algo.get(algo, {}).get(name, {})
            candidate_cache[name] = {
                "revise_state": revise_state,
                "intrinsic_reward_base": intrinsic_reward,
                "intrinsic_reward_eval": intrinsic_reward_eval,
                "intrinsic_reward_probe_eval": intrinsic_reward_probe_eval,
                "policy_state_fn": _build_policy_state_fn(revise_state, algorithm=algo),
                "code": candidate_code_map_by_algo.get(algo, {}).get(name, ""),
                "origin": str(meta.get("origin", _candidate_origin_from_name(name))),
                "family": str(meta.get("family", "")),
                "design_mode": str(meta.get("design_mode", "")),
                "feature_groups": list(meta.get("feature_groups", [])),
                "revise_hash": str(meta.get("revise_hash", "")),
                "intrinsic_hash": str(meta.get("intrinsic_hash", "")),
                "lipschitz": _estimate_intrinsic_lipschitz(
                    reference_states=reference_states,
                    revise_state_fn=revise_state,
                    intrinsic_reward_fn=intrinsic_reward_eval,
                    max_pairs=int(candidate_scoring_cfg["lipschitz_max_pairs"]),
                    quantile=float(candidate_scoring_cfg["lipschitz_quantile"]),
                ),
                "intrinsic_signal_stats_raw": _estimate_intrinsic_signal_stats(
                    revise_state_fn=revise_state,
                    intrinsic_reward_fn=intrinsic_reward_probe_eval,
                    reference_states=reference_states,
                    input_mode="raw",
                ),
            }

    def _score_candidate_for_algo(algo: str, runtime: dict, candidate_name: str, seeds: List[int]) -> dict:
        cache = candidate_cache[candidate_name]
        score_payload = _score_candidate_payload_for_algo_external(
            cfg=cfg,
            algo=algo,
            runtime=runtime,
            revise_state=cache["revise_state"],
            intrinsic_reward_eval=cache["intrinsic_reward_eval"],
            intrinsic_reward_probe_eval=cache["intrinsic_reward_probe_eval"],
            policy_state_fn_candidate=cache["policy_state_fn"],
            seeds=seeds,
            train_df=train_df,
            val_df=val_df,
            schema=schema,
            env_cfg=env_cfg,
            state_fn_raw=state_fn_raw,
            finagent=finagent,
            candidate_scoring_cfg=candidate_scoring_cfg,
            reference_states=reference_states,
            drl_backend=drl_backend,
        )
        return {
            "name": candidate_name,
            "origin": str(cache.get("origin", _candidate_origin_from_name(candidate_name))),
            "family": str(cache.get("family", "")),
            "design_mode": str(cache.get("design_mode", "")),
            "feature_groups": list(cache.get("feature_groups", [])),
            "revise_hash": str(cache.get("revise_hash", "")),
            "intrinsic_hash": str(cache.get("intrinsic_hash", "")),
            **score_payload,
        }

    candidate_scores_by_algo: Dict[str, List[dict]] = {}
    best_candidate_by_algo: Dict[str, str] = {}
    candidate_pool_size_by_algo = {algo: int(len(candidate_funcs_by_algo[algo])) for algo in eval_algos}
    candidate_final_selection_artifacts: Dict[str, object] = {
        "mode": final_selection_mode,
        "cfg": _json_safe(final_selection_cfg),
        "candidate_pool_size_input_by_algo": _json_safe(candidate_pool_size_input_by_algo),
        "candidate_pool_size_after_prefilter_by_algo": _json_safe(candidate_pool_size_by_algo),
        "prefilter": _json_safe(candidate_prefilter_summary_by_algo),
        "worker_errors": {},
    }

    def _score_all_candidates_for_algo(algo: str) -> tuple[str, List[dict], List[dict]]:
        runtime = algo_runtime_cache[algo]
        rows: List[dict] = []
        candidate_errors: List[dict] = []
        for name, _, _, _ in candidate_funcs_by_algo[algo]:
            try:
                rows.append(_score_candidate_for_algo(algo, runtime, name, selection_seeds))
            except Exception as exc:
                candidate_errors.append(
                    {
                        "candidate": name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        rows = _rank_candidate_rows(rows, candidate_scoring_cfg)
        return algo, rows, candidate_errors

    if final_selection_mode == "subprocess":
        candidate_scores_by_algo, subprocess_artifacts = _score_candidates_by_algo_subprocess(
            run_dir=run_dir,
            cfg=cfg,
            eval_algos=eval_algos,
            candidate_funcs_by_algo=candidate_funcs_by_algo,
            candidate_meta_by_algo=candidate_meta_by_algo,
            algo_runtime_cache=algo_runtime_cache,
            candidate_scoring_cfg=candidate_scoring_cfg,
            selection_seeds=selection_seeds,
            train_df=train_df,
            val_df=val_df,
            schema=schema,
            env_cfg=env_cfg,
            reference_states=reference_states,
            drl_backend=drl_backend,
            native_validation_states=native_validation_states,
            native_raw_dim=native_raw_dim,
            timeout_s=int(final_selection_cfg["timeout_s"]),
            poll_s=int(final_selection_cfg["poll_s"]),
            heartbeat_timeout_s=int(final_selection_cfg["heartbeat_timeout_s"]),
            bootstrap_timeout_s=int(final_selection_cfg["bootstrap_timeout_s"]),
        )
        candidate_final_selection_artifacts.update(_json_safe(subprocess_artifacts))
    elif final_selection_mode == "threadpool":
        with ThreadPoolExecutor(
            max_workers=min(branch_parallel_workers, len(eval_algos)),
            thread_name_prefix="lesr-final-select",
        ) as executor:
            future_to_algo = {
                executor.submit(_score_all_candidates_for_algo, algo): algo
                for algo in eval_algos
            }
            for future in as_completed(future_to_algo):
                algo, rows, candidate_errors = future.result()
                candidate_scores_by_algo[algo] = rows
                if candidate_errors:
                    candidate_final_selection_artifacts["worker_errors"][f"{algo}__candidate_errors"] = _json_safe(candidate_errors)
    else:
        for algo in eval_algos:
            algo_name, rows, candidate_errors = _score_all_candidates_for_algo(algo)
            candidate_scores_by_algo[algo_name] = rows
            if candidate_errors:
                candidate_final_selection_artifacts["worker_errors"][f"{algo_name}__candidate_errors"] = _json_safe(candidate_errors)

    for algo in eval_algos:
        candidate_scores_by_algo.setdefault(algo, [])

    search_best_candidate_by_algo: Dict[str, str] = {}
    for algo in eval_algos:
        rows = candidate_scores_by_algo.get(algo, [])
        if rows:
            search_best_candidate_by_algo[algo] = str(rows[0]["name"])
        elif candidate_funcs_by_algo[algo]:
            search_best_candidate_by_algo[algo] = str(candidate_funcs_by_algo[algo][0][0])
    best_candidate_by_algo = dict(search_best_candidate_by_algo)

    best_name = search_best_candidate_by_algo.get(primary_algo, candidate_funcs_by_algo[primary_algo][0][0])
    cand_scores = [
        (str(row["name"]), float(row["score"]))
        for row in candidate_scores_by_algo.get(primary_algo, [])
    ]
    if not cand_scores:
        cand_scores = [(best_name, 0.0)]

    candidate_fingerprint = _build_candidate_fingerprint(
        best_name,
        candidate_code_map_by_algo.get(primary_algo, {}),
    )
    candidate_fingerprint_by_algo = {
        algo: _build_candidate_fingerprint(
            search_best_candidate_by_algo.get(algo, candidate_funcs_by_algo[algo][0][0]),
            candidate_code_map_by_algo[algo],
        )
        for algo in eval_algos
    }

    consensus_promotion_cfg = _resolve_consensus_promotion_cfg(llm_cfg)

    def _short_eval_metrics_once(
        algo: str,
        runtime: dict,
        revise_state_fn,
        intrinsic_reward_eval,
        policy_state_fn,
        use_revised: bool,
        use_intrinsic: bool,
        seed: int,
        intrinsic_input_mode: str = "revised",
        steps_small_override: int | None = None,
    ) -> dict:
        algo_env_cfg = _env_cfg_for_algo(algo)
        steps_small = int(steps_small_override) if steps_small_override is not None else _effective_steps(
            cfg.n_small,
            int(train_df["date"].nunique()),
        )
        native_backend = str(drl_backend or "current").strip().lower() == "finsaber_native"
        if native_backend:
            native_cfg = runtime.get("native_cfg")
            native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {})
            if native_cfg is None:
                raise ValueError("finsaber_native runtime missing native_cfg")
            native_cfg_small = replace(native_cfg, total_timesteps=int(steps_small))
            native_algo_kwargs_small = _native_small_budget_algo_kwargs(algo, native_algo_kwargs, int(steps_small))
            result = train_finsaber_native(
                algo=algo,
                train_df=train_df,
                eval_df=val_df,
                eval_history_df=native_selection_history_df,
                cfg=native_cfg_small,
                seed=int(seed),
                algo_kwargs=native_algo_kwargs_small,
                revise_state=revise_state_fn,
                intrinsic_reward=intrinsic_reward_eval,
                policy_state_fn=policy_state_fn,
                use_revised=use_revised,
                use_intrinsic=use_intrinsic,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                intrinsic_input_mode=intrinsic_input_mode,
            )
            metrics, _ = _sb3_metrics_from_eval(result)
            return metrics
        if algo == "td3" and runtime["is_td3_legacy"]:
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
            eval_env = TradingEnv(val_df, schema.assets, schema, algo_env_cfg)
            result = train_td3(
                env=train_env,
                state_dim=policy_state_fn(np.zeros(schema.dim(), dtype=np.float32)).shape[0],
                action_dim=len(schema.assets),
                cfg=td3_cfg_small,
                max_steps=steps_small,
                state_fn=policy_state_fn,
                revise_state=revise_state_fn,
                intrinsic_reward=intrinsic_reward_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                use_intrinsic=use_intrinsic,
                intrinsic_timing=cfg.intrinsic_timing,
                finagent=finagent,
                finagent_weight=cfg.finagent_weight,
                seed=seed,
                eval_env=eval_env,
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_input_mode=intrinsic_input_mode,
            )
            return compute_metrics(np.array(result.eval_values_final))
        if algo == "td3":
            td3_cfg_small = _resolve_td3_cfg(runtime["td3_algo_base_cfg"], steps_small, cfg.warmup_ratio, cfg.evaluation)
            sb3_cfg_small = _td3_cfg_to_sb3_cfg(runtime["sb3_algo_base_cfg"], td3_cfg_small, steps_small)
            sb3_kwargs_small = dict(runtime["sb3_algo_kwargs"])
            sb3_kwargs_small.update(_td3_cfg_to_sb3_kwargs(td3_cfg_small))
            result = train_sb3(
                algo="td3",
                train_df=train_df,
                eval_df=val_df,
                assets=schema.assets,
                schema=schema,
                env_cfg=algo_env_cfg,
                cfg=sb3_cfg_small,
                action_space_type="continuous",
                policy_action_bound=runtime["td3_policy_action_bound"],
                revise_state=revise_state_fn,
                intrinsic_reward=intrinsic_reward_eval,
                intrinsic_w=float(cfg.intrinsic_w),
                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                intrinsic_timing=cfg.intrinsic_timing,
                use_revised=use_revised,
                use_intrinsic=use_intrinsic,
                intrinsic_input_mode=intrinsic_input_mode,
                policy_state_fn=policy_state_fn,
                seed=seed,
                algo_kwargs=sb3_kwargs_small,
            )
            metrics, _ = _sb3_metrics_from_eval(result)
            return metrics
        sb3_cfg_small = replace(runtime["sb3_algo_base_cfg"], total_timesteps=int(steps_small))
        result = train_sb3(
            algo=algo,
            train_df=train_df,
            eval_df=val_df,
            assets=schema.assets,
            schema=schema,
            env_cfg=algo_env_cfg,
            cfg=sb3_cfg_small,
            action_space_type=_action_space_type(algo),
            policy_action_bound=None,
            revise_state=revise_state_fn,
            intrinsic_reward=intrinsic_reward_eval,
            intrinsic_w=float(cfg.intrinsic_w),
            intrinsic_scale_mode=cfg.intrinsic_scale_mode,
            intrinsic_timing=cfg.intrinsic_timing,
            use_revised=use_revised,
            use_intrinsic=use_intrinsic,
            intrinsic_input_mode=intrinsic_input_mode,
            policy_state_fn=policy_state_fn,
            seed=seed,
            algo_kwargs=runtime["sb3_algo_kwargs"],
        )
        metrics, _ = _sb3_metrics_from_eval(result)
        return metrics

    def _component_delta_payload_for_algo(
        algo: str,
        runtime: dict,
        revise_state_fn,
        intrinsic_reward_eval,
        policy_state_fn,
        use_revised: bool,
        use_intrinsic: bool,
        seeds: List[int],
        intrinsic_input_mode: str = "revised",
        steps_small_override: int | None = None,
    ) -> dict:
        baseline_metrics_by_seed = _baseline_metrics_for_algo(
            algo,
            runtime,
            seeds,
            steps_small_override=steps_small_override,
        )
        delta_sharpes: list[float] = []
        delta_scores: list[float] = []
        per_seed_rows: list[dict] = []
        for sd in seeds:
            metrics = _short_eval_metrics_once(
                algo=algo,
                runtime=runtime,
                revise_state_fn=revise_state_fn,
                intrinsic_reward_eval=intrinsic_reward_eval,
                policy_state_fn=policy_state_fn,
                use_revised=use_revised,
                use_intrinsic=use_intrinsic,
                seed=int(sd),
                intrinsic_input_mode=intrinsic_input_mode,
                steps_small_override=steps_small_override,
            )
            baseline = baseline_metrics_by_seed.get(int(sd), {})
            d_sh = float(metrics.get("Sharpe", 0.0) - baseline.get("Sharpe", 0.0))
            d_sc = float(_score_from_metrics(metrics) - _score_from_metrics(baseline))
            delta_sharpes.append(d_sh)
            delta_scores.append(d_sc)
            per_seed_rows.append(
                {
                    "seed": int(sd),
                    "Sharpe": float(metrics.get("Sharpe", 0.0)),
                    "CR": float(metrics.get("CR", 0.0)),
                    "delta_sharpe": d_sh,
                    "delta_score": d_sc,
                }
            )
        return {
            "delta_sharpe_mean": float(np.mean(delta_sharpes)) if delta_sharpes else 0.0,
            "delta_score_mean": float(np.mean(delta_scores)) if delta_scores else 0.0,
            "delta_sharpes": [float(x) for x in delta_sharpes],
            "delta_scores": [float(x) for x in delta_scores],
            "per_seed": per_seed_rows,
        }

    def _evaluate_shared_candidate(
        *,
        label: str,
        source_type: str,
        origin: str,
        family: str,
        design_mode: str,
        feature_groups: list[str],
        support_count: int,
        revise_state_fn,
        intrinsic_reward_eval,
        policy_state_fn,
        use_revised: bool,
        use_intrinsic: bool,
        intrinsic_input_mode: str = "revised",
        intrinsic_signal_stats_raw: dict | None = None,
    ) -> dict:
        target_group = {
            "state_core": "G1_revise_only",
            "intrinsic_core": "G2_intrinsic_only",
            "joint_pair": "G3_revise_intrinsic",
        }.get(str(source_type), "unknown")
        per_algo = {}
        delta_sharpes = []
        delta_scores = []
        for algo in eval_algos:
            runtime = algo_runtime_cache[algo]
            algo_scoring_seeds, algo_steps_small = _resolve_candidate_scoring_budget(
                cfg=cfg,
                algo=algo,
                candidate_scoring_cfg=candidate_scoring_cfg,
                requested_seeds=selection_seeds,
                train_df=train_df,
            )
            algo_payload = _component_delta_payload_for_algo(
                algo=algo,
                runtime=runtime,
                revise_state_fn=revise_state_fn,
                intrinsic_reward_eval=intrinsic_reward_eval,
                policy_state_fn=policy_state_fn,
                use_revised=use_revised,
                use_intrinsic=use_intrinsic,
                seeds=algo_scoring_seeds,
                intrinsic_input_mode=intrinsic_input_mode,
                steps_small_override=algo_steps_small,
            )
            per_algo[algo] = algo_payload
            delta_sharpes.append(float(algo_payload["delta_sharpe_mean"]))
            delta_scores.append(float(algo_payload["delta_score_mean"]))
        consensus = _consensus_sharpe_score(delta_sharpes)
        raw_intrinsic_stats = dict(intrinsic_signal_stats_raw or {})
        raw_intrinsic_nontrivial = bool(raw_intrinsic_stats.get("nontrivial", False))
        raw_g2_delta_anchor = max(delta_sharpes) if delta_sharpes else 0.0
        finite_delta_sharpes = [float(v) for v in delta_sharpes if np.isfinite(float(v))]
        candidate_reject_floor = float(consensus_promotion_cfg["candidate_reject_floor"])
        intrinsic_nontrivial = bool(
            source_type != "intrinsic_core"
            or (
                raw_intrinsic_nontrivial
                and (
                    raw_g2_delta_anchor >= float(consensus_promotion_cfg["intrinsic_pick_floor"])
                    or any(
                        abs(float(v)) >= float(consensus_promotion_cfg["intrinsic_nontrivial_floor"])
                        for v in delta_sharpes
                    )
                )
            )
        )
        return {
            "name": label,
            "source_type": source_type,
            "origin": origin,
            "family": family,
            "design_mode": design_mode,
            "feature_groups": feature_groups,
            "support_count": int(support_count),
            "target_group": target_group,
            "per_algo": per_algo,
            "delta_sharpes": {algo: float(payload["delta_sharpe_mean"]) for algo, payload in per_algo.items()},
            "delta_scores": {algo: float(payload["delta_score_mean"]) for algo, payload in per_algo.items()},
            "target_delta_sharpes": {algo: float(payload["delta_sharpe_mean"]) for algo, payload in per_algo.items()},
            "target_delta_scores": {algo: float(payload["delta_score_mean"]) for algo, payload in per_algo.items()},
            "target_delta_sharpe_mean": float(np.mean(delta_sharpes)) if delta_sharpes else 0.0,
            "target_delta_score_mean": float(np.mean(delta_scores)) if delta_scores else 0.0,
            "intrinsic_nontrivial": intrinsic_nontrivial,
            "intrinsic_signal_stats_raw": raw_intrinsic_stats,
            "intrinsic_signal_nontrivial_raw": raw_intrinsic_nontrivial,
            "raw_g2_delta_anchor": float(raw_g2_delta_anchor),
            **consensus,
            "eligible": bool(
                origin == "llm"
                and int(support_count) >= int(consensus_promotion_cfg["support_min"])
                and bool(finite_delta_sharpes)
                and max(finite_delta_sharpes) >= candidate_reject_floor
                and intrinsic_nontrivial
            ),
        }

    def _select_pooled_rows_for_algo(algo_rows: list[dict], top_k: int) -> list[dict]:
        llm_rows = [dict(row) for row in algo_rows if str(row.get("origin", "")) == "llm"]
        if not llm_rows:
            return []
        preferred_rows = [row for row in llm_rows if bool(row.get("eligible", True))]
        if not preferred_rows:
            preferred_rows = list(llm_rows)

        def _row_intrinsic_stats(row: dict) -> dict:
            return dict(row.get("intrinsic_signal_stats_raw", {}) or {})

        def _row_key(row: dict, mode: str | None = None) -> tuple[float, ...]:
            intrinsic_stats = _row_intrinsic_stats(row)
            intrinsic_nontrivial_raw = float(bool(row.get("intrinsic_signal_nontrivial_raw", False)))
            intrinsic_std_raw = float(_sanitize_float(intrinsic_stats.get("std", 0.0)))
            intrinsic_span_raw = float(_sanitize_float(intrinsic_stats.get("span", 0.0)))
            g1_delta = float(_sanitize_float(row.get("state_probe_delta_sharpe", row.get("state_probe_score_delta", 0.0))))
            g2_delta = float(_sanitize_float(row.get("intrinsic_probe_delta_sharpe", row.get("intrinsic_probe_score_delta", 0.0))))
            g3_delta = float(_sanitize_float(row.get("performance_delta_sharpe", row.get("performance_score_delta", 0.0))))
            if mode == "intrinsic_first":
                return (
                    intrinsic_nontrivial_raw,
                    g2_delta,
                    intrinsic_std_raw,
                    intrinsic_span_raw,
                    float(_sanitize_float(row.get("intrinsic_probe_score", 0.0))),
                    float(_sanitize_float(row.get("score", 0.0))),
                )
            if mode == "state_first":
                return (
                    g1_delta,
                    float(_sanitize_float(row.get("state_probe_score", 0.0))),
                    float(_sanitize_float(row.get("score", 0.0))),
                )
            return (
                intrinsic_nontrivial_raw,
                float((g1_delta + g2_delta + g3_delta) / 3.0),
                g3_delta,
                float(_sanitize_float(row.get("score", 0.0))),
                intrinsic_std_raw,
                float(_sanitize_float(row.get("state_probe_score", 0.0))),
            )

        selected: list[dict] = []
        selected_names: set[str] = set()
        for mode in ["intrinsic_first", "state_first", "balanced"]:
            mode_rows = [row for row in preferred_rows if str(row.get("design_mode", "")) == mode]
            if not mode_rows:
                mode_rows = [row for row in llm_rows if str(row.get("design_mode", "")) == mode]
            if not mode_rows:
                continue
            best_row = sorted(mode_rows, key=lambda row: _row_key(row, mode), reverse=True)[0]
            cur_name = str(best_row.get("name", ""))
            if cur_name and cur_name not in selected_names:
                selected.append(best_row)
                selected_names.add(cur_name)
            if len(selected) >= int(top_k):
                return selected[: int(top_k)]

        fallback_rows = preferred_rows if preferred_rows else llm_rows
        for row in sorted(fallback_rows, key=lambda cur: _row_key(cur, None), reverse=True):
            cur_name = str(row.get("name", ""))
            if cur_name and cur_name not in selected_names:
                selected.append(row)
                selected_names.add(cur_name)
            if len(selected) >= int(top_k):
                break
        return selected[: int(top_k)]

    def _component_support_count(rows: list[dict], target_row: dict, component_type: str) -> int:
        hash_key = "revise_hash" if component_type == "state_core" else "intrinsic_hash"
        target_hash = str(target_row.get(hash_key, ""))
        target_family = str(target_row.get("family", ""))
        target_mode = str(target_row.get("design_mode", ""))
        target_groups = tuple(sorted(str(x) for x in (target_row.get("feature_groups", []) or [])))

        exact_support = sum(1 for row in rows if target_hash and str(row.get(hash_key, "")) == target_hash)

        semantic_rows = [
            row
            for row in rows
            if str(row.get("family", "")) == target_family
            and str(row.get("design_mode", "")) == target_mode
            and tuple(sorted(str(x) for x in (row.get("feature_groups", []) or []))) == target_groups
        ]
        semantic_support = len(
            {
                (
                    str(row.get("algorithm", "")),
                    int(row.get("iteration", -1)),
                )
                for row in semantic_rows
            }
        )
        algo_support = len({str(row.get("algorithm", "")) for row in semantic_rows})
        return int(max(exact_support, semantic_support, algo_support, 1))

    per_algo_selected_rows: dict[str, list[dict]] = {}
    if consensus_promotion_cfg["enabled"]:
        for algo in eval_algos:
            algo_rows = candidate_scores_by_algo.get(algo, [])
            per_algo_selected_rows[algo] = _select_pooled_rows_for_algo(
                [dict(row, algorithm=algo) for row in algo_rows],
                int(consensus_promotion_cfg["top_k_per_algo"]),
            )
    else:
        per_algo_selected_rows = {algo: [] for algo in eval_algos}

    def _promotion_sort_key(row: dict) -> tuple[float, float, float]:
        return (
            float(_sanitize_float(row.get("consensus_score", 0.0))),
            float(_sanitize_float(row.get("min_delta_sharpe", 0.0))),
            float(_sanitize_float(row.get("target_delta_sharpe_mean", row.get("mean_delta_sharpe", 0.0)))),
        )

    def _row_state_delta(row: dict) -> float:
        return float(
            _sanitize_float(
                row.get("state_probe_delta_sharpe", row.get("state_probe_score_delta", 0.0))
            )
        )

    def _row_intrinsic_delta(row: dict) -> float:
        return float(
            _sanitize_float(
                row.get("intrinsic_probe_delta_sharpe", row.get("intrinsic_probe_score_delta", 0.0))
            )
        )

    def _state_source_sort_key(row: dict) -> tuple[float, ...]:
        return (
            _row_state_delta(row),
            float(
                _sanitize_float(
                    row.get("performance_delta_sharpe", row.get("performance_score_delta", 0.0))
                )
            ),
            float(_sanitize_float(row.get("state_probe_score", 0.0))),
            float(_sanitize_float(row.get("score", 0.0))),
        )

    def _intrinsic_source_sort_key(row: dict) -> tuple[float, ...]:
        intrinsic_stats = dict(row.get("intrinsic_signal_stats_raw", {}) or {})
        return (
            _row_intrinsic_delta(row),
            float(bool(row.get("intrinsic_signal_nontrivial_raw", False))),
            float(_sanitize_float(intrinsic_stats.get("std", 0.0))),
            float(
                _sanitize_float(
                    row.get(
                        "performance_delta_sharpe",
                        row.get("performance_score_delta", 0.0),
                    )
                )
            ),
            float(_sanitize_float(row.get("score", 0.0))),
        )

    def _td3_intrinsic_source_ok(row: dict) -> bool:
        if str(row.get("algorithm", "")).lower() != "td3":
            return True
        if str(row.get("design_mode", "")) not in {"intrinsic_first", "balanced"}:
            return False
        if not bool(row.get("intrinsic_signal_nontrivial_raw", False)):
            return False
        probe_delta = _row_intrinsic_delta(row)
        probe_score_delta = float(_sanitize_float(row.get("intrinsic_probe_score_delta", 0.0)))
        td3_probe_floor = float(consensus_promotion_cfg["td3_intrinsic_probe_floor"])
        return probe_delta > td3_probe_floor or probe_score_delta > td3_probe_floor

    def _algo_row_sort_key(row: dict, algo: str) -> tuple[float, ...]:
        per_algo = (row.get("per_algo", {}) or {}).get(algo, {}) or {}
        if str(row.get("source_type", "")) == "intrinsic_core":
            intrinsic_stats = dict(row.get("intrinsic_signal_stats_raw", {}) or {})
            return (
                float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0))),
                float(bool(row.get("intrinsic_signal_nontrivial_raw", False))),
                float(_sanitize_float(per_algo.get("delta_score_mean", 0.0))),
                float(_sanitize_float(row.get("raw_g2_delta_anchor", 0.0))),
                float(_sanitize_float(intrinsic_stats.get("std", 0.0))),
                float(_sanitize_float(intrinsic_stats.get("span", 0.0))),
                float(_sanitize_float(row.get("consensus_score", 0.0))),
                float(_sanitize_float(row.get("target_delta_sharpe_mean", row.get("mean_delta_sharpe", 0.0)))),
            )
        return (
            float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0))),
            float(_sanitize_float(row.get("consensus_score", 0.0))),
            float(_sanitize_float(per_algo.get("delta_score_mean", 0.0))),
            float(_sanitize_float(row.get("target_delta_sharpe_mean", row.get("mean_delta_sharpe", 0.0)))),
        )

    def _pick_algo_promoted_row(
        rows: list[dict],
        algo: str,
        *,
        require_nontrivial_intrinsic: bool = False,
    ) -> tuple[dict | None, bool]:
        if not rows:
            return None, True
        soft_ranked_rows = []
        for row in rows:
            if str(row.get("origin", "")) != "llm":
                continue
            if int(row.get("support_count", 0)) < int(consensus_promotion_cfg["support_min"]):
                continue
            per_algo = (row.get("per_algo", {}) or {}).get(algo, {}) or {}
            cur_delta = float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0)))
            cur_score = float(_sanitize_float(per_algo.get("delta_score_mean", 0.0)))
            if require_nontrivial_intrinsic and not bool(row.get("intrinsic_signal_nontrivial_raw", False)):
                continue
            if require_nontrivial_intrinsic and str(algo) == "td3":
                if str(row.get("source_type", "")) == "intrinsic_core":
                    probe_delta = float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0)))
                    probe_score_delta = float(_sanitize_float(per_algo.get("delta_score_mean", 0.0)))
                else:
                    probe_delta = float(
                        _sanitize_float(
                            row.get("intrinsic_probe_delta_sharpe", row.get("intrinsic_probe_score_delta", 0.0))
                        )
                    )
                    probe_score_delta = float(_sanitize_float(row.get("intrinsic_probe_score_delta", 0.0)))
                td3_probe_floor = float(consensus_promotion_cfg["td3_intrinsic_probe_floor"])
                if probe_delta <= td3_probe_floor and probe_score_delta <= td3_probe_floor:
                    continue
            if str(row.get("source_type", "")) == "intrinsic_core" or require_nontrivial_intrinsic:
                cur_floor = float(consensus_promotion_cfg["intrinsic_pick_floor"])
            elif str(row.get("source_type", "")) == "joint_pair":
                cur_floor = float(consensus_promotion_cfg["joint_pick_floor"])
            else:
                cur_floor = float(consensus_promotion_cfg["state_pick_floor"])
            if cur_delta < cur_floor and cur_score < cur_floor:
                continue
            soft_ranked_rows.append(row)
        if soft_ranked_rows:
            soft_ranked_rows = sorted(soft_ranked_rows, key=lambda cur: _algo_row_sort_key(cur, algo), reverse=True)
            return soft_ranked_rows[0], False
        return None, True

    def _build_algo_component_inputs(algo: str) -> tuple[list[dict], list[dict], list[dict]]:
        selected_rows = per_algo_selected_rows.get(algo, [])
        revise_support = Counter(str(row.get("revise_hash", "")) for row in selected_rows if str(row.get("revise_hash", "")))
        intrinsic_support = Counter(str(row.get("intrinsic_hash", "")) for row in selected_rows if str(row.get("intrinsic_hash", "")))

        # G1 should come from pure state-oriented rows instead of mixed intrinsic-led candidates.
        state_source_rows = [
            row
            for row in selected_rows
            if str(row.get("design_mode", "")) in {"state_first", "balanced"}
        ]
        state_source_rows = sorted(state_source_rows, key=_state_source_sort_key, reverse=True)

        state_inputs: list[dict] = []
        seen_revise: set[str] = set()
        for row in state_source_rows:
            revise_hash = str(row.get("revise_hash", ""))
            if not revise_hash or revise_hash in seen_revise:
                continue
            seen_revise.add(revise_hash)
            cache = candidate_cache.get(str(row.get("name", "")))
            if not cache:
                continue
            state_inputs.append(
                _evaluate_shared_candidate(
                    label=str(row.get("name", "")),
                    source_type="state_core",
                    origin=str(row.get("origin", "")),
                    family=str(row.get("family", "")),
                    design_mode=str(row.get("design_mode", "")),
                    feature_groups=list(row.get("feature_groups", [])),
                    support_count=int(max(int(revise_support.get(revise_hash, 0)), 1)),
                    revise_state_fn=cache["revise_state"],
                    intrinsic_reward_eval=_zero_intrinsic_reward,
                    policy_state_fn=cache["policy_state_fn"],
                    use_revised=True,
                    use_intrinsic=False,
                )
            )

        intrinsic_source_rows = list(selected_rows)
        if str(algo).lower() == "td3":
            intrinsic_source_rows = [row for row in intrinsic_source_rows if _td3_intrinsic_source_ok(row)]
        intrinsic_source_rows = sorted(intrinsic_source_rows, key=_intrinsic_source_sort_key, reverse=True)

        intrinsic_inputs: list[dict] = []
        seen_intrinsic: set[str] = set()
        for row in intrinsic_source_rows:
            intrinsic_hash = str(row.get("intrinsic_hash", ""))
            if not intrinsic_hash or intrinsic_hash in seen_intrinsic:
                continue
            seen_intrinsic.add(intrinsic_hash)
            cache = candidate_cache.get(str(row.get("name", "")))
            if not cache:
                continue
            intrinsic_inputs.append(
                _evaluate_shared_candidate(
                    label=str(row.get("name", "")),
                    source_type="intrinsic_core",
                    origin=str(row.get("origin", "")),
                    family=str(row.get("family", "")),
                    design_mode=str(row.get("design_mode", "")),
                    feature_groups=list(row.get("feature_groups", [])),
                    support_count=int(max(int(intrinsic_support.get(intrinsic_hash, 0)), 1)),
                    revise_state_fn=cache["revise_state"],
                    intrinsic_reward_eval=cache["intrinsic_reward_probe_eval"],
                    policy_state_fn=_state_fn_raw_for_algo(algo),
                    use_revised=False,
                    use_intrinsic=True,
                    intrinsic_input_mode="raw",
                    intrinsic_signal_stats_raw=cache.get("intrinsic_signal_stats_raw"),
                )
            )

        state_inputs = sorted(state_inputs, key=_promotion_sort_key, reverse=True)
        intrinsic_inputs = sorted(intrinsic_inputs, key=_promotion_sort_key, reverse=True)
        state_inputs = state_inputs[: int(consensus_promotion_cfg["max_state_cores"])]
        intrinsic_inputs = intrinsic_inputs[: int(consensus_promotion_cfg["max_intrinsic_cores"])]

        joint_inputs: list[dict] = []
        for state_row in state_inputs:
            for intrinsic_row in intrinsic_inputs:
                state_name = str(state_row.get("name", ""))
                intrinsic_name = str(intrinsic_row.get("name", ""))
                combined_code = _build_combined_candidate_code(
                    candidate_cache.get(state_name, {}).get("code", ""),
                    candidate_cache.get(intrinsic_name, {}).get("code", ""),
                )
                if not combined_code:
                    continue
                try:
                    revise_state_combo, intrinsic_reward_combo = load_functions_from_code(combined_code)
                    _validate_candidate_pair(revise_state_combo, intrinsic_reward_combo)
                except Exception:
                    continue
                intrinsic_combo_eval = _prepare_intrinsic(revise_state_combo, intrinsic_reward_combo)
                policy_state_fn_combo = _build_policy_state_fn(
                    revise_state_combo,
                    summary_key=f"{algo}_joint_pair_{state_name}_{intrinsic_name}",
                )
                joint_inputs.append(
                    _evaluate_shared_candidate(
                        label=f"{state_name}__X__{intrinsic_name}",
                        source_type="joint_pair",
                        origin="llm",
                        family=f"{state_row.get('family', '')}|{intrinsic_row.get('family', '')}",
                        design_mode=f"{state_row.get('design_mode', '')}|{intrinsic_row.get('design_mode', '')}",
                        feature_groups=sorted(
                            set(list(state_row.get("feature_groups", [])) + list(intrinsic_row.get("feature_groups", [])))
                        ),
                        support_count=int(
                            max(
                                int(state_row.get("support_count", 0)),
                                int(intrinsic_row.get("support_count", 0)),
                            )
                        ),
                        revise_state_fn=revise_state_combo,
                        intrinsic_reward_eval=intrinsic_combo_eval,
                        policy_state_fn=policy_state_fn_combo,
                        use_revised=True,
                        use_intrinsic=True,
                        intrinsic_input_mode="revised",
                    )
                )
        joint_inputs = sorted(joint_inputs, key=_promotion_sort_key, reverse=True)[
            : int(consensus_promotion_cfg["joint_top_pairs"])
        ]
        return state_inputs, intrinsic_inputs, joint_inputs

    official_shared_cores = {
        "mode": "per_algo_independent",
        "state_core": None,
        "intrinsic_core": None,
        "joint_pair": None,
        "low_confidence": {
            "state_core": True,
            "intrinsic_core": True,
            "joint_pair": True,
        },
        "pooled_candidate_count": 0,
    }

    def _identity_bundle(name: str, *, state_fn_raw_override=None) -> dict:
        raw_state_fn = state_fn_raw_override or state_fn_raw
        return {
            "name": name,
            "origin": "identity",
            "family": "",
            "design_mode": "",
            "feature_groups": [],
            "revise_state": _identity_revise_state,
            "intrinsic_reward_effective": _zero_intrinsic_reward,
            "state_fn_revised": raw_state_fn,
            "postprocess_summary": {
                "mode": intrinsic_postprocess_cfg.get("mode", "raw"),
                "available": False,
            },
        }

    def _bundle_from_candidate_name(
        name: str,
        *,
        policy_uses_revised: bool,
        use_intrinsic: bool,
        intrinsic_input_mode: str = "revised",
        state_fn_raw_override=None,
    ) -> dict:
        raw_state_fn = state_fn_raw_override or state_fn_raw
        cache = candidate_cache.get(name, {})
        revise_state_fn = cache.get("revise_state", _identity_revise_state)
        intrinsic_reward_fn = cache.get("intrinsic_reward_base", _zero_intrinsic_reward)
        intrinsic_reward_effective, post_summary = _build_intrinsic_postprocessed_fn(
            intrinsic_reward=intrinsic_reward_fn if use_intrinsic else _zero_intrinsic_reward,
            revise_state=revise_state_fn,
            reference_states=reference_states,
            post_cfg=intrinsic_postprocess_cfg,
            input_mode=intrinsic_input_mode,
        )
        if intrinsic_reward_effective is None:
            intrinsic_reward_effective = intrinsic_reward_fn if use_intrinsic else _zero_intrinsic_reward
            post_summary = {
                "mode": intrinsic_postprocess_cfg.get("mode", "raw"),
                "available": bool(use_intrinsic),
            }
        return {
            "name": name,
            "origin": str(cache.get("origin", _candidate_origin_from_name(name))),
            "family": str(cache.get("family", "")),
            "design_mode": str(cache.get("design_mode", "")),
            "feature_groups": list(cache.get("feature_groups", [])),
            "revise_state": revise_state_fn,
            "intrinsic_reward_effective": intrinsic_reward_effective if use_intrinsic else _zero_intrinsic_reward,
            "state_fn_revised": (
                cache.get("policy_state_fn", raw_state_fn) if policy_uses_revised else raw_state_fn
            ),
            "postprocess_summary": post_summary,
        }

    def _bundle_from_joint_row(row: dict | None, *, algorithm: str | None = None, state_fn_raw_override=None) -> dict:
        raw_state_fn = state_fn_raw_override or state_fn_raw
        if row is None:
            return _identity_bundle("joint_pair_identity", state_fn_raw_override=raw_state_fn)
        label = str(row.get("name", ""))
        if "__X__" not in label:
            return _bundle_from_candidate_name(
                label,
                policy_uses_revised=True,
                use_intrinsic=True,
                state_fn_raw_override=raw_state_fn,
            )
        state_name, intrinsic_name = label.split("__X__", 1)
        combined_code = _build_combined_candidate_code(
            candidate_cache.get(state_name, {}).get("code", ""),
            candidate_cache.get(intrinsic_name, {}).get("code", ""),
        )
        if not combined_code:
            return _identity_bundle("joint_pair_identity", state_fn_raw_override=raw_state_fn)
        try:
            revise_state_combo, intrinsic_reward_combo = load_functions_from_code(combined_code)
        except Exception:
            return _identity_bundle("joint_pair_identity", state_fn_raw_override=raw_state_fn)
        intrinsic_reward_effective, post_summary = _build_intrinsic_postprocessed_fn(
            intrinsic_reward=intrinsic_reward_combo,
            revise_state=revise_state_combo,
            reference_states=reference_states,
            post_cfg=intrinsic_postprocess_cfg,
        )
        if intrinsic_reward_effective is None:
            intrinsic_reward_effective = intrinsic_reward_combo
            post_summary = {
                "mode": intrinsic_postprocess_cfg.get("mode", "raw"),
                "available": True,
            }
        return {
            "name": label,
            "origin": "llm",
            "family": str(row.get("family", "")),
            "design_mode": str(row.get("design_mode", "")),
            "feature_groups": list(row.get("feature_groups", [])),
            "revise_state": revise_state_combo,
            "intrinsic_reward_effective": intrinsic_reward_effective,
            "state_fn_revised": _build_policy_state_fn(
                revise_state_combo,
                summary_key=f"official_joint_{state_name}_{intrinsic_name}",
                algorithm=algorithm,
            ),
            "postprocess_summary": post_summary,
        }

    official_shared_library = {
        "mode": "per_algo_independent",
        "state_cores": [],
        "intrinsic_cores": [],
        "joint_pairs": [],
        "per_algo": {},
    }

    intrinsic_efficacy_gate_cfg = {
        "enabled": bool(drl_backend == "finsaber_native"),
        "probe_seed_count": int(max(1, len(selection_seeds))),
        "probe_steps_floor": 40,
        "algo_probe_steps_floor": {
            "sac": 120,
            "td3": 180,
        },
        "algo_probe_seed_floor": {
            "sac": 2,
            "td3": 3,
        },
        "action_equal_ratio_threshold": 0.999,
        "action_near_equal_ratio_threshold": 0.97,
        "action_soft_equal_ratio_threshold": 0.90,
        "reward_total_delta_eps": 1e-9,
        "reward_total_negative_guard": -0.05,
        "intrinsic_mean_delta_eps": 1e-9,
        "retain_positive_intrinsic_algos": ["sac", "td3"],
        "retain_positive_intrinsic_floor": 0.05,
        "retain_positive_intrinsic_score_floor": 0.05,
        "retain_positive_intrinsic_reward_guard": -0.02,
    }

    def _bundle_with_gate_fallback(
        bundle: dict,
        *,
        fallback_from: str,
        fallback_reason: str,
        fallback_source_name: str,
        gate_probe: dict,
    ) -> dict:
        out = dict(bundle)
        out["fallback_from"] = str(fallback_from)
        out["fallback_reason"] = str(fallback_reason)
        out["fallback_source_name"] = str(fallback_source_name)
        out["gate_probe"] = _json_safe(gate_probe)
        return out

    def _group_probe_settings(group_name: str) -> dict:
        return {
            "G0_baseline": {"use_revised": False, "use_intrinsic": False, "intrinsic_input_mode": "raw"},
            "G1_revise_only": {"use_revised": True, "use_intrinsic": False, "intrinsic_input_mode": "revised"},
            "G2_intrinsic_only": {"use_revised": False, "use_intrinsic": True, "intrinsic_input_mode": "raw"},
            "G3_revise_intrinsic": {"use_revised": True, "use_intrinsic": True, "intrinsic_input_mode": "revised"},
        }.get(str(group_name), {"use_revised": False, "use_intrinsic": False, "intrinsic_input_mode": "raw"})

    def _probe_intrinsic_pair_gate(
        algo: str,
        *,
        base_group_name: str,
        intrinsic_group_name: str,
        base_bundle: dict,
        intrinsic_bundle: dict,
    ) -> dict:
        if not intrinsic_efficacy_gate_cfg["enabled"]:
            return {"available": False, "enabled": False, "diagnosis": "gate_disabled"}
        runtime = algo_runtime_cache.get(algo, {})
        runtime_native_cfg = runtime.get("native_cfg")
        runtime_native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {})
        if runtime_native_cfg is None or native_selection_history_df is None:
            return {"available": False, "enabled": True, "diagnosis": "missing_native_probe_context"}
        train_days = int(train_df["date"].nunique())
        algo_probe_steps_floor = int(
            max(
                0,
                _sanitize_float((intrinsic_efficacy_gate_cfg.get("algo_probe_steps_floor", {}) or {}).get(str(algo), 0.0))
                or 0.0,
            )
        )
        probe_steps = int(
            max(
                _effective_steps(cfg.n_small, train_days),
                int(intrinsic_efficacy_gate_cfg.get("probe_steps_floor", 0)),
                algo_probe_steps_floor,
            )
        )
        probe_cfg = replace(runtime_native_cfg, total_timesteps=probe_steps)
        runtime_native_algo_kwargs_probe = _native_small_budget_algo_kwargs(
            algo,
            runtime_native_algo_kwargs,
            int(probe_steps),
        )
        algo_probe_seed_floor = int(
            max(
                1,
                _sanitize_float((intrinsic_efficacy_gate_cfg.get("algo_probe_seed_floor", {}) or {}).get(str(algo), 1.0))
                or 1.0,
            )
        )
        seed_count = int(max(1, intrinsic_efficacy_gate_cfg["probe_seed_count"], algo_probe_seed_floor))
        probe_seed_pool: List[int] = []
        for raw_seed in list(selection_seeds or []) + list(cfg.seeds or []) + [int(cfg.seed)]:
            try:
                cur_seed = int(raw_seed)
            except Exception:
                continue
            if cur_seed not in probe_seed_pool:
                probe_seed_pool.append(cur_seed)
        probe_seeds = list(probe_seed_pool[:seed_count]) if probe_seed_pool else [int(cfg.seed)]
        base_cfg = _group_probe_settings(base_group_name)
        intrinsic_cfg = _group_probe_settings(intrinsic_group_name)
        base_rows: List[dict] = []
        intrinsic_rows: List[dict] = []
        probe_errors: List[dict] = []
        for sd in probe_seeds:
            try:
                base_result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=probe_cfg,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_probe,
                    revise_state=base_bundle["revise_state"],
                    intrinsic_reward=base_bundle["intrinsic_reward_effective"],
                    policy_state_fn=base_bundle["state_fn_revised"],
                    use_revised=bool(base_cfg["use_revised"]),
                    use_intrinsic=bool(base_cfg["use_intrinsic"]),
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode=str(base_cfg["intrinsic_input_mode"]),
                )
                intrinsic_result = train_finsaber_native(
                    algo=algo,
                    train_df=train_df,
                    eval_df=val_df,
                    eval_history_df=native_selection_history_df,
                    cfg=probe_cfg,
                    seed=int(sd),
                    algo_kwargs=runtime_native_algo_kwargs_probe,
                    revise_state=intrinsic_bundle["revise_state"],
                    intrinsic_reward=intrinsic_bundle["intrinsic_reward_effective"],
                    policy_state_fn=intrinsic_bundle["state_fn_revised"],
                    use_revised=bool(intrinsic_cfg["use_revised"]),
                    use_intrinsic=bool(intrinsic_cfg["use_intrinsic"]),
                    intrinsic_w=float(cfg.intrinsic_w),
                    intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                    intrinsic_timing=cfg.intrinsic_timing,
                    intrinsic_input_mode=str(intrinsic_cfg["intrinsic_input_mode"]),
                )
                base_rows.append(_sb3_seed_trace_from_result(base_result, int(sd)))
                intrinsic_rows.append(_sb3_seed_trace_from_result(intrinsic_result, int(sd)))
            except Exception as exc:
                probe_errors.append({"seed": int(sd), "error": str(exc)})
        base_by_seed = {int(row.get("seed", -1)): row for row in base_rows}
        intrinsic_by_seed = {int(row.get("seed", -1)): row for row in intrinsic_rows}
        shared_seeds = sorted(set(base_by_seed.keys()) & set(intrinsic_by_seed.keys()))
        seed_diffs: List[dict] = []
        for sd in shared_seeds:
            base_row = base_by_seed[sd]
            intrinsic_row = intrinsic_by_seed[sd]
            base_actions = base_row.get("eval_actions_policy", []) or base_row.get("eval_actions_final", []) or []
            intrinsic_actions = intrinsic_row.get("eval_actions_policy", []) or intrinsic_row.get("eval_actions_final", []) or []
            n_action = min(len(base_actions), len(intrinsic_actions))
            action_equal = 0
            for idx in range(n_action):
                arr_base = np.asarray(base_actions[idx], dtype=float)
                arr_intr = np.asarray(intrinsic_actions[idx], dtype=float)
                if arr_base.shape == arr_intr.shape and np.allclose(arr_base, arr_intr, atol=1e-12, rtol=0.0):
                    action_equal += 1
            action_equal_ratio = float(action_equal / n_action) if n_action > 0 else 0.0
            base_values = _to_float_list(base_row.get("eval_values", []) or base_row.get("eval_values_final", []))
            intrinsic_values = _to_float_list(intrinsic_row.get("eval_values", []) or intrinsic_row.get("eval_values_final", []))
            n_value = min(len(base_values), len(intrinsic_values))
            eval_value_mae = (
                float(np.mean(np.abs(np.asarray(intrinsic_values[:n_value], dtype=float) - np.asarray(base_values[:n_value], dtype=float))))
                if n_value > 0
                else 0.0
            )
            reward_total_delta_mean = _mean_delta(
                _to_float_list(base_row.get("eval_reward_total", [])),
                _to_float_list(intrinsic_row.get("eval_reward_total", [])),
            )
            intrinsic_mean_delta = _mean_delta(
                _to_float_list(base_row.get("eval_intrinsic_values", [])),
                _to_float_list(intrinsic_row.get("eval_intrinsic_values", [])),
            )
            seed_diffs.append(
                {
                    "seed": int(sd),
                    "action_equal_ratio": action_equal_ratio,
                    "eval_value_mae": eval_value_mae,
                    "eval_reward_total_delta_mean": reward_total_delta_mean,
                    "intrinsic_mean_delta": intrinsic_mean_delta,
                    "len_actions_base": int(len(base_actions)),
                    "len_actions_intrinsic": int(len(intrinsic_actions)),
                }
            )
        summary = {
            "seed_count": int(len(seed_diffs)),
            "action_equal_ratio_mean": float(np.mean([x["action_equal_ratio"] for x in seed_diffs])) if seed_diffs else 0.0,
            "eval_value_mae_mean": float(np.mean([x["eval_value_mae"] for x in seed_diffs])) if seed_diffs else 0.0,
            "eval_reward_total_delta_mean": float(np.mean([x["eval_reward_total_delta_mean"] for x in seed_diffs])) if seed_diffs else 0.0,
            "intrinsic_mean_delta_mean": float(np.mean([x["intrinsic_mean_delta"] for x in seed_diffs])) if seed_diffs else 0.0,
        }
        action_equal_ratio_threshold = float(intrinsic_efficacy_gate_cfg["action_equal_ratio_threshold"])
        action_near_equal_ratio_threshold = float(intrinsic_efficacy_gate_cfg["action_near_equal_ratio_threshold"])
        action_soft_equal_ratio_threshold = float(intrinsic_efficacy_gate_cfg["action_soft_equal_ratio_threshold"])
        reward_total_delta_eps = float(intrinsic_efficacy_gate_cfg["reward_total_delta_eps"])
        reward_total_negative_guard = float(intrinsic_efficacy_gate_cfg["reward_total_negative_guard"])
        intrinsic_mean_delta_eps = float(intrinsic_efficacy_gate_cfg["intrinsic_mean_delta_eps"])
        if seed_diffs:
            if summary["action_equal_ratio_mean"] >= action_equal_ratio_threshold and abs(summary["eval_reward_total_delta_mean"]) > reward_total_delta_eps:
                diagnosis = "intrinsic_changed_reward_total_but_policy_actions_remained_identical"
            elif summary["action_equal_ratio_mean"] >= action_equal_ratio_threshold and abs(summary["intrinsic_mean_delta_mean"]) > intrinsic_mean_delta_eps:
                diagnosis = "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_identical"
            elif summary["action_equal_ratio_mean"] >= action_equal_ratio_threshold:
                diagnosis = "intrinsic_has_no_effect_on_reward_total_and_policy"
            elif summary["action_equal_ratio_mean"] >= action_near_equal_ratio_threshold and abs(summary["eval_reward_total_delta_mean"]) > reward_total_delta_eps:
                diagnosis = "intrinsic_changed_reward_total_but_policy_actions_remained_near_identical"
            elif summary["action_equal_ratio_mean"] >= action_near_equal_ratio_threshold and abs(summary["intrinsic_mean_delta_mean"]) > intrinsic_mean_delta_eps:
                diagnosis = "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_near_identical"
            elif (
                summary["action_equal_ratio_mean"] >= action_soft_equal_ratio_threshold
                and summary["eval_reward_total_delta_mean"] <= reward_total_negative_guard
            ):
                diagnosis = "intrinsic_degraded_probe_reward_total_without_clear_policy_separation"
            elif summary["eval_value_mae_mean"] <= 1e-9:
                diagnosis = "policy_actions_changed_but_eval_values_not_sensitive"
            else:
                diagnosis = "mixed_or_partial_separation"
        else:
            diagnosis = "insufficient_seed_overlap"
        should_fallback = bool(
            seed_diffs
            and diagnosis in {
                "intrinsic_changed_reward_total_but_policy_actions_remained_identical",
                "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_identical",
                "intrinsic_changed_reward_total_but_policy_actions_remained_near_identical",
                "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_near_identical",
                "intrinsic_degraded_probe_reward_total_without_clear_policy_separation",
                "intrinsic_has_no_effect_on_reward_total_and_policy",
            }
        )
        return {
            "enabled": True,
            "available": bool(seed_diffs),
            "algo": str(algo),
            "base_group": str(base_group_name),
            "intrinsic_group": str(intrinsic_group_name),
            "probe_steps": int(probe_steps),
            "probe_seeds": [int(x) for x in probe_seeds],
            "diagnosis": diagnosis,
            "should_fallback": should_fallback,
            "summary": summary,
            "seed_diffs": seed_diffs,
            "errors": probe_errors,
        }

    def _should_keep_positive_intrinsic_after_gate(
        algo: str,
        promotion_row: dict | None,
        gate_report: dict,
        *,
        group_name: str,
    ) -> bool:
        if not bool(gate_report.get("should_fallback")):
            return False
        if str(group_name) != "G2_intrinsic_only":
            return False
        retain_algos = {
            str(x).strip().lower()
            for x in (intrinsic_efficacy_gate_cfg.get("retain_positive_intrinsic_algos", []) or [])
            if str(x).strip()
        }
        if str(algo).strip().lower() not in retain_algos:
            return False
        if not promotion_row or str(promotion_row.get("origin", "")) != "llm":
            return False
        diagnosis = str(gate_report.get("diagnosis", ""))
        if diagnosis not in {
            "intrinsic_changed_reward_total_but_policy_actions_remained_identical",
            "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_identical",
            "intrinsic_changed_reward_total_but_policy_actions_remained_near_identical",
            "intrinsic_changed_intrinsic_signal_but_policy_actions_remained_near_identical",
        }:
            return False
        if not bool(promotion_row.get("intrinsic_signal_nontrivial_raw", False)):
            return False
        per_algo = (promotion_row.get("per_algo", {}) or {}).get(algo, {}) or {}
        target_delta = float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0)))
        target_score = float(_sanitize_float(per_algo.get("delta_score_mean", 0.0)))
        target_delta_floor = float(intrinsic_efficacy_gate_cfg.get("retain_positive_intrinsic_floor", 0.0))
        target_score_floor = float(intrinsic_efficacy_gate_cfg.get("retain_positive_intrinsic_score_floor", 0.0))
        if target_delta < target_delta_floor and target_score < target_score_floor:
            return False
        summary = dict(gate_report.get("summary", {}) or {})
        reward_total_delta = float(_sanitize_float(summary.get("eval_reward_total_delta_mean", 0.0)))
        intrinsic_mean_delta = float(_sanitize_float(summary.get("intrinsic_mean_delta_mean", 0.0)))
        if (
            abs(reward_total_delta) <= float(intrinsic_efficacy_gate_cfg.get("reward_total_delta_eps", 0.0))
            and abs(intrinsic_mean_delta) <= float(intrinsic_efficacy_gate_cfg.get("intrinsic_mean_delta_eps", 0.0))
        ):
            return False
        if reward_total_delta <= float(intrinsic_efficacy_gate_cfg.get("retain_positive_intrinsic_reward_guard", -0.02)):
            return False
        return True

    def _mark_gate_keep_override(
        gate_report: dict,
        algo: str,
        promotion_row: dict | None,
    ) -> dict:
        out = dict(gate_report)
        per_algo = ((promotion_row or {}).get("per_algo", {}) or {}).get(algo, {}) or {}
        out["should_fallback"] = False
        out["kept_after_gate"] = True
        out["keep_reason"] = "positive_target_algo_intrinsic_signal_under_short_probe"
        out["keep_source_name"] = str((promotion_row or {}).get("name", ""))
        out["keep_target_delta_sharpe"] = float(_sanitize_float(per_algo.get("delta_sharpe_mean", 0.0)))
        out["keep_target_delta_score"] = float(_sanitize_float(per_algo.get("delta_score_mean", 0.0)))
        return out

    official_group_bundles_by_algo: Dict[str, dict] = {}
    official_intrinsic_postprocess_summary_by_algo: Dict[str, dict] = {}
    official_best_joint_pair_by_algo: Dict[str, str] = {}
    intrinsic_efficacy_gate_by_algo: Dict[str, dict] = {}
    for algo in eval_algos:
        algo_state_fn_raw = _state_fn_raw_for_algo(algo)
        state_core_inputs, intrinsic_core_inputs, joint_pair_inputs = _build_algo_component_inputs(algo)
        official_shared_library["per_algo"][algo] = {
            "state_cores": _json_safe(state_core_inputs),
            "intrinsic_cores": _json_safe(intrinsic_core_inputs),
            "joint_pairs": _json_safe(joint_pair_inputs),
            "selected_rows": _json_safe(per_algo_selected_rows.get(algo, [])),
        }
        algo_state_row, algo_state_low_conf = _pick_algo_promoted_row(state_core_inputs, algo)
        algo_intrinsic_row, algo_intrinsic_low_conf = _pick_algo_promoted_row(
            intrinsic_core_inputs,
            algo,
            require_nontrivial_intrinsic=True,
        )
        algo_joint_row, algo_joint_low_conf = _pick_algo_promoted_row(joint_pair_inputs, algo)
        algo_group_bundles = {
            "G1_revise_only": (
                _bundle_from_candidate_name(
                    str(algo_state_row.get("name", "")),
                    policy_uses_revised=True,
                    use_intrinsic=False,
                    state_fn_raw_override=algo_state_fn_raw,
                )
                if algo_state_row is not None
                else _identity_bundle(f"{algo}_state_core_identity", state_fn_raw_override=algo_state_fn_raw)
            ),
            "G2_intrinsic_only": (
                _bundle_from_candidate_name(
                    str(algo_intrinsic_row.get("name", "")),
                    policy_uses_revised=False,
                    use_intrinsic=True,
                    intrinsic_input_mode="raw",
                    state_fn_raw_override=algo_state_fn_raw,
                )
                if algo_intrinsic_row is not None
                else _identity_bundle(f"{algo}_intrinsic_core_identity", state_fn_raw_override=algo_state_fn_raw)
            ),
            "G3_revise_intrinsic": (
                _bundle_from_joint_row(
                    algo_joint_row,
                    algorithm=algo,
                    state_fn_raw_override=algo_state_fn_raw,
                )
                if algo_joint_row is not None
                else _identity_bundle(f"{algo}_joint_pair_identity", state_fn_raw_override=algo_state_fn_raw)
            ),
            "_promotion": {
                "state_core": _json_safe(algo_state_row) if algo_state_row is not None else None,
                "intrinsic_core": _json_safe(algo_intrinsic_row) if algo_intrinsic_row is not None else None,
                "joint_pair": _json_safe(algo_joint_row) if algo_joint_row is not None else None,
                "low_confidence": {
                    "state_core": bool(algo_state_low_conf),
                    "intrinsic_core": bool(algo_intrinsic_low_conf),
                    "joint_pair": bool(algo_joint_low_conf),
                },
            },
        }
        gate_reports: Dict[str, dict] = {}
        if intrinsic_efficacy_gate_cfg["enabled"]:
            g0_bundle = _identity_bundle(f"{algo}_g0_baseline_identity", state_fn_raw_override=algo_state_fn_raw)
            g2_report = _probe_intrinsic_pair_gate(
                algo,
                base_group_name="G0_baseline",
                intrinsic_group_name="G2_intrinsic_only",
                base_bundle=g0_bundle,
                intrinsic_bundle=algo_group_bundles["G2_intrinsic_only"],
            )
            if _should_keep_positive_intrinsic_after_gate(
                algo,
                algo_intrinsic_row,
                g2_report,
                group_name="G2_intrinsic_only",
            ):
                g2_report = _mark_gate_keep_override(g2_report, algo, algo_intrinsic_row)
            gate_reports["G2_intrinsic_only"] = _json_safe(g2_report)
            if g2_report.get("should_fallback"):
                algo_group_bundles["G2_intrinsic_only"] = _bundle_with_gate_fallback(
                    dict(g0_bundle),
                    fallback_from="G2_intrinsic_only",
                    fallback_reason=str(g2_report.get("diagnosis", "")),
                    fallback_source_name=str(algo_group_bundles["G2_intrinsic_only"].get("name", "")),
                    gate_probe=g2_report,
                )
            g3_report = _probe_intrinsic_pair_gate(
                algo,
                base_group_name="G1_revise_only",
                intrinsic_group_name="G3_revise_intrinsic",
                base_bundle=algo_group_bundles["G1_revise_only"],
                intrinsic_bundle=algo_group_bundles["G3_revise_intrinsic"],
            )
            if _should_keep_positive_intrinsic_after_gate(
                algo,
                algo_joint_row,
                g3_report,
                group_name="G3_revise_intrinsic",
            ):
                g3_report = _mark_gate_keep_override(g3_report, algo, algo_joint_row)
            gate_reports["G3_revise_intrinsic"] = _json_safe(g3_report)
            if g3_report.get("should_fallback"):
                algo_group_bundles["G3_revise_intrinsic"] = _bundle_with_gate_fallback(
                    dict(algo_group_bundles["G1_revise_only"]),
                    fallback_from="G3_revise_intrinsic",
                    fallback_reason=str(g3_report.get("diagnosis", "")),
                    fallback_source_name=str(algo_group_bundles["G3_revise_intrinsic"].get("name", "")),
                    gate_probe=g3_report,
                )
        algo_group_bundles["_promotion"]["intrinsic_efficacy_gate"] = _json_safe(gate_reports)
        intrinsic_efficacy_gate_by_algo[algo] = _json_safe(gate_reports)
        official_group_bundles_by_algo[algo] = algo_group_bundles
        official_best_joint_pair_by_algo[algo] = str(algo_group_bundles["G3_revise_intrinsic"].get("name", ""))
        official_intrinsic_postprocess_summary_by_algo[algo] = {
            group_name: _json_safe(
                bundle.get("postprocess_summary", {"mode": intrinsic_postprocess_cfg.get("mode", "raw"), "available": False})
            )
            for group_name, bundle in algo_group_bundles.items()
            if not str(group_name).startswith("_")
        }
    intrinsic_postprocess_summary = {
        algo: payload.get("G3_revise_intrinsic", {"mode": intrinsic_postprocess_cfg.get("mode", "raw"), "available": False})
        for algo, payload in official_intrinsic_postprocess_summary_by_algo.items()
    }

    def _bundle_public_summary(bundle: dict) -> dict:
        return {
            "name": str(bundle.get("name", "")),
            "origin": str(bundle.get("origin", "")),
            "family": str(bundle.get("family", "")),
            "design_mode": str(bundle.get("design_mode", "")),
            "feature_groups": list(bundle.get("feature_groups", [])),
            "fallback_from": str(bundle.get("fallback_from", "")),
            "fallback_reason": str(bundle.get("fallback_reason", "")),
            "fallback_source_name": str(bundle.get("fallback_source_name", "")),
        }

    official_group_bundle_summary_by_algo = {
        algo: {
            group_name: _bundle_public_summary(bundle)
            for group_name, bundle in bundles.items()
            if not str(group_name).startswith("_")
        }
        for algo, bundles in official_group_bundles_by_algo.items()
    }
    official_promotion_by_algo = {
        algo: _json_safe(bundles.get("_promotion", {}))
        for algo, bundles in official_group_bundles_by_algo.items()
    }

    state_scale_summary = {
        "state_norm_effective": state_norm_effective,
        "raw_state_dim": raw_state_dim,
        "reference_sample_count": int(reference_states.shape[0]),
        "volume_indices": volume_indices,
        "raw_reference_stats": matrix_stats(reference_states),
        "policy_spaces": state_norm_policy_spaces,
    }

    # ablations with best candidate
    groups = {
        "G0_baseline": dict(use_revised=False, use_intrinsic=False),
        "G1_revise_only": dict(use_revised=True, use_intrinsic=False),
        "G2_intrinsic_only": dict(use_revised=False, use_intrinsic=True),
        "G3_revise_intrinsic": dict(use_revised=True, use_intrinsic=True),
    }
    if cfg.groups:
        groups = {k: v for k, v in groups.items() if k in cfg.groups}

    results = {
        "candidate_scores": cand_scores,
        "best_candidate": best_name,
        "candidate_scores_by_algo": candidate_scores_by_algo,
        "best_candidate_by_algo": best_candidate_by_algo,
        "search_best_candidate_by_algo": search_best_candidate_by_algo,
        "candidate_pool_size_input_by_algo": candidate_pool_size_input_by_algo,
        "candidate_pool_size_by_algo": candidate_pool_size_by_algo,
        "branch_iteration_effective": {"mode": branch_iteration_mode, **_json_safe(branch_iteration_cfg)},
        "branch_iteration_artifacts": _json_safe(branch_iteration_artifacts),
        "final_selection_effective": {
            "mode": final_selection_mode,
            **_json_safe(final_selection_cfg),
        },
        "final_selection_artifacts": _json_safe(candidate_final_selection_artifacts),
        "official_shared_cores": _json_safe(official_shared_cores),
        "official_shared_library": _json_safe(official_shared_library),
        "official_best_joint_pair_by_algo": _json_safe(official_best_joint_pair_by_algo),
        "official_group_bundles_by_algo": _json_safe(official_group_bundle_summary_by_algo),
        "official_promotion_by_algo": _json_safe(official_promotion_by_algo),
        "intrinsic_efficacy_gate_effective": _json_safe(intrinsic_efficacy_gate_cfg),
        "intrinsic_efficacy_gate_by_algo": _json_safe(intrinsic_efficacy_gate_by_algo),
    }
    bootstrap_cfg = _resolve_bootstrap_cfg(cfg.bootstrap)
    intrinsic_w_schedule: List[float] = []
    for w in (cfg.intrinsic_w_schedule or []):
        wv = _sanitize_float(w)
        if np.isfinite(wv):
            intrinsic_w_schedule.append(float(wv))
    if cfg.intrinsic_w not in intrinsic_w_schedule:
        intrinsic_w_schedule.append(float(cfg.intrinsic_w))
    intrinsic_w_schedule = sorted(set(intrinsic_w_schedule))
    algo_results = {}
    reward_trace: Dict[str, Dict[str, List[dict]]] = {}
    td3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    sb3_seed_trace: Dict[str, Dict[str, List[dict]]] = {}
    metrics_source_map: Dict[str, Dict[str, str]] = {}
    intrinsic_w_effective_map: Dict[str, Dict[str, float]] = {}
    intrinsic_w_selection_trace: Dict[str, Dict[str, dict]] = {}
    evaluation_warnings: List[dict] = []
    for algo in eval_algos:
        runtime = algo_runtime_cache[algo]
        algo_env_cfg = _env_cfg_for_algo(algo)
        is_td3_legacy = bool(runtime["is_td3_legacy"])
        td3_algo_base_cfg = runtime["td3_algo_base_cfg"]
        sb3_algo_base_cfg = runtime["sb3_algo_base_cfg"]
        sb3_algo_kwargs = dict(runtime["sb3_algo_kwargs"])
        td3_policy_action_bound = runtime["td3_policy_action_bound"]
        native_backend = drl_backend == "finsaber_native"
        runtime_native_cfg = runtime.get("native_cfg")
        runtime_native_algo_kwargs = dict(runtime.get("native_algo_kwargs", {}) or {})
        algo_groups = {}
        reward_trace[algo] = {}
        if algo == "td3":
            td3_seed_trace[algo] = {}
        else:
            sb3_seed_trace[algo] = {}
        metrics_source_map[algo] = {}
        intrinsic_w_effective_map[algo] = {}
        intrinsic_w_selection_trace[algo] = {}
        algo_group_bundles = official_group_bundles_by_algo.get(algo, {})
        for gname, gcfg in groups.items():
            group_bundle = algo_group_bundles.get(gname, _identity_bundle(f"{algo}_{gname}_identity"))
            best_revise_state = group_bundle["revise_state"]
            best_intrinsic_reward_effective = group_bundle["intrinsic_reward_effective"]
            state_fn_revised = group_bundle["state_fn_revised"]
            per_seed = []
            reward_trace[algo][gname] = []
            if algo == "td3":
                td3_seed_trace[algo][gname] = []
            else:
                sb3_seed_trace[algo][gname] = []
            metrics_source = "unknown"
            intrinsic_w_effective = float(cfg.intrinsic_w)
            if gcfg["use_intrinsic"] and len(intrinsic_w_schedule) > 1:
                probe_seed_count = int(min(len(cfg.seeds or []), intrinsic_w_tuning_cfg["probe_seed_count"]))
                if probe_seed_count <= 0:
                    probe_seed_count = 1
                probe_seed_list = list(cfg.seeds[:probe_seed_count]) if cfg.seeds else [0]
                probe_scores: List[dict] = []
                for w in intrinsic_w_schedule:
                    w_seed_scores: List[float] = []
                    for sd in probe_seed_list:
                        if native_backend:
                            policy_state_fn_probe = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                            result_probe = train_finsaber_native(
                                algo=algo,
                                train_df=train_df,
                                eval_df=val_df,
                                eval_history_df=native_selection_history_df,
                                cfg=runtime_native_cfg,
                                seed=int(sd),
                                algo_kwargs=runtime_native_algo_kwargs,
                                revise_state=best_revise_state,
                                intrinsic_reward=best_intrinsic_reward_effective,
                                policy_state_fn=policy_state_fn_probe,
                                use_revised=gcfg["use_revised"],
                                use_intrinsic=True,
                                intrinsic_w=float(w),
                                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                                intrinsic_timing=cfg.intrinsic_timing,
                                intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                            )
                            probe_metrics, _ = _sb3_metrics_from_eval(result_probe)
                        elif is_td3_legacy:
                            train_env_probe = TradingEnv(train_df, schema.assets, schema, env_cfg)
                            eval_env_probe = TradingEnv(val_df, schema.assets, schema, env_cfg)
                            state_fn_probe = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                            steps_small_probe = _effective_steps(cfg.n_small, int(train_df["date"].nunique()))
                            td3_cfg_probe = _resolve_td3_cfg(td3_algo_base_cfg, steps_small_probe, cfg.warmup_ratio, cfg.evaluation)
                            result_probe = train_td3(
                                env=train_env_probe,
                                state_dim=(state_fn_probe(np.zeros(schema.dim(), dtype=np.float32)).shape[0]),
                                action_dim=len(schema.assets),
                                cfg=td3_cfg_probe,
                                max_steps=steps_small_probe,
                                state_fn=state_fn_probe,
                                revise_state=best_revise_state,
                                intrinsic_reward=best_intrinsic_reward_effective,
                                intrinsic_w=float(w),
                                use_intrinsic=True,
                                intrinsic_timing=cfg.intrinsic_timing,
                                finagent=finagent,
                                finagent_weight=cfg.finagent_weight,
                                seed=sd,
                                eval_env=eval_env_probe,
                                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                                intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                            )
                            probe_metrics = compute_metrics(np.array(result_probe.eval_values_final))
                        else:
                            action_space_type_probe = _action_space_type(algo)
                            policy_state_fn_probe = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                            sb3_cfg_probe = replace(
                                sb3_algo_base_cfg,
                                total_timesteps=int(_effective_steps(cfg.n_small, int(train_df["date"].nunique()))),
                            )
                            result_probe = train_sb3(
                                algo=algo,
                                train_df=train_df,
                                eval_df=val_df,
                                assets=schema.assets,
                                schema=schema,
                                env_cfg=algo_env_cfg,
                                cfg=sb3_cfg_probe,
                                action_space_type=action_space_type_probe,
                                policy_action_bound=(td3_policy_action_bound if algo == "td3" else None),
                                revise_state=best_revise_state,
                                intrinsic_reward=best_intrinsic_reward_effective,
                                intrinsic_w=float(w),
                                intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                                intrinsic_timing=cfg.intrinsic_timing,
                                use_revised=gcfg["use_revised"],
                                use_intrinsic=True,
                                intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                                policy_state_fn=policy_state_fn_probe,
                                seed=sd,
                                algo_kwargs=sb3_algo_kwargs,
                            )
                            probe_metrics, _ = _sb3_metrics_from_eval(result_probe)
                        w_seed_scores.append(_score_from_metrics(probe_metrics))
                    probe_scores.append(
                        {
                            "intrinsic_w": float(w),
                            "mean_score": float(np.mean(w_seed_scores)) if w_seed_scores else 0.0,
                            "seed_scores": [float(x) for x in w_seed_scores],
                        }
                    )
                probe_scores.sort(key=lambda x: x["mean_score"], reverse=True)
                if probe_scores:
                    top_score = float(probe_scores[0]["mean_score"])
                    tie_tol = float(intrinsic_w_tuning_cfg["tie_tolerance"])
                    tied = [row for row in probe_scores if float(row["mean_score"]) >= (top_score - tie_tol)]
                    if intrinsic_w_tuning_cfg["prefer_smallest_w"] and tied:
                        intrinsic_w_effective = float(min(tied, key=lambda x: float(x["intrinsic_w"]))["intrinsic_w"])
                    else:
                        intrinsic_w_effective = float(probe_scores[0]["intrinsic_w"])
                intrinsic_w_selection_trace[algo][gname] = {
                    "enabled": True,
                    "schedule": intrinsic_w_schedule,
                    "probe_seed_list": probe_seed_list,
                    "tuning_cfg": intrinsic_w_tuning_cfg,
                    "selected_intrinsic_w": intrinsic_w_effective,
                    "probe_scores": probe_scores,
                }
            else:
                intrinsic_w_selection_trace[algo][gname] = {
                    "enabled": False,
                    "schedule": intrinsic_w_schedule,
                    "probe_seed_list": [],
                    "tuning_cfg": intrinsic_w_tuning_cfg,
                    "selected_intrinsic_w": intrinsic_w_effective,
                    "probe_scores": [],
                }
            intrinsic_w_effective_map[algo][gname] = float(intrinsic_w_effective)
            for sd in cfg.seeds:
                if native_backend:
                    policy_state_fn = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                    result = train_finsaber_native(
                        algo=algo,
                        train_df=train_df,
                        eval_df=test_df,
                        eval_history_df=native_eval_history_df,
                        cfg=runtime_native_cfg,
                        seed=int(sd),
                        algo_kwargs=runtime_native_algo_kwargs,
                        revise_state=best_revise_state,
                        intrinsic_reward=best_intrinsic_reward_effective,
                        policy_state_fn=policy_state_fn,
                        use_revised=gcfg["use_revised"],
                        use_intrinsic=gcfg["use_intrinsic"],
                        intrinsic_w=float(intrinsic_w_effective),
                        intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                        intrinsic_timing=cfg.intrinsic_timing,
                        intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                    )
                    metrics, metrics_source = _sb3_metrics_from_eval(result)
                    intrinsic_vals = result["intrinsic"]
                    reward_env_vals = result.get("reward_env", [])
                    action_penalty_vals = result.get("action_penalty", [])
                    reward_total_vals = result.get("reward_total", result.get("rewards", []))
                    if not action_penalty_vals and reward_env_vals:
                        action_penalty_vals = [0.0 for _ in range(len(reward_env_vals))]
                    intrinsic_ratio_vals = [
                        abs(float(intrinsic_w_effective) * float(ri))
                        / (abs(float(re)) + abs(float(ap)) + 1e-6)
                        for ri, re, ap in zip(intrinsic_vals, reward_env_vals, action_penalty_vals)
                    ]
                    if algo == "td3":
                        td3_seed_trace[algo][gname].append(_td3_seed_trace_from_sb3_result(result, int(sd)))
                    else:
                        sb3_seed_trace[algo][gname].append(_sb3_seed_trace_from_result(result, int(sd)))
                elif is_td3_legacy:
                    train_env = TradingEnv(train_df, schema.assets, schema, algo_env_cfg)
                    eval_env = TradingEnv(test_df, schema.assets, schema, algo_env_cfg)
                    state_fn = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                    steps_full = _effective_steps(cfg.n_full, int(train_df["date"].nunique()))
                    td3_cfg = _resolve_td3_cfg(td3_algo_base_cfg, steps_full, cfg.warmup_ratio, cfg.evaluation)
                    result = train_td3(
                        env=train_env,
                        state_dim=(state_fn(np.zeros(schema.dim(), dtype=np.float32)).shape[0]),
                        action_dim=len(schema.assets),
                        cfg=td3_cfg,
                        max_steps=steps_full,
                        state_fn=state_fn,
                        revise_state=best_revise_state,
                        intrinsic_reward=best_intrinsic_reward_effective,
                        intrinsic_w=float(intrinsic_w_effective),
                        use_intrinsic=gcfg["use_intrinsic"],
                        intrinsic_timing=cfg.intrinsic_timing,
                        finagent=finagent,
                        finagent_weight=cfg.finagent_weight,
                        seed=sd,
                        eval_env=eval_env,
                        intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                        intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                    )
                    metrics = compute_metrics(np.array(result.eval_values_final))
                    metrics_source = "td3.eval_values_final"
                    intrinsic_vals = result.eval_intrinsic_values
                    reward_env_vals = result.eval_reward_env
                    action_penalty_vals = result.eval_action_penalties
                    reward_total_vals = result.eval_reward_total
                    intrinsic_ratio_vals = result.eval_intrinsic_ratio
                    td3_seed_trace[algo][gname].append(_td3_seed_trace_from_result(result, int(sd)))
                else:
                    action_space_type = _action_space_type(algo, drl_backend)
                    policy_state_fn = state_fn_revised if gcfg["use_revised"] else state_fn_raw
                    sb3_cfg_run = replace(
                        sb3_algo_base_cfg,
                        total_timesteps=int(_effective_steps(cfg.n_full, int(train_df["date"].nunique()))),
                    )
                    result = train_sb3(
                        algo=algo,
                        train_df=train_df,
                        eval_df=test_df,
                        assets=schema.assets,
                        schema=schema,
                        env_cfg=algo_env_cfg,
                        cfg=sb3_cfg_run,
                        action_space_type=action_space_type,
                        policy_action_bound=(td3_policy_action_bound if algo == "td3" else None),
                        revise_state=best_revise_state,
                        intrinsic_reward=best_intrinsic_reward_effective,
                        intrinsic_w=float(intrinsic_w_effective),
                        intrinsic_scale_mode=cfg.intrinsic_scale_mode,
                        intrinsic_timing=cfg.intrinsic_timing,
                        use_revised=gcfg["use_revised"],
                        use_intrinsic=gcfg["use_intrinsic"],
                        intrinsic_input_mode=("revised" if gcfg["use_revised"] else "raw"),
                        policy_state_fn=policy_state_fn,
                        seed=sd,
                        algo_kwargs=sb3_algo_kwargs,
                    )
                    metrics, metrics_source = _sb3_metrics_from_eval(result)
                    intrinsic_vals = result["intrinsic"]
                    reward_env_vals = result.get("reward_env", [])
                    action_penalty_vals = result.get("action_penalty", [])
                    reward_total_vals = result.get("reward_total", result.get("rewards", []))
                    if not action_penalty_vals and reward_env_vals:
                        action_penalty_vals = [0.0 for _ in range(len(reward_env_vals))]
                    intrinsic_ratio_vals = [
                        abs(float(intrinsic_w_effective) * float(ri))
                        / (abs(float(re)) + abs(float(ap)) + 1e-6)
                        for ri, re, ap in zip(intrinsic_vals, reward_env_vals, action_penalty_vals)
                    ]
                    if algo == "td3":
                        td3_seed_trace[algo][gname].append(_td3_seed_trace_from_sb3_result(result, int(sd)))
                    else:
                        sb3_seed_trace[algo][gname].append(_sb3_seed_trace_from_result(result, int(sd)))

                intrinsic_vals = intrinsic_vals or []
                reward_env_vals = reward_env_vals or []
                action_penalty_vals = action_penalty_vals or []
                reward_total_vals = reward_total_vals or []
                intrinsic_ratio_vals = intrinsic_ratio_vals or []
                if not action_penalty_vals and reward_env_vals:
                    action_penalty_vals = [0.0 for _ in range(len(reward_env_vals))]
                if not reward_env_vals and reward_total_vals and intrinsic_vals:
                    n = min(len(reward_total_vals), len(intrinsic_vals))
                    reward_total_vals = reward_total_vals[:n]
                    intrinsic_vals = intrinsic_vals[:n]
                    reward_env_vals = [
                        float(rt) - float(intrinsic_w_effective) * float(ri)
                        for rt, ri in zip(reward_total_vals, intrinsic_vals)
                    ]
                    action_penalty_vals = [0.0 for _ in range(n)]
                    intrinsic_ratio_vals = [
                        abs(float(intrinsic_w_effective) * float(ri)) / (abs(float(re)) + 1e-6)
                        for ri, re in zip(intrinsic_vals, reward_env_vals)
                    ]
                intrinsic_ratio_robust_vals, env_near_zero_ratio = _robust_intrinsic_ratio_vals(
                    intrinsic_vals=intrinsic_vals,
                    reward_env_vals=reward_env_vals,
                    action_penalty_vals=action_penalty_vals,
                    intrinsic_w=float(intrinsic_w_effective),
                    floor=float(diagnostics_cfg["robust_ratio_floor"]),
                )
                n_delta = min(len(reward_total_vals), len(reward_env_vals))
                reward_total_minus_env_vals = [
                    float(reward_total_vals[i]) - float(reward_env_vals[i])
                    for i in range(n_delta)
                ]

                intrinsic_summary = {
                    "mean": float(np.mean(intrinsic_vals)) if intrinsic_vals else 0.0,
                    "std": float(np.std(intrinsic_vals)) if intrinsic_vals else 0.0,
                    "count": int(len(intrinsic_vals)),
                }
                reward_trace[algo][gname].append(
                    {
                        "seed": sd,
                        "reward_env": _reward_stats(reward_env_vals),
                        "action_penalty": _reward_stats(action_penalty_vals),
                        "reward_total": _reward_stats(reward_total_vals),
                        "intrinsic": _reward_stats(intrinsic_vals),
                        "intrinsic_w_effective": float(intrinsic_w_effective),
                        "intrinsic_effect_ratio": _reward_stats(intrinsic_ratio_vals),
                        "intrinsic_effect_ratio_robust": _reward_stats(intrinsic_ratio_robust_vals),
                        "env_near_zero_ratio": float(env_near_zero_ratio),
                        "reward_total_minus_env": _reward_stats(reward_total_minus_env_vals),
                    }
                )
                per_seed.append(
                    {
                        "seed": sd,
                        "metrics": metrics,
                        "intrinsic": intrinsic_summary,
                        "intrinsic_w_effective": float(intrinsic_w_effective),
                        "metrics_source": metrics_source,
                    }
                )
            # aggregate
            agg = {}
            for k in per_seed[0]["metrics"].keys():
                vals = [p["metrics"][k] for p in per_seed]
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                if bootstrap_cfg["enabled"] and k in {"Sharpe", "CR"}:
                    bs = bootstrap_mean_ci(
                        vals,
                        n_resamples=bootstrap_cfg["n_resamples"],
                        alpha=bootstrap_cfg["alpha"],
                        random_seed=_stable_seed(
                            bootstrap_cfg["random_seed"],
                            f"{algo}:{gname}:{k}",
                        ),
                    )
                    agg[k]["bootstrap"] = {
                        "ci_low": bs["ci_low"],
                        "ci_high": bs["ci_high"],
                        "n_resamples": bs["n_resamples"],
                        "alpha": bs["alpha"],
                    }
            algo_groups[gname] = {"per_seed": per_seed, "summary": agg}
            metrics_source_map[algo][gname] = metrics_source

        # Diagnostics for suspicious metric equality / intrinsic no-op behavior.
        if "G1_revise_only" in algo_groups and "G3_revise_intrinsic" in algo_groups:
            g1 = algo_groups["G1_revise_only"]["summary"]
            g3 = algo_groups["G3_revise_intrinsic"]["summary"]
            core_keys = ["Sharpe", "CR", "MDD", "AV"]
            same_metrics = all(abs(float(g1[k]["mean"]) - float(g3[k]["mean"])) <= 1e-12 for k in core_keys)
            if same_metrics:
                g3_intr_means = [
                    float(row.get("intrinsic", {}).get("mean", 0.0))
                    for row in reward_trace[algo].get("G3_revise_intrinsic", [])
                ]
                evaluation_warnings.append(
                    {
                        "algorithm": algo,
                        "type": "g1_g3_identical_metrics",
                        "message": "G1_revise_only and G3_revise_intrinsic have identical aggregate core metrics.",
                        "g3_intrinsic_mean_avg": float(np.mean(g3_intr_means)) if g3_intr_means else 0.0,
                        "metrics": {k: {"G1": g1[k]["mean"], "G3": g3[k]["mean"]} for k in core_keys},
                    }
                )

        for base_group, intrinsic_group in [("G0_baseline", "G2_intrinsic_only"), ("G1_revise_only", "G3_revise_intrinsic")]:
            if base_group not in reward_trace[algo] or intrinsic_group not in reward_trace[algo]:
                continue
            base_rows = reward_trace[algo][base_group]
            intrinsic_rows = reward_trace[algo][intrinsic_group]
            for idx in range(min(len(base_rows), len(intrinsic_rows))):
                b = base_rows[idx]
                i = intrinsic_rows[idx]
                intrinsic_mean = float(i["intrinsic"]["mean"])
                if abs(intrinsic_mean) <= 1e-8:
                    continue
                total_delta = abs(float(i["reward_total"]["mean"]) - float(b["reward_total"]["mean"]))
                if total_delta <= 1e-12:
                    evaluation_warnings.append(
                        {
                            "algorithm": algo,
                            "type": "intrinsic_effect_zero_total_reward",
                            "group_pair": [base_group, intrinsic_group],
                            "seed": i["seed"],
                            "message": "Intrinsic mean is non-zero but reward_total mean equals the paired non-intrinsic group.",
                            "intrinsic_mean": intrinsic_mean,
                            "reward_total_base": b["reward_total"]["mean"],
                            "reward_total_intrinsic": i["reward_total"]["mean"],
                        }
                    )

        algo_results[algo] = {"groups": algo_groups, "metrics_source": metrics_source_map[algo]}

    td3_g1_g3_diff = {}
    td3_diff_summary = {}
    for algo, algo_trace in td3_seed_trace.items():
        diff_payload = _build_td3_g1_g3_diff(algo_trace)
        td3_g1_g3_diff[algo] = diff_payload
        td3_diff_summary[algo] = {
            "diagnosis": diff_payload.get("diagnosis"),
            "seed_count": diff_payload.get("summary", {}).get("seed_count", 0),
            "action_equal_ratio_mean": diff_payload.get("summary", {}).get("action_equal_ratio_mean", 0.0),
            "eval_value_mae_mean": diff_payload.get("summary", {}).get("eval_value_mae_mean", 0.0),
            "eval_reward_total_delta_mean": diff_payload.get("summary", {}).get("eval_reward_total_delta_mean", 0.0),
        }
    train_days = int(train_df["date"].nunique())
    small_steps = _effective_steps(cfg.n_small, train_days)
    full_steps = _effective_steps(cfg.n_full, train_days)
    td3_cfg_small = _resolve_td3_cfg(td3_base_cfg, small_steps, cfg.warmup_ratio, cfg.evaluation)
    td3_cfg_full = _resolve_td3_cfg(td3_base_cfg, full_steps, cfg.warmup_ratio, cfg.evaluation)
    actor_max_action_effective = (
        float(td3_cfg_full.actor_max_action)
        if td3_cfg_full.actor_max_action is not None
        else float(cfg.max_trade)
    )
    td3_action_saturation_summary = _build_td3_action_saturation_summary(
        td3_seed_trace,
        action_bound=actor_max_action_effective,
        collapse_threshold=0.95,
    )
    policy_behavior_summary = _build_policy_behavior_summary(
        td3_seed_trace=td3_seed_trace,
        sb3_seed_trace=sb3_seed_trace,
        action_bound=float(env_cfg.max_trade),
        td3_action_bound=float(actor_max_action_effective),
        collapse_threshold=0.95,
    )
    for algo in eval_algos:
        algo_group_payload = algo_results.get(algo, {}).get("groups", {}) or {}
        if not algo_group_payload:
            continue
        zero_core_groups = {}
        core_keys = ["Sharpe", "CR", "MDD", "AV"]
        for group_name, group_payload in algo_group_payload.items():
            summary = (group_payload or {}).get("summary", {}) or {}
            if not summary:
                continue
            if all(abs(float((summary.get(metric_name) or {}).get("mean", 0.0))) <= 1e-12 for metric_name in core_keys):
                zero_core_groups[group_name] = {
                    metric_name: float((summary.get(metric_name) or {}).get("mean", 0.0))
                    for metric_name in core_keys
                }
        if zero_core_groups and len(zero_core_groups) == len(algo_group_payload):
            overall_behavior = (policy_behavior_summary.get(algo, {}) or {}).get("_overall", {}) or {}
            unique_mean = float(overall_behavior.get("unique_action_count_mean", 0.0))
            entropy_mean = float(overall_behavior.get("action_entropy_mean", 0.0))
            turnover_mean = float(overall_behavior.get("avg_daily_portfolio_weight_change_mean", 0.0))
            if unique_mean >= 2.0 or entropy_mean >= 0.1 or abs(turnover_mean) >= 1e-6:
                evaluation_warnings.append(
                    {
                        "algorithm": algo,
                        "type": "all_groups_zero_core_metrics_with_nontrivial_behavior",
                        "message": "All G0-G3 core metrics are zero while policy behavior remains non-trivial.",
                        "group_metrics": zero_core_groups,
                        "behavior_summary": {
                            "unique_action_count_mean": unique_mean,
                            "action_entropy_mean": entropy_mean,
                            "avg_daily_portfolio_weight_change_mean": turnover_mean,
                            "near_bound_ratio_mean": float(overall_behavior.get("near_bound_ratio_mean", 0.0)),
                        },
                    }
                )
    actor_collapse_detected = bool(
        td3_action_saturation_summary.get("td3", {})
        .get("_overall", {})
        .get("actor_collapse_detected", False)
    )

    if len(eval_algos) == 1:
        results["groups"] = algo_results[eval_algos[0]]["groups"]
    else:
        results["algorithms"] = algo_results
    results["protocol"] = {
        "eval_protocol": split_meta["protocol"],
        "split": split_meta,
        "selected_assets": schema.assets,
        "window_setup": cfg.window_setup,
        "universe_snapshot": universe_snapshot,
        "decision_ts_rule": decision_ts_rule,
        "action_quantization_mode": action_quantization_mode,
        "discrete_action_levels": _resolve_discrete_action_levels(cfg),
        "action_bound_penalty_effective": action_bound_penalty_cfg,
        "action_bound_penalty_effective_by_algo": {
            algo: {
                **_resolve_action_bound_penalty_cfg(cfg, algo),
                "reference_bound": float(_resolve_action_bound_penalty_reference_bound(cfg, algo)),
            }
            for algo in eval_algos
        },
        "metrics_source": metrics_source_map,
        "evaluation_warning": evaluation_warnings,
        "scoring_objective": candidate_scoring_objective,
        "candidate_scoring_effective": candidate_scoring_cfg,
        "llm_iteration_mode": llm_iteration_mode,
        "llm_branch_algorithms": llm_branch_algos,
        "llm_branch_parallel_workers": int(branch_parallel_workers),
        "branch_iteration_effective": {"mode": branch_iteration_mode, **_json_safe(branch_iteration_cfg)},
        "branch_iteration_artifacts": _json_safe(branch_iteration_artifacts),
        "candidate_selection_seeds": list(selection_seeds),
        "candidate_selection_seed_count": int(len(selection_seeds)),
        "candidate_pool_size_input_by_algo": candidate_pool_size_input_by_algo,
        "candidate_pool_size_by_algo": candidate_pool_size_by_algo,
        "final_selection_effective": {
            "mode": final_selection_mode,
            **_json_safe(final_selection_cfg),
        },
        "final_selection_artifacts": _json_safe(candidate_final_selection_artifacts),
        "best_candidate_by_algo": best_candidate_by_algo,
        "search_best_candidate_by_algo": search_best_candidate_by_algo,
        "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
        "official_shared_cores": _json_safe(official_shared_cores),
        "official_shared_library": _json_safe(official_shared_library),
        "official_best_joint_pair_by_algo": _json_safe(official_best_joint_pair_by_algo),
        "official_group_bundles_by_algo": _json_safe(official_group_bundle_summary_by_algo),
        "official_promotion_by_algo": _json_safe(official_promotion_by_algo),
        "intrinsic_efficacy_gate_effective": _json_safe(intrinsic_efficacy_gate_cfg),
        "intrinsic_efficacy_gate_by_algo": _json_safe(intrinsic_efficacy_gate_by_algo),
        "intrinsic_scale_mode": cfg.intrinsic_scale_mode,
        "intrinsic_timing": cfg.intrinsic_timing,
        "intrinsic_timing_effective": cfg.intrinsic_timing,
        "intrinsic_w_effective": intrinsic_w_effective_map,
        "intrinsic_w_selection_trace": intrinsic_w_selection_trace,
        "intrinsic_w_tuning_effective": intrinsic_w_tuning_cfg,
        "state_norm_effective": state_norm_effective,
        "intrinsic_postprocess_effective": intrinsic_postprocess_summary,
        "intrinsic_postprocess_effective_by_algo": _json_safe(intrinsic_postprocess_summary),
        "intrinsic_postprocess_effective_by_algo_by_group": _json_safe(official_intrinsic_postprocess_summary_by_algo),
        "algo_tuning_effective": algo_tuning_effective,
        "diagnostics_effective": diagnostics_cfg,
        "max_trade_effective": int(cfg.max_trade),
        "warmup_ratio": cfg.warmup_ratio,
        "td3_backend": td3_backend,
        "td3_diagnostic_trace_enabled": True,
        "td3_diff_summary": td3_diff_summary,
        "td3_action_saturation_summary": td3_action_saturation_summary,
        "policy_behavior_summary": {
            algo: payload.get("_overall", {})
            for algo, payload in policy_behavior_summary.items()
        },
        "actor_collapse_detected": actor_collapse_detected,
        "actor_max_action_effective": actor_max_action_effective,
        "warmup_ratio_effective": {
            "n_small": {
                "max_steps": small_steps,
                "start_timesteps": td3_cfg_small.start_timesteps,
                "ratio": float(td3_cfg_small.start_timesteps / max(1, small_steps)),
            },
            "n_full": {
                "max_steps": full_steps,
                "start_timesteps": td3_cfg_full.start_timesteps,
                "ratio": float(td3_cfg_full.start_timesteps / max(1, full_steps)),
            },
        },
        "bootstrap_ci": bootstrap_cfg,
        "evaluation": cfg.evaluation or {},
        "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
        "split_date_filter_summary": _json_safe(split_date_filter_summary),
        "stub": {
            "use_finagent_signal": cfg.use_finagent_signal,
            "finagent_weight": cfg.finagent_weight,
        },
        "walk_forward": cfg.walk_forward or {"enabled": False},
        "experiment_phase": experiment_cfg["phase"],
        "experiment_frozen": bool(experiment_cfg["frozen"]),
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
    }

    # save metrics
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2))
    state_scale_path = run_dir / "state_scale_summary.json"
    state_scale_path.write_text(json.dumps(state_scale_summary, indent=2))
    td3_trace_path = run_dir / "td3_seed_trace.json"
    sb3_trace_path = run_dir / "sb3_action_trace.json"
    td3_diff_path = run_dir / "td3_g1_g3_diff.json"
    td3_sat_path = run_dir / "td3_action_saturation.json"
    policy_behavior_path = run_dir / "policy_behavior_summary.json"
    td3_trace_path.write_text(json.dumps(td3_seed_trace, indent=2))
    sb3_trace_path.write_text(json.dumps(sb3_seed_trace, indent=2))
    td3_diff_path.write_text(json.dumps(td3_g1_g3_diff, indent=2))
    td3_sat_path.write_text(json.dumps(td3_action_saturation_summary, indent=2))
    policy_behavior_path.write_text(json.dumps(policy_behavior_summary, indent=2))

    run_manifest = {
        "protocol_version": "trading-lesr-v2",
        "eval_protocol": split_meta["protocol"],
        "split": split_meta,
        "experiment_phase": experiment_cfg["phase"],
        "claim_id": experiment_cfg["claim_id"],
        "hypothesis_id": experiment_cfg["hypothesis_id"],
        "is_confirmatory": _is_confirmatory(experiment_cfg),
        "experiment_frozen": bool(experiment_cfg["frozen"]),
        "config_fingerprint": config_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "candidate_fingerprint_by_algo": candidate_fingerprint_by_algo,
        "best_candidate_by_algo": best_candidate_by_algo,
        "search_best_candidate_by_algo": search_best_candidate_by_algo,
        "llm_iteration_mode": llm_iteration_mode,
        "llm_branch_algorithms": llm_branch_algos,
        "llm_branch_parallel_workers": int(branch_parallel_workers),
        "branch_iteration_effective": {"mode": branch_iteration_mode, **_json_safe(branch_iteration_cfg)},
        "branch_iteration_artifacts": _json_safe(branch_iteration_artifacts),
        "candidate_selection_seeds": list(selection_seeds),
        "candidate_selection_seed_count": int(len(selection_seeds)),
        "candidate_pool_size_input_by_algo": candidate_pool_size_input_by_algo,
        "candidate_pool_size_by_algo": candidate_pool_size_by_algo,
        "final_selection_effective": {
            "mode": final_selection_mode,
            **_json_safe(final_selection_cfg),
        },
        "final_selection_artifacts": _json_safe(candidate_final_selection_artifacts),
        "official_shared_cores": _json_safe(official_shared_cores),
        "official_shared_library": _json_safe(official_shared_library),
        "official_best_joint_pair_by_algo": _json_safe(official_best_joint_pair_by_algo),
        "official_group_bundles_by_algo": _json_safe(official_group_bundle_summary_by_algo),
        "official_promotion_by_algo": _json_safe(official_promotion_by_algo),
        "intrinsic_efficacy_gate_effective": _json_safe(intrinsic_efficacy_gate_cfg),
        "intrinsic_efficacy_gate_by_algo": _json_safe(intrinsic_efficacy_gate_by_algo),
        "selected_assets": schema.assets,
        "window_setup": cfg.window_setup,
        "universe_snapshot": universe_snapshot,
        "decision_ts_rule": decision_ts_rule,
        "action_quantization_mode": action_quantization_mode,
        "discrete_action_levels": _resolve_discrete_action_levels(cfg),
        "action_bound_penalty_effective": action_bound_penalty_cfg,
        "action_bound_penalty_effective_by_algo": {
            algo: {
                **_resolve_action_bound_penalty_cfg(cfg, algo),
                "reference_bound": float(_resolve_action_bound_penalty_reference_bound(cfg, algo)),
            }
            for algo in eval_algos
        },
        "metrics_source": metrics_source_map,
        "evaluation_warning": evaluation_warnings,
        "scoring_objective": candidate_scoring_objective,
        "candidate_scoring_effective": candidate_scoring_cfg,
        "algorithm": cfg.algorithm,
        "eval_algorithms": eval_algos,
        "llm_generation_target": generation_target,
        "llm_scenario_family": {
            "enabled": bool(scenario_enabled),
            "families": list(scenario_families),
            "candidates_per_family_per_iter": int(candidates_per_family),
            "router": _json_safe(router_cfg),
        },
        "scenario_profile": scenario_profile,
        "groups": list(groups.keys()),
        "intrinsic_scale_mode": cfg.intrinsic_scale_mode,
        "intrinsic_timing": cfg.intrinsic_timing,
        "intrinsic_timing_effective": cfg.intrinsic_timing,
        "intrinsic_w_effective": intrinsic_w_effective_map,
        "intrinsic_w_selection_trace": intrinsic_w_selection_trace,
        "intrinsic_w_tuning_effective": intrinsic_w_tuning_cfg,
        "state_norm_effective": state_norm_effective,
        "intrinsic_postprocess_effective": intrinsic_postprocess_summary,
        "intrinsic_postprocess_effective_by_algo": _json_safe(intrinsic_postprocess_summary),
        "intrinsic_postprocess_effective_by_algo_by_group": _json_safe(official_intrinsic_postprocess_summary_by_algo),
        "algo_tuning_effective": algo_tuning_effective,
        "diagnostics_effective": diagnostics_cfg,
        "intrinsic_w": cfg.intrinsic_w,
        "intrinsic_w_schedule": list(cfg.intrinsic_w_schedule or []),
        "max_trade_effective": int(cfg.max_trade),
        "warmup_ratio": cfg.warmup_ratio,
        "td3_backend": td3_backend,
        "td3_diagnostic_trace_enabled": True,
        "td3_diff_summary": td3_diff_summary,
        "td3_action_saturation_summary": td3_action_saturation_summary,
        "policy_behavior_summary": {
            algo: payload.get("_overall", {})
            for algo, payload in policy_behavior_summary.items()
        },
        "actor_collapse_detected": actor_collapse_detected,
        "actor_max_action_effective": actor_max_action_effective,
        "warmup_ratio_effective": {
            "n_small": {
                "max_steps": small_steps,
                "start_timesteps": td3_cfg_small.start_timesteps,
                "ratio": float(td3_cfg_small.start_timesteps / max(1, small_steps)),
            },
            "n_full": {
                "max_steps": full_steps,
                "start_timesteps": td3_cfg_full.start_timesteps,
                "ratio": float(td3_cfg_full.start_timesteps / max(1, full_steps)),
            },
        },
        "bootstrap_ci": bootstrap_cfg,
        "evaluation": cfg.evaluation or {},
        "walk_forward": cfg.walk_forward or {"enabled": False},
        "split_date_filters_effective": _json_safe(cfg.split_date_filters or {}),
        "split_date_filter_summary": _json_safe(split_date_filter_summary),
        "stub": {
            "use_finagent_signal": cfg.use_finagent_signal,
            "finagent_weight": cfg.finagent_weight,
        },
    }
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    # write summary artifacts
    metrics_table_path = _write_metrics_table(run_dir, results)
    summary_path = _write_run_summary(run_dir, cfg, results, iter_trace, llm_errors, split_meta)

    required_files = [
        Path(metrics_table_path),
        run_manifest_path,
        run_dir / "reward_trace.json",
        state_scale_path,
        policy_behavior_path,
        scenario_profile_path,
    ]
    if "td3" in [str(a).lower() for a in eval_algos]:
        required_files.append(td3_diff_path)
        required_files.append(td3_sat_path)
    if any(str(a).lower() != "td3" for a in eval_algos):
        required_files.append(sb3_trace_path)
    run_manifest["completeness_check"] = _build_completeness_check(required_files)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    # artifacts manifest
    root = repo_root()
    artifacts = {
        "raw_data": str(raw_path.relative_to(root)),
        "processed_data": str(processed_path.relative_to(root)),
        "system_prompt": str((run_dir / "system_prompt.txt").relative_to(root)),
        "prompt": str((run_dir / "prompt.txt").relative_to(root)),
        "metrics": str((run_dir / "metrics.json").relative_to(root)),
        "reward_trace": str((run_dir / "reward_trace.json").relative_to(root)),
        "state_scale_summary": str(state_scale_path.relative_to(root)),
        "td3_seed_trace": str(td3_trace_path.relative_to(root)),
        "sb3_action_trace": str(sb3_trace_path.relative_to(root)),
        "td3_g1_g3_diff": str(td3_diff_path.relative_to(root)),
        "td3_action_saturation": str(td3_sat_path.relative_to(root)),
        "policy_behavior_summary": str(policy_behavior_path.relative_to(root)),
        "scenario_profile": str(scenario_profile_path.relative_to(root)),
        "revision_candidates": str(cand_dir.relative_to(root)),
        "metrics_table": str(Path(metrics_table_path).relative_to(root)),
        "run_summary": str(Path(summary_path).relative_to(root)),
        "llm_errors": str((run_dir / "llm_errors.json").relative_to(root)),
        "run_manifest": str(run_manifest_path.relative_to(root)),
    }
    if (run_dir / "llm_iter_trace.json").exists():
        artifacts["llm_iter_trace"] = str((run_dir / "llm_iter_trace.json").relative_to(root))
    if (run_dir / "llm_responses.json").exists():
        artifacts["llm_responses"] = str((run_dir / "llm_responses.json").relative_to(root))
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

    # hashes
    hashes = {
        "raw_data": sha256_file(raw_path),
        "processed_data": sha256_file(processed_path),
        "system_prompt": sha256_file(run_dir / "system_prompt.txt"),
        "prompt": sha256_file(run_dir / "prompt.txt"),
        "metrics": sha256_file(run_dir / "metrics.json"),
        "reward_trace": sha256_file(run_dir / "reward_trace.json"),
        "state_scale_summary": sha256_file(state_scale_path),
        "td3_seed_trace": sha256_file(td3_trace_path),
        "sb3_action_trace": sha256_file(sb3_trace_path),
        "td3_g1_g3_diff": sha256_file(td3_diff_path),
        "td3_action_saturation": sha256_file(td3_sat_path),
        "policy_behavior_summary": sha256_file(policy_behavior_path),
        "scenario_profile": sha256_file(scenario_profile_path),
        "metrics_table": sha256_file(Path(metrics_table_path)),
        "run_summary": sha256_file(Path(summary_path)),
        "llm_errors": sha256_file(run_dir / "llm_errors.json"),
        "run_manifest": sha256_file(run_manifest_path),
    }
    if (run_dir / "llm_iter_trace.json").exists():
        hashes["llm_iter_trace"] = sha256_file(run_dir / "llm_iter_trace.json")
    if (run_dir / "llm_responses.json").exists():
        hashes["llm_responses"] = sha256_file(run_dir / "llm_responses.json")
    (run_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))

    return results
