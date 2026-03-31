from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.finsaber_data import load_finsaber_prices
from src.pipeline.demo import DemoConfig, _json_safe, run_demo
from src.pipeline.regime_specialist import build_causal_regime_labels
from src.utils.paths import ensure_dir, repo_root


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_bull_windows(bull_dates: list[str], train_bars: int, val_bars: int, test_bars: int, step_bars: int) -> list[dict]:
    windows: list[dict] = []
    total = int(train_bars + val_bars + test_bars)
    idx = 0
    while idx + total <= len(bull_dates):
        train_dates = bull_dates[idx : idx + train_bars]
        val_dates = bull_dates[idx + train_bars : idx + train_bars + val_bars]
        test_dates = bull_dates[idx + train_bars + val_bars : idx + total]
        windows.append(
            {
                "train_dates": train_dates,
                "val_dates": val_dates,
                "test_dates": test_dates,
                "train": {"start": train_dates[0], "end": train_dates[-1], "days": len(train_dates)},
                "val": {"start": val_dates[0], "end": val_dates[-1], "days": len(val_dates)},
                "test": {"start": test_dates[0], "end": test_dates[-1], "days": len(test_dates)},
            }
        )
        idx += int(step_bars)
    return windows


def _aggregate_metrics(run_dir: Path, window_names: list[str]) -> tuple[pd.DataFrame, dict]:
    rows: list[pd.DataFrame] = []
    for window_name in window_names:
        table_path = run_dir / window_name / "metrics_table.csv"
        if not table_path.exists():
            continue
        frame = pd.read_csv(table_path)
        frame.insert(0, "window_name", window_name)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(), {"window_count": 0}

    full = pd.concat(rows, ignore_index=True)
    metric_cols = ["Sharpe_mean", "CR_mean", "MDD_mean", "AV_mean", "intrinsic_mean", "intrinsic_w_effective_mean"]
    agg = (
        full.groupby(["algorithm", "group", "metrics_source"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    agg.insert(0, "window_name", "aggregate")
    out = pd.concat([full, agg], ignore_index=True)
    summary = {
        "window_count": int(len(window_names)),
        "algorithms": sorted(full["algorithm"].dropna().unique().tolist()),
        "groups": sorted(full["group"].dropna().unique().tolist()),
    }
    return out, summary


def run_from_spec(spec_path: Path, run_dir: Path) -> None:
    root = repo_root()
    spec = load_config(spec_path)
    base_cfg_path = root / str(spec["base_config"])
    base_cfg_dict = load_config(base_cfg_path)

    effective_cfg = _deep_update(base_cfg_dict, spec.get("overrides") or {})
    experiment_cfg = spec.get("experiment") or {}
    effective_cfg["task_description"] = str(
        experiment_cfg.get("task_description", effective_cfg.get("task_description", "bull_regime_long_window"))
    )
    effective_cfg["experiment"] = _deep_update(effective_cfg.get("experiment") or {}, experiment_cfg)

    demo_cfg = DemoConfig(**copy.deepcopy(effective_cfg))
    price_path = (root / str(demo_cfg.finsaber_price_path)).resolve()
    raw_prices = load_finsaber_prices(price_path, None, demo_cfg.start_date, demo_cfg.end_date or demo_cfg.start_date)

    label_cfg = spec.get("regime_labeling") or {}
    labels = build_causal_regime_labels(
        raw_prices,
        label_start=str(label_cfg.get("label_start", demo_cfg.start_date)),
        label_end=str(label_cfg.get("label_end", demo_cfg.end_date or demo_cfg.start_date)),
        lookback_days=int(label_cfg.get("lookback_days", 63)),
        persistence_days=int(label_cfg.get("persistence_days", 5)),
    )
    target_regime = str(label_cfg.get("target_regime", "bull"))
    bull_dates = sorted(labels.loc[labels["final_label"] == target_regime, "date"].astype(str).tolist())

    bull_window_cfg = spec.get("bull_windowing") or {}
    windows = _build_bull_windows(
        bull_dates,
        int(bull_window_cfg.get("train_bars", 504)),
        int(bull_window_cfg.get("val_bars", 126)),
        int(bull_window_cfg.get("test_bars", 126)),
        int(bull_window_cfg.get("step_bars", 126)),
    )
    if not windows:
        raise ValueError("No bull-only windows could be generated from the labeled history.")

    ensure_dir(run_dir)
    (run_dir / "config.yaml").write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "effective_demo_config.yaml").write_text(
        yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    labels.to_csv(run_dir / "regime_labels.csv", index=False)
    pd.DataFrame({"date": bull_dates}).to_csv(run_dir / "bull_dates.csv", index=False)
    (run_dir / "bull_windows.json").write_text(
        json.dumps(_json_safe(windows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    window_names: list[str] = []
    for idx, window in enumerate(windows):
        window_name = f"wf_window_{idx:02d}"
        window_names.append(window_name)
        window_dir = run_dir / window_name
        ensure_dir(window_dir)
        if (window_dir / "metrics_table.csv").exists() and (window_dir / "run_manifest.json").exists():
            continue
        window_cfg = copy.deepcopy(effective_cfg)
        window_cfg["walk_forward"] = {"enabled": False}
        window_cfg["eval_protocol"] = "temporal_split"
        window_cfg["data_split"] = {
            "train": {"start": window["train"]["start"], "end": window["train"]["end"]},
            "val": {"start": window["val"]["start"], "end": window["val"]["end"]},
            "test": {"start": window["test"]["start"], "end": window["test"]["end"]},
        }
        window_cfg["split_date_filters"] = {
            "train": {"include_dates": window["train_dates"]},
            "val": {"include_dates": window["val_dates"]},
            "test": {"include_dates": window["test_dates"]},
        }
        window_cfg["task_description"] = f"{effective_cfg['task_description']} [{target_regime}] {window_name}"
        run_demo(DemoConfig(**window_cfg), run_dir=window_dir, data_dir=root / "data")

    metrics_table, summary = _aggregate_metrics(run_dir, window_names)
    if not metrics_table.empty:
        metrics_table.to_csv(run_dir / "walk_forward_metrics_table.csv", index=False)
    (run_dir / "walk_forward_summary.json").write_text(
        json.dumps(
            {
                "target_regime": target_regime,
                "window_count": int(len(window_names)),
                "bull_date_count": int(len(bull_dates)),
                "windowing": _json_safe(bull_window_cfg),
                "summary": _json_safe(summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_type": "bull_regime_long_window",
                "base_config": str(spec["base_config"]),
                "target_regime": target_regime,
                "bull_date_count": int(len(bull_dates)),
                "window_count": int(len(window_names)),
                "window_names": window_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
