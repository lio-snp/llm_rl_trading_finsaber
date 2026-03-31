from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REGIME_ORDER = ("bull", "bear", "sideways")


def _rolling_max_drawdown(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return 0.0
    running_max = np.maximum.accumulate(arr)
    safe_max = np.maximum(running_max, 1e-12)
    drawdowns = arr / safe_max - 1.0
    return float(np.min(drawdowns))


def _raw_regime_label(ret63: float, dd63: float) -> str:
    if np.isfinite(ret63) and np.isfinite(dd63):
        if ret63 >= 0.08 and dd63 > -0.08:
            return "bull"
        if ret63 <= -0.08 or dd63 <= -0.12:
            return "bear"
    return "sideways"


def apply_regime_persistence_filter(raw_labels: Iterable[str], persistence_days: int = 5) -> list[str]:
    final_labels: list[str] = []
    prev_raw = ""
    streak = 0
    for label in raw_labels:
        if label == prev_raw:
            streak += 1
        else:
            prev_raw = label
            streak = 1

        prev_final = final_labels[-1] if final_labels else "sideways"
        if label == "sideways":
            final_labels.append("sideways")
        elif prev_final == label:
            final_labels.append(label)
        elif streak >= int(max(1, persistence_days)):
            final_labels.append(label)
        else:
            final_labels.append("sideways")
    return final_labels


def build_causal_regime_labels(
    price_df: pd.DataFrame,
    *,
    label_start: str,
    label_end: str,
    lookback_days: int = 63,
    persistence_days: int = 5,
) -> pd.DataFrame:
    required_cols = {"date", "asset", "close"}
    missing = required_cols.difference(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing columns: {sorted(missing)}")

    prices = price_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    pivot = (
        prices.pivot_table(index="date", columns="asset", values="close", aggfunc="last")
        .sort_index()
        .astype(float)
    )
    if pivot.empty:
        raise ValueError("price_df produced an empty close pivot for regime labeling.")

    asset_returns = pivot.pct_change(fill_method=None)
    investable_asset_count = pivot.notna().sum(axis=1).astype(int)
    ew_return = asset_returns.mean(axis=1, skipna=True).fillna(0.0)
    market_value = (1.0 + ew_return).cumprod()
    market_value_prev = market_value.shift(1)

    ret63 = market_value_prev / market_value_prev.shift(int(lookback_days)) - 1.0
    dd63 = market_value_prev.rolling(int(lookback_days), min_periods=int(lookback_days)).apply(
        _rolling_max_drawdown,
        raw=True,
    )

    raw_labels = [_raw_regime_label(float(r), float(d)) for r, d in zip(ret63.to_numpy(), dd63.to_numpy())]
    final_labels = apply_regime_persistence_filter(raw_labels, persistence_days=int(persistence_days))

    out = pd.DataFrame(
        {
            "date": pivot.index.strftime("%Y-%m-%d"),
            "market_return_ew": ew_return.astype(float).to_numpy(),
            "market_value_ew": market_value.astype(float).to_numpy(),
            "investable_asset_count": investable_asset_count.to_numpy(),
            "ret63": ret63.astype(float).to_numpy(),
            "dd63": dd63.astype(float).to_numpy(),
            "raw_label": raw_labels,
            "final_label": final_labels,
        }
    )
    mask = (pd.to_datetime(out["date"]) >= pd.to_datetime(label_start)) & (
        pd.to_datetime(out["date"]) <= pd.to_datetime(label_end)
    )
    return out.loc[mask].reset_index(drop=True)


def split_dates_by_regime(dates: Iterable[str], labels_df: pd.DataFrame) -> dict[str, list[str]]:
    label_map = {
        str(row["date"]): str(row["final_label"])
        for _, row in labels_df[["date", "final_label"]].iterrows()
    }
    out = {regime: [] for regime in REGIME_ORDER}
    for date in dates:
        regime = label_map.get(str(date), "sideways")
        if regime not in out:
            regime = "sideways"
        out[regime].append(str(date))
    return out


def summarize_window_regime_coverage(
    *,
    window_name: str,
    train_dates: Iterable[str],
    val_dates: Iterable[str],
    test_dates: Iterable[str],
    labels_df: pd.DataFrame,
    min_train_days: int,
    min_val_days: int,
    min_test_days: int,
) -> dict:
    train_map = split_dates_by_regime(train_dates, labels_df)
    val_map = split_dates_by_regime(val_dates, labels_df)
    test_map = split_dates_by_regime(test_dates, labels_df)

    per_regime: dict[str, dict] = {}
    for regime in REGIME_ORDER:
        train_count = int(len(train_map[regime]))
        val_count = int(len(val_map[regime]))
        test_count = int(len(test_map[regime]))
        eligible = (
            train_count >= int(min_train_days)
            and val_count >= int(min_val_days)
            and test_count >= int(min_test_days)
        )
        per_regime[regime] = {
            "train_days": train_count,
            "val_days": val_count,
            "test_days": test_count,
            "eligible": bool(eligible),
            "fallback_test_days": 0 if eligible else test_count,
        }

    total_test_days = int(sum(v["test_days"] for v in per_regime.values()))
    fallback_test_days = int(sum(v["fallback_test_days"] for v in per_regime.values()))
    return {
        "window_name": window_name,
        "min_train_days": int(min_train_days),
        "min_val_days": int(min_val_days),
        "min_test_days": int(min_test_days),
        "per_regime": per_regime,
        "total_test_days": total_test_days,
        "fallback_test_days": fallback_test_days,
        "fallback_ratio": float(fallback_test_days / total_test_days) if total_test_days > 0 else 0.0,
    }


def load_algo_seed_traces(run_dir: Path, algo: str) -> dict[str, list[dict]]:
    algo = str(algo).lower()
    if algo == "td3":
        path = run_dir / "td3_seed_trace.json"
    else:
        path = run_dir / "sb3_action_trace.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get(algo, {})
    return groups if isinstance(groups, dict) else {}


def aligned_daily_returns_from_seed_row(seed_row: dict, test_dates: list[str]) -> pd.DataFrame:
    raw_values = seed_row.get("eval_values_final")
    if raw_values is None:
        raw_values = seed_row.get("eval_values", [])
    values = np.asarray(raw_values or [], dtype=float).reshape(-1)
    if values.size < 2 or len(test_dates) < 2:
        return pd.DataFrame(columns=["date", "daily_return"])

    daily_returns = values[1:] / np.maximum(values[:-1], 1e-12) - 1.0
    realized_dates = [str(d) for d in test_dates[1:]]
    n = int(min(len(realized_dates), daily_returns.size))
    return pd.DataFrame(
        {
            "date": realized_dates[:n],
            "daily_return": daily_returns[:n].astype(float),
        }
    )


def route_seed_row_by_regime(
    *,
    shared_seed_row: dict,
    specialist_seed_rows_by_regime: dict[str, dict],
    label_by_date: dict[str, str],
    eligible_regimes: set[str],
    test_dates: list[str],
    initial_value: float = 100000.0,
) -> dict:
    shared_df = aligned_daily_returns_from_seed_row(shared_seed_row, test_dates)
    shared_map = dict(zip(shared_df["date"], shared_df["daily_return"]))
    specialist_maps: dict[str, dict[str, float]] = {}
    for regime, row in specialist_seed_rows_by_regime.items():
        specialist_df = aligned_daily_returns_from_seed_row(row, test_dates)
        specialist_maps[regime] = dict(zip(specialist_df["date"], specialist_df["daily_return"]))

    ordered_dates = list(shared_df["date"].tolist())
    selected_returns: list[float] = []
    routing_trace: list[dict] = []
    for date in ordered_dates:
        regime = str(label_by_date.get(date, "sideways"))
        specialist_map = specialist_maps.get(regime, {})
        use_specialist = regime in eligible_regimes and date in specialist_map
        daily_return = float(specialist_map[date] if use_specialist else shared_map.get(date, 0.0))
        selected_returns.append(daily_return)
        routing_trace.append(
            {
                "date": date,
                "label": regime,
                "source": ("specialist" if use_specialist else "shared"),
                "daily_return": daily_return,
            }
        )

    values = [float(initial_value)]
    for ret in selected_returns:
        values.append(float(values[-1] * (1.0 + ret)))

    return {
        "dates": ordered_dates,
        "daily_returns": selected_returns,
        "values": values,
        "routing_trace": routing_trace,
        "fallback_count": int(sum(1 for row in routing_trace if row["source"] != "specialist")),
    }
