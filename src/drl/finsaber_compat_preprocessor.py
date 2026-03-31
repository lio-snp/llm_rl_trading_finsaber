from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.paths import repo_root


@lru_cache(maxsize=1)
def _finsaber_root() -> Path:
    parent = repo_root().parent.resolve()
    direct = parent / "FINSABER-main"
    if direct.exists():
        return direct
    for child in parent.iterdir():
        candidate = child / "FINSABER-main"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"FINSABER-main not found under {parent}")


@lru_cache(maxsize=1)
def _ensure_finsaber_import_path() -> str:
    root_path = _finsaber_root()
    for path in [root_path, root_path / "rl_traders"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return str(root_path)


@lru_cache(maxsize=1)
def _get_feature_engineer_class():
    _ensure_finsaber_import_path()
    try:
        from rl_traders.finrl.meta.preprocessor.preprocessors import FeatureEngineer

        return FeatureEngineer
    except ModuleNotFoundError as exc:
        if exc.name != "stockstats":
            raise
        return None


@lru_cache(maxsize=1)
def load_default_finrl_indicators() -> tuple[str, ...]:
    _ensure_finsaber_import_path()
    from rl_traders.finrl.config import INDICATORS

    return tuple(str(item) for item in INDICATORS)


def format_price_frame_for_finrl(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "open", "high", "low", "close", "volume"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns for finsaber_compat: {sorted(missing)}")
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    df = df.rename(columns={"asset": "tic"})
    df["day"] = df["date"].dt.dayofweek
    return df[["date", "tic", "open", "high", "low", "close", "volume", "day"]]


def _causal_fill_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["tic", "date"]).copy()
    non_fill_cols = {"date", "tic"}
    fill_cols = [col for col in out.columns if col not in non_fill_cols]
    out[fill_cols] = out.groupby("tic", group_keys=False)[fill_cols].ffill()
    out[fill_cols] = out[fill_cols].fillna(0.0)
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    out.index = pd.factorize(out["date"])[0]
    return out


def _clean_data_fallback(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    return df[df.tic.isin(tics)].copy()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_indicator(group: pd.DataFrame, indicator: str) -> pd.Series:
    close = group["close"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    if indicator == "macd":
        return _ema(close, 12) - _ema(close, 26)
    if indicator == "boll_ub":
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        return ma + 2.0 * std
    if indicator == "boll_lb":
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        return ma - 2.0 * std
    if indicator.startswith("rsi_"):
        window = int(indicator.split("_")[1])
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta.clip(upper=0.0)).abs()
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))
    if indicator.startswith("cci_"):
        window = int(indicator.split("_")[1])
        typical = (high + low + close) / 3.0
        sma = typical.rolling(window).mean()
        mad = typical.rolling(window).apply(lambda values: np.mean(np.abs(values - np.mean(values))), raw=True)
        return (typical - sma) / (0.015 * mad + 1e-8)
    if indicator.startswith("dx_"):
        window = int(indicator.split("_")[1])
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=group.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=group.index,
        )
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        plus_di = 100.0 * plus_dm.rolling(window).mean() / (atr + 1e-8)
        minus_di = 100.0 * minus_dm.rolling(window).mean() / (atr + 1e-8)
        return 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    if indicator.startswith("close_") and indicator.endswith("_sma"):
        window = int(indicator.split("_")[1])
        return close.rolling(window).mean()
    raise ValueError(f"Unsupported finsaber_compat indicator fallback: {indicator}")


def _add_technical_indicator_fallback(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    out = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    for indicator in indicators:
        series = out.groupby("tic", group_keys=False).apply(lambda group: _compute_indicator(group, indicator))
        out[indicator] = series.reset_index(level=0, drop=True).sort_index()
    return out.sort_values(["date", "tic"]).reset_index(drop=True)


def _calculate_turbulence_fallback(df: pd.DataFrame) -> pd.DataFrame:
    price_pivot = df.pivot(index="date", columns="tic", values="close").pct_change()
    unique_dates = list(price_pivot.index)
    if len(unique_dates) <= 252:
        return pd.DataFrame({"date": unique_dates, "turbulence": [0.0] * len(unique_dates)})
    turbulence_index = [0.0] * 252
    count = 0
    for idx in range(252, len(unique_dates)):
        current_price = price_pivot.loc[[unique_dates[idx]]]
        hist_price = price_pivot.loc[
            (price_pivot.index < unique_dates[idx]) & (price_pivot.index >= unique_dates[idx - 252])
        ]
        filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min() :].dropna(axis=1)
        if filtered_hist_price.empty:
            turbulence_index.append(0.0)
            continue
        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[column for column in filtered_hist_price.columns]] - np.mean(
            filtered_hist_price, axis=0
        )
        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            turbulence_temp = float(temp[0][0]) if count > 2 else 0.0
        else:
            turbulence_temp = 0.0
        turbulence_index.append(turbulence_temp)
    return pd.DataFrame({"date": unique_dates, "turbulence": turbulence_index})


def preprocess_price_frame(
    raw_df: pd.DataFrame,
    *,
    tech_indicator_list: Iterable[str] | None = None,
    use_turbulence: bool = True,
    use_vix: bool = False,
    user_defined_feature: bool = False,
) -> tuple[pd.DataFrame, dict]:
    indicators = list(tech_indicator_list or load_default_finrl_indicators())
    formatted = format_price_frame_for_finrl(raw_df)
    fe_cls = _get_feature_engineer_class()
    turbulence_applied = False
    processed = _clean_data_fallback(formatted)
    if fe_cls is None:
        if indicators:
            processed = _add_technical_indicator_fallback(processed, indicators)
        if bool(use_turbulence):
            try:
                processed = processed.merge(_calculate_turbulence_fallback(processed), on="date", how="left")
                turbulence_applied = True
            except Exception:
                turbulence_applied = False
    else:
        fe = fe_cls(
            use_technical_indicator=True,
            tech_indicator_list=indicators,
            use_vix=bool(use_vix),
            use_turbulence=bool(use_turbulence),
            user_defined_feature=bool(user_defined_feature),
        )
        if indicators:
            processed = fe.add_technical_indicator(processed)
        if bool(use_vix):
            try:
                processed = fe.add_vix(processed)
            except Exception:
                pass
        if bool(use_turbulence):
            try:
                processed = fe.add_turbulence(processed)
                turbulence_applied = True
            except Exception:
                turbulence_applied = False
        if bool(user_defined_feature):
            try:
                processed = fe.add_user_defined_feature(processed)
            except Exception:
                pass

    processed = _causal_fill_by_ticker(processed)
    summary = {
        "rows": int(len(processed)),
        "asset_count": int(processed["tic"].nunique()) if not processed.empty else 0,
        "date_count": int(processed["date"].nunique()) if not processed.empty else 0,
        "start": str(processed["date"].min().date()) if not processed.empty else "",
        "end": str(processed["date"].max().date()) if not processed.empty else "",
        "tech_indicator_list": indicators,
        "use_vix": bool(use_vix),
        "use_turbulence_requested": bool(use_turbulence),
        "use_turbulence_applied": bool(turbulence_applied),
        "causal_fill": "ffill_then_zero",
    }
    return processed, summary


def align_processed_frames(*frames: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return [frame.copy() for frame in frames], []
    common_tics = sorted(set.intersection(*[set(frame["tic"].unique().tolist()) for frame in non_empty]))
    if not common_tics:
        raise ValueError("No overlapping assets remain after compat preprocessing.")
    aligned: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty:
            aligned.append(frame.copy() if frame is not None else pd.DataFrame())
            continue
        out = frame[frame["tic"].isin(common_tics)].copy()
        out = out.sort_values(["date", "tic"]).reset_index(drop=True)
        out.index = pd.factorize(out["date"])[0]
        aligned.append(out)
    return aligned, common_tics
