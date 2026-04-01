from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _apply_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "adjusted_close" not in out.columns:
        return out

    close = pd.to_numeric(out["close"], errors="coerce")
    adjusted_close = pd.to_numeric(out["adjusted_close"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = adjusted_close / close
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    return out


def load_finsaber_prices(
    path: Path,
    assets: List[str] | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _apply_adjusted_ohlc(df)
    if assets:
        df = df[df["symbol"].isin(assets)]
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.rename(columns={"symbol": "asset"})
    df = df[["date", "asset", "open", "high", "low", "close", "volume"]]
    df["date"] = df["date"].dt.date.astype(str)
    return df.reset_index(drop=True)
