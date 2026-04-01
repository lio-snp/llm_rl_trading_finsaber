from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / (roll_down + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def add_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df.groupby("asset")["close"].pct_change().fillna(0.0)

    for ind in indicators:
        if ind.startswith("sma_"):
            window = int(ind.split("_")[1])
            df[ind] = df.groupby("asset")["close"].transform(lambda s: s.rolling(window).mean())
        elif ind.startswith("vol_"):
            window = int(ind.split("_")[1])
            df[ind] = df.groupby("asset")["ret_1d"].transform(lambda s: s.rolling(window).std())
        elif ind.startswith("rsi_"):
            window = int(ind.split("_")[1])
            df[ind] = df.groupby("asset")["close"].transform(lambda s: _rsi(s, window))
        else:
            raise ValueError(f"Unsupported indicator: {ind}")

    df = df.fillna(0.0)
    return df
