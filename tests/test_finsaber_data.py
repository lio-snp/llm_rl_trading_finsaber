from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.finsaber_data import load_finsaber_prices


def test_load_finsaber_prices_uses_adjusted_close_for_ohlc(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "symbol": ["A", "A"],
            "open": [110.0, 220.0],
            "high": [120.0, 240.0],
            "low": [90.0, 180.0],
            "close": [100.0, 200.0],
            "adjusted_close": [50.0, 100.0],
            "volume": [10, 20],
        }
    ).to_csv(path, index=False)

    out = load_finsaber_prices(path, ["A"], "2020-01-01", "2020-01-02")

    assert out["open"].tolist() == [55.0, 110.0]
    assert out["high"].tolist() == [60.0, 120.0]
    assert out["low"].tolist() == [45.0, 90.0]
    assert out["close"].tolist() == [50.0, 100.0]
    assert out["volume"].tolist() == [10, 20]


def test_load_finsaber_prices_keeps_raw_ohlc_when_adjusted_close_missing(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "symbol": ["A"],
            "open": [11.0],
            "high": [12.0],
            "low": [9.0],
            "close": [10.0],
            "volume": [10],
        }
    ).to_csv(path, index=False)

    out = load_finsaber_prices(path, ["A"], "2020-01-01", "2020-01-01")

    assert out["open"].tolist() == [11.0]
    assert out["high"].tolist() == [12.0]
    assert out["low"].tolist() == [9.0]
    assert out["close"].tolist() == [10.0]

