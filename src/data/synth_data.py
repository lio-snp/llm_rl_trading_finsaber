from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.utils.paths import ensure_dir


@dataclass
class SynthConfig:
    assets: List[str]
    start_date: str
    days: int
    seed: int


def generate_synth_ohlcv(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start_date, periods=cfg.days)
    rows = []
    for asset in cfg.assets:
        price = 100.0 + rng.normal(0, 1.0)
        for d in dates:
            drift = rng.normal(0.0002, 0.0005)
            shock = rng.normal(0.0, 0.01)
            close = max(0.1, price * (1.0 + drift + shock))
            open_p = max(0.1, price * (1.0 + rng.normal(0.0, 0.002)))
            high = max(open_p, close) * (1.0 + abs(rng.normal(0.0, 0.002)))
            low = min(open_p, close) * (1.0 - abs(rng.normal(0.0, 0.002)))
            volume = int(rng.integers(1e5, 1e6))
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "asset": asset,
                    "open": float(open_p),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": volume,
                }
            )
            price = close
    return pd.DataFrame(rows)


def save_raw_data(df: pd.DataFrame, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    if not out_path.exists():
        df.to_csv(out_path, index=False)
    return out_path
