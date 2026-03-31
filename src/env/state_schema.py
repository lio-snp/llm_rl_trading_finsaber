from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd


@dataclass
class StateSchema:
    assets: List[str]
    indicators: List[str]
    global_features: List[str]

    def dim(self) -> int:
        per_asset = 6 + len(self.indicators)  # OHLCV + holding
        return len(self.global_features) + len(self.assets) * per_asset

    def describe(self) -> List[str]:
        desc = []
        idx = 0
        for gf in self.global_features:
            desc.append(f"s[{idx}] = {gf}")
            idx += 1
        for asset in self.assets:
            for field in ["open", "high", "low", "close", "volume", "holding"]:
                desc.append(f"s[{idx}] = {asset}:{field}")
                idx += 1
            for ind in self.indicators:
                desc.append(f"s[{idx}] = {asset}:{ind}")
                idx += 1
        return desc

    def build_state(
        self,
        day_df: pd.DataFrame,
        holdings: Dict[str, float],
        cash: float,
        risk: float = 0.0,
        online_features: Dict[str, float] | None = None,
    ) -> np.ndarray:
        online = dict(online_features or {})
        state = []
        for gf in self.global_features:
            if gf == "cash":
                state.append(cash)
            elif gf == "portfolio_value":
                pv = cash
                for asset in self.assets:
                    price = float(day_df.loc[day_df["asset"] == asset, "close"].iloc[0])
                    pv += holdings.get(asset, 0.0) * price
                state.append(pv)
            elif gf == "risk":
                state.append(risk)
            elif gf in {"ret_ema_20", "ret_sq_ema_20", "drawdown_20", "turnover_ema_20"}:
                state.append(float(online.get(gf, 0.0)))
            else:
                raise ValueError(f"Unknown global feature: {gf}")

        for asset in self.assets:
            row = day_df.loc[day_df["asset"] == asset].iloc[0]
            state.extend(
                [
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    float(holdings.get(asset, 0.0)),
                ]
            )
            for ind in self.indicators:
                state.append(float(row[ind]))

        return np.array(state, dtype=np.float32)
