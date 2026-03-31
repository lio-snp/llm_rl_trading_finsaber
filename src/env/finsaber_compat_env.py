from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import repo_root


def _finsaber_repo_root() -> Path:
    parent = repo_root().parent.resolve()
    direct = parent / "FINSABER-main"
    if direct.exists():
        return direct
    for child in parent.iterdir():
        candidate = child / "FINSABER-main"
        if candidate.exists():
            return candidate.resolve()
    return parent / "FINSABER-main"


@lru_cache(maxsize=1)
def _load_stock_trading_env():
    finsaber_root = _finsaber_repo_root()
    if str(finsaber_root) not in sys.path:
        sys.path.insert(0, str(finsaber_root))
    from rl_traders.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # type: ignore

    return StockTradingEnv


@dataclass(frozen=True)
class FinsaberCompatEnvConfig:
    hmax: int = 1000
    initial_amount: float = 100000.0
    buy_cost_pct: float = 0.0049
    sell_cost_pct: float = 0.0049
    reward_scaling: float = 1e-4
    tech_indicator_list: list[str] | None = None
    turbulence_threshold: float | None = None


def infer_state_space(df: pd.DataFrame, tech_indicator_list: list[str]) -> int:
    stock_dim = int(df["tic"].nunique())
    return int(1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim)


def build_finsaber_stock_env(
    df: pd.DataFrame,
    cfg: FinsaberCompatEnvConfig,
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
):
    StockTradingEnv = _load_stock_trading_env()
    tech_indicator_list = list(cfg.tech_indicator_list or [])
    env_df = df.copy()
    if "_day_idx" not in env_df.columns:
        env_df["date"] = pd.to_datetime(env_df["date"]).dt.strftime("%Y-%m-%d")
        env_df = env_df.sort_values(["date", "tic"]).reset_index(drop=True)
        env_df["_day_idx"] = pd.factorize(env_df["date"])[0]
    stock_dim = int(env_df["tic"].nunique())
    if stock_dim <= 0:
        raise ValueError("Finsaber compat env requires at least one ticker.")
    env = StockTradingEnv(
        df=env_df.sort_values(["_day_idx", "tic"]).reset_index(drop=True).set_index("_day_idx"),
        stock_dim=stock_dim,
        hmax=int(cfg.hmax),
        initial_amount=float(cfg.initial_amount),
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[float(cfg.buy_cost_pct)] * stock_dim,
        sell_cost_pct=[float(cfg.sell_cost_pct)] * stock_dim,
        reward_scaling=float(cfg.reward_scaling),
        state_space=infer_state_space(df, tech_indicator_list),
        action_space=stock_dim,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=cfg.turbulence_threshold,
        initial=bool(initial),
        previous_state=list(previous_state or []),
    )
    return env


def portfolio_weights_from_state(state: np.ndarray, stock_dim: int) -> tuple[list[float], float]:
    arr = np.asarray(state, dtype=float).reshape(-1)
    cash = float(arr[0]) if arr.size else 0.0
    prices = arr[1 : 1 + stock_dim]
    holdings = arr[1 + stock_dim : 1 + 2 * stock_dim]
    asset_values = np.asarray(prices, dtype=float) * np.asarray(holdings, dtype=float)
    total = max(1e-12, float(cash + np.sum(asset_values)))
    weights = (asset_values / total).astype(float).tolist()
    cash_weight = float(cash / total)
    return weights, cash_weight
