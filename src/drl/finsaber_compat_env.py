from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
def _get_stock_trading_env_class():
    _ensure_finsaber_import_path()
    from rl_traders.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    return StockTradingEnv


@dataclass
class FinsaberCompatEnvConfig:
    initial_amount: float
    hmax: int
    buy_cost_pct: float
    sell_cost_pct: float
    reward_scaling: float
    tech_indicator_list: list[str]
    turbulence_threshold: float | None = None


def _cost_vector(value: float, stock_dim: int) -> list[float]:
    return [float(value)] * int(stock_dim)


def build_env_kwargs(df: pd.DataFrame, cfg: FinsaberCompatEnvConfig) -> dict:
    stock_dim = int(df["tic"].nunique())
    if stock_dim <= 0:
        raise ValueError("finsaber_compat env requires at least one asset.")
    return {
        "hmax": int(cfg.hmax),
        "initial_amount": float(cfg.initial_amount),
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": _cost_vector(cfg.buy_cost_pct, stock_dim),
        "sell_cost_pct": _cost_vector(cfg.sell_cost_pct, stock_dim),
        "state_space": int(1 + 2 * stock_dim + len(cfg.tech_indicator_list) * stock_dim),
        "stock_dim": stock_dim,
        "tech_indicator_list": list(cfg.tech_indicator_list),
        "action_space": stock_dim,
        "reward_scaling": float(cfg.reward_scaling),
        "turbulence_threshold": cfg.turbulence_threshold,
    }


def make_env(df: pd.DataFrame, cfg: FinsaberCompatEnvConfig):
    env_cls = _get_stock_trading_env_class()
    return env_cls(df=df, **build_env_kwargs(df, cfg))


def evaluate_online(model, env, *, deterministic: bool = True) -> dict:
    obs, _ = env.reset()
    reward_total: list[float] = []
    eval_trace: list[dict] = []
    eval_actions_policy: list[list[float]] = []
    eval_actions_executed: list[list[float]] = []
    step_idx = 0
    done = False
    prev_action_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=bool(deterministic))
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        obs, reward, terminated, truncated, _ = env.step(action_arr)
        done = bool(terminated or truncated)
        if len(env.actions_memory) > prev_action_count:
            executed = np.asarray(env.actions_memory[-1], dtype=float).reshape(-1)
            prev_action_count = len(env.actions_memory)
        else:
            executed = np.zeros_like(action_arr, dtype=float)
        eval_actions_policy.append(action_arr.tolist())
        eval_actions_executed.append(executed.tolist())
        current_value = float(env.asset_memory[-1]) if env.asset_memory else 0.0
        current_date = str(env.date_memory[-1]) if getattr(env, "date_memory", None) else ""
        reward_total.append(float(reward))
        eval_trace.append(
            {
                "step": int(step_idx),
                "date": current_date,
                "action_policy": action_arr.tolist(),
                "action_executed": executed.tolist(),
                "reward_total": float(reward),
                "portfolio_value": current_value,
                "done": bool(done),
            }
        )
        step_idx += 1
    return {
        "values": [float(v) for v in list(env.asset_memory)],
        "reward_total": reward_total,
        "eval_trace": eval_trace,
        "eval_actions_policy": eval_actions_policy,
        "eval_actions_executed": eval_actions_executed,
    }
