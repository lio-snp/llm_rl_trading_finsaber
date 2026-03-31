from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.env.state_schema import StateSchema


@dataclass
class EnvConfig:
    initial_cash: float
    max_trade: int
    fee_rate: float
    decision_ts_rule: str = "close_t_to_open_t1"
    action_quantization_mode: str = "integer"
    discrete_action_levels: int = 3
    action_bound_penalty_coef: float = 0.0
    action_bound_penalty_threshold: float = 0.95
    action_bound_penalty_power: float = 2.0
    action_bound_penalty_reference_bound: float | None = None


class TradingEnv:
    def __init__(self, df: pd.DataFrame, assets: List[str], schema: StateSchema, cfg: EnvConfig):
        self.df = df.copy()
        self.assets = assets
        self.schema = schema
        self.cfg = cfg
        self.dates = sorted(self.df["date"].unique())
        self.day_idx = 0
        self.cash = cfg.initial_cash
        self.holdings: Dict[str, float] = {a: 0.0 for a in assets}
        self.last_value = cfg.initial_cash
        self._online_alpha = float(2.0 / (20.0 + 1.0))
        self._online_features: Dict[str, float] = {
            "ret_ema_20": 0.0,
            "ret_sq_ema_20": 0.0,
            "drawdown_20": 0.0,
            "turnover_ema_20": 0.0,
        }
        self._recent_portfolio_values: List[float] = [float(cfg.initial_cash)]
        self._prev_asset_weights = np.zeros(len(self.assets), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.day_idx = 0
        self.cash = self.cfg.initial_cash
        self.holdings = {a: 0.0 for a in self.assets}
        self.last_value = self.cfg.initial_cash
        self._online_features = {
            "ret_ema_20": 0.0,
            "ret_sq_ema_20": 0.0,
            "drawdown_20": 0.0,
            "turnover_ema_20": 0.0,
        }
        self._recent_portfolio_values = [float(self.cfg.initial_cash)]
        self._prev_asset_weights = np.zeros(len(self.assets), dtype=np.float32)
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.array(action, dtype=float).reshape(-1)
        action_requested = np.clip(action, -self.cfg.max_trade, self.cfg.max_trade)
        action = action_requested.copy()
        quant_mode = str(self.cfg.action_quantization_mode or "integer").lower()
        if quant_mode in {"integer", "integer_shares", "int"}:
            # Keep default behavior identical to prior implementation.
            action = action.astype(int)
        elif quant_mode in {"none", "continuous", "raw"}:
            action = action.astype(float)
        else:
            action = action.astype(int)
        action_executed = np.asarray(action, dtype=float).reshape(-1)
        action_bound_penalty = self._action_bound_penalty(action_executed)

        rule = str(self.cfg.decision_ts_rule or "close_t_to_open_t1").lower()
        if rule == "close_t_to_open_t1":
            # state at close_t, action filled at open_{t+1}, reward on close_{t+1}
            if self.day_idx >= len(self.dates) - 1:
                cur_day_df = self._day_df()
                asset_weights, cash_weight = self._portfolio_weights(cur_day_df, price_col="close")
                return self._get_state(), 0.0, True, {
                    "portfolio_value": self.last_value,
                    "action_requested": action_requested.tolist(),
                    "action_executed": action_executed.tolist(),
                    "action_bound_penalty": 0.0,
                    "portfolio_weights": asset_weights.tolist(),
                    "cash_weight": float(cash_weight),
                    "portfolio_weight_change": 0.0,
                }
            trade_day_df = self._day_df(self.day_idx + 1)
            prev_value = float(self.last_value)
            for asset, act in zip(self.assets, action):
                price = float(trade_day_df.loc[trade_day_df["asset"] == asset, "open"].iloc[0])
                if act > 0:
                    cost = price * act * (1.0 + self.cfg.fee_rate)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.holdings[asset] += act
                elif act < 0:
                    sell_qty = min(-act, self.holdings[asset])
                    proceeds = price * sell_qty * (1.0 - self.cfg.fee_rate)
                    self.cash += proceeds
                    self.holdings[asset] -= sell_qty

            cur_value = self._portfolio_value(trade_day_df, price_col="close")
            reward = cur_value - self.last_value
            self.last_value = cur_value
            asset_weights, cash_weight = self._portfolio_weights(trade_day_df, price_col="close")
            turnover = self._update_online_features(prev_value, cur_value, asset_weights)
            self.day_idx += 1
            done = self.day_idx >= len(self.dates) - 1
            return self._get_state(), reward, done, {
                "portfolio_value": cur_value,
                "action_requested": action_requested.tolist(),
                "action_executed": action_executed.tolist(),
                "action_bound_penalty": float(action_bound_penalty),
                "portfolio_weights": asset_weights.tolist(),
                "cash_weight": float(cash_weight),
                "portfolio_weight_change": float(turnover),
            }

        # compatibility mode: fill and reward at close_t
        day_df = self._day_df(self.day_idx)
        prev_value = float(self.last_value)
        for asset, act in zip(self.assets, action):
            price = float(day_df.loc[day_df["asset"] == asset, "close"].iloc[0])
            if act > 0:
                cost = price * act * (1.0 + self.cfg.fee_rate)
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[asset] += act
            elif act < 0:
                sell_qty = min(-act, self.holdings[asset])
                proceeds = price * sell_qty * (1.0 - self.cfg.fee_rate)
                self.cash += proceeds
                self.holdings[asset] -= sell_qty

        cur_value = self._portfolio_value(day_df, price_col="close")
        reward = cur_value - self.last_value
        self.last_value = cur_value
        asset_weights, cash_weight = self._portfolio_weights(day_df, price_col="close")
        turnover = self._update_online_features(prev_value, cur_value, asset_weights)
        self.day_idx += 1
        done = self.day_idx >= len(self.dates)
        return self._get_state(), reward, done, {
            "portfolio_value": cur_value,
            "action_requested": action_requested.tolist(),
            "action_executed": action_executed.tolist(),
            "action_bound_penalty": float(action_bound_penalty),
            "portfolio_weights": asset_weights.tolist(),
            "cash_weight": float(cash_weight),
            "portfolio_weight_change": float(turnover),
        }

    def _action_bound_penalty(self, action_executed: np.ndarray) -> float:
        coef = float(self.cfg.action_bound_penalty_coef)
        if coef <= 0:
            return 0.0
        reference_bound = (
            float(self.cfg.action_bound_penalty_reference_bound)
            if self.cfg.action_bound_penalty_reference_bound is not None
            else float(self.cfg.max_trade)
        )
        bound = max(1.0, reference_bound)
        threshold = float(np.clip(self.cfg.action_bound_penalty_threshold, 0.0, 0.999999))
        power = max(1.0, float(self.cfg.action_bound_penalty_power))
        ratio = np.abs(np.asarray(action_executed, dtype=float).reshape(-1)) / bound
        excess = np.clip((ratio - threshold) / max(1e-9, (1.0 - threshold)), 0.0, None)
        return float(coef * np.mean(np.power(excess, power)))

    def _day_df(self, day_idx: int | None = None) -> pd.DataFrame:
        if day_idx is None:
            day_idx = self.day_idx
        date = self.dates[min(day_idx, len(self.dates) - 1)]
        return self.df[self.df["date"] == date]

    def _get_state(self) -> np.ndarray:
        day_df = self._day_df()
        return self.schema.build_state(
            day_df,
            self.holdings,
            self.cash,
            online_features=self._online_features,
        )

    def _portfolio_value(self, day_df: pd.DataFrame, price_col: str = "close") -> float:
        value = self.cash
        for asset in self.assets:
            price = float(day_df.loc[day_df["asset"] == asset, price_col].iloc[0])
            value += self.holdings[asset] * price
        return value

    def _portfolio_weights(self, day_df: pd.DataFrame, price_col: str = "close") -> Tuple[np.ndarray, float]:
        total_value = max(1e-12, float(self._portfolio_value(day_df, price_col=price_col)))
        asset_weights = []
        for asset in self.assets:
            price = float(day_df.loc[day_df["asset"] == asset, price_col].iloc[0])
            asset_value = float(self.holdings[asset]) * price
            asset_weights.append(asset_value / total_value)
        cash_weight = float(self.cash / total_value)
        return np.asarray(asset_weights, dtype=np.float32), float(cash_weight)

    def _update_online_features(self, prev_value: float, cur_value: float, asset_weights: np.ndarray) -> float:
        prev_value = max(1e-12, float(prev_value))
        cur_value = float(cur_value)
        ret = float(cur_value / prev_value - 1.0)
        alpha = float(self._online_alpha)
        prev_ret_ema = float(self._online_features.get("ret_ema_20", 0.0))
        prev_ret_sq_ema = float(self._online_features.get("ret_sq_ema_20", 0.0))
        prev_turnover_ema = float(self._online_features.get("turnover_ema_20", 0.0))

        turnover = float(np.sum(np.abs(np.asarray(asset_weights, dtype=np.float32) - self._prev_asset_weights)))
        self._online_features["ret_ema_20"] = float((1.0 - alpha) * prev_ret_ema + alpha * ret)
        self._online_features["ret_sq_ema_20"] = float((1.0 - alpha) * prev_ret_sq_ema + alpha * (ret * ret))
        self._online_features["turnover_ema_20"] = float((1.0 - alpha) * prev_turnover_ema + alpha * turnover)

        self._recent_portfolio_values.append(cur_value)
        if len(self._recent_portfolio_values) > 20:
            self._recent_portfolio_values = self._recent_portfolio_values[-20:]
        rolling_peak = max(self._recent_portfolio_values) if self._recent_portfolio_values else cur_value
        drawdown = 0.0 if rolling_peak <= 0 else max(0.0, 1.0 - cur_value / rolling_peak)
        self._online_features["drawdown_20"] = float(drawdown)
        self._prev_asset_weights = np.asarray(asset_weights, dtype=np.float32).copy()
        return turnover
