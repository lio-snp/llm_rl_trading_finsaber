from __future__ import annotations

from typing import List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.trading_env import TradingEnv, EnvConfig
from src.env.state_schema import StateSchema


class TradingGymEnv(gym.Env):
    def __init__(
        self,
        df,
        assets: List[str],
        schema: StateSchema,
        cfg: EnvConfig,
        action_space_type: str = "continuous",
        policy_action_bound: float | None = None,
        revise_state=None,
        intrinsic_reward=None,
        intrinsic_w: float = 0.0,
        intrinsic_scale_mode: str = "raw",
        use_revised: bool = False,
        use_intrinsic: bool = False,
        intrinsic_timing: str = "post_action_state",
        intrinsic_input_mode: str = "revised",
        policy_state_fn=None,
    ):
        super().__init__()
        self._env = TradingEnv(df, assets, schema, cfg)
        self.assets = assets
        self.schema = schema
        self.cfg = cfg
        self.action_space_type = action_space_type
        self.policy_action_bound = float(policy_action_bound) if policy_action_bound is not None else None
        if self.policy_action_bound is not None and self.policy_action_bound <= 0:
            self.policy_action_bound = None
        self.revise_state = revise_state
        self.intrinsic_reward = intrinsic_reward
        self.intrinsic_w = intrinsic_w
        self.intrinsic_scale_mode = intrinsic_scale_mode
        self.use_revised = use_revised
        self.use_intrinsic = use_intrinsic
        self.intrinsic_timing = intrinsic_timing
        self.intrinsic_input_mode = str(intrinsic_input_mode or "revised").strip().lower()
        self.policy_state_fn = policy_state_fn
        self.intrinsic_values: List[float] = []
        self.reward_env_values: List[float] = []
        self.reward_total_values: List[float] = []
        self.action_penalty_values: List[float] = []
        self.portfolio_weight_values: List[List[float]] = []
        self.cash_weight_values: List[float] = []
        self.portfolio_weight_change_values: List[float] = []
        self._last_state: np.ndarray | None = None

        obs_dim = schema.dim()
        if policy_state_fn is not None:
            try:
                test_state = np.zeros(schema.dim(), dtype=np.float32)
                normalized = policy_state_fn(test_state)
                obs_dim = int(np.array(normalized).shape[0])
            except Exception:
                obs_dim = schema.dim()
        elif use_revised and revise_state is not None:
            try:
                test_state = np.zeros(schema.dim(), dtype=np.float32)
                revised = revise_state(test_state)
                obs_dim = int(np.array(revised).shape[0])
            except Exception:
                obs_dim = schema.dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if action_space_type == "discrete":
            self._base = max(3, int(getattr(cfg, "discrete_action_levels", 3) or 3))
            if self._base % 2 == 0:
                raise ValueError(f"discrete_action_levels must be odd, got {self._base}")
            self._discrete_action_values = np.linspace(
                -float(self.cfg.max_trade),
                float(self.cfg.max_trade),
                self._base,
                dtype=np.float32,
            )
            self._n_actions = self._base ** len(assets)
            self.action_space = spaces.Discrete(self._n_actions)
        else:
            bound = float(self.policy_action_bound) if self.policy_action_bound is not None else float(cfg.max_trade)
            self.action_space = spaces.Box(
                low=-bound,
                high=bound,
                shape=(len(assets),),
                dtype=np.float32,
            )

    def _sanitize(self, value) -> float:
        try:
            out = float(value) if value is not None else 0.0
        except Exception:
            out = 0.0
        if not np.isfinite(out):
            out = 0.0
        return out

    def _scale_intrinsic(self, value: float) -> float:
        mode = (self.intrinsic_scale_mode or "raw").lower()
        if mode == "bounded_100":
            return float(np.clip(value, -100.0, 100.0))
        if mode == "normalized":
            return float(np.tanh(value) * 100.0)
        return float(value)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            super().reset(seed=seed)
        self.intrinsic_values = []
        self.reward_env_values = []
        self.reward_total_values = []
        self.action_penalty_values = []
        self.portfolio_weight_values = []
        self.cash_weight_values = []
        self.portfolio_weight_change_values = []
        state = self._env.reset()
        self._last_state = np.array(state, dtype=np.float32)
        obs = self._obs(state)
        return obs, {}

    def _decode_action(self, action: int) -> np.ndarray:
        # Base-N encoding across assets using symmetric action levels.
        digits = []
        x = int(action)
        for _ in range(len(self.assets)):
            digits.append(x % self._base)
            x //= self._base
        acts = [float(self._discrete_action_values[int(d)]) for d in digits]
        return np.array(acts, dtype=np.float32)

    def step(self, action):
        prev_state = np.array(self._last_state, dtype=np.float32) if self._last_state is not None else None
        if self.action_space_type == "discrete":
            action_vec = self._decode_action(action)
        else:
            action_vec = np.array(action, dtype=np.float32)
        state, reward_env, done, info = self._env.step(action_vec)
        self._last_state = np.array(state, dtype=np.float32)
        action_penalty = self._sanitize(info.get("action_bound_penalty", 0.0))
        r_int = 0.0
        if self.use_intrinsic and self.intrinsic_reward is not None:
            if str(self.intrinsic_timing).lower() == "pre_action_state" and prev_state is not None:
                intrinsic_state = prev_state
            else:
                intrinsic_state = state
            if self.intrinsic_input_mode == "raw":
                revised = np.nan_to_num(np.asarray(intrinsic_state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                revised = self._revise(intrinsic_state)
            try:
                r_int = self._sanitize(self.intrinsic_reward(revised))
            except Exception:
                r_int = 0.0
            r_int = self._scale_intrinsic(r_int)
        reward = float(reward_env) + self.intrinsic_w * r_int - float(action_penalty)
        self.intrinsic_values.append(float(r_int))
        self.reward_env_values.append(float(reward_env))
        self.reward_total_values.append(float(reward))
        self.action_penalty_values.append(float(action_penalty))
        portfolio_weights = info.get("portfolio_weights")
        if portfolio_weights is not None:
            self.portfolio_weight_values.append(
                np.asarray(portfolio_weights, dtype=float).reshape(-1).tolist()
            )
        self.cash_weight_values.append(float(info.get("cash_weight", 0.0)))
        self.portfolio_weight_change_values.append(float(info.get("portfolio_weight_change", 0.0)))
        obs = self._obs(state)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def _revise(self, state: np.ndarray) -> np.ndarray:
        if self.revise_state is None:
            return np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            revised = self.revise_state(state)
            if revised is None:
                return np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            revised = np.asarray(revised, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(revised)):
                revised = np.nan_to_num(revised, nan=0.0, posinf=0.0, neginf=0.0)
            return revised
        except Exception:
            return np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _obs(self, state: np.ndarray) -> np.ndarray:
        if self.policy_state_fn is not None:
            try:
                out = np.asarray(self.policy_state_fn(state), dtype=np.float32).reshape(-1)
                return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                return np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_revised and self.revise_state is not None:
            return self._revise(state)
        return np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
