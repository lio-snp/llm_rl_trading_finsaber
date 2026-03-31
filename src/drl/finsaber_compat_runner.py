from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from src.drl.finsaber_compat_preprocessor import load_default_finrl_indicators, preprocess_price_frame
from src.env.finsaber_compat_env import (
    FinsaberCompatEnvConfig,
    build_finsaber_stock_env,
    portfolio_weights_from_state,
)


@dataclass
class FinsaberCompatConfig:
    total_timesteps: int
    initial_amount: float
    hmax: int = 1000
    buy_cost_pct: float = 0.0049
    sell_cost_pct: float = 0.0049
    reward_scaling: float = 1e-4
    tech_indicator_list: list[str] | None = None
    use_turbulence: bool = True
    use_vix: bool = False
    user_defined_feature: bool = False
    deterministic_eval: bool = True
    eval_episodes: int = 1


ALGOS = {
    "a2c": A2C,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
    "ou": OrnsteinUhlenbeckActionNoise,
}

MODEL_DEFAULTS: dict[str, dict] = {
    "a2c": {
        "n_steps": 100,
        "learning_rate": 1e-5,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    },
    "ppo": {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 2.5e-4,
        "ent_coef": 0.1,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    },
    "sac": {
        "learning_rate": 2e-2,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "learning_starts": 100,
        "ent_coef": 0.1,
        "tau": 0.005,
        "gamma": 0.99,
        "action_noise": "normal",
    },
    "td3": {
        "learning_rate": 3e-2,
        "buffer_size": 1_000_000,
        "tau": 0.005,
        "gamma": 0.99,
        "policy_delay": 2,
        "target_policy_noise": 0.5,
        "target_noise_clip": 0.5,
        "action_noise": "normal",
    },
}


def resolve_model_kwargs(algo: str, overrides: dict | None = None) -> dict:
    algo_key = str(algo).lower()
    if algo_key not in MODEL_DEFAULTS:
        raise ValueError(f"Unsupported finsaber_compat algo: {algo}")
    payload = dict(MODEL_DEFAULTS[algo_key])
    for key, value in dict(overrides or {}).items():
        if value is not None:
            payload[key] = value
    return payload


def _prepare_model_kwargs(algo: str, env, raw_kwargs: dict, seed: int) -> dict:
    kwargs = dict(raw_kwargs or {})
    if "action_noise" in kwargs:
        noise_key = str(kwargs.get("action_noise")).strip().lower()
        noise_cls = NOISE.get(noise_key)
        if noise_cls is not None:
            action_dim = int(env.action_space.shape[-1])
            kwargs["action_noise"] = noise_cls(
                mean=np.zeros(action_dim, dtype=np.float32),
                sigma=0.1 * np.ones(action_dim, dtype=np.float32),
            )
        else:
            kwargs.pop("action_noise", None)
    kwargs.setdefault("seed", int(seed))
    return kwargs


def _filter_processed_to_eval_dates(processed_eval: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    eval_dates = {
        str(pd.to_datetime(value).date())
        for value in pd.to_datetime(eval_df["date"]).unique().tolist()
    }
    out = processed_eval.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out[out["date"].isin(eval_dates)].copy()
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    out["_day_idx"] = pd.factorize(out["date"])[0]
    return out


def _evaluate_online(model, env, *, deterministic: bool, eval_episodes: int) -> dict:
    values: list[float] = []
    values_episodes: list[list[float]] = []
    reward_total: list[float] = []
    reward_env: list[float] = []
    intrinsic: list[float] = []
    action_penalty: list[float] = []
    eval_trace: list[dict] = []
    eval_actions_policy: list[list[float]] = []
    eval_actions_executed: list[list[float]] = []

    for _ in range(max(1, int(eval_episodes))):
        obs, _ = env.reset()
        episode_values = [float(env.asset_memory[-1])]
        prev_weights: list[float] | None = None
        done = False
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            action_policy = np.asarray(action, dtype=float).reshape(-1).tolist()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            action_exec = env.actions_memory[-1] if env.actions_memory else np.zeros(len(action_policy), dtype=float)
            action_exec = np.asarray(action_exec, dtype=float).reshape(-1).tolist()
            state = np.asarray(env.state, dtype=float).reshape(-1)
            stock_dim = int(env.stock_dim)
            weights, cash_weight = portfolio_weights_from_state(state, stock_dim)
            if prev_weights is None:
                weight_change = 0.0
            else:
                weight_change = float(
                    np.sum(np.abs(np.asarray(weights, dtype=float) - np.asarray(prev_weights, dtype=float)))
                )
            prev_weights = list(weights)
            portfolio_value = float(env.asset_memory[-1]) if env.asset_memory else 0.0
            reward_total.append(float(reward))
            reward_env.append(float(reward))
            intrinsic.append(0.0)
            action_penalty.append(0.0)
            eval_actions_policy.append(action_policy)
            eval_actions_executed.append(action_exec)
            eval_trace.append(
                {
                    "step": int(step_idx),
                    "action_policy": action_policy,
                    "action_executed": action_exec,
                    "reward_env": float(reward),
                    "intrinsic": 0.0,
                    "action_bound_penalty": 0.0,
                    "reward_total": float(reward),
                    "portfolio_value": portfolio_value,
                    "portfolio_weights": list(weights),
                    "cash_weight": float(cash_weight),
                    "portfolio_weight_change": float(weight_change),
                    "done": bool(done),
                }
            )
            episode_values.append(portfolio_value)
            step_idx += 1
        values_episodes.append(episode_values)

    if values_episodes:
        min_len = min(len(v) for v in values_episodes)
        stacked = np.array([v[:min_len] for v in values_episodes], dtype=float)
        values = stacked.mean(axis=0).tolist()
    return {
        "values": values,
        "values_episodes": values_episodes,
        "rewards": reward_total,
        "reward_total": reward_total,
        "reward_env": reward_env,
        "intrinsic": intrinsic,
        "action_penalty": action_penalty,
        "eval_trace": eval_trace,
        "eval_actions_policy": eval_actions_policy,
        "eval_actions_executed": eval_actions_executed,
        "eval_metadata": {
            "backend": "finsaber_compat",
            "deterministic": bool(deterministic),
            "eval_episodes": int(max(1, int(eval_episodes))),
        },
    }


def train_finsaber_compat(
    *,
    algo: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_history_df: pd.DataFrame | None,
    cfg: FinsaberCompatConfig,
    seed: int = 0,
    algo_kwargs: dict | None = None,
) -> dict:
    algo_key = str(algo).lower()
    if algo_key not in ALGOS:
        raise ValueError(f"Unsupported finsaber_compat algo: {algo}")

    indicators = list(cfg.tech_indicator_list or load_default_finrl_indicators())
    train_processed, train_summary = preprocess_price_frame(
        train_df,
        tech_indicator_list=indicators,
        use_turbulence=cfg.use_turbulence,
        use_vix=cfg.use_vix,
        user_defined_feature=cfg.user_defined_feature,
    )
    eval_history = eval_history_df if eval_history_df is not None else eval_df
    eval_processed_full, eval_summary = preprocess_price_frame(
        eval_history,
        tech_indicator_list=indicators,
        use_turbulence=cfg.use_turbulence,
        use_vix=cfg.use_vix,
        user_defined_feature=cfg.user_defined_feature,
    )
    eval_processed = _filter_processed_to_eval_dates(eval_processed_full, eval_df)
    train_tics = sorted(train_processed["tic"].unique().tolist())
    eval_processed = eval_processed[eval_processed["tic"].isin(train_tics)].copy()
    common_tics = sorted(eval_processed["tic"].unique().tolist())
    if train_processed.empty or eval_processed.empty:
        raise ValueError("finsaber_compat preprocessing produced empty train/eval frame.")

    env_cfg = FinsaberCompatEnvConfig(
        initial_amount=float(cfg.initial_amount),
        hmax=int(cfg.hmax),
        buy_cost_pct=float(cfg.buy_cost_pct),
        sell_cost_pct=float(cfg.sell_cost_pct),
        reward_scaling=float(cfg.reward_scaling),
        tech_indicator_list=indicators,
    )
    train_env = build_finsaber_stock_env(train_processed, env_cfg)
    vec_env, _ = train_env.get_sb_env()
    kwargs = _prepare_model_kwargs(algo_key, vec_env, resolve_model_kwargs(algo_key, overrides=algo_kwargs), seed)
    model_cls = ALGOS[algo_key]
    model = model_cls(
        "MlpPolicy",
        vec_env,
        verbose=0,
        **kwargs,
    )
    vec_env.seed(int(seed))
    model.set_random_seed(int(seed))
    model.learn(total_timesteps=int(cfg.total_timesteps))

    eval_env = build_finsaber_stock_env(eval_processed, env_cfg)
    eval_payload = _evaluate_online(
        model,
        eval_env,
        deterministic=cfg.deterministic_eval,
        eval_episodes=int(cfg.eval_episodes),
    )
    return {
        "values": eval_payload["values"],
        "values_episodes": eval_payload["values_episodes"],
        "rewards": eval_payload["reward_total"],
        "reward_total": eval_payload["reward_total"],
        "reward_env": eval_payload["reward_env"],
        "intrinsic": [0.0] * len(eval_payload["reward_total"]),
        "action_penalty": [0.0] * len(eval_payload["reward_total"]),
        "eval_trace": eval_payload["eval_trace"],
        "eval_actions_policy": eval_payload["eval_actions_policy"],
        "eval_actions_executed": eval_payload["eval_actions_executed"],
        "eval_metadata": {
            "backend": "finsaber_compat",
            "deterministic": bool(cfg.deterministic_eval),
            "seed": int(seed),
            "tech_indicator_list": indicators,
            "common_assets": list(common_tics),
            "train_preprocess": train_summary,
            "eval_preprocess": eval_summary,
            "model_kwargs": kwargs,
            "env": {
                "hmax": int(cfg.hmax),
                "buy_cost_pct": float(cfg.buy_cost_pct),
                "sell_cost_pct": float(cfg.sell_cost_pct),
                "reward_scaling": float(cfg.reward_scaling),
            },
        },
        "action_space_type": "continuous",
        "backend": "finsaber_compat",
        "processed_train_rows": int(len(train_processed)),
        "processed_eval_rows": int(len(eval_processed)),
    }
