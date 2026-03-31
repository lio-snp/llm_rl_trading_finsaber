from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.finsaber_native import config as finrl_config
from src.finsaber_native.preprocessors import FeatureEngineer
from src.finsaber_native.env_stocktrading import StockTradingEnv
from src.finsaber_native.finrl_strategy import FinRLStrategy


@dataclass
class FinsaberNativeConfig:
    total_timesteps: int
    initial_amount: float
    hmax: int = 1000
    buy_cost_pct: float = 0.0049
    sell_cost_pct: float = 0.0049
    reward_scaling: float = 1e-4
    tech_indicator_list: list[str] | None = None
    use_vix: bool = False
    use_turbulence: bool = True
    user_defined_feature: bool = False
    deterministic_eval: bool = True
    eval_episodes: int = 1
    print_verbosity: int = 10


TIMING_MODEL_PARAMS: dict[str, dict] = {
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
    "a2c": {
        "n_steps": 100,
        "learning_rate": 1e-5,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "gamma": 0.99,
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


def load_default_finrl_indicators() -> tuple[str, ...]:
    return tuple(str(item) for item in finrl_config.INDICATORS)


def resolve_model_kwargs(algo: str, overrides: dict | None = None) -> dict:
    algo_key = str(algo).lower()
    if algo_key not in TIMING_MODEL_PARAMS:
        raise ValueError(f"Unsupported finsaber_native algo: {algo}")
    payload = dict(TIMING_MODEL_PARAMS[algo_key])
    for key, value in dict(overrides or {}).items():
        if value is not None:
            payload[key] = value
    return payload


def format_raw_data_for_fe(raw_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "open", "high", "low", "close", "volume"}
    missing = required - set(raw_data.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns for finsaber_native: {sorted(missing)}")
    df = raw_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    df = df.rename(columns={"asset": "tic"})
    df["day"] = df["date"].dt.dayofweek
    return df[["date", "tic", "open", "high", "low", "close", "volume", "day"]]


def preprocess_data(
    formatted_raw: pd.DataFrame,
    *,
    tech_indicator_list: list[str],
    use_vix: bool,
    use_turbulence: bool,
    user_defined_feature: bool,
) -> tuple[pd.DataFrame, dict]:
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicator_list,
        use_vix=bool(use_vix),
        use_turbulence=bool(use_turbulence),
        user_defined_feature=bool(user_defined_feature),
    )
    processed = fe.preprocess_data(formatted_raw)
    processed = processed.sort_values(["date", "tic"], ignore_index=True)
    processed.index = processed["date"].factorize()[0]
    summary = {
        "rows": int(len(processed)),
        "asset_count": int(processed["tic"].nunique()) if not processed.empty else 0,
        "date_count": int(processed["date"].nunique()) if not processed.empty else 0,
        "start": str(pd.to_datetime(processed["date"]).min().date()) if not processed.empty else "",
        "end": str(pd.to_datetime(processed["date"]).max().date()) if not processed.empty else "",
        "tech_indicator_list": list(tech_indicator_list),
        "use_vix": bool(use_vix),
        "use_turbulence": bool(use_turbulence),
        "user_defined_feature": bool(user_defined_feature),
    }
    return processed, summary


def _filter_processed_to_eval_dates(processed_eval: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    eval_dates = {
        str(pd.to_datetime(value).date())
        for value in pd.to_datetime(eval_df["date"]).unique().tolist()
    }
    out = processed_eval.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out[out["date"].isin(eval_dates)].copy()
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    out.index = out["date"].factorize()[0]
    return out


def _extract_actions_memory(df_actions: pd.DataFrame) -> list[list[float]]:
    if df_actions is None or df_actions.empty:
        return []
    if "date" in df_actions.columns and "actions" in df_actions.columns:
        rows = []
        for _, row in df_actions.iterrows():
            arr = np.asarray(row["actions"], dtype=float).reshape(-1).tolist()
            rows.append(arr)
        return rows
    action_cols = [col for col in df_actions.columns if str(col).lower() != "date"]
    if not action_cols:
        action_cols = list(df_actions.columns)
    return df_actions[action_cols].astype(float).values.tolist()


def _extract_policy_actions(environment: StockTradingEnv, snapshot: dict | None = None) -> list[list[float]]:
    snapshot = snapshot or {}
    rows = snapshot.get("actions_policy_memory", getattr(environment, "actions_policy_memory", []))
    out: list[list[float]] = []
    for row in list(rows or []):
        out.append(np.asarray(row, dtype=float).reshape(-1).tolist())
    return out


def _extract_eval_trace(
    environment: StockTradingEnv,
    policy_actions: list[list[float]],
    executed_actions: list[list[float]],
    reward_scaling: float,
    snapshot: dict | None = None,
) -> list[dict]:
    trace: list[dict] = []
    snapshot = snapshot or {}
    rewards_raw = list(snapshot.get("rewards_memory", getattr(environment, "rewards_memory", [])))
    values = list(snapshot.get("asset_memory", getattr(environment, "asset_memory", [])))
    state_memory = list(snapshot.get("state_memory", getattr(environment, "state_memory", [])))
    reward_env_values = list(snapshot.get("reward_env_values", getattr(environment, "reward_env_values", [])))
    intrinsic_values = list(snapshot.get("intrinsic_values", getattr(environment, "intrinsic_values", [])))
    reward_total_values = list(snapshot.get("reward_total_values", getattr(environment, "reward_total_values", [])))
    action_penalty_values = list(snapshot.get("action_penalty_values", getattr(environment, "action_penalty_values", [])))
    portfolio_weight_values = list(snapshot.get("portfolio_weight_values", getattr(environment, "portfolio_weight_values", [])))
    cash_weight_values = list(snapshot.get("cash_weight_values", getattr(environment, "cash_weight_values", [])))
    portfolio_weight_change_values = list(
        snapshot.get("portfolio_weight_change_values", getattr(environment, "portfolio_weight_change_values", []))
    )
    for idx, action_row in enumerate(executed_actions):
        action_policy_row = policy_actions[idx] if idx < len(policy_actions) else list(action_row)
        portfolio_value = float(values[idx + 1]) if idx + 1 < len(values) else float(values[-1])
        reward_raw = float(rewards_raw[idx]) if idx < len(rewards_raw) else 0.0
        reward_scaled = (
            float(reward_env_values[idx])
            if idx < len(reward_env_values)
            else float(reward_raw * reward_scaling)
        )
        intrinsic_val = float(intrinsic_values[idx]) if idx < len(intrinsic_values) else 0.0
        reward_total = float(reward_total_values[idx]) if idx < len(reward_total_values) else reward_scaled
        action_penalty = float(action_penalty_values[idx]) if idx < len(action_penalty_values) else 0.0
        state_row = np.asarray(state_memory[idx], dtype=float).reshape(-1) if idx < len(state_memory) else np.asarray([], dtype=float)
        trace.append(
            {
                "step": int(idx),
                "action_policy": list(action_policy_row),
                "action_executed": list(action_row),
                "reward_env": reward_scaled,
                "intrinsic": intrinsic_val,
                "action_bound_penalty": action_penalty,
                "reward_total": reward_total,
                "portfolio_value": portfolio_value,
                "portfolio_weights": list(portfolio_weight_values[idx]) if idx < len(portfolio_weight_values) else [],
                "cash_weight": float(cash_weight_values[idx]) if idx < len(cash_weight_values) else 0.0,
                "portfolio_weight_change": (
                    float(portfolio_weight_change_values[idx])
                    if idx < len(portfolio_weight_change_values)
                    else 0.0
                ),
                "state_dim": int(state_row.size),
            }
        )
    return trace


def train_finsaber_native(
    *,
    algo: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_history_df: pd.DataFrame | None,
    cfg: FinsaberNativeConfig,
    seed: int = 0,
    algo_kwargs: dict | None = None,
    revise_state=None,
    intrinsic_reward=None,
    policy_state_fn=None,
    use_revised: bool = False,
    use_intrinsic: bool = False,
    intrinsic_w: float = 0.0,
    intrinsic_scale_mode: str = "raw",
    intrinsic_timing: str = "pre_action_state",
    intrinsic_input_mode: str = "raw",
) -> dict:
    algo_key = str(algo).lower()
    indicators = list(cfg.tech_indicator_list or load_default_finrl_indicators())

    formatted_train = format_raw_data_for_fe(train_df)
    train_data, train_summary = preprocess_data(
        formatted_train,
        tech_indicator_list=indicators,
        use_vix=cfg.use_vix,
        use_turbulence=cfg.use_turbulence,
        user_defined_feature=cfg.user_defined_feature,
    )

    train_tics = sorted(train_data["tic"].unique().tolist())
    eval_history_source = eval_history_df if eval_history_df is not None else train_df
    formatted_eval_history = format_raw_data_for_fe(eval_history_source)
    formatted_eval_history = formatted_eval_history[formatted_eval_history["tic"].isin(train_tics)].copy()
    formatted_eval = format_raw_data_for_fe(eval_df)
    formatted_eval = formatted_eval[formatted_eval["tic"].isin(train_tics)].copy()

    eval_tics = sorted(formatted_eval["tic"].unique().tolist())
    missing_eval_tics = sorted(set(train_tics) - set(eval_tics))
    if missing_eval_tics:
        raise ValueError(f"Missing eval tickers after preprocessing: {missing_eval_tics}")

    env_kwargs = {
        "hmax": int(cfg.hmax),
        "initial_amount": float(cfg.initial_amount),
        "buy_cost_pct": [float(cfg.buy_cost_pct)] * len(train_tics),
        "sell_cost_pct": [float(cfg.sell_cost_pct)] * len(train_tics),
        "reward_scaling": float(cfg.reward_scaling),
        "print_verbosity": int(cfg.print_verbosity),
        "revise_state": revise_state,
        "intrinsic_reward": intrinsic_reward,
        "policy_state_fn": policy_state_fn,
        "use_revised_for_policy": bool(use_revised),
        "use_intrinsic": bool(use_intrinsic),
        "intrinsic_w": float(intrinsic_w),
        "intrinsic_scale_mode": str(intrinsic_scale_mode),
        "intrinsic_timing": str(intrinsic_timing),
        "intrinsic_input_mode": str(intrinsic_input_mode),
    }

    model_kwargs = resolve_model_kwargs(algo_key, overrides=algo_kwargs)
    strategy = FinRLStrategy.create_standalone(
        train_data=train_data.copy(),
        algorithm=algo_key,
        total_timesteps=int(cfg.total_timesteps),
        initial_amount=float(cfg.initial_amount),
    )
    strategy.formatted_raw = formatted_train.copy()
    trained_model, default_env_kwargs = strategy.train_drl_model(
        algorithm=algo_key,
        total_timesteps=int(cfg.total_timesteps),
        env_kwargs=env_kwargs,
        seed=int(seed),
        model_kwargs_override=model_kwargs,
        test_frames=[formatted_eval],
        history_frame=formatted_eval_history,
        deterministic=bool(cfg.deterministic_eval),
    )

    eval_data = strategy.test_data[0].copy() if strategy.test_data else pd.DataFrame()
    eval_summary = {
        "rows": int(len(eval_data)),
        "asset_count": int(eval_data["tic"].nunique()) if not eval_data.empty else 0,
        "date_count": int(eval_data["date"].nunique()) if not eval_data.empty else 0,
        "start": str(pd.to_datetime(eval_data["date"]).min().date()) if not eval_data.empty else "",
        "end": str(pd.to_datetime(eval_data["date"]).max().date()) if not eval_data.empty else "",
        "tech_indicator_list": list(indicators),
        "use_vix": bool(cfg.use_vix),
        "use_turbulence": bool(cfg.use_turbulence),
        "user_defined_feature": bool(cfg.user_defined_feature),
    }

    values_episodes: list[list[float]] = []
    reward_total: list[float] = []
    reward_env: list[float] = []
    eval_trace: list[dict] = []
    eval_actions_executed: list[list[float]] = []
    for df_account_value, df_actions, processed_eval, trade_env in zip(
        strategy.df_account_value,
        strategy.df_actions,
        strategy.test_data,
        strategy.trade_envs,
    ):
        trace_snapshot = dict(getattr(trade_env, "_lesr_trace_snapshot", {}) or {})
        episode_values = df_account_value["account_value"].astype(float).tolist()
        episode_actions = _extract_actions_memory(df_actions)
        episode_policy_actions = _extract_policy_actions(trade_env, trace_snapshot)
        episode_reward_env = [float(x) for x in list(getattr(trade_env, "reward_env_values", []))]
        if not episode_reward_env and trace_snapshot:
            episode_reward_env = [float(x) for x in list(trace_snapshot.get("reward_env_values", []))]
        episode_reward_total = [float(x) for x in list(getattr(trade_env, "reward_total_values", []))]
        if not episode_reward_total and trace_snapshot:
            episode_reward_total = [float(x) for x in list(trace_snapshot.get("reward_total_values", []))]
        if not episode_reward_env:
            episode_reward_env = [float(x) * float(cfg.reward_scaling) for x in trade_env.rewards_memory]
        if not episode_reward_total:
            episode_reward_total = list(episode_reward_env)
        values_episodes.append(episode_values)
        reward_total.extend(episode_reward_total)
        reward_env.extend(episode_reward_env)
        eval_actions_executed.extend(episode_actions)
        eval_trace.extend(
            _extract_eval_trace(
                trade_env,
                episode_policy_actions,
                episode_actions,
                float(cfg.reward_scaling),
                snapshot=trace_snapshot,
            )
        )

    if values_episodes:
        min_len = min(len(v) for v in values_episodes)
        values = np.array([v[:min_len] for v in values_episodes], dtype=float).mean(axis=0).tolist()
    else:
        values = []

    return {
        "values": values,
        "values_episodes": values_episodes,
        "rewards": reward_total,
        "reward_total": reward_total,
        "reward_env": reward_env,
        "intrinsic": [
            float(x)
            for trade_env in strategy.trade_envs
            for x in list(
                getattr(trade_env, "intrinsic_values", [])
                or list((getattr(trade_env, "_lesr_trace_snapshot", {}) or {}).get("intrinsic_values", []))
            )
        ],
        "action_penalty": [
            float(x)
            for trade_env in strategy.trade_envs
            for x in list(
                getattr(trade_env, "action_penalty_values", [])
                or list((getattr(trade_env, "_lesr_trace_snapshot", {}) or {}).get("action_penalty_values", []))
            )
        ],
        "eval_trace": eval_trace,
        "eval_actions_policy": [
            list(row.get("action_policy", []))
            for row in eval_trace
            if isinstance(row, dict)
        ],
        "eval_actions_executed": eval_actions_executed,
        "eval_metadata": {
            "backend": "finsaber_native",
            "deterministic": bool(cfg.deterministic_eval),
            "seed": int(seed),
            "tech_indicator_list": indicators,
            "train_preprocess": train_summary,
            "eval_preprocess": eval_summary,
            "default_env_kwargs": default_env_kwargs,
            "model_kwargs": model_kwargs,
            "lesr_effective": {
                "use_revised": bool(use_revised),
                "use_intrinsic": bool(use_intrinsic),
                "intrinsic_w": float(intrinsic_w),
                "intrinsic_scale_mode": str(intrinsic_scale_mode),
                "intrinsic_timing": str(intrinsic_timing),
                "intrinsic_input_mode": str(intrinsic_input_mode),
                "policy_state_fn": "custom" if policy_state_fn is not None else "identity",
                "revise_state": "custom" if revise_state is not None else "identity",
                "intrinsic_reward": "custom" if intrinsic_reward is not None else "zero",
            },
            "class_names": {
                "FeatureEngineer": "FeatureEngineer",
                "StockTradingEnv": "StockTradingEnv",
                "DRLAgent": "DRLAgent",
                "FinRLStrategy": "FinRLStrategy",
            },
            "strategy_variables": {
                "model_params": list(strategy.model_params.keys()),
                "formatted_raw_rows": int(len(strategy.formatted_raw)),
                "train_data_rows": int(len(strategy.train_data)),
                "test_data_count": int(len(strategy.test_data)),
            },
        },
        "action_space_type": "continuous",
        "backend": "finsaber_native",
        "processed_train_rows": int(len(train_data)),
        "processed_eval_rows": int(len(eval_data)),
        "state_space": int(default_env_kwargs["state_space"]),
        "stock_dim": int(default_env_kwargs["stock_dim"]),
    }
