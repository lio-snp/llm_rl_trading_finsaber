from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.gym_trading_env import TradingGymEnv
from src.env.trading_env import EnvConfig
from src.env.state_schema import StateSchema


@dataclass
class SB3Config:
    total_timesteps: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    ent_coef: float = 0.0
    eval_episodes: int = 1


ALGOS = {
    "a2c": A2C,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def _make_vec_env(
    df,
    assets,
    schema,
    cfg,
    action_space_type: str,
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
    return DummyVecEnv(
        [
            lambda: TradingGymEnv(
                df,
                assets,
                schema,
                cfg,
                action_space_type=action_space_type,
                policy_action_bound=policy_action_bound,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward,
                intrinsic_w=intrinsic_w,
                intrinsic_scale_mode=intrinsic_scale_mode,
                use_revised=use_revised,
                use_intrinsic=use_intrinsic,
                intrinsic_timing=intrinsic_timing,
                intrinsic_input_mode=intrinsic_input_mode,
                policy_state_fn=policy_state_fn,
            )
        ]
    )


def _evaluate(
    model,
    df,
    assets,
    schema: StateSchema,
    cfg: EnvConfig,
    action_space_type: str,
    policy_action_bound: float | None,
    eval_episodes: int,
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
    env = TradingGymEnv(
        df,
        assets,
        schema,
        cfg,
        action_space_type=action_space_type,
        policy_action_bound=policy_action_bound,
        revise_state=revise_state,
        intrinsic_reward=intrinsic_reward,
        intrinsic_w=intrinsic_w,
        intrinsic_scale_mode=intrinsic_scale_mode,
        use_revised=use_revised,
        use_intrinsic=use_intrinsic,
        intrinsic_timing=intrinsic_timing,
        intrinsic_input_mode=intrinsic_input_mode,
        policy_state_fn=policy_state_fn,
    )
    values: List[float] = []
    values_episodes: List[List[float]] = []
    reward_env: List[float] = []
    reward_total: List[float] = []
    intrinsic: List[float] = []
    action_penalty: List[float] = []
    eval_trace: List[dict] = []
    eval_actions_executed: List[List[float]] = []
    eval_actions_policy: List[List[float]] = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        values = [env._env.last_value]
        done = False
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isinstance(action, np.ndarray):
                action_policy = np.asarray(action, dtype=float).reshape(-1).tolist()
            else:
                action_policy = [float(action)]
            action_executed = info.get("action_executed", action_policy)
            action_executed = np.asarray(action_executed, dtype=float).reshape(-1).tolist()
            reward_total.append(float(reward))
            if env.reward_env_values:
                reward_env.append(float(env.reward_env_values[-1]))
            if env.intrinsic_values:
                intrinsic.append(float(env.intrinsic_values[-1]))
            if env.action_penalty_values:
                action_penalty.append(float(env.action_penalty_values[-1]))
            eval_actions_policy.append(action_policy)
            eval_actions_executed.append(action_executed)
            eval_trace.append(
                {
                    "step": int(step_idx),
                    "action_policy": action_policy,
                    "action_executed": action_executed,
                    "reward_env": float(env.reward_env_values[-1]) if env.reward_env_values else 0.0,
                    "intrinsic": float(env.intrinsic_values[-1]) if env.intrinsic_values else 0.0,
                    "action_bound_penalty": float(env.action_penalty_values[-1]) if env.action_penalty_values else 0.0,
                    "reward_total": float(reward),
                    "portfolio_value": float(info.get("portfolio_value", values[-1])),
                    "portfolio_weights": np.asarray(info.get("portfolio_weights", []), dtype=float).reshape(-1).tolist(),
                    "cash_weight": float(info.get("cash_weight", 0.0)),
                    "portfolio_weight_change": float(info.get("portfolio_weight_change", 0.0)),
                    "done": bool(done),
                }
            )
            values.append(float(info.get("portfolio_value", values[-1])))
            step_idx += 1
        values_episodes.append(values)

    if values_episodes:
        min_len = min(len(v) for v in values_episodes)
        stacked = np.array([v[:min_len] for v in values_episodes], dtype=float)
        values = stacked.mean(axis=0).tolist()
    else:
        values = []
    return (
        values,
        values_episodes,
        reward_total,
        intrinsic,
        reward_env,
        action_penalty,
        eval_trace,
        eval_actions_policy,
        eval_actions_executed,
    )


def _prepare_td3_kwargs(kwargs: dict, action_dim: int) -> dict:
    out = dict(kwargs)
    raw_policy_kwargs = out.get("policy_kwargs")
    if raw_policy_kwargs is not None and not isinstance(raw_policy_kwargs, dict):
        raw_policy_kwargs = {}

    if "discount" in out and "gamma" not in out:
        out["gamma"] = float(out["discount"])
    out.pop("discount", None)

    if "start_timesteps" in out and "learning_starts" not in out:
        out["learning_starts"] = int(max(0, out["start_timesteps"]))
    out.pop("start_timesteps", None)

    if "policy_noise" in out and "target_policy_noise" not in out:
        out["target_policy_noise"] = float(out["policy_noise"])
    out.pop("policy_noise", None)

    if "noise_clip" in out and "target_noise_clip" not in out:
        out["target_noise_clip"] = float(out["noise_clip"])
    out.pop("noise_clip", None)

    if "policy_freq" in out and "policy_delay" not in out:
        out["policy_delay"] = int(max(1, out["policy_freq"]))
    out.pop("policy_freq", None)

    hidden_dim = out.pop("hidden_dim", None)
    if hidden_dim is not None:
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            policy_kwargs = dict(raw_policy_kwargs or {})
            if "net_arch" not in policy_kwargs:
                policy_kwargs["net_arch"] = [hidden_dim, hidden_dim]
            out["policy_kwargs"] = policy_kwargs
    elif raw_policy_kwargs is not None:
        out["policy_kwargs"] = dict(raw_policy_kwargs)

    expl_noise = out.pop("expl_noise", None)
    if expl_noise is not None and "action_noise" not in out:
        sigma = float(expl_noise)
        if sigma > 0:
            out["action_noise"] = NormalActionNoise(
                mean=np.zeros(action_dim, dtype=np.float32),
                sigma=np.full(action_dim, sigma, dtype=np.float32),
            )

    out.pop("actor_max_action", None)
    out.pop("corr_tau", None)
    out.pop("eval_freq", None)
    out.pop("eval_episodes", None)

    return out


def train_sb3(
    algo: str,
    train_df,
    eval_df,
    assets,
    schema: StateSchema,
    env_cfg: EnvConfig,
    cfg: SB3Config,
    action_space_type: str,
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
    seed: int = 0,
    algo_kwargs: dict | None = None,
) -> Dict[str, List[float]]:
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}")

    vec_env = _make_vec_env(
        train_df,
        assets,
        schema,
        env_cfg,
        action_space_type,
        policy_action_bound=policy_action_bound,
        revise_state=revise_state,
        intrinsic_reward=intrinsic_reward,
        intrinsic_w=intrinsic_w,
        intrinsic_scale_mode=intrinsic_scale_mode,
        use_revised=use_revised,
        use_intrinsic=use_intrinsic,
        intrinsic_timing=intrinsic_timing,
        intrinsic_input_mode=intrinsic_input_mode,
        policy_state_fn=policy_state_fn,
    )
    model_cls = ALGOS[algo]

    kwargs = {
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "verbose": 0,
    }
    if algo in {"a2c", "ppo", "sac"}:
        kwargs["ent_coef"] = cfg.ent_coef
    if algo in {"ppo", "sac", "td3"}:
        kwargs["batch_size"] = cfg.batch_size
    if algo_kwargs:
        for key, value in dict(algo_kwargs).items():
            if value is None:
                continue
            kwargs[key] = value
    if algo == "td3":
        kwargs = _prepare_td3_kwargs(kwargs, action_dim=len(assets))
    kwargs.setdefault("seed", int(seed))

    model = model_cls("MlpPolicy", vec_env, **kwargs)
    vec_env.seed(seed)
    model.set_random_seed(seed)
    model.learn(total_timesteps=cfg.total_timesteps)

    model.policy.set_training_mode(False)
    (
        values,
        values_episodes,
        reward_total,
        intrinsic,
        reward_env,
        action_penalty,
        eval_trace,
        eval_actions_policy,
        eval_actions_executed,
    ) = _evaluate(
        model,
        eval_df,
        assets,
        schema,
        env_cfg,
        action_space_type,
        policy_action_bound,
        cfg.eval_episodes,
        revise_state=revise_state,
        intrinsic_reward=intrinsic_reward,
        intrinsic_w=intrinsic_w,
        intrinsic_scale_mode=intrinsic_scale_mode,
        use_revised=use_revised,
        use_intrinsic=use_intrinsic,
        intrinsic_timing=intrinsic_timing,
        intrinsic_input_mode=intrinsic_input_mode,
        policy_state_fn=policy_state_fn,
    )
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
            "deterministic": True,
            "model_mode": "eval",
            "eval_episodes": int(cfg.eval_episodes),
        },
    }
