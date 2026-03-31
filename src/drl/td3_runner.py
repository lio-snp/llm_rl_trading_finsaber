from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.drl.td3 import TD3
from src.drl.replay_buffer import ReplayBuffer
from src.env.trading_env import TradingEnv
from src.llm.finagent_stub import FinAgentStub


@dataclass
class TD3Config:
    max_action: float
    actor_max_action: float | None = None
    start_timesteps: int = 100
    batch_size: int = 64
    expl_noise: float = 0.1
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    hidden_dim: int = 256
    corr_tau: float = 0.005
    eval_freq: int = 5000
    eval_episodes: int = 1


@dataclass
class RunResult:
    values: List[float]
    rewards: List[float]
    reward_total: List[float]
    states: List[np.ndarray]
    corrs: List[float]
    eval_steps: List[List[float]]
    eval_values: List[float]
    eval_values_initial: List[float]
    eval_values_final: List[float]
    intrinsic_values: List[float]
    action_penalties: List[float]
    intrinsic_ratio: List[float]
    eval_reward_env: List[float]
    eval_action_penalties: List[float]
    eval_reward_total: List[float]
    eval_intrinsic_values: List[float]
    eval_intrinsic_ratio: List[float]
    eval_actions_final: List[List[float]]
    eval_states_final: List[str]
    eval_q1_final: List[float]
    eval_trace_final: List[dict]


def _sanitize_scalar(value: float | int | np.ndarray | None) -> float:
    try:
        out = float(value) if value is not None else 0.0
    except Exception:
        out = 0.0
    if not np.isfinite(out):
        out = 0.0
    return out


def _scale_intrinsic(value: float, mode: str) -> float:
    mode = (mode or "raw").lower()
    if mode == "bounded_100":
        return float(np.clip(value, -100.0, 100.0))
    if mode == "normalized":
        return float(np.tanh(value) * 100.0)
    return float(value)


def _state_signature(state: np.ndarray) -> str:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.round(arr, 6)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _cal_lipschitz(states: List[np.ndarray], rewards: List[float], state_dim: int) -> np.ndarray:
    if len(states) < 2 or len(rewards) < 2:
        return np.zeros([state_dim], dtype=float)
    state_change = np.stack(states, axis=1)
    reward_change = np.array(rewards, dtype=float).reshape(1, -1)
    lipschitz = np.zeros([state_dim], dtype=float)
    for i in range(state_dim):
        order = np.argsort(state_change[i])
        cur_s = state_change[i].copy()[order]
        cur_r = reward_change.squeeze().copy()[order]
        diff_s = np.abs(cur_s[:-1] - cur_s[1:])
        diff_r = np.abs(cur_r[:-1] - cur_r[1:])
        lipschitz[i] = (diff_r / (diff_s + 1e-2)).max()
    return lipschitz


def _eval_policy(
    env: TradingEnv,
    policy: TD3,
    state_fn,
    eval_episodes: int,
    revise_state=None,
    intrinsic_reward=None,
    intrinsic_w: float = 0.0,
    use_intrinsic: bool = False,
    intrinsic_scale_mode: str = "raw",
    intrinsic_timing: str = "pre_action_state",
    intrinsic_input_mode: str = "revised",
) -> dict:
    avg_reward = 0.0
    all_values: List[float] = []
    reward_env_vals: List[float] = []
    reward_total_vals: List[float] = []
    intrinsic_vals: List[float] = []
    action_penalty_vals: List[float] = []
    intrinsic_ratio_vals: List[float] = []
    all_actions: List[List[float]] = []
    all_state_sigs: List[str] = []
    all_q1_vals: List[float] = []
    all_trace: List[dict] = []
    for _ in range(eval_episodes):
        eval_env = TradingEnv(env.df, env.assets, env.schema, env.cfg)
        state = eval_env.reset()
        values = [eval_env.last_value]
        actions: List[List[float]] = []
        state_sigs: List[str] = []
        q1_vals: List[float] = []
        trace_rows: List[dict] = []
        step_idx = 0
        done = False
        while not done:
            state_sig = _state_signature(state)
            policy_state = state_fn(state)
            action = policy.select_action(policy_state)
            q1_val = 0.0
            try:
                critic_device = next(policy.critic.parameters()).device
                s_tensor = torch.tensor(policy_state.reshape(1, -1), dtype=torch.float32, device=critic_device)
                a_tensor = torch.tensor(action.reshape(1, -1), dtype=torch.float32, device=critic_device)
                q1_val = float(policy.critic.Q1(s_tensor, a_tensor).detach().cpu().item())
            except Exception:
                q1_val = 0.0
            next_state, reward_env, done, info = eval_env.step(action)
            action_penalty = _sanitize_scalar(info.get("action_bound_penalty", 0.0))
            r_int = 0.0
            if use_intrinsic and intrinsic_reward is not None:
                if str(intrinsic_timing).lower() == "post_action_state":
                    intrinsic_state = next_state
                else:
                    intrinsic_state = state
                if str(intrinsic_input_mode or "revised").lower() == "raw":
                    revised_intrinsic = intrinsic_state
                else:
                    revised_intrinsic = revise_state(intrinsic_state) if revise_state is not None else intrinsic_state
                r_int = _scale_intrinsic(_sanitize_scalar(intrinsic_reward(revised_intrinsic)), intrinsic_scale_mode)
            reward_total = float(reward_env) + intrinsic_w * float(r_int) - float(action_penalty)
            ratio = abs(intrinsic_w * float(r_int)) / (abs(float(reward_env)) + abs(float(action_penalty)) + 1e-6)

            reward_env_vals.append(float(reward_env))
            action_penalty_vals.append(float(action_penalty))
            reward_total_vals.append(float(reward_total))
            intrinsic_vals.append(float(r_int))
            intrinsic_ratio_vals.append(float(ratio))
            action_policy = np.asarray(action, dtype=float).reshape(-1).tolist()
            action_executed = np.asarray(info.get("action_executed", action_policy), dtype=float).reshape(-1).tolist()
            actions.append(action_executed)
            state_sigs.append(state_sig)
            q1_vals.append(float(q1_val))
            trace_rows.append(
                {
                    "step": int(step_idx),
                    "state_signature": state_sig,
                    "action_policy": action_policy,
                    "action_executed": action_executed,
                    "q1": float(q1_val),
                    "reward_env": float(reward_env),
                    "action_bound_penalty": float(action_penalty),
                    "intrinsic": float(r_int),
                    "reward_total": float(reward_total),
                    "intrinsic_effect_ratio": float(ratio),
                    "portfolio_value": float(eval_env.last_value),
                    "portfolio_weights": np.asarray(info.get("portfolio_weights", []), dtype=float).reshape(-1).tolist(),
                    "cash_weight": float(info.get("cash_weight", 0.0)),
                    "portfolio_weight_change": float(info.get("portfolio_weight_change", 0.0)),
                    "done": bool(done),
                }
            )
            avg_reward += float(reward_total)
            values.append(float(eval_env.last_value))
            state = next_state
            step_idx += 1
        if not all_values:
            all_values = values
            all_actions = actions
            all_state_sigs = state_sigs
            all_q1_vals = q1_vals
            all_trace = trace_rows
    return {
        "avg_reward": float(avg_reward / max(eval_episodes, 1)),
        "values": all_values,
        "reward_env": reward_env_vals,
        "action_penalty": action_penalty_vals,
        "reward_total": reward_total_vals,
        "intrinsic": intrinsic_vals,
        "intrinsic_ratio": intrinsic_ratio_vals,
        "actions": all_actions,
        "state_signatures": all_state_sigs,
        "q1_values": all_q1_vals,
        "trace": all_trace,
    }


def train_td3(
    env: TradingEnv,
    state_dim: int,
    action_dim: int,
    cfg: TD3Config,
    max_steps: int,
    state_fn,
    revise_state,
    intrinsic_reward,
    intrinsic_w: float,
    use_intrinsic: bool,
    intrinsic_timing: str = "pre_action_state",
    intrinsic_input_mode: str = "revised",
    finagent: FinAgentStub | None = None,
    finagent_weight: float = 0.0,
    seed: int = 0,
    eval_env: TradingEnv | None = None,
    intrinsic_scale_mode: str = "raw",
) -> RunResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor_max_action = float(cfg.actor_max_action) if cfg.actor_max_action is not None else float(cfg.max_action)
    if actor_max_action <= 0:
        actor_max_action = float(cfg.max_action)

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=actor_max_action,
        discount=cfg.discount,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_freq=cfg.policy_freq,
        hidden_dim=cfg.hidden_dim,
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state = env.reset()
    values = [env.last_value]
    rewards = []
    reward_total = []
    states = []
    soft_corr = np.zeros([state_dim], dtype=float)
    episode_states: List[np.ndarray] = []
    episode_rewards: List[float] = []
    intrinsic_values: List[float] = []
    action_penalties: List[float] = []
    intrinsic_ratio: List[float] = []
    evaluations_steps: List[List[float]] = []

    eval_base_env = eval_env if eval_env is not None else env
    eval_out = _eval_policy(
        eval_base_env,
        policy,
        state_fn,
        cfg.eval_episodes,
        revise_state=revise_state,
        intrinsic_reward=intrinsic_reward,
        intrinsic_w=intrinsic_w,
        use_intrinsic=use_intrinsic,
        intrinsic_scale_mode=intrinsic_scale_mode,
        intrinsic_timing=intrinsic_timing,
        intrinsic_input_mode=intrinsic_input_mode,
    )
    eval_reward = float(eval_out["avg_reward"])
    eval_values_initial = list(eval_out["values"])
    eval_reward_env_initial = list(eval_out["reward_env"])
    eval_action_penalty_initial = list(eval_out["action_penalty"])
    eval_reward_total_initial = list(eval_out["reward_total"])
    eval_intrinsic_values_initial = list(eval_out["intrinsic"])
    eval_intrinsic_ratio_initial = list(eval_out["intrinsic_ratio"])
    eval_actions_initial = list(eval_out["actions"])
    eval_states_initial = list(eval_out["state_signatures"])
    eval_q1_initial = list(eval_out["q1_values"])
    eval_trace_initial = list(eval_out["trace"])
    evaluations_steps.append([float(eval_reward), 0.0])

    latest_eval_reward = float(eval_reward)
    latest_eval_values = list(eval_values_initial)
    latest_eval_reward_env = list(eval_reward_env_initial)
    latest_eval_action_penalty = list(eval_action_penalty_initial)
    latest_eval_reward_total = list(eval_reward_total_initial)
    latest_eval_intrinsic_values = list(eval_intrinsic_values_initial)
    latest_eval_intrinsic_ratio = list(eval_intrinsic_ratio_initial)
    latest_eval_actions = list(eval_actions_initial)
    latest_eval_states = list(eval_states_initial)
    latest_eval_q1 = list(eval_q1_initial)
    latest_eval_trace = list(eval_trace_initial)

    warmup_steps = int(np.clip(cfg.start_timesteps, 0, max(0, max_steps - 1)))

    for t in range(max_steps):
        revised = revise_state(state) if revise_state is not None else state
        states.append(revised.copy())
        episode_states.append(revised.copy())
        policy_state = state_fn(state)
        if t < warmup_steps:
            action = np.random.uniform(-actor_max_action, actor_max_action, size=action_dim)
        else:
            action = policy.select_action(policy_state)
            if cfg.expl_noise > 0:
                action = action + np.random.normal(0, cfg.expl_noise, size=action_dim)

        if finagent is not None and finagent_weight > 0.0:
            prices = {row.asset: float(row.close) for row in env._day_df().itertuples()}
            fa_actions = finagent.step(prices)
            fa_vec = np.array([fa_actions[a] for a in env.assets], dtype=float)
            action = action + finagent_weight * fa_vec * cfg.max_action

        action = np.clip(action, -cfg.max_action, cfg.max_action)
        next_state, reward_env, done, info = env.step(action)
        action_penalty = _sanitize_scalar(info.get("action_bound_penalty", 0.0))
        episode_rewards.append(float(reward_env))
        if use_intrinsic and intrinsic_reward is not None:
            try:
                if str(intrinsic_timing).lower() == "post_action_state":
                    intrinsic_state = next_state
                else:
                    intrinsic_state = state
                if str(intrinsic_input_mode or "revised").lower() == "raw":
                    revised_intrinsic = intrinsic_state
                else:
                    revised_intrinsic = revise_state(intrinsic_state) if revise_state is not None else intrinsic_state
                r_int = _sanitize_scalar(intrinsic_reward(revised_intrinsic))
            except Exception:
                r_int = 0.0
            r_int = _scale_intrinsic(r_int, intrinsic_scale_mode)
            reward = reward_env + intrinsic_w * r_int - action_penalty
        else:
            reward = reward_env - action_penalty
            r_int = 0.0
        action_penalties.append(float(action_penalty))
        intrinsic_values.append(float(r_int))
        intrinsic_ratio.append(
            abs(intrinsic_w * float(r_int)) / (abs(float(reward_env)) + abs(float(action_penalty)) + 1e-6)
        )
        reward_total.append(float(reward))

        replay_buffer.add(policy_state, action, state_fn(next_state), reward, done)

        if t >= warmup_steps:
            policy.train(replay_buffer, batch_size=cfg.batch_size)

        rewards.append(float(reward_env))
        values.append(float(info.get("portfolio_value", values[-1])))

        state = next_state
        if done:
            if episode_states:
                corr = _cal_lipschitz(episode_states, episode_rewards, state_dim)
                soft_corr = cfg.corr_tau * corr + (1 - cfg.corr_tau) * soft_corr
            episode_states = []
            episode_rewards = []
            state = env.reset()

        if (t + 1) % cfg.eval_freq == 0:
            eval_out = _eval_policy(
                eval_base_env,
                policy,
                state_fn,
                cfg.eval_episodes,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward,
                intrinsic_w=intrinsic_w,
                use_intrinsic=use_intrinsic,
                intrinsic_scale_mode=intrinsic_scale_mode,
                intrinsic_timing=intrinsic_timing,
            )
            latest_eval_reward = float(eval_out["avg_reward"])
            latest_eval_values = list(eval_out["values"])
            latest_eval_reward_env = list(eval_out["reward_env"])
            latest_eval_action_penalty = list(eval_out["action_penalty"])
            latest_eval_reward_total = list(eval_out["reward_total"])
            latest_eval_intrinsic_values = list(eval_out["intrinsic"])
            latest_eval_intrinsic_ratio = list(eval_out["intrinsic_ratio"])
            latest_eval_actions = list(eval_out["actions"])
            latest_eval_states = list(eval_out["state_signatures"])
            latest_eval_q1 = list(eval_out["q1_values"])
            latest_eval_trace = list(eval_out["trace"])
            evaluations_steps.append([float(latest_eval_reward), float(t + 1)])

    eval_out = _eval_policy(
        eval_base_env,
        policy,
        state_fn,
        cfg.eval_episodes,
        revise_state=revise_state,
        intrinsic_reward=intrinsic_reward,
        intrinsic_w=intrinsic_w,
        use_intrinsic=use_intrinsic,
        intrinsic_scale_mode=intrinsic_scale_mode,
        intrinsic_timing=intrinsic_timing,
    )
    latest_eval_reward = float(eval_out["avg_reward"])
    latest_eval_values = list(eval_out["values"])
    latest_eval_reward_env = list(eval_out["reward_env"])
    latest_eval_action_penalty = list(eval_out["action_penalty"])
    latest_eval_reward_total = list(eval_out["reward_total"])
    latest_eval_intrinsic_values = list(eval_out["intrinsic"])
    latest_eval_intrinsic_ratio = list(eval_out["intrinsic_ratio"])
    latest_eval_actions = list(eval_out["actions"])
    latest_eval_states = list(eval_out["state_signatures"])
    latest_eval_q1 = list(eval_out["q1_values"])
    latest_eval_trace = list(eval_out["trace"])
    evaluations_steps.append([float(latest_eval_reward), float(max_steps)])

    return RunResult(
        values=values,
        rewards=rewards,
        reward_total=reward_total,
        states=states,
        corrs=list(soft_corr),
        eval_steps=evaluations_steps,
        eval_values=list(latest_eval_values),
        eval_values_initial=list(eval_values_initial),
        eval_values_final=list(latest_eval_values),
        intrinsic_values=intrinsic_values,
        action_penalties=action_penalties,
        intrinsic_ratio=intrinsic_ratio,
        eval_reward_env=list(latest_eval_reward_env),
        eval_action_penalties=list(latest_eval_action_penalty),
        eval_reward_total=list(latest_eval_reward_total),
        eval_intrinsic_values=list(latest_eval_intrinsic_values),
        eval_intrinsic_ratio=list(latest_eval_intrinsic_ratio),
        eval_actions_final=list(latest_eval_actions),
        eval_states_final=list(latest_eval_states),
        eval_q1_final=list(latest_eval_q1),
        eval_trace_final=list(latest_eval_trace),
    )
