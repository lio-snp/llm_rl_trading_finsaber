from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.features import add_indicators
from src.data.finsaber_data import load_finsaber_prices
from src.drl.metrics import compute_metrics
from src.drl.replay_buffer import ReplayBuffer
from src.drl.td3 import TD3
from src.env.state_schema import StateSchema
from src.env.trading_env import EnvConfig, TradingEnv
from src.pipeline.demo import _build_td3_g1_g3_diff
from src.utils.code_loader import load_functions_from_code


@dataclass
class TempTD3Config:
    max_action: float
    start_timesteps: int = 100
    batch_size: int = 64
    expl_noise: float = 0.1
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    hidden_dim: int = 256
    eval_freq: int = 5000


@dataclass
class TempRunResult:
    eval_values_final: List[float]
    eval_reward_env: List[float]
    eval_reward_total: List[float]
    eval_intrinsic_values: List[float]
    eval_intrinsic_ratio: List[float]
    eval_actions_final: List[List[float]]
    eval_states_final: List[str]
    eval_q1_final: List[float]
    eval_trace_final: List[dict]


def _state_signature(state: np.ndarray) -> str:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.round(arr, 6)
    return str(hash(arr.tobytes()))


def _sanitize_scalar(value) -> float:
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


def _to_float_list(vals) -> List[float]:
    if vals is None:
        return []
    return [float(v) for v in vals]


def _eval_policy_temp(
    env: TradingEnv,
    policy: TD3,
    state_fn,
    revise_state,
    intrinsic_reward,
    intrinsic_w: float,
    use_intrinsic: bool,
    intrinsic_scale_mode: str,
    intrinsic_timing: str,
) -> dict:
    eval_env = TradingEnv(env.df, env.assets, env.schema, env.cfg)
    state = eval_env.reset()

    values = [float(eval_env.last_value)]
    reward_env_vals: List[float] = []
    reward_total_vals: List[float] = []
    intrinsic_vals: List[float] = []
    intrinsic_ratio_vals: List[float] = []
    actions: List[List[float]] = []
    states: List[str] = []
    q1_vals: List[float] = []
    trace: List[dict] = []

    done = False
    step_idx = 0
    while not done:
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

        next_state, reward_env, done, _ = eval_env.step(action)
        r_int = 0.0
        if use_intrinsic and intrinsic_reward is not None:
            intrinsic_state = next_state if str(intrinsic_timing).lower() == "post_action_state" else state
            revised_intrinsic = revise_state(intrinsic_state) if revise_state is not None else intrinsic_state
            r_int = _scale_intrinsic(_sanitize_scalar(intrinsic_reward(revised_intrinsic)), intrinsic_scale_mode)

        reward_total = float(reward_env) + intrinsic_w * float(r_int)
        ratio = abs(intrinsic_w * float(r_int)) / (abs(float(reward_env)) + 1e-6)
        sig = _state_signature(state)

        reward_env_vals.append(float(reward_env))
        reward_total_vals.append(float(reward_total))
        intrinsic_vals.append(float(r_int))
        intrinsic_ratio_vals.append(float(ratio))
        actions.append(np.asarray(action, dtype=float).reshape(-1).tolist())
        states.append(sig)
        q1_vals.append(float(q1_val))
        values.append(float(eval_env.last_value))

        trace.append(
            {
                "step": int(step_idx),
                "state_signature": sig,
                "action": np.asarray(action, dtype=float).reshape(-1).tolist(),
                "q1": float(q1_val),
                "reward_env": float(reward_env),
                "intrinsic": float(r_int),
                "reward_total": float(reward_total),
                "intrinsic_effect_ratio": float(ratio),
                "portfolio_value": float(eval_env.last_value),
                "done": bool(done),
            }
        )

        state = next_state
        step_idx += 1

    return {
        "values": values,
        "reward_env": reward_env_vals,
        "reward_total": reward_total_vals,
        "intrinsic": intrinsic_vals,
        "intrinsic_ratio": intrinsic_ratio_vals,
        "actions": actions,
        "state_signatures": states,
        "q1_values": q1_vals,
        "trace": trace,
    }


def train_td3_temp(
    train_env: TradingEnv,
    eval_env: TradingEnv,
    state_dim: int,
    action_dim: int,
    cfg: TempTD3Config,
    max_steps: int,
    state_fn,
    revise_state,
    intrinsic_reward,
    intrinsic_w: float,
    use_intrinsic: bool,
    intrinsic_timing: str,
    intrinsic_scale_mode: str,
    seed: int,
) -> TempRunResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=cfg.max_action,
        discount=cfg.discount,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_freq=cfg.policy_freq,
        hidden_dim=cfg.hidden_dim,
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    state = train_env.reset()

    warmup_steps = int(np.clip(cfg.start_timesteps, 0, max(0, max_steps - 1)))
    for t in range(max_steps):
        policy_state = state_fn(state)
        if t < warmup_steps:
            action = np.random.uniform(-cfg.max_action, cfg.max_action, size=action_dim)
        else:
            action = policy.select_action(policy_state)
            if cfg.expl_noise > 0:
                action = action + np.random.normal(0, cfg.expl_noise, size=action_dim)

        action = np.clip(action, -cfg.max_action, cfg.max_action)
        next_state, reward_env, done, _ = train_env.step(action)

        r_int = 0.0
        if use_intrinsic and intrinsic_reward is not None:
            intrinsic_state = next_state if str(intrinsic_timing).lower() == "post_action_state" else state
            revised_intrinsic = revise_state(intrinsic_state) if revise_state is not None else intrinsic_state
            r_int = _scale_intrinsic(_sanitize_scalar(intrinsic_reward(revised_intrinsic)), intrinsic_scale_mode)
        reward = float(reward_env) + intrinsic_w * float(r_int)

        replay_buffer.add(policy_state, action, state_fn(next_state), reward, done)
        if t >= warmup_steps:
            policy.train(replay_buffer, batch_size=cfg.batch_size)

        state = next_state
        if done:
            state = train_env.reset()

    eval_out = _eval_policy_temp(
        eval_env,
        policy,
        state_fn,
        revise_state,
        intrinsic_reward,
        intrinsic_w,
        use_intrinsic,
        intrinsic_scale_mode,
        intrinsic_timing,
    )
    return TempRunResult(
        eval_values_final=list(eval_out["values"]),
        eval_reward_env=list(eval_out["reward_env"]),
        eval_reward_total=list(eval_out["reward_total"]),
        eval_intrinsic_values=list(eval_out["intrinsic"]),
        eval_intrinsic_ratio=list(eval_out["intrinsic_ratio"]),
        eval_actions_final=list(eval_out["actions"]),
        eval_states_final=list(eval_out["state_signatures"]),
        eval_q1_final=list(eval_out["q1_values"]),
        eval_trace_final=list(eval_out["trace"]),
    )


def _seed_trace_from_result(result: TempRunResult, seed: int) -> dict:
    return {
        "seed": int(seed),
        "eval_values_final": _to_float_list(result.eval_values_final),
        "eval_reward_env": _to_float_list(result.eval_reward_env),
        "eval_reward_total": _to_float_list(result.eval_reward_total),
        "eval_intrinsic_values": _to_float_list(result.eval_intrinsic_values),
        "eval_intrinsic_ratio": _to_float_list(result.eval_intrinsic_ratio),
        "eval_actions_final": [[float(x) for x in step] for step in (result.eval_actions_final or [])],
        "eval_states_final": [str(x) for x in (result.eval_states_final or [])],
        "eval_q1_final": _to_float_list(result.eval_q1_final),
        "eval_trace_final": result.eval_trace_final or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base YAML config, e.g. configs/td3_selected4_mt500_w300.yaml")
    parser.add_argument("--window-manifest", required=True, help="Window run_manifest.json path for split/assets")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--actor-max-action", type=float, default=50.0, help="Temporary TD3 actor output bound")
    parser.add_argument("--intrinsic-w", type=float, default=None, help="Override intrinsic_w")
    parser.add_argument("--steps", type=int, default=None, help="Override train steps")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / args.config).read_text())
    wm = json.loads((repo / args.window_manifest).read_text())

    selected_assets = wm["selected_assets"]
    split = wm["split"]
    start_date = split["train"]["start"]
    end_date = split["test"]["end"]

    price_path = (repo / cfg["finsaber_price_path"]).resolve()
    df = load_finsaber_prices(
        path=price_path,
        assets=selected_assets,
        start_date=start_date,
        end_date=end_date,
    )
    df = add_indicators(df, cfg["indicators"])

    train_df = df[(df["date"] >= split["train"]["start"]) & (df["date"] <= split["train"]["end"])].copy()
    test_df = df[(df["date"] >= split["test"]["start"]) & (df["date"] <= split["test"]["end"])].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("train/test dataframe is empty for chosen split")

    schema = StateSchema(selected_assets, cfg["indicators"], cfg["global_features"])
    env_cfg = EnvConfig(
        initial_cash=float(cfg["initial_cash"]),
        max_trade=int(cfg["max_trade"]),
        fee_rate=float(cfg["fee_rate"]),
        decision_ts_rule=str(cfg.get("execution", {}).get("decision_ts_rule", "close_t_to_open_t1")),
    )

    candidate_code = (repo / cfg["fixed_candidate_path"]).read_text()
    revise_state, intrinsic_reward = load_functions_from_code(candidate_code)

    intrinsic_w = float(cfg["intrinsic_w"] if args.intrinsic_w is None else args.intrinsic_w)
    steps = int(args.steps if args.steps is not None else min(int(cfg["n_full"]), int(train_df["date"].nunique())))

    td3_cfg = cfg["td3"]
    temp_cfg = TempTD3Config(
        max_action=float(args.actor_max_action),
        start_timesteps=min(int(td3_cfg["start_timesteps"]), max(0, steps - 1)),
        batch_size=int(td3_cfg["batch_size"]),
        expl_noise=float(td3_cfg["expl_noise"]),
        discount=float(td3_cfg["discount"]),
        tau=float(td3_cfg["tau"]),
        policy_noise=float(td3_cfg["policy_noise"]),
        noise_clip=float(td3_cfg["noise_clip"]),
        policy_freq=int(td3_cfg["policy_freq"]),
        hidden_dim=int(td3_cfg["hidden_dim"]),
        eval_freq=max(1, int(steps / 5)),
    )

    groups = {
        "G1_revise_only": dict(use_revised=True, use_intrinsic=False),
        "G3_revise_intrinsic": dict(use_revised=True, use_intrinsic=True),
    }
    seeds = [int(s) for s in cfg["seeds"]]

    td3_trace = {"td3": {k: [] for k in groups.keys()}}
    metrics_rows = []
    reward_trace = {"td3": {k: [] for k in groups.keys()}}
    for gname, gcfg in groups.items():
        for sd in seeds:
            train_env = TradingEnv(train_df, schema.assets, schema, env_cfg)
            eval_env = TradingEnv(test_df, schema.assets, schema, env_cfg)
            state_fn = revise_state if gcfg["use_revised"] else (lambda s: s)
            result = train_td3_temp(
                train_env=train_env,
                eval_env=eval_env,
                state_dim=int(state_fn(np.zeros(schema.dim(), dtype=np.float32)).shape[0]),
                action_dim=len(schema.assets),
                cfg=temp_cfg,
                max_steps=steps,
                state_fn=state_fn,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward,
                intrinsic_w=intrinsic_w,
                use_intrinsic=bool(gcfg["use_intrinsic"]),
                intrinsic_timing=str(cfg.get("intrinsic_timing", "pre_action_state")),
                intrinsic_scale_mode=str(cfg.get("intrinsic_scale_mode", "bounded_100")),
                seed=int(sd),
            )
            td3_trace["td3"][gname].append(_seed_trace_from_result(result, int(sd)))
            m = compute_metrics(np.array(result.eval_values_final, dtype=float))
            metrics_rows.append(
                {
                    "group": gname,
                    "seed": int(sd),
                    "Sharpe": float(m["Sharpe"]),
                    "CR": float(m["CR"]),
                    "MDD": float(m["MDD"]),
                    "AV": float(m["AV"]),
                }
            )
            reward_trace["td3"][gname].append(
                {
                    "seed": int(sd),
                    "reward_env": {
                        "mean": float(np.mean(result.eval_reward_env)) if result.eval_reward_env else 0.0,
                        "std": float(np.std(result.eval_reward_env)) if result.eval_reward_env else 0.0,
                        "count": len(result.eval_reward_env),
                    },
                    "reward_total": {
                        "mean": float(np.mean(result.eval_reward_total)) if result.eval_reward_total else 0.0,
                        "std": float(np.std(result.eval_reward_total)) if result.eval_reward_total else 0.0,
                        "count": len(result.eval_reward_total),
                    },
                    "intrinsic": {
                        "mean": float(np.mean(result.eval_intrinsic_values)) if result.eval_intrinsic_values else 0.0,
                        "std": float(np.std(result.eval_intrinsic_values)) if result.eval_intrinsic_values else 0.0,
                        "count": len(result.eval_intrinsic_values),
                    },
                    "intrinsic_effect_ratio": {
                        "mean": float(np.mean(result.eval_intrinsic_ratio)) if result.eval_intrinsic_ratio else 0.0,
                        "std": float(np.std(result.eval_intrinsic_ratio)) if result.eval_intrinsic_ratio else 0.0,
                        "count": len(result.eval_intrinsic_ratio),
                    },
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "td3_seed_trace.json").write_text(json.dumps(td3_trace, indent=2))
    td3_diff = {"td3": _build_td3_g1_g3_diff(td3_trace["td3"])}
    (out_dir / "td3_g1_g3_diff.json").write_text(json.dumps(td3_diff, indent=2))
    (out_dir / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2))

    with (out_dir / "metrics_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "seed", "Sharpe", "CR", "MDD", "AV"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {}
    for g in groups.keys():
        rows = [r for r in metrics_rows if r["group"] == g]
        summary[g] = {
            "Sharpe_mean": float(np.mean([r["Sharpe"] for r in rows])) if rows else 0.0,
            "CR_mean": float(np.mean([r["CR"] for r in rows])) if rows else 0.0,
            "MDD_mean": float(np.mean([r["MDD"] for r in rows])) if rows else 0.0,
            "AV_mean": float(np.mean([r["AV"] for r in rows])) if rows else 0.0,
        }
    payload = {
        "config": args.config,
        "window_manifest": args.window_manifest,
        "selected_assets": selected_assets,
        "split": split,
        "actor_max_action": float(args.actor_max_action),
        "env_max_trade": int(cfg["max_trade"]),
        "intrinsic_w": float(intrinsic_w),
        "steps": int(steps),
        "summary": summary,
        "td3_diff": td3_diff,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
