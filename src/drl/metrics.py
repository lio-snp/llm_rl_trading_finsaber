from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return {
            "AR": 0.0,
            "CR": 0.0,
            "AV": 0.0,
            "MDD": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
        }

    returns = pd.Series(values).pct_change().fillna(0.0)
    cr = float(values[-1] / values[0] - 1.0)
    ar = float((1.0 + cr) ** (252.0 / max(len(values), 1)) - 1.0)
    av = float(returns.std() * np.sqrt(252.0))

    cumulative = np.maximum.accumulate(values)
    drawdown = (values - cumulative) / cumulative
    mdd = float(drawdown.min())

    sharpe = 0.0
    if returns.std() > 1e-8:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252.0))

    downside = returns[returns < 0]
    sortino = 0.0
    if downside.std() > 1e-8:
        sortino = float(returns.mean() / downside.std() * np.sqrt(252.0))

    return {
        "AR": ar,
        "CR": cr,
        "AV": av,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
    }


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_resamples": int(n_resamples),
            "alpha": float(alpha),
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if arr.size == 1:
        return {
            "count": 1,
            "mean": mean,
            "std": std,
            "ci_low": mean,
            "ci_high": mean,
            "n_resamples": int(n_resamples),
            "alpha": float(alpha),
        }

    rng = np.random.default_rng(int(random_seed))
    n_resamples = int(max(100, n_resamples))
    sample_idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    sample_means = arr[sample_idx].mean(axis=1)
    q_low = float(np.quantile(sample_means, alpha / 2.0))
    q_high = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "ci_low": q_low,
        "ci_high": q_high,
        "n_resamples": n_resamples,
        "alpha": float(alpha),
    }
