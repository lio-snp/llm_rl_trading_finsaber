from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class TD3StateNormConfig:
    mode: str = "none"
    eps: float = 1e-6
    log_volume: bool = True


def resolve_td3_state_norm_config(td3_cfg: dict | None) -> TD3StateNormConfig:
    cfg = (td3_cfg or {}).get("state_norm", {})
    mode = str(cfg.get("mode", "none")).lower()
    log_volume = bool(cfg.get("log_volume", True))
    if mode == "log_volume":
        mode = "none"
        log_volume = True
    if mode not in {"none", "zscore", "robust"}:
        mode = "none"
    eps = float(max(1e-12, cfg.get("eps", 1e-6)))
    return TD3StateNormConfig(mode=mode, eps=eps, log_volume=log_volume)


def matrix_stats(arr: np.ndarray) -> dict:
    data = np.asarray(arr, dtype=float)
    if data.size == 0:
        return {
            "shape": [0, 0],
            "global": {"min": 0.0, "max": 0.0, "p95_abs": 0.0, "p99_abs": 0.0},
            "per_dim": [],
        }
    if data.ndim == 1:
        data = data.reshape(1, -1)
    per_dim = []
    for i in range(data.shape[1]):
        col = data[:, i]
        per_dim.append(
            {
                "idx": int(i),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "p95_abs": float(np.quantile(np.abs(col), 0.95)),
                "p99_abs": float(np.quantile(np.abs(col), 0.99)),
            }
        )
    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "global": {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "p95_abs": float(np.quantile(np.abs(data), 0.95)),
            "p99_abs": float(np.quantile(np.abs(data), 0.99)),
        },
        "per_dim": per_dim,
    }


def _sanitize_vec(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _coerce_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    arr = _sanitize_vec(vec)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        return arr[:dim]
    out = np.zeros(dim, dtype=float)
    out[: arr.shape[0]] = arr
    return out


def _apply_log_volume(arr: np.ndarray, raw_dim: int, volume_indices: List[int], enabled: bool) -> np.ndarray:
    if not enabled:
        return arr
    out = np.asarray(arr, dtype=float).copy()
    max_idx = min(int(raw_dim), int(out.shape[0]))
    for idx in volume_indices:
        if 0 <= int(idx) < max_idx:
            v = float(out[idx])
            out[idx] = np.sign(v) * np.log1p(abs(v))
    return out


def _fit_center_scale(arr: np.ndarray, mode: str, eps: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "zscore":
        center = np.mean(arr, axis=0)
        scale = np.std(arr, axis=0)
        scale = np.where(scale < eps, 1.0, scale)
        return center, scale
    if mode == "robust":
        center = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        scale = q75 - q25
        scale = np.where(scale < eps, 1.0, scale)
        return center, scale
    return None, None


def build_td3_state_fn(
    base_state_fn: Callable[[np.ndarray], np.ndarray],
    reference_states: np.ndarray,
    raw_dim: int,
    volume_indices: List[int],
    norm_cfg: TD3StateNormConfig,
) -> tuple[Callable[[np.ndarray], np.ndarray], Dict]:
    raw_samples = np.asarray(reference_states, dtype=float)
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.reshape(1, -1)

    transformed_rows = []
    expected_dim: int | None = None
    for s in raw_samples:
        try:
            y = _sanitize_vec(base_state_fn(np.asarray(s, dtype=np.float32)))
        except Exception:
            continue
        if expected_dim is None:
            expected_dim = int(y.shape[0])
        if int(y.shape[0]) != int(expected_dim):
            continue
        y = _apply_log_volume(y, raw_dim, volume_indices, norm_cfg.log_volume)
        transformed_rows.append(y)

    if not transformed_rows:
        try:
            fallback = _sanitize_vec(base_state_fn(np.zeros(raw_dim, dtype=np.float32)))
        except Exception:
            fallback = np.zeros(raw_dim, dtype=float)
        fallback = _apply_log_volume(fallback, raw_dim, volume_indices, norm_cfg.log_volume)
        expected_dim = int(fallback.shape[0])
        transformed_rows = [fallback]

    fit_arr = np.stack(transformed_rows, axis=0)
    fit_dim = int(fit_arr.shape[1])
    center, scale = _fit_center_scale(fit_arr, norm_cfg.mode, norm_cfg.eps)

    def _fn(state: np.ndarray) -> np.ndarray:
        try:
            out = _sanitize_vec(base_state_fn(state))
        except Exception:
            out = _sanitize_vec(state)
        out = _coerce_dim(out, fit_dim)
        out = _apply_log_volume(out, raw_dim, volume_indices, norm_cfg.log_volume)
        if center is not None and scale is not None:
            out = (out - center) / scale
        return out.astype(np.float32)

    post_rows = []
    for s in raw_samples:
        try:
            post_rows.append(_fn(s))
        except Exception:
            post_rows.append(np.zeros(fit_dim, dtype=np.float32))
    post_arr = np.stack(post_rows, axis=0)
    summary = {
        "mode": norm_cfg.mode,
        "eps": norm_cfg.eps,
        "log_volume": norm_cfg.log_volume,
        "fitted_dim": int(fit_arr.shape[1]),
        "fit_sample_count": int(fit_arr.shape[0]),
        "pre_stats": matrix_stats(fit_arr),
        "post_stats": matrix_stats(post_arr),
    }
    if center is not None and scale is not None:
        summary["center_stats"] = {
            "mean_abs": float(np.mean(np.abs(center))),
            "p95_abs": float(np.quantile(np.abs(center), 0.95)),
        }
        summary["scale_stats"] = {
            "mean": float(np.mean(scale)),
            "p95": float(np.quantile(scale, 0.95)),
        }
    return _fn, summary
