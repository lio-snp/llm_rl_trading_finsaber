from __future__ import annotations

from typing import List, Tuple

from src.env.state_schema import StateSchema


def _idx_map(schema: StateSchema):
    g = len(schema.global_features)
    per_asset = 6 + len(schema.indicators)
    field_offset = {
        "open": 0,
        "high": 1,
        "low": 2,
        "close": 3,
        "volume": 4,
        "holding": 5,
    }
    return g, per_asset, field_offset


def generate_candidate_codes(schema: StateSchema) -> List[Tuple[str, str]]:
    g, per_asset, field_offset = _idx_map(schema)

    close_idxs = [g + i * per_asset + field_offset["close"] for i in range(len(schema.assets))]
    open_idxs = [g + i * per_asset + field_offset["open"] for i in range(len(schema.assets))]
    high_idxs = [g + i * per_asset + field_offset["high"] for i in range(len(schema.assets))]
    low_idxs = [g + i * per_asset + field_offset["low"] for i in range(len(schema.assets))]
    holding_idxs = [g + i * per_asset + field_offset["holding"] for i in range(len(schema.assets))]

    # Candidate 0: identity + zero intrinsic reward
    code0 = """
import numpy as np

def revise_state(s):
    return np.array(s, dtype=float)

def intrinsic_reward(updated_s):
    return 0.0
"""

    # Candidate 1: momentum + exposure
    code1 = f"""
import numpy as np

CLOSE_IDXS = {close_idxs}
OPEN_IDXS = {open_idxs}
HOLDING_IDXS = {holding_idxs}


def revise_state(s):
    s = np.array(s, dtype=float)
    momentum = (s[CLOSE_IDXS] - s[OPEN_IDXS]) / (s[OPEN_IDXS] + 1e-8)
    exposure = np.sum(np.abs(s[HOLDING_IDXS]))
    updated_s = np.concatenate([s, momentum, [exposure]])
    return updated_s


def intrinsic_reward(updated_s):
    # encourage positive momentum, penalize excessive exposure
    extra_start = updated_s.shape[0] - ({len(close_idxs)} + 1)
    momentum = updated_s[extra_start: extra_start + {len(close_idxs)}]
    exposure = updated_s[-1]
    return float(np.mean(momentum) - 0.01 * exposure)
"""

    # Candidate 2: volatility proxy + concentration
    code2 = f"""
import numpy as np

HIGH_IDXS = {high_idxs}
LOW_IDXS = {low_idxs}
CLOSE_IDXS = {close_idxs}
HOLDING_IDXS = {holding_idxs}


def revise_state(s):
    s = np.array(s, dtype=float)
    spread = (s[HIGH_IDXS] - s[LOW_IDXS]) / (s[CLOSE_IDXS] + 1e-8)
    concentration = np.max(np.abs(s[HOLDING_IDXS]))
    updated_s = np.concatenate([s, spread, [concentration]])
    return updated_s


def intrinsic_reward(updated_s):
    extra_start = updated_s.shape[0] - ({len(close_idxs)} + 1)
    spread = updated_s[extra_start: extra_start + {len(close_idxs)}]
    concentration = updated_s[-1]
    return float(-np.mean(spread) - 0.01 * concentration)
"""

    return [
        ("identity", code0),
        ("momentum_exposure", code1),
        ("volatility_concentration", code2),
    ]
