from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.env.state_schema import StateSchema


@dataclass
class PolicyConfig:
    max_trade: int


class HeuristicPolicy:
    def __init__(self, schema: StateSchema, cfg: PolicyConfig):
        self.schema = schema
        self.cfg = cfg

    def act(self, raw_state: np.ndarray, revised_state: Optional[np.ndarray] = None) -> np.ndarray:
        # If revised_state provides extra per-asset signals, use them.
        if revised_state is None:
            revised_state = raw_state

        n_assets = len(self.schema.assets)
        raw_dim = self.schema.dim()
        extra_dim = revised_state.shape[0] - raw_dim

        if extra_dim >= n_assets:
            # use first n_assets extra dims as signals
            signals = revised_state[raw_dim: raw_dim + n_assets]
        else:
            # fallback: compute momentum from open/close in raw state
            g = len(self.schema.global_features)
            per_asset = 6 + len(self.schema.indicators)
            signals = []
            for i in range(n_assets):
                open_idx = g + i * per_asset + 0
                close_idx = g + i * per_asset + 3
                open_p = raw_state[open_idx]
                close_p = raw_state[close_idx]
                signals.append((close_p - open_p) / (open_p + 1e-8))
            signals = np.array(signals, dtype=float)

        actions = np.zeros(n_assets, dtype=int)
        actions[signals > 0] = self.cfg.max_trade
        actions[signals < 0] = -self.cfg.max_trade
        return actions
