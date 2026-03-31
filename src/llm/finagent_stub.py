from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.llm.finmem_stub import MemoryBuffer


@dataclass
class FinAgentStubConfig:
    preference: str = "aggressive"


class FinAgentStub:
    def __init__(self, cfg: FinAgentStubConfig):
        self.cfg = cfg
        self.memory = MemoryBuffer(max_len=30)

    def step(self, prices: Dict[str, float]) -> Dict[str, int]:
        # simple heuristic: follow short-term mean return proxy
        mean_price = np.mean(list(prices.values()))
        self.memory.update(mean_price)
        trend = mean_price - self.memory.mean()
        actions = {}
        for asset, price in prices.items():
            if trend > 0:
                actions[asset] = 1
            elif trend < 0:
                actions[asset] = -1
            else:
                actions[asset] = 0
        return actions
