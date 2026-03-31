from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MemoryBuffer:
    max_len: int = 30
    values: List[float] = field(default_factory=list)

    def update(self, v: float) -> None:
        self.values.append(v)
        if len(self.values) > self.max_len:
            self.values = self.values[-self.max_len :]

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))
