from __future__ import annotations

import pandas as pd

from src.drl.finsaber_compat_preprocessor import align_processed_frames, format_price_frame_for_finrl
from src.pipeline.demo import _resolve_drl_backend, _resolve_finsaber_compat_cfg


class _Cfg:
    def __init__(self):
        self.execution = {
            "drl_backend": "finsaber_compat",
            "finsaber_compat": {
                "hmax": 123,
                "reward_scaling": 1e-4,
                "eval_episodes": 2,
            },
        }
        self.initial_cash = 100000.0
        self.max_trade = 1000
        self.fee_rate = 0.001


def test_format_price_frame_for_finrl_basic():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "asset": ["A", "B", "A", "B"],
            "open": [1, 2, 1.1, 2.1],
            "high": [1, 2, 1.1, 2.1],
            "low": [1, 2, 1.1, 2.1],
            "close": [1, 2, 1.1, 2.1],
            "volume": [10, 20, 11, 21],
        }
    )
    out = format_price_frame_for_finrl(raw)
    assert list(out.columns) == ["date", "tic", "open", "high", "low", "close", "volume", "day"]
    assert out["tic"].tolist() == ["A", "B", "A", "B"]


def test_align_processed_frames_uses_common_assets():
    left = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "tic": ["A", "B"],
            "close": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02"],
            "tic": ["B", "C"],
            "close": [2.1, 3.1],
        }
    )
    (left_aligned, right_aligned), common = align_processed_frames(left, right)
    assert common == ["B"]
    assert left_aligned["tic"].unique().tolist() == ["B"]
    assert right_aligned["tic"].unique().tolist() == ["B"]


def test_resolve_finsaber_backend_cfg():
    cfg = _Cfg()
    assert _resolve_drl_backend(cfg) == "finsaber_compat"
    compat_cfg, summary = _resolve_finsaber_compat_cfg(cfg, total_timesteps=77)
    assert compat_cfg.total_timesteps == 77
    assert compat_cfg.hmax == 123
    assert compat_cfg.eval_episodes == 2
    assert summary["hmax"] == 123
