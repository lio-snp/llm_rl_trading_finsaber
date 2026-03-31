from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FinsaberNativeStateContract:
    asset_order: list[str]
    indicator_order: list[str]

    @property
    def stock_dim(self) -> int:
        return int(len(self.asset_order))

    @property
    def state_dim(self) -> int:
        return int(1 + 2 * self.stock_dim + self.stock_dim * len(self.indicator_order))

    @property
    def cash_index(self) -> int:
        return 0

    @property
    def close_slice(self) -> tuple[int, int]:
        return (1, 1 + self.stock_dim)

    @property
    def holdings_slice(self) -> tuple[int, int]:
        start = self.close_slice[1]
        return (start, start + self.stock_dim)

    def indicator_slices(self) -> dict[str, tuple[int, int]]:
        start = self.holdings_slice[1]
        out: dict[str, tuple[int, int]] = {}
        for name in self.indicator_order:
            end = start + self.stock_dim
            out[str(name)] = (start, end)
            start = end
        return out

    def describe(self) -> list[str]:
        rows = ["s[0] = cash"]
        idx = 1
        for asset in self.asset_order:
            rows.append(f"s[{idx}] = close_price:{asset}")
            idx += 1
        for asset in self.asset_order:
            rows.append(f"s[{idx}] = holding:{asset}")
            idx += 1
        for indicator in self.indicator_order:
            for asset in self.asset_order:
                rows.append(f"s[{idx}] = indicator:{indicator}:{asset}")
                idx += 1
        return rows

    def describe_compact(self) -> list[str]:
        rows = ["s[0] = cash"]
        rows.append(
            f"s[{self.close_slice[0]}:{self.close_slice[1]}] = close-price block "
            f"(asset order: {', '.join(self.asset_order)})"
        )
        rows.append(
            f"s[{self.holdings_slice[0]}:{self.holdings_slice[1]}] = holdings block "
            f"(same asset order)"
        )
        for indicator, (start, end) in self.indicator_slices().items():
            rows.append(
                f"s[{start}:{end}] = indicator block `{indicator}` "
                f"(asset order: {', '.join(self.asset_order)})"
            )
        return rows

    def prompt_note(self) -> str:
        return (
            "Native backend contract rules:\n"
            "- Treat the state description above as authoritative FINSABER-native layout.\n"
            "- Do not assume the generic OHLCV-per-asset LESR schema.\n"
            "- `intrinsic_reward(s)` must work directly on this raw native layout.\n"
            "- `revise_state(s)` may append extra dims, but it must preserve the original native prefix semantics.\n"
            "- If you compute portfolio-level features, derive them from cash, close-price block, and holdings block in this native layout."
        )

    def summary(self) -> dict:
        return {
            "state_contract": "finsaber_native_v1",
            "state_semantics": "cash + close_prices + holdings + indicator_major_blocks",
            "asset_order": list(self.asset_order),
            "indicator_order": list(self.indicator_order),
            "state_dim": int(self.state_dim),
            "cash_index": int(self.cash_index),
            "close_slice": list(self.close_slice),
            "holdings_slice": list(self.holdings_slice),
            "indicator_slices": {key: list(val) for key, val in self.indicator_slices().items()},
        }


def build_finsaber_native_state_contract(
    assets: list[str],
    tech_indicator_list: list[str],
) -> FinsaberNativeStateContract:
    asset_order = [str(asset) for asset in assets]
    indicator_order = [str(ind) for ind in tech_indicator_list]
    return FinsaberNativeStateContract(asset_order=asset_order, indicator_order=indicator_order)


def _resolve_asset_column(df: pd.DataFrame) -> str:
    if "asset" in df.columns:
        return "asset"
    if "tic" in df.columns:
        return "tic"
    raise ValueError("Native state contract requires either 'asset' or 'tic' column.")


def build_finsaber_native_state(
    day_df: pd.DataFrame,
    *,
    contract: FinsaberNativeStateContract,
    cash: float,
    holdings: dict[str, float] | None = None,
) -> np.ndarray:
    asset_col = _resolve_asset_column(day_df)
    rows = day_df.copy()
    rows[asset_col] = rows[asset_col].astype(str)
    by_asset = rows.set_index(asset_col)
    missing_assets = [asset for asset in contract.asset_order if asset not in by_asset.index]
    if missing_assets:
        raise ValueError(f"Missing assets for native state build: {missing_assets}")
    holdings = {str(k): float(v) for k, v in dict(holdings or {}).items()}
    state = [float(cash)]
    for asset in contract.asset_order:
        state.append(float(by_asset.loc[asset, "close"]))
    for asset in contract.asset_order:
        state.append(float(holdings.get(asset, 0.0)))
    for indicator in contract.indicator_order:
        for asset in contract.asset_order:
            state.append(float(by_asset.loc[asset, indicator]))
    return np.asarray(state, dtype=np.float32)


def collect_finsaber_native_reference_states(
    processed_df: pd.DataFrame,
    *,
    contract: FinsaberNativeStateContract,
    initial_cash: float,
    max_samples: int | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    if processed_df is None or processed_df.empty:
        raise ValueError("processed native dataframe is empty; cannot build validation states.")
    df = processed_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    asset_col = _resolve_asset_column(df)
    required_cols = {"date", asset_col, "close", *contract.indicator_order}
    missing_cols = sorted(col for col in required_cols if col not in df.columns)
    if missing_cols:
        raise ValueError(f"processed native dataframe missing state-contract columns: {missing_cols}")
    for _, day_df in df.groupby("date", sort=True):
        try:
            rows.append(
                build_finsaber_native_state(
                    day_df,
                    contract=contract,
                    cash=float(initial_cash),
                    holdings={asset: 0.0 for asset in contract.asset_order},
                )
            )
        except Exception:
            continue
    if not rows:
        raise ValueError("unable to build any native reference states from processed dataframe")
    stacked = np.stack(rows, axis=0).astype(np.float32)
    if max_samples is not None and stacked.shape[0] > max_samples:
        idx = np.linspace(0, stacked.shape[0] - 1, num=int(max_samples), dtype=int)
        stacked = stacked[idx]
    return stacked


def select_native_validation_states(reference_states: np.ndarray, max_states: int = 3) -> list[np.ndarray]:
    arr = np.asarray(reference_states, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return []
    if arr.shape[0] <= max_states:
        return [arr[i].copy() for i in range(arr.shape[0])]
    idx = np.linspace(0, arr.shape[0] - 1, num=int(max_states), dtype=int)
    return [arr[int(i)].copy() for i in idx]
