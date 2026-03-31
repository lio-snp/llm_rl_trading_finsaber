from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.paths import repo_root


def load_filter_dates_from_path(path_like: str) -> set[str]:
    path = Path(path_like)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"split_date_filters path not found: {path}")

    suffix = path.suffix.lower()
    values: list[object]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("dates", [])
        if not isinstance(payload, list):
            raise ValueError(f"split_date_filters json must contain a list of dates: {path}")
        values = payload
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise ValueError(f"split_date_filters csv requires a 'date' column: {path}")
        values = frame["date"].dropna().tolist()
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out: set[str] = set()
    for value in values:
        try:
            out.add(str(pd.to_datetime(value).date()))
        except Exception:
            continue
    return out


def normalize_split_date_filter(raw_filter: object) -> set[str] | None:
    if raw_filter in (None, "", [], {}, ()):
        return None
    if isinstance(raw_filter, dict):
        if "include_dates" in raw_filter:
            return normalize_split_date_filter(raw_filter.get("include_dates"))
        if "dates" in raw_filter:
            return normalize_split_date_filter(raw_filter.get("dates"))
        if "path" in raw_filter:
            return load_filter_dates_from_path(str(raw_filter.get("path", "")))
        if "include_dates_path" in raw_filter:
            return load_filter_dates_from_path(str(raw_filter.get("include_dates_path", "")))
        return None
    if isinstance(raw_filter, str):
        return load_filter_dates_from_path(raw_filter)
    if isinstance(raw_filter, (list, tuple, set)):
        out: set[str] = set()
        for value in raw_filter:
            try:
                out.add(str(pd.to_datetime(value).date()))
            except Exception:
                continue
        return out
    return None


def apply_split_date_filter(df: pd.DataFrame, raw_filter: object) -> tuple[pd.DataFrame, dict]:
    allowed_dates = normalize_split_date_filter(raw_filter)
    requested_dates = sorted(df["date"].unique().tolist()) if not df.empty else []
    requested_date_count = int(len(requested_dates))
    if not allowed_dates:
        return df.copy(), {
            "filtered": False,
            "requested_date_count": requested_date_count,
            "allowed_date_count": requested_date_count,
            "effective_date_count": requested_date_count,
            "dropped_date_count": 0,
        }

    filtered = df[df["date"].isin(allowed_dates)].copy()
    effective_date_count = int(filtered["date"].nunique()) if not filtered.empty else 0
    return filtered, {
        "filtered": True,
        "requested_date_count": requested_date_count,
        "allowed_date_count": int(len(allowed_dates)),
        "effective_date_count": effective_date_count,
        "dropped_date_count": int(max(0, requested_date_count - effective_date_count)),
    }


def apply_split_date_filters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date_filters: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    filters = split_date_filters or {}
    if not isinstance(filters, dict):
        filters = {}
    train_out, train_summary = apply_split_date_filter(train_df, filters.get("train"))
    val_out, val_summary = apply_split_date_filter(val_df, filters.get("val"))
    test_out, test_summary = apply_split_date_filter(test_df, filters.get("test"))
    return train_out, val_out, test_out, {
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }


def split_meta_block_from_df(df: pd.DataFrame, original_block: dict | None) -> dict:
    block = dict(original_block or {})
    block["requested_start"] = str(block.get("start", ""))
    block["requested_end"] = str(block.get("end", ""))
    block["requested_days"] = int(block.get("days", 0) or 0)
    unique_dates = sorted(df["date"].unique().tolist()) if not df.empty else []
    if unique_dates:
        block["start"] = unique_dates[0]
        block["end"] = unique_dates[-1]
    block["days"] = int(len(unique_dates))
    return block
