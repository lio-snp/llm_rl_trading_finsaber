from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _download_with_stooq(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    for symbol in tickers:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        cur = pd.read_csv(url)
        if cur.empty:
            continue
        cur = cur.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        cur["date"] = pd.to_datetime(cur["date"])
        cur = cur[(cur["date"] >= start_ts) & (cur["date"] <= end_ts)]
        if cur.empty:
            continue
        cur["symbol"] = symbol
        rows.append(cur[["date", "open", "high", "low", "close", "volume", "symbol"]])
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out["volume"] = out["volume"].astype(float)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _download_with_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        try:
            return _download_with_stooq(tickers=tickers, start=start, end=end)
        except Exception as stooq_exc:
            raise RuntimeError(
                "download failed: install `yfinance` or ensure internet access to stooq."
            ) from stooq_exc

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])

    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        for symbol in tickers:
            if symbol not in df.columns.get_level_values(0):
                continue
            cur = df[symbol].reset_index()
            cur = cur.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            cur["symbol"] = symbol
            rows.append(cur[["date", "open", "high", "low", "close", "volume", "symbol"]])
    else:
        cur = df.reset_index()
        cur = cur.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        cur["symbol"] = tickers[0]
        rows.append(cur[["date", "open", "high", "low", "close", "volume", "symbol"]])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out["volume"] = out["volume"].astype(float)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="SPY,QQQ,DIA,IWM",
        help="Comma-separated ticker list",
    )
    parser.add_argument("--start", default="2000-01-03")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--out",
        default="data/raw/index_etf_prices_2000_2024.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    tickers = [x.strip().upper() for x in str(args.tickers).split(",") if x.strip()]
    if not tickers:
        raise ValueError("no valid tickers")

    out = _download_with_yfinance(tickers=tickers, start=str(args.start), end=str(args.end))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parents[1] / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[ok] rows={len(out)} path={out_path}")
    if not out.empty:
        coverage = out.groupby("symbol")["date"].agg(["min", "max", "count"]).reset_index()
        print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
