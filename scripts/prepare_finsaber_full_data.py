from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to all_sp500_prices_2000_2024_delisted_include.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst = root / "data" / "full" / "all_sp500_prices_2000_2024_delisted_include.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
