from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.demo import DemoConfig, run_demo
from src.utils.paths import repo_root, ensure_dir


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_unique_run_dir(root: Path, target: str) -> Path:
    # Keep timestamp prefix for readability, add millisecond + short nonce to avoid parallel collisions.
    now = dt.datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{int(now.microsecond / 1000):03d}"
    nonce = uuid.uuid4().hex[:4]
    run_dir = root / "runs" / f"{timestamp}_{ms}_{nonce}_{target}"
    while run_dir.exists():
        nonce = uuid.uuid4().hex[:4]
        run_dir = root / "runs" / f"{timestamp}_{ms}_{nonce}_{target}"
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["demo", "bull_regime_long_window"], help="Run target")
    parser.add_argument("--config", default=None, help="Config path")
    args = parser.parse_args()

    root = repo_root()
    default_config = (
        "configs/demo.yaml"
        if args.target == "demo"
        else "configs/current_baseline/bull_regime_long_window_5level.yaml"
    )
    cfg_path = root / (args.config or default_config)
    cfg_dict = load_config(cfg_path)

    run_dir = _build_unique_run_dir(root, args.target)
    ensure_dir(run_dir)

    # snapshot config
    (run_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.target == "demo":
        cfg = DemoConfig(**cfg_dict)
        run_demo(cfg, run_dir=run_dir, data_dir=root / "data")
    elif args.target == "bull_regime_long_window":
        from scripts.run_bull_regime_long_window import run_from_spec

        run_from_spec(cfg_path, run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
