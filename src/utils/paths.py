from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # scripts/ or src/ are expected to be two levels under repo root
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
