from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.demo import _build_cross_window_distillation
from src.utils.hash import sha256_file
from src.utils.paths import repo_root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def backfill_run(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    root = repo_root()

    run_manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    artifacts_path = run_dir / "artifacts.json"
    hashes_path = run_dir / "hashes.json"
    cross_path = run_dir / "cross_window_distillation.json"

    if not run_manifest_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json: {run_manifest_path}")

    run_manifest = _load_json(run_manifest_path)
    walk_forward = run_manifest.get("walk_forward") or {}
    window_infos = walk_forward.get("windows") or []
    if not window_infos:
        raise ValueError(f"No walk_forward.windows found in {run_manifest_path}")

    distillation = _build_cross_window_distillation(root, window_infos)
    _write_json(cross_path, distillation)

    if metrics_path.exists():
        metrics = _load_json(metrics_path)
    else:
        metrics = {}
    metrics["cross_window_distillation"] = distillation
    _write_json(metrics_path, metrics)

    run_manifest["cross_window_distillation"] = distillation
    _write_json(run_manifest_path, run_manifest)

    artifacts = _load_json(artifacts_path) if artifacts_path.exists() else {}
    artifacts["cross_window_distillation"] = str(cross_path.relative_to(root))
    _write_json(artifacts_path, artifacts)

    hashes = _load_json(hashes_path) if hashes_path.exists() else {}
    hashes["cross_window_distillation"] = sha256_file(cross_path)
    if metrics_path.exists():
        hashes["metrics"] = sha256_file(metrics_path)
    hashes["run_manifest"] = sha256_file(run_manifest_path)
    hashes["artifacts"] = sha256_file(artifacts_path)
    _write_json(hashes_path, hashes)

    return cross_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cross_window_distillation.json for walk-forward runs.")
    parser.add_argument("--run-dir", action="append", required=True, help="Full path to a completed walk-forward run dir.")
    args = parser.parse_args()

    for run_dir_raw in args.run_dir:
        cross_path = backfill_run(Path(run_dir_raw))
        print(cross_path)


if __name__ == "__main__":
    main()
