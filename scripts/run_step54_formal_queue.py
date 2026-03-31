from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.demo import DemoConfig, run_demo
from src.utils.paths import ensure_dir, repo_root


@dataclass(frozen=True)
class Task:
    name: str
    label: str
    config_rel: str


ROOT = repo_root()
RUNS_DIR = ROOT / "runs"
STATE_PATH = RUNS_DIR / "step54_formal_queue_state.json"

TASKS = [
    Task(
        name="step54_selected4_full",
        label="selected4",
        config_rel="configs/step54_regime_signal_unpacking/selected4_per_algo_regime_signal_unpacking_full.yaml",
    ),
    Task(
        name="step54_composite_full",
        label="composite",
        config_rel="configs/step54_regime_signal_unpacking/composite_per_algo_regime_signal_unpacking_full.yaml",
    ),
]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _load_state() -> dict[str, Any]:
    return _load_json(
        STATE_PATH,
        {
            "created_at": _now(),
            "updated_at": _now(),
            "tasks": {},
        },
    )


def _build_unique_run_dir(target: str = "demo") -> Path:
    now = dt.datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{int(now.microsecond / 1000):03d}"
    nonce = uuid.uuid4().hex[:4]
    run_dir = RUNS_DIR / f"{timestamp}_{ms}_{nonce}_{target}"
    while run_dir.exists():
        nonce = uuid.uuid4().hex[:4]
        run_dir = RUNS_DIR / f"{timestamp}_{ms}_{nonce}_{target}"
    return run_dir


def _load_cfg_dict(config_rel: str) -> dict[str, Any]:
    path = ROOT / config_rel
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_new_config(config_rel: str, run_dir: Path) -> None:
    cfg_dict = _load_cfg_dict(config_rel)
    ensure_dir(run_dir)
    (run_dir / "config.yaml").write_text((ROOT / config_rel).read_text())
    cfg = DemoConfig(**cfg_dict)
    run_demo(cfg, run_dir=run_dir, data_dir=ROOT / "data")


def _task_status_from_artifacts(run_dir: Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return "missing_manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return "invalid_manifest"
    check = manifest.get("completeness_check", {}) or {}
    return str(check.get("status", "unknown"))


def _resume_existing(run_dir: Path) -> None:
    import subprocess

    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "scripts/resume_walk_forward.py", "--run-dir", str(run_dir)],
        cwd=str(ROOT),
        env=env,
        check=True,
    )


def _run_or_resume(task: Task, state: dict[str, Any]) -> Path:
    task_state = state["tasks"].get(task.name, {})
    run_dir_str = task_state.get("run_dir")
    run_dir = Path(run_dir_str) if run_dir_str else None
    if run_dir and run_dir.exists():
        if _task_status_from_artifacts(run_dir) == "complete":
            return run_dir
        _resume_existing(run_dir)
        return run_dir

    run_dir = _build_unique_run_dir("demo")
    task_state["run_dir"] = str(run_dir)
    task_state["status"] = "running"
    task_state["started_at"] = _now()
    task_state["updated_at"] = _now()
    state["tasks"][task.name] = task_state
    state["updated_at"] = _now()
    _save_state(state)
    _run_new_config(task.config_rel, run_dir)
    return run_dir


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run_manifest.json", {})
    wf = manifest.get("walk_forward", {}) or {}
    return {
        "run_dir": str(run_dir),
        "completeness": str((manifest.get("completeness_check", {}) or {}).get("status", "unknown")),
        "window_count": int(wf.get("window_count", 0) or 0),
        "actor_collapse_detected": bool(manifest.get("actor_collapse_detected", False)),
        "cross_window_distillation_present": bool((run_dir / "cross_window_distillation.json").exists()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-api-key", default="", help="Temporary DeepSeek API key for this queue run only.")
    args = parser.parse_args()

    api_key = args.deepseek_api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required for Step54 formal queue execution.")
    os.environ["DEEPSEEK_API_KEY"] = api_key

    state = _load_state()
    state["updated_at"] = _now()
    _save_state(state)

    for task in TASKS:
        task_state = state["tasks"].setdefault(task.name, {})
        task_state["config_rel"] = task.config_rel
        task_state["label"] = task.label
        task_state["status"] = task_state.get("status", "pending")
        task_state["updated_at"] = _now()
        state["updated_at"] = _now()
        _save_state(state)

        run_dir = _run_or_resume(task, state)
        task_state = state["tasks"].setdefault(task.name, {})
        task_state["run_dir"] = str(run_dir)
        task_state["completed_at"] = _now()
        task_state["status"] = "completed" if _task_status_from_artifacts(run_dir) == "complete" else "incomplete"
        task_state["summary"] = _summarize_run(run_dir)
        state["updated_at"] = _now()
        _save_state(state)

    print(json.dumps({"status": "ok", "state_path": str(STATE_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
