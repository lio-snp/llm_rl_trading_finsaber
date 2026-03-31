from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.dashboard.models import BoundProcess, IterationSnapshot, RunSnapshot, WindowSnapshot

_DIALOG_RE = re.compile(r"^dialogs(?:_(?P<algo>[a-z0-9]+))?_it(?P<it>\d+)\.txt$")


@dataclass(frozen=True)
class _WindowSpec:
    name: str
    path: Path
    split: dict[str, str]
    is_root_window: bool


def format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def read_json_safe(path: Path | None) -> tuple[Any | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{path.name}: {exc}"


def read_yaml_safe(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{path.name}: {exc}"
    if isinstance(payload, dict):
        return payload, None
    return None, f"{path.name}: expected mapping, got {type(payload).__name__}"


def read_csv_safe(path: Path | None) -> tuple[pd.DataFrame | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{path.name}: {exc}"


def read_text_safe(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{path.name}: {exc}"


def collect_processes() -> list[BoundProcess]:
    cmd = ["ps", "-axo", "pid=,etime=,command="]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    records: list[BoundProcess] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid_text, etime, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        records.append(BoundProcess(pid=pid, etime=etime, command=command))
    return records


def discover_runs(
    runs_root: Path,
    stale_minutes: int = 15,
    processes: list[BoundProcess] | None = None,
) -> list[RunSnapshot]:
    root = runs_root.resolve()
    proc_list = collect_processes() if processes is None else processes
    snapshots: list[RunSnapshot] = []
    for run_dir in sorted(root.iterdir()):
        if not _is_run_dir(run_dir):
            continue
        snapshots.append(_summarize_run_dir(run_dir=run_dir, runs_root=root, stale_minutes=stale_minutes, processes=proc_list))
    return snapshots


def load_run_snapshot(
    run_dir: Path,
    runs_root: Path,
    stale_minutes: int = 15,
    processes: list[BoundProcess] | None = None,
    detailed: bool = True,
    detailed_window_names: set[str] | None = None,
) -> RunSnapshot:
    root = runs_root.resolve()
    project_root = root.parent
    proc_list = collect_processes() if processes is None else processes

    config_path = run_dir / "config.yaml"
    root_manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.md"
    metrics_path = run_dir / "metrics.json"
    walk_forward_summary_path = run_dir / "walk_forward_summary.json"
    walk_forward_table_path = run_dir / "walk_forward_metrics_table.csv"
    artifacts_path = run_dir / "artifacts.json"
    cross_window_distillation_path = run_dir / "cross_window_distillation.json"

    config, config_error = read_yaml_safe(config_path)
    root_manifest, manifest_error = read_json_safe(root_manifest_path)
    walk_forward_summary, wf_error = read_json_safe(walk_forward_summary_path)
    load_warnings = [msg for msg in [config_error, manifest_error, wf_error] if msg]

    algorithms = _collect_algorithms(config, root_manifest, walk_forward_summary)
    window_specs = _collect_window_specs(
        run_dir=run_dir,
        project_root=project_root,
        config=config or {},
        root_manifest=root_manifest if isinstance(root_manifest, dict) else None,
        walk_forward_summary=walk_forward_summary if isinstance(walk_forward_summary, dict) else None,
    )

    windows = [
        _build_window_snapshot(
            spec=spec,
            algorithms=algorithms,
            detailed=detailed and (detailed_window_names is None or spec.name in detailed_window_names),
        )
        for spec in window_specs
    ]
    total_windows = len(windows)
    completed_windows = sum(1 for window in windows if window.completed)

    root_mtime = _latest_immediate_mtime(run_dir)
    latest_candidates = [ts for ts in [root_mtime, *[window.last_updated_ts for window in windows]] if ts is not None]
    last_updated_ts = max(latest_candidates) if latest_candidates else None

    run_bound_processes = [
        process
        for process in proc_list
        if "resume_walk_forward.py" in process.command and str(run_dir.resolve()) in process.command
    ]
    recent_update = bool(
        last_updated_ts is not None and (time.time() - last_updated_ts) <= float(stale_minutes * 60)
    )
    root_completed = bool(
        isinstance(root_manifest, dict)
        and isinstance(root_manifest.get("completeness_check"), dict)
        and root_manifest["completeness_check"].get("status") == "complete"
    )
    if not root_completed and total_windows == 1 and windows and windows[0].completed:
        root_completed = root_manifest_path.exists() or summary_path.exists()
    has_structured_artifacts = any(
        path.exists()
        for path in [
            root_manifest_path,
            summary_path,
            metrics_path,
            walk_forward_summary_path,
            walk_forward_table_path,
            artifacts_path,
        ]
    ) or any(window.exists for window in windows)

    if root_completed:
        state = "completed"
    elif run_bound_processes or recent_update:
        state = "running"
    elif total_windows > 1 and 0 < completed_windows < total_windows and has_structured_artifacts:
        state = "partial"
    elif total_windows == 1 and has_structured_artifacts and not windows[0].completed and windows[0].exists:
        state = "partial"
    else:
        state = "stale_or_failed"

    metadata: dict[str, Any] = {}
    if detailed:
        if isinstance(root_manifest, dict):
            metadata["root_manifest"] = root_manifest
        if isinstance(walk_forward_summary, dict):
            metadata["walk_forward_summary"] = walk_forward_summary

    return RunSnapshot(
        run_id=run_dir.name,
        path=run_dir,
        state=state,
        last_updated_ts=last_updated_ts,
        algorithms=algorithms,
        total_windows=total_windows,
        completed_windows=completed_windows,
        windows=windows,
        bound_processes=run_bound_processes,
        config=config or {},
        config_path=config_path if config_path.exists() else None,
        root_manifest_path=root_manifest_path if root_manifest_path.exists() else None,
        summary_path=summary_path if summary_path.exists() else None,
        metrics_path=metrics_path if metrics_path.exists() else None,
        walk_forward_summary_path=walk_forward_summary_path if walk_forward_summary_path.exists() else None,
        walk_forward_table_path=walk_forward_table_path if walk_forward_table_path.exists() else None,
        artifacts_path=artifacts_path if artifacts_path.exists() else None,
        cross_window_distillation_path=(
            cross_window_distillation_path if cross_window_distillation_path.exists() else None
        ),
        load_warnings=load_warnings,
        metadata=metadata,
    )


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"logs", "_batch_logs"}:
        return False
    if (path / "config.yaml").exists():
        return True
    if any((path / name).exists() for name in ["run_manifest.json", "metrics.json", "walk_forward_summary.json"]):
        return True
    return any(child.is_dir() and child.name.startswith("wf_window_") for child in path.iterdir())


def _summarize_run_dir(
    run_dir: Path,
    runs_root: Path,
    stale_minutes: int,
    processes: list[BoundProcess],
) -> RunSnapshot:
    config_path = run_dir / "config.yaml"
    root_manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.md"
    metrics_path = run_dir / "metrics.json"
    walk_forward_summary_path = run_dir / "walk_forward_summary.json"
    walk_forward_table_path = run_dir / "walk_forward_metrics_table.csv"

    algorithms: list[str] = []
    actual_windows = sorted(child for child in run_dir.glob("wf_window_*") if child.is_dir())
    total_windows = max(len(actual_windows), 1)

    if actual_windows:
        completed_windows = sum(
            1
            for window_dir in actual_windows
            if (window_dir / "metrics.json").exists()
            and ((window_dir / "run_summary.md").exists() or (window_dir / "run_manifest.json").exists())
        )
    else:
        completed_windows = int(
            metrics_path.exists() and (summary_path.exists() or root_manifest_path.exists())
        )

    root_completed = bool(
        root_manifest_path.exists()
        and (walk_forward_summary_path.exists() or walk_forward_table_path.exists())
    )
    if not root_completed and total_windows == 1 and completed_windows == 1:
        root_completed = summary_path.exists() or root_manifest_path.exists()

    last_updated_candidates = [_latest_immediate_mtime(run_dir)]
    last_updated_candidates.extend(window_dir.stat().st_mtime for window_dir in actual_windows)
    last_updated_ts = max(ts for ts in last_updated_candidates if ts is not None)

    bound_processes = [
        process
        for process in processes
        if "resume_walk_forward.py" in process.command and str(run_dir.resolve()) in process.command
    ]
    recent_update = bool(
        last_updated_ts is not None and (time.time() - last_updated_ts) <= float(stale_minutes * 60)
    )
    has_structured_artifacts = bool(actual_windows) or any(
        path.exists()
        for path in [root_manifest_path, summary_path, metrics_path, walk_forward_summary_path]
    )
    if root_completed:
        state = "completed"
    elif bound_processes or recent_update:
        state = "running"
    elif 0 < completed_windows < total_windows and has_structured_artifacts:
        state = "partial"
    else:
        state = "stale_or_failed"

    return RunSnapshot(
        run_id=run_dir.name,
        path=run_dir,
        state=state,
        last_updated_ts=last_updated_ts,
        algorithms=algorithms,
        total_windows=total_windows,
        completed_windows=completed_windows,
        windows=[],
        bound_processes=bound_processes,
        config={},
        config_path=config_path if config_path.exists() else None,
        root_manifest_path=root_manifest_path if root_manifest_path.exists() else None,
        summary_path=summary_path if summary_path.exists() else None,
        metrics_path=metrics_path if metrics_path.exists() else None,
        walk_forward_summary_path=walk_forward_summary_path if walk_forward_summary_path.exists() else None,
        walk_forward_table_path=walk_forward_table_path if walk_forward_table_path.exists() else None,
    )


def _collect_algorithms(
    config: dict[str, Any] | None,
    root_manifest: dict[str, Any] | None,
    walk_forward_summary: dict[str, Any] | None,
) -> list[str]:
    candidates: list[str] = []
    for payload in [config or {}, root_manifest or {}, walk_forward_summary or {}]:
        eval_algorithms = payload.get("eval_algorithms")
        if isinstance(eval_algorithms, list):
            candidates.extend([str(item).lower() for item in eval_algorithms])
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, str):
            candidates.append(algorithm.lower())
    seen: set[str] = set()
    normalized: list[str] = []
    for algo in candidates:
        if algo not in seen:
            normalized.append(algo)
            seen.add(algo)
    return normalized


def _collect_window_specs(
    run_dir: Path,
    project_root: Path,
    config: dict[str, Any],
    root_manifest: dict[str, Any] | None,
    walk_forward_summary: dict[str, Any] | None,
) -> list[_WindowSpec]:
    specs_by_name: dict[str, _WindowSpec] = {}

    manifest_windows = None
    if isinstance(root_manifest, dict):
        walk_forward_meta = root_manifest.get("walk_forward")
        if isinstance(walk_forward_meta, dict):
            manifest_windows = walk_forward_meta.get("windows")
    if isinstance(manifest_windows, list):
        for entry in manifest_windows:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("window_name") or Path(str(entry.get("run_dir", ""))).name)
            if not name:
                continue
            rel_run_dir = entry.get("run_dir")
            path = project_root / str(rel_run_dir) if rel_run_dir else run_dir / name
            specs_by_name[name] = _WindowSpec(
                name=name,
                path=path,
                split=_normalize_split(entry.get("split")),
                is_root_window=False,
            )

    if not specs_by_name and isinstance(walk_forward_summary, dict):
        summary_windows = walk_forward_summary.get("windows")
        if isinstance(summary_windows, list):
            for entry in summary_windows:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("window_name") or Path(str(entry.get("run_dir", ""))).name)
                if not name:
                    continue
                rel_run_dir = entry.get("run_dir")
                path = project_root / str(rel_run_dir) if rel_run_dir else run_dir / name
                specs_by_name[name] = _WindowSpec(
                    name=name,
                    path=path,
                    split=_normalize_split(entry.get("split")),
                    is_root_window=False,
                )

    if not specs_by_name:
        walk_forward_cfg = config.get("walk_forward")
        if isinstance(walk_forward_cfg, dict) and bool(walk_forward_cfg.get("enabled", False)):
            windows = walk_forward_cfg.get("windows") or []
            if isinstance(windows, list):
                for index, entry in enumerate(windows):
                    name = f"wf_window_{index:02d}"
                    specs_by_name[name] = _WindowSpec(
                        name=name,
                        path=run_dir / name,
                        split=_normalize_split(entry),
                        is_root_window=False,
                    )

    for child in run_dir.glob("wf_window_*"):
        if child.name not in specs_by_name:
            specs_by_name[child.name] = _WindowSpec(
                name=child.name,
                path=child,
                split={},
                is_root_window=False,
            )

    if specs_by_name:
        return [specs_by_name[name] for name in sorted(specs_by_name)]

    return [
        _WindowSpec(
            name="root",
            path=run_dir,
            split={},
            is_root_window=True,
        )
    ]


def _normalize_split(raw_split: Any) -> dict[str, str]:
    if not isinstance(raw_split, dict):
        return {}
    normalized: dict[str, str] = {}
    for key in ["train", "val", "test"]:
        value = raw_split.get(key)
        if isinstance(value, dict):
            start = value.get("start")
            end = value.get("end")
            if start or end:
                normalized[key] = f"{start or '?'} -> {end or '?'}"
        elif isinstance(value, str):
            normalized[key] = value
    return normalized


def _build_window_snapshot(
    spec: _WindowSpec,
    algorithms: list[str],
    detailed: bool,
) -> WindowSnapshot:
    path = spec.path
    exists = path.exists()
    file_status = {
        "metrics": (path / "metrics.json").exists(),
        "metrics_table": (path / "metrics_table.csv").exists(),
        "reward_trace": (path / "reward_trace.json").exists(),
        "run_manifest": (path / "run_manifest.json").exists(),
        "run_summary": (path / "run_summary.md").exists(),
        "policy_behavior_summary": (path / "policy_behavior_summary.json").exists(),
        "td3_action_saturation": (path / "td3_action_saturation.json").exists(),
        "llm_iter_trace": (path / "llm_iter_trace.json").exists(),
        "llm_responses": (path / "llm_responses.json").exists(),
        "llm_errors": (path / "llm_errors.json").exists(),
        "scenario_profile": (path / "scenario_profile.json").exists(),
        "artifacts": (path / "artifacts.json").exists(),
        "revision_candidates": (path / "revision_candidates").exists(),
    }
    dialog_map_by_algo, generic_dialogs, has_dialogs = _discover_dialog_files(path)
    if not exists:
        state = "pending"
    elif file_status["metrics"] and (file_status["run_summary"] or file_status["run_manifest"]):
        state = "completed"
    elif has_dialogs and not file_status["llm_iter_trace"]:
        state = "running_llm"
    elif file_status["llm_iter_trace"] and not file_status["metrics"]:
        state = "running_rl"
    else:
        state = "incomplete"

    window_algorithms = list(algorithms)
    for algo in sorted(dialog_map_by_algo):
        if algo not in window_algorithms:
            window_algorithms.append(algo)

    iterations_by_algo: dict[str, list[IterationSnapshot]] = {algo: [] for algo in window_algorithms}
    latest_iteration_by_algo: dict[str, int | None] = {
        algo: _latest_iteration(dialog_map_by_algo.get(algo, {}), generic_dialogs if not dialog_map_by_algo.get(algo) else {})
        for algo in window_algorithms
    }
    load_warnings: list[str] = []
    metadata: dict[str, Any] = {}

    if detailed and exists:
        trace_payload, trace_error = read_json_safe(path / "llm_iter_trace.json")
        responses_payload, responses_error = read_json_safe(path / "llm_responses.json")
        errors_payload, errors_error = read_json_safe(path / "llm_errors.json")
        manifest_payload, manifest_error = read_json_safe(path / "run_manifest.json")
        load_warnings.extend(
            [message for message in [trace_error, responses_error, errors_error, manifest_error] if message]
        )
        if isinstance(manifest_payload, dict):
            metadata["run_manifest"] = manifest_payload

        grouped_trace = _group_by_algo_iteration(trace_payload if isinstance(trace_payload, list) else [])
        grouped_responses = _group_by_algo_iteration(
            responses_payload if isinstance(responses_payload, list) else [],
            default_algorithm=None,
        )
        grouped_errors = _group_by_algo_iteration(
            errors_payload if isinstance(errors_payload, list) else [],
            default_algorithm=None,
        )

        trace_algorithms = {algo for algo, _ in grouped_trace}
        response_algorithms = {algo for algo, _ in grouped_responses}
        error_algorithms = {algo for algo, _ in grouped_errors}
        for algo in sorted(trace_algorithms | response_algorithms | error_algorithms):
            if algo and algo not in window_algorithms:
                window_algorithms.append(algo)
                iterations_by_algo[algo] = []
                latest_iteration_by_algo.setdefault(algo, None)

        for algo in window_algorithms:
            dialog_candidates = dialog_map_by_algo.get(algo, {}) or generic_dialogs
            iteration_ids = set(dialog_candidates)
            iteration_ids.update(it for trace_algo, it in grouped_trace if trace_algo == algo)
            iteration_ids.update(it for response_algo, it in grouped_responses if response_algo == algo)
            iteration_ids.update(it for error_algo, it in grouped_errors if error_algo == algo)
            snapshots: list[IterationSnapshot] = []
            for iteration in sorted(iteration_ids):
                trace_entry = grouped_trace.get((algo, iteration))
                response_items = grouped_responses.get((algo, iteration), [])
                error_items = grouped_errors.get((algo, iteration), [])
                dialog_path = dialog_candidates.get(iteration)
                last_updated_ts = _max_mtime(
                    [
                        dialog_path,
                        path / "llm_iter_trace.json" if file_status["llm_iter_trace"] else None,
                        path / "llm_responses.json" if file_status["llm_responses"] else None,
                        path / "llm_errors.json" if file_status["llm_errors"] else None,
                    ]
                )
                snapshots.append(
                    IterationSnapshot(
                        algorithm=algo,
                        iteration=iteration,
                        candidate_count=len(trace_entry.get("candidates", [])) if isinstance(trace_entry, dict) else 0,
                        error_count=len(error_items),
                        response_count=len(response_items),
                        has_dialog=dialog_path is not None and dialog_path.exists(),
                        has_finalized_trace=isinstance(trace_entry, dict),
                        dialog_path=dialog_path if dialog_path and dialog_path.exists() else None,
                        trace_entry=trace_entry if isinstance(trace_entry, dict) else None,
                        response_items=response_items,
                        error_items=error_items,
                        last_updated_ts=last_updated_ts,
                    )
                )
            iterations_by_algo[algo] = snapshots
            latest_iteration_by_algo[algo] = snapshots[-1].iteration if snapshots else latest_iteration_by_algo.get(algo)

    last_updated_ts = _latest_immediate_mtime(path) if exists else None
    return WindowSnapshot(
        name=spec.name,
        path=path,
        state=state,
        last_updated_ts=last_updated_ts,
        split=spec.split,
        exists=exists,
        is_root_window=spec.is_root_window,
        algorithms=window_algorithms,
        latest_iteration_by_algo=latest_iteration_by_algo,
        iterations_by_algo=iterations_by_algo,
        file_status=file_status,
        metrics_path=(path / "metrics.json") if file_status["metrics"] else None,
        metrics_table_path=(path / "metrics_table.csv") if file_status["metrics_table"] else None,
        reward_trace_path=(path / "reward_trace.json") if file_status["reward_trace"] else None,
        run_manifest_path=(path / "run_manifest.json") if file_status["run_manifest"] else None,
        run_summary_path=(path / "run_summary.md") if file_status["run_summary"] else None,
        policy_behavior_path=(
            (path / "policy_behavior_summary.json") if file_status["policy_behavior_summary"] else None
        ),
        action_saturation_path=(
            (path / "td3_action_saturation.json") if file_status["td3_action_saturation"] else None
        ),
        llm_iter_trace_path=(path / "llm_iter_trace.json") if file_status["llm_iter_trace"] else None,
        llm_responses_path=(path / "llm_responses.json") if file_status["llm_responses"] else None,
        llm_errors_path=(path / "llm_errors.json") if file_status["llm_errors"] else None,
        scenario_profile_path=(path / "scenario_profile.json") if file_status["scenario_profile"] else None,
        artifacts_path=(path / "artifacts.json") if file_status["artifacts"] else None,
        candidate_dir=(path / "revision_candidates") if file_status["revision_candidates"] else None,
        load_warnings=load_warnings,
        metadata=metadata,
    )


def _discover_dialog_files(path: Path) -> tuple[dict[str, dict[int, Path]], dict[int, Path], bool]:
    dialog_map_by_algo: dict[str, dict[int, Path]] = {}
    generic_dialogs: dict[int, Path] = {}
    if not path.exists():
        return dialog_map_by_algo, generic_dialogs, False
    for child in path.iterdir():
        match = _DIALOG_RE.match(child.name)
        if not child.is_file() or match is None:
            continue
        iteration = int(match.group("it"))
        algo = match.group("algo")
        if algo:
            dialog_map_by_algo.setdefault(algo, {})[iteration] = child
        else:
            generic_dialogs[iteration] = child
    has_dialogs = bool(dialog_map_by_algo or generic_dialogs)
    return dialog_map_by_algo, generic_dialogs, has_dialogs


def _group_by_algo_iteration(
    payload: list[dict[str, Any]],
    default_algorithm: str | None = None,
) -> dict[tuple[str, int], Any]:
    grouped: dict[tuple[str, int], Any] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        algorithm = item.get("algorithm", default_algorithm)
        iteration = item.get("iteration")
        if not isinstance(algorithm, str) or not isinstance(iteration, int):
            continue
        key = (algorithm.lower(), iteration)
        if key not in grouped:
            grouped[key] = [] if "content" in item or "error_type" in item else item
        if isinstance(grouped[key], list):
            grouped[key].append(item)
        else:
            grouped[key] = item
    return grouped


def _latest_iteration(primary: dict[int, Path], fallback: dict[int, Path]) -> int | None:
    available = set(primary)
    if not available:
        available = set(fallback)
    return max(available) if available else None


def _latest_immediate_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    mtimes = [path.stat().st_mtime]
    try:
        for child in path.iterdir():
            mtimes.append(child.stat().st_mtime)
    except OSError:  # pragma: no cover - defensive
        return max(mtimes)
    return max(mtimes)


def _max_mtime(paths: list[Path | None]) -> float | None:
    mtimes: list[float] = []
    for path in paths:
        if path is not None and path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else None
