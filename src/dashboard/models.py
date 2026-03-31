from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RunState = Literal["completed", "running", "partial", "stale_or_failed"]
WindowState = Literal["pending", "running_llm", "running_rl", "completed", "incomplete"]


@dataclass(frozen=True)
class BoundProcess:
    pid: int
    etime: str
    command: str


@dataclass
class IterationSnapshot:
    algorithm: str
    iteration: int
    candidate_count: int
    error_count: int
    response_count: int
    has_dialog: bool
    has_finalized_trace: bool
    dialog_path: Path | None = None
    trace_entry: dict[str, Any] | None = None
    response_items: list[dict[str, Any]] = field(default_factory=list)
    error_items: list[dict[str, Any]] = field(default_factory=list)
    last_updated_ts: float | None = None


@dataclass
class WindowSnapshot:
    name: str
    path: Path
    state: WindowState
    last_updated_ts: float | None
    split: dict[str, str]
    exists: bool
    is_root_window: bool
    algorithms: list[str]
    latest_iteration_by_algo: dict[str, int | None]
    iterations_by_algo: dict[str, list[IterationSnapshot]]
    file_status: dict[str, bool]
    metrics_path: Path | None = None
    metrics_table_path: Path | None = None
    reward_trace_path: Path | None = None
    run_manifest_path: Path | None = None
    run_summary_path: Path | None = None
    policy_behavior_path: Path | None = None
    action_saturation_path: Path | None = None
    llm_iter_trace_path: Path | None = None
    llm_responses_path: Path | None = None
    llm_errors_path: Path | None = None
    scenario_profile_path: Path | None = None
    artifacts_path: Path | None = None
    candidate_dir: Path | None = None
    load_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.state == "completed"

    @property
    def display_name(self) -> str:
        return "root" if self.is_root_window else self.name


@dataclass
class RunSnapshot:
    run_id: str
    path: Path
    state: RunState
    last_updated_ts: float | None
    algorithms: list[str]
    total_windows: int
    completed_windows: int
    windows: list[WindowSnapshot]
    bound_processes: list[BoundProcess]
    config: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None
    root_manifest_path: Path | None = None
    summary_path: Path | None = None
    metrics_path: Path | None = None
    walk_forward_summary_path: Path | None = None
    walk_forward_table_path: Path | None = None
    artifacts_path: Path | None = None
    cross_window_distillation_path: Path | None = None
    load_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.state == "running"
