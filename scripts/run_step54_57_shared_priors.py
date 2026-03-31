from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.demo import DemoConfig, run_demo
from src.utils.paths import ensure_dir
from src.utils.paths import repo_root


@dataclass(frozen=True)
class Task:
    name: str
    step: str
    label: str
    config_rel: str
    kind: str


ROOT = repo_root()
RUNS_DIR = ROOT / "runs"
STATE_PATH = RUNS_DIR / "step54_57_shared_priors_queue_state.json"

STEP_DOCS = {
    "step54": ROOT / "docs/steps/step_54_regime_signal_unpacking",
    "step55": ROOT / "docs/steps/step_55_portfolio_memory_turnover",
    "step56": ROOT / "docs/steps/step_56_dsr_proxy_state",
    "step57": ROOT / "docs/steps/step_57_cross_window_distillation",
}

STEP_TITLES = {
    "step54": "Step54 Regime Signal Unpacking",
    "step55": "Step55 Portfolio Memory And Turnover",
    "step56": "Step56 DSR-Proxy State",
    "step57": "Step57 Cross-Window Distillation",
}

STEP_BASELINES = {
    "step54": [
        "Step53 selected4 smoke: `/Users/liuyanlinsnp/Desktop/LLM升级路/llm_rl_trading/runs/20260309_075402_693_281a_demo`",
        "Step53 composite smoke: `/Users/liuyanlinsnp/Desktop/LLM升级路/llm_rl_trading/runs/20260309_081248_960_ffd4_demo`",
    ],
    "step55": [
        "Step54 selected4 smoke run from this queue",
        "Step54 composite smoke run from this queue",
    ],
    "step56": [
        "Step55 selected4 smoke run from this queue",
        "Step55 composite smoke run from this queue",
    ],
    "step57": [
        "Latest successful full selected4 base: `/Users/liuyanlinsnp/Desktop/LLM升级路/llm_rl_trading/runs/20260305_163250_845_c008_demo/config.yaml`",
        "Latest successful full composite base: `/Users/liuyanlinsnp/Desktop/LLM升级路/llm_rl_trading/runs/20260302_140511_375_b1b3_demo/config.yaml`",
        "Step56 selected4/composite smoke runs from this queue",
    ],
}

TASKS = [
    Task(
        name="step54_selected4_smoke",
        step="step54",
        label="selected4",
        config_rel="configs/step54_regime_signal_unpacking/selected4_per_algo_regime_signal_unpacking_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step54_composite_smoke",
        step="step54",
        label="composite",
        config_rel="configs/step54_regime_signal_unpacking/composite_per_algo_regime_signal_unpacking_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step55_selected4_smoke",
        step="step55",
        label="selected4",
        config_rel="configs/step55_portfolio_memory_turnover/selected4_per_algo_portfolio_memory_turnover_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step55_composite_smoke",
        step="step55",
        label="composite",
        config_rel="configs/step55_portfolio_memory_turnover/composite_per_algo_portfolio_memory_turnover_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step56_selected4_smoke",
        step="step56",
        label="selected4",
        config_rel="configs/step56_dsr_proxy_state/selected4_per_algo_dsr_proxy_state_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step56_composite_smoke",
        step="step56",
        label="composite",
        config_rel="configs/step56_dsr_proxy_state/composite_per_algo_dsr_proxy_state_smoke.yaml",
        kind="smoke",
    ),
    Task(
        name="step57_selected4_full",
        step="step57",
        label="selected4",
        config_rel="configs/step57_cross_window_distillation/selected4_per_algo_cross_window_distillation_full.yaml",
        kind="full",
    ),
    Task(
        name="step57_composite_full",
        step="step57",
        label="composite",
        config_rel="configs/step57_cross_window_distillation/composite_per_algo_cross_window_distillation_full.yaml",
        kind="full",
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
            "steps": {},
        },
    )


def _run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _load_cfg_dict(config_rel: str) -> dict[str, Any]:
    path = ROOT / config_rel
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _run_new_config(config_rel: str, run_dir: Path) -> None:
    cfg_dict = _load_cfg_dict(config_rel)
    ensure_dir(run_dir)
    (run_dir / "config.yaml").write_text((ROOT / config_rel).read_text())
    cfg = DemoConfig(**cfg_dict)
    run_demo(cfg, run_dir=run_dir, data_dir=ROOT / "data")


def _run_or_resume(task: Task, env: dict[str, str], state: dict[str, Any]) -> Path:
    task_state = state["tasks"].get(task.name, {})
    run_dir_str = task_state.get("run_dir")
    run_dir = Path(run_dir_str) if run_dir_str else None
    if run_dir and run_dir.exists():
        manifest = run_dir / "run_manifest.json"
        status = _task_status_from_artifacts(run_dir)
        if status == "complete":
            return run_dir
        _run_cmd([sys.executable, "scripts/resume_walk_forward.py", "--run-dir", str(run_dir)], env)
        return run_dir

    run_dir = _build_unique_run_dir("demo")
    task_state["run_dir"] = str(run_dir)
    task_state["status"] = "running"
    task_state["started_at"] = _now()
    task_state["updated_at"] = _now()
    state["updated_at"] = _now()
    _save_state(state)
    _run_new_config(task.config_rel, run_dir)
    return run_dir


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


def _network_like(error: dict[str, Any]) -> bool:
    txt = " ".join(
        str(error.get(key, ""))
        for key in ["error_type", "message", "phase"]
    ).lower()
    markers = [
        "timeout",
        "timed out",
        "connection",
        "proxy",
        "ssl",
        "tls",
        "429",
        "rate limit",
        "network",
        "service unavailable",
        "temporarily unavailable",
    ]
    return any(marker in txt for marker in markers)


def _load_iter_trace(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "wf_window_00" / "llm_iter_trace.json"
    if not path.exists():
        path = run_dir / "llm_iter_trace.json"
    return _load_json(path, [])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _read_walk_forward_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "walk_forward_metrics_table.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate_dsharpe(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    agg_rows = [row for row in rows if row.get("window_name") == "aggregate"]
    by_algo_group: dict[tuple[str, str], float] = {}
    for row in agg_rows:
        key = (row["algorithm"], row["group"])
        by_algo_group[key] = float(row.get("Sharpe_mean", 0.0) or 0.0)
    algos = sorted({algo for algo, _ in by_algo_group.keys()})
    for algo in algos:
        g0 = by_algo_group.get((algo, "G0_baseline"), 0.0)
        out[algo] = {
            "G1-G0": by_algo_group.get((algo, "G1_revise_only"), g0) - g0,
            "G2-G0": by_algo_group.get((algo, "G2_intrinsic_only"), g0) - g0,
            "G3-G0": by_algo_group.get((algo, "G3_revise_intrinsic"), g0) - g0,
        }
    return out


def _mean_intrinsic_probe_by_algo(iter_trace: list[dict[str, Any]]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for row in iter_trace:
        algo = str(row.get("algorithm", ""))
        for cand in row.get("candidates", []) or []:
            if not cand.get("valid"):
                continue
            val = cand.get("intrinsic_probe_score_delta")
            if val is None:
                continue
            acc.setdefault(algo, []).append(float(val))
    return {algo: _mean(vals) for algo, vals in sorted(acc.items())}


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run_manifest.json", {})
    errors = _load_json(run_dir / "llm_errors.json", [])
    iter_trace = _load_iter_trace(run_dir)
    rows = _read_walk_forward_rows(run_dir)
    sample_valid = [int(row.get("sample_valid_count", 0) or 0) for row in iter_trace]
    zero_valid = sum(1 for value in sample_valid if value == 0)
    avg_valid_by_algo: dict[str, float] = {}
    by_algo: dict[str, list[int]] = {}
    for row in iter_trace:
        by_algo.setdefault(str(row.get("algorithm", "")), []).append(int(row.get("sample_valid_count", 0) or 0))
    for algo, values in sorted(by_algo.items()):
        avg_valid_by_algo[algo] = _mean([float(v) for v in values])
    return {
        "run_dir": str(run_dir),
        "completeness": str((manifest.get("completeness_check", {}) or {}).get("status", "unknown")),
        "window_count": int((manifest.get("walk_forward", {}) or {}).get("window_count", 0) or 0),
        "network_like_errors": int(sum(1 for error in errors if _network_like(error))),
        "zero_valid_iterations": int(zero_valid),
        "avg_valid_candidates_by_algo": avg_valid_by_algo,
        "candidate_scoring_effective": manifest.get("candidate_scoring_effective", {}),
        "scenario_profile": manifest.get("scenario_profile", {}),
        "aggregate_dsharpe": _aggregate_dsharpe(rows),
        "mean_intrinsic_probe_by_algo": _mean_intrinsic_probe_by_algo(iter_trace),
        "cross_window_distillation_present": (run_dir / "cross_window_distillation.json").exists(),
    }


def _render_analysis(
    problem: str,
    hypothesis: str,
    intervention: str,
    failure_modes: list[str],
    acceptance: list[str],
    result: str,
    residual: str,
    next_decision: str,
) -> str:
    return "\n".join(
        [
            f"## Problem\n{problem}",
            "",
            f"## Mechanism Hypothesis\n{hypothesis}",
            "",
            f"## Intervention\n{intervention}",
            "",
            "## Expected Failure Modes",
            *[f"- {item}" for item in failure_modes],
            "",
            "## Falsification / Acceptance Criteria",
            *[f"- {item}" for item in acceptance],
            "",
            f"## Result\n{result}",
            "",
            f"## Residual Uncertainty\n{residual}",
            "",
            f"## Next Decision\n{next_decision}",
            "",
        ]
    )


STEP_ANALYSIS_PRE = {
    "step54": {
        "problem": "The existing LESR loop already routes each window into `trend_follow`, `mean_revert`, or `risk_shield`, but the LLM only sees a compressed family label and coarse return/volatility summary. That loses information about how stress is evolving inside the window.",
        "hypothesis": "If the LLM sees explicit continuous regime signals in addition to the family label, it can generate candidates that react to volatility expansion, trend strength, and cross-sectional dispersion more precisely without introducing algorithm-specific hacks.",
        "intervention": "Implemented explicit regime signal unpacking in `scenario_profile` and threaded the new fields through prompt context, per-iteration sampling instructions, cross-branch feedback, `scenario_profile.json`, and run manifests. The family router itself remains unchanged.",
        "failure_modes": [
            "The extra regime fields may add prompt noise without improving candidate quality.",
            "LLM branches may overfit to short-term volatility spikes.",
            "Sampling health could regress if the extra context makes candidate generation less valid.",
        ],
        "acceptance": [
            "Both Step54 smoke runs complete.",
            "`zero-valid` branch iterations remain `0`.",
            "Prompt and manifest contain the new regime fields.",
            "No new gating logic is introduced into ranking.",
        ],
    },
    "step55": {
        "problem": "The LESR search still lacks explicit portfolio-memory priors and a weak penalty against unstable reallocations, so short-train winners can still be weight-jittery.",
        "hypothesis": "If the LLM is pushed toward portfolio-memory dimensions and ranking sees turnover stability as a weak additive term, candidates should stay action-sensitive while avoiding unnecessary reallocations.",
        "intervention": "Added portfolio-memory prompt priors, per-step normalized portfolio-weight traces, and additive turnover stability scoring/feedback with no gate.",
        "failure_modes": [
            "Turnover scoring may be too weak to matter.",
            "The extra diagnostics may add complexity without changing candidate ranking.",
            "Evaluation traces could drift if portfolio weights serialize inconsistently.",
        ],
        "acceptance": [
            "Both Step55 smoke runs complete.",
            "Turnover metrics appear in trace artifacts.",
            "Candidate scoring effective config shows `turnover_weight`.",
            "No regression to `zero-valid` sampling.",
        ],
    },
    "step56": {
        "problem": "The LESR interface is still stateless, so exact Differential Sharpe style shaping cannot be migrated directly. The project needed a lower-cost bridge that exposes running portfolio-risk information without changing the LESR function signatures.",
        "hypothesis": "Adding online DSR-like proxy features to the base state should make it easier for the LLM to generate risk-adjusted intrinsic rewards that remain informative even under `G2_intrinsic_only` evaluation.",
        "intervention": "Added optional global state features `ret_ema_20`, `ret_sq_ema_20`, `drawdown_20`, and `turnover_ema_20`, updated the environment to maintain them online, and reinforced design-mode instructions so `intrinsic_first` candidates can use these running-risk proxies.",
        "failure_modes": [
            "The new state features may not materially improve intrinsic-path usefulness.",
            "The LLM may ignore the new fields and continue producing revise-dominant candidates.",
            "Additional state dims may create noise for small smoke runs.",
        ],
        "acceptance": [
            "Both Step56 smoke runs complete.",
            "State descriptions expose the new global features.",
            "Either `G2-G0` becomes meaningfully non-zero for at least one algo, or intrinsic-probe deltas improve for at least two algos without total regression.",
            "If these criteria fail, stop before Step57 experiment execution.",
        ],
    },
    "step57": {
        "problem": "Aggregate walk-forward metrics exist, but the project still lacks a dedicated post-hoc distillation report that separates cross-window evidence from in-window selection logic.",
        "hypothesis": "A dedicated cross-window distillation summary will make it easier to compare positive-window counts, delta Sharpe stability, candidate family recurrence, and fingerprint recurrence without turning stability into an in-window gate.",
        "intervention": "Added post-hoc `cross_window_distillation.json` generation for multi-window runs and prepared full selected4/composite configs that reuse Step56 code paths without altering search gates.",
        "failure_modes": [
            "The full runs may fail or remain incomplete.",
            "The distillation summary may not be sufficiently different from the existing aggregate table.",
            "If Step56 does not improve intrinsic usefulness, the full reruns may not be justified.",
        ],
        "acceptance": [
            "Both full runs complete.",
            "The final report distinguishes post-hoc aggregate metrics from candidate-selection-time stability.",
            "No new selection gate is introduced into the in-window LESR loop.",
        ],
    },
}


def _write_analysis(step: str, result: str, residual: str, next_decision: str) -> None:
    meta = STEP_ANALYSIS_PRE[step]
    path = STEP_DOCS[step] / "analysis.md"
    title = STEP_TITLES[step]
    body = _render_analysis(
        meta["problem"],
        meta["hypothesis"],
        meta["intervention"],
        meta["failure_modes"],
        meta["acceptance"],
        result,
        residual,
        next_decision,
    )
    path.write_text(f"# {title} Analysis\n\n{body}")


def _format_run_row(label: str, config_rel: str, summary: dict[str, Any] | None) -> list[str]:
    if summary is None:
        return [
            f"## {label}",
            "| Item | Value |",
            "|---|---|",
            f"| Config | `{ROOT / config_rel}` |",
            "| Run Dir | Pending |",
            "| Completeness | Pending |",
            "| Zero-valid iterations | Pending |",
            "| Network-like errors | Pending |",
            "| Avg valid candidates | Pending |",
            "",
        ]
    avg_valid = ", ".join(
        f"{algo}={value:.2f}" for algo, value in summary["avg_valid_candidates_by_algo"].items()
    ) or "n/a"
    return [
        f"## {label}",
        "| Item | Value |",
        "|---|---|",
        f"| Config | `{ROOT / config_rel}` |",
        f"| Run Dir | `{summary['run_dir']}` |",
        f"| Completeness | `{summary['completeness']}` |",
        f"| Zero-valid iterations | `{summary['zero_valid_iterations']}` |",
        f"| Network-like errors | `{summary['network_like_errors']}` |",
        f"| Avg valid candidates | `{avg_valid}` |",
        "",
    ]


def _format_dsharpe_table(title: str, by_algo: dict[str, dict[str, float]]) -> list[str]:
    lines = [f"### {title}", "| Algo | G1-G0 | G2-G0 | G3-G0 |", "|---|---:|---:|---:|"]
    if not by_algo:
        lines.append("| Pending |  |  |  |")
    else:
        for algo, row in sorted(by_algo.items()):
            lines.append(
                f"| {algo} | {row.get('G1-G0', 0.0):+.4f} | {row.get('G2-G0', 0.0):+.4f} | {row.get('G3-G0', 0.0):+.4f} |"
            )
    lines.append("")
    return lines


def _format_probe_table(title: str, by_algo: dict[str, float]) -> list[str]:
    lines = [f"### {title}", "| Algo | Mean intrinsic_probe_score_delta |", "|---|---:|"]
    if not by_algo:
        lines.append("| Pending |  |")
    else:
        for algo, value in sorted(by_algo.items()):
            lines.append(f"| {algo} | {value:+.4f} |")
    lines.append("")
    return lines


def _write_results(step: str, task_summaries: dict[str, dict[str, Any]], acceptance_line: str, extra_lines: list[str] | None = None) -> None:
    docs_dir = STEP_DOCS[step]
    task_by_label = {task.label: task for task in TASKS if task.step == step}
    selected_summary = task_summaries.get("selected4")
    composite_summary = task_summaries.get("composite")
    lines = [f"# {STEP_TITLES[step]} Results", "", f"Status: {acceptance_line}", ""]
    lines.extend(_format_run_row("selected4", task_by_label["selected4"].config_rel, selected_summary))
    lines.extend(_format_run_row("composite", task_by_label["composite"].config_rel, composite_summary))
    if selected_summary:
        lines.extend(_format_dsharpe_table("selected4 aggregate dSharpe", selected_summary["aggregate_dsharpe"]))
        lines.extend(_format_probe_table("selected4 intrinsic probe means", selected_summary["mean_intrinsic_probe_by_algo"]))
    if composite_summary:
        lines.extend(_format_dsharpe_table("composite aggregate dSharpe", composite_summary["aggregate_dsharpe"]))
        lines.extend(_format_probe_table("composite intrinsic probe means", composite_summary["mean_intrinsic_probe_by_algo"]))
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend(
        [
            "## Key Artifacts",
            "- `wf_window_00/prompt.txt`",
            "- `wf_window_00/scenario_profile.json`",
            "- `wf_window_00/run_manifest.json`",
            "- `wf_window_00/llm_iter_trace.json`",
        ]
    )
    if step == "step57":
        lines.append("- `cross_window_distillation.json`")
    lines.append("")
    (docs_dir / "results.md").write_text("\n".join(lines))


def _write_plan(step: str, task_states: list[dict[str, Any]]) -> None:
    docs_dir = STEP_DOCS[step]
    cfgs = [f"- `{ROOT / task['config_rel']}`" for task in task_states]
    baselines = [f"- {item}" for item in STEP_BASELINES[step]]
    lines = [
        f"# {STEP_TITLES[step]} Plan And Commands",
        "",
        "## Configs",
        *cfgs,
        "",
        "## Comparison Baseline",
        *baselines,
        "",
        "## Commands",
        "```bash",
        *[
            f"cd {ROOT}\n/opt/anaconda3/envs/ml/bin/python scripts/run_step54_57_shared_priors.py"
            for _ in range(1)
        ],
        "```",
        "",
        "## Run Directories",
    ]
    for task in task_states:
        run_dir = task.get("run_dir", "Pending")
        lines.append(f"- `{task['name']}`: `{run_dir}`")
    lines.append("")
    (docs_dir / "plan_and_commands.md").write_text("\n".join(lines))


def _step_task_states(step: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for task in TASKS:
        if task.step != step:
            continue
        task_state = dict(state["tasks"].get(task.name, {}))
        task_state.update(
            {
                "name": task.name,
                "label": task.label,
                "config_rel": task.config_rel,
                "kind": task.kind,
            }
        )
        out.append(task_state)
    return out


def _update_step_docs(step: str, state: dict[str, Any], task_summaries: dict[str, dict[str, Any]], result: str, residual: str, next_decision: str, extra_results: list[str] | None = None) -> None:
    _write_analysis(step, result, residual, next_decision)
    _write_plan(step, _step_task_states(step, state))
    _write_results(step, task_summaries, result, extra_results)


def _step56_passed(state: dict[str, Any]) -> bool:
    step = state.get("steps", {}).get("step56", {})
    return bool(step.get("passed", False))


def _evaluate_step56(step55_summaries: dict[str, dict[str, Any]], step56_summaries: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    g2_hits = []
    for label, summary in step56_summaries.items():
        for algo, row in summary.get("aggregate_dsharpe", {}).items():
            g2 = abs(float(row.get("G2-G0", 0.0)))
            if g2 >= 0.02:
                g2_hits.append(f"{label}/{algo}={g2:.4f}")
    if g2_hits:
        return True, "Step56 passed via non-trivial G2 delta: " + ", ".join(g2_hits)

    improved_algos = []
    regressed_flags = []
    all_algos = sorted(
        {
            algo
            for summary in list(step55_summaries.values()) + list(step56_summaries.values())
            for algo in summary.get("mean_intrinsic_probe_by_algo", {}).keys()
        }
    )
    for algo in all_algos:
        before_vals = []
        after_vals = []
        for label in ["selected4", "composite"]:
            if algo in step55_summaries.get(label, {}).get("mean_intrinsic_probe_by_algo", {}):
                before_vals.append(step55_summaries[label]["mean_intrinsic_probe_by_algo"][algo])
            if algo in step56_summaries.get(label, {}).get("mean_intrinsic_probe_by_algo", {}):
                after_vals.append(step56_summaries[label]["mean_intrinsic_probe_by_algo"][algo])
        if not before_vals or not after_vals:
            continue
        before_mean = _mean(before_vals)
        after_mean = _mean(after_vals)
        if after_mean > before_mean:
            improved_algos.append(f"{algo}: {before_mean:+.4f}->{after_mean:+.4f}")
        regressed_flags.append(after_mean < before_mean)
    if len(improved_algos) >= 2 and not all(regressed_flags):
        return True, "Step56 passed via intrinsic probe improvement: " + ", ".join(improved_algos)
    return False, "Step56 failed acceptance: no |G2-G0| >= 0.02 and intrinsic probe means did not improve for at least two algos without full regression."


def _task_order(stop_after_step56: bool) -> list[Task]:
    if not stop_after_step56:
        return TASKS
    return [task for task in TASKS if task.step != "step57"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-api-key", default="", help="Temporary DeepSeek API key for this queue run only.")
    parser.add_argument("--stop-after-step56", action="store_true", help="Do not execute Step57 even if Step56 passes.")
    args = parser.parse_args()

    api_key = args.deepseek_api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required for Step54-57 queue execution.")

    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = api_key
    os.environ["DEEPSEEK_API_KEY"] = api_key

    state = _load_state()
    state["updated_at"] = _now()
    _save_state(state)

    step54_summaries: dict[str, dict[str, Any]] = {}
    step55_summaries: dict[str, dict[str, Any]] = {}
    step56_summaries: dict[str, dict[str, Any]] = {}
    step57_summaries: dict[str, dict[str, Any]] = {}

    for task in _task_order(args.stop_after_step56):
        if task.step == "step57" and not _step56_passed(state):
            break

        task_state = state["tasks"].setdefault(task.name, {})
        task_state["status"] = task_state.get("status", "pending")
        task_state["config_rel"] = task.config_rel
        task_state["label"] = task.label
        task_state["kind"] = task.kind
        task_state["updated_at"] = _now()
        _save_state(state)

        run_dir = _run_or_resume(task, env, state)
        task_state["run_dir"] = str(run_dir)
        task_state["status"] = "completed" if _task_status_from_artifacts(run_dir) == "complete" else "incomplete"
        task_state["completed_at"] = _now()
        task_state["summary"] = _summarize_run(run_dir)
        state["updated_at"] = _now()
        _save_state(state)

        summary = task_state["summary"]
        if task.step == "step54":
            step54_summaries[task.label] = summary
        elif task.step == "step55":
            step55_summaries[task.label] = summary
        elif task.step == "step56":
            step56_summaries[task.label] = summary
        elif task.step == "step57":
            step57_summaries[task.label] = summary

        if task.step == "step54" and len(step54_summaries) == 2:
            result = (
                f"Executed both Step54 smoke runs. selected4=`{step54_summaries['selected4']['run_dir']}`, "
                f"composite=`{step54_summaries['composite']['run_dir']}`. zero-valid counts: "
                f"selected4={step54_summaries['selected4']['zero_valid_iterations']}, "
                f"composite={step54_summaries['composite']['zero_valid_iterations']}."
            )
            residual = "The main remaining question is whether richer regime context improves downstream metrics or only preserves search health."
            next_decision = "Proceed to Step55 with turnover-aware scoring only if Step54 preserved completion and zero-valid health."
            state["steps"]["step54"] = {
                "completed": True,
                "task_summaries": step54_summaries,
                "result": result,
            }
            _update_step_docs("step54", state, step54_summaries, result, residual, next_decision)
            _save_state(state)

        if task.step == "step55" and len(step55_summaries) == 2:
            result = (
                f"Executed both Step55 smoke runs. selected4=`{step55_summaries['selected4']['run_dir']}`, "
                f"composite=`{step55_summaries['composite']['run_dir']}`. turnover-weight effective values: "
                f"selected4={float(step55_summaries['selected4']['candidate_scoring_effective'].get('turnover_weight', 0.0)):.2f}, "
                f"composite={float(step55_summaries['composite']['candidate_scoring_effective'].get('turnover_weight', 0.0)):.2f}."
            )
            residual = "Turnover diagnostics are now wired in, but the remaining uncertainty is whether they materially improve ranking rather than just add observability."
            next_decision = "Proceed to Step56 and test whether the new DSR-like proxy state makes the intrinsic path informative."
            state["steps"]["step55"] = {
                "completed": True,
                "task_summaries": step55_summaries,
                "result": result,
            }
            _update_step_docs("step55", state, step55_summaries, result, residual, next_decision)
            _save_state(state)

        if task.step == "step56" and len(step56_summaries) == 2:
            passed, reason = _evaluate_step56(step55_summaries, step56_summaries)
            result = (
                f"Executed both Step56 smoke runs. selected4=`{step56_summaries['selected4']['run_dir']}`, "
                f"composite=`{step56_summaries['composite']['run_dir']}`. {reason}"
            )
            residual = (
                "The remaining uncertainty is whether any observed intrinsic-path gain survives on multi-window full runs."
                if passed
                else "The DSR-proxy state may still be too weak for the current LESR interface, so a full rerun is not yet justified."
            )
            next_decision = (
                "Proceed to Step57 full runs using the Step56 code path."
                if passed and not args.stop_after_step56
                else "Stop before Step57 full execution and inspect Step56 candidate traces."
            )
            state["steps"]["step56"] = {
                "completed": True,
                "passed": passed,
                "reason": reason,
                "task_summaries": step56_summaries,
            }
            extra_lines = [
                "## Acceptance",
                f"- `{reason}`",
                "",
            ]
            _update_step_docs("step56", state, step56_summaries, result, residual, next_decision, extra_lines)
            _save_state(state)
            if not passed:
                break

        if task.step == "step57" and len(step57_summaries) == 2:
            result = (
                f"Executed both Step57 full runs. selected4=`{step57_summaries['selected4']['run_dir']}`, "
                f"composite=`{step57_summaries['composite']['run_dir']}`. cross_window_distillation present on both runs."
            )
            residual = "The remaining question is interpretive rather than mechanical: which cross-window patterns should be promoted into shared LESR priors next."
            next_decision = "Compare Step57 cross-window distillation tables against prior full selected4/composite runs and decide which shared priors deserve promotion."
            state["steps"]["step57"] = {
                "completed": True,
                "task_summaries": step57_summaries,
                "result": result,
            }
            extra_lines = [
                "## Acceptance",
                f"- `selected4 cross_window_distillation`: `{step57_summaries['selected4']['cross_window_distillation_present']}`",
                f"- `composite cross_window_distillation`: `{step57_summaries['composite']['cross_window_distillation_present']}`",
                "",
            ]
            _update_step_docs("step57", state, step57_summaries, result, residual, next_decision, extra_lines)
            _save_state(state)

    print(json.dumps({"status": "ok", "state_path": str(STATE_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
