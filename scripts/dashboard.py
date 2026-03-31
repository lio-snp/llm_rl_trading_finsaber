from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import discover_runs, format_timestamp, load_run_snapshot, read_csv_safe, read_json_safe, read_text_safe

RUN_STATE_ORDER = {
    "running": 0,
    "partial": 1,
    "completed": 2,
    "stale_or_failed": 3,
}
STATE_COLOR = {
    "running": "#f59e0b",
    "partial": "#2563eb",
    "completed": "#16a34a",
    "stale_or_failed": "#dc2626",
    "pending": "#6b7280",
    "running_llm": "#f59e0b",
    "running_rl": "#0ea5e9",
    "incomplete": "#ef4444",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--refresh-seconds", type=int, default=5)
    parser.add_argument("--stale-minutes", type=int, default=15)
    return parser.parse_args()


def status_badge(label: str, state: str) -> str:
    color = STATE_COLOR.get(state, "#6b7280")
    return (
        f"<span style='display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;"
        f"background:{color};color:white;font-size:0.85rem;font-weight:600'>{label}</span>"
    )


def sort_runs(runs: list[Any]) -> list[Any]:
    return sorted(
        runs,
        key=lambda run: (
            RUN_STATE_ORDER.get(run.state, 99),
            -(run.last_updated_ts or 0.0),
            run.run_id,
        ),
    )


def build_run_label(run: Any) -> str:
    return f"{run.run_id} | {run.state} | {format_timestamp(run.last_updated_ts)}"


def build_window_label(window: Any) -> str:
    split = " | ".join(f"{key}:{value}" for key, value in window.split.items() if value)
    suffix = f" | {split}" if split else ""
    return f"{window.display_name} | {window.state}{suffix}"


def config_preview(config: dict[str, Any]) -> dict[str, Any]:
    llm_cfg = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    walk_forward_cfg = config.get("walk_forward") if isinstance(config.get("walk_forward"), dict) else {}
    preview = {
        "data_source": config.get("data_source"),
        "task_description": config.get("task_description"),
        "window_setup": config.get("window_setup"),
        "algorithm": config.get("algorithm"),
        "eval_algorithms": config.get("eval_algorithms"),
        "assets": config.get("assets"),
        "llm": {
            key: llm_cfg.get(key)
            for key in ["enabled", "model", "k", "iterations", "iteration_mode", "branch_parallel_workers"]
            if key in llm_cfg
        },
        "walk_forward": {
            "enabled": walk_forward_cfg.get("enabled", False),
            "aggregate": walk_forward_cfg.get("aggregate"),
            "window_count": len(walk_forward_cfg.get("windows", []) or []),
        },
    }
    return {key: value for key, value in preview.items() if value not in [None, {}, []]}


def render_overview(run: Any) -> None:
    st.markdown(status_badge(run.state, run.state), unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Run", run.run_id)
    metric_cols[1].metric("Windows", str(run.total_windows))
    metric_cols[2].metric("Completed", str(run.completed_windows))
    metric_cols[3].metric("Last Update", format_timestamp(run.last_updated_ts))
    metric_cols[4].metric("Algorithms", ", ".join(run.algorithms) if run.algorithms else "-")

    progress_rows = [
        {
            "window": window.display_name,
            "state": window.state,
            "value": 1,
        }
        for window in run.windows
    ]
    if progress_rows:
        progress_df = pd.DataFrame(progress_rows)
        fig = px.bar(
            progress_df,
            x="window",
            y="value",
            color="state",
            color_discrete_map=STATE_COLOR,
            title="Window Completion",
        )
        fig.update_layout(showlegend=True, yaxis_visible=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if run.bound_processes:
        st.subheader("Bound Processes")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "pid": proc.pid,
                        "elapsed": proc.etime,
                        "command": proc.command,
                    }
                    for proc in run.bound_processes
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No bound `resume_walk_forward.py --run-dir ...` process found for this run.")

    if run.load_warnings:
        for warning in run.load_warnings:
            st.warning(warning)

    overview_cols = st.columns(2)
    with overview_cols[0]:
        st.subheader("Config Preview")
        st.json(config_preview(run.config), expanded=False)
    with overview_cols[1]:
        st.subheader("Aggregate Paths")
        st.json(
            {
                "path": str(run.path),
                "run_manifest": str(run.root_manifest_path) if run.root_manifest_path else None,
                "run_summary": str(run.summary_path) if run.summary_path else None,
                "metrics": str(run.metrics_path) if run.metrics_path else None,
                "walk_forward_summary": (
                    str(run.walk_forward_summary_path) if run.walk_forward_summary_path else None
                ),
                "walk_forward_metrics_table": (
                    str(run.walk_forward_table_path) if run.walk_forward_table_path else None
                ),
            },
            expanded=False,
        )

    if "root_manifest" in run.metadata:
        with st.expander("Root Manifest", expanded=False):
            st.json(run.metadata["root_manifest"], expanded=False)
    if "walk_forward_summary" in run.metadata:
        with st.expander("Walk-Forward Summary", expanded=False):
            st.json(run.metadata["walk_forward_summary"], expanded=False)


def build_windows_table(run: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in run.windows:
        row = {
            "window": window.display_name,
            "state": window.state,
            "train": window.split.get("train", "-"),
            "val": window.split.get("val", "-"),
            "test": window.split.get("test", "-"),
            "last_update": format_timestamp(window.last_updated_ts),
            "has_metrics": window.file_status.get("metrics", False),
            "has_trace": window.file_status.get("llm_iter_trace", False),
        }
        for algo in run.algorithms:
            latest_it = window.latest_iteration_by_algo.get(algo)
            row[f"{algo}_latest_it"] = "-" if latest_it is None else f"it{latest_it}"
        rows.append(row)
    return pd.DataFrame(rows)


def render_windows(run: Any) -> None:
    windows_df = build_windows_table(run)
    if windows_df.empty:
        st.info("No window data found.")
        return
    st.dataframe(windows_df, use_container_width=True, hide_index=True)


def reward_trace_frame(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return pd.DataFrame()
    for algo, groups in payload.items():
        if not isinstance(groups, dict):
            continue
        for group, seeds in groups.items():
            if not isinstance(seeds, list):
                continue
            for seed_payload in seeds:
                if not isinstance(seed_payload, dict):
                    continue
                rows.append(
                    {
                        "algorithm": algo,
                        "group": group,
                        "seed": seed_payload.get("seed"),
                        "reward_total_mean": _nested_mean(seed_payload, "reward_total"),
                        "reward_env_mean": _nested_mean(seed_payload, "reward_env"),
                        "intrinsic_mean": _nested_mean(seed_payload, "intrinsic"),
                        "intrinsic_ratio_mean": _nested_mean(seed_payload, "intrinsic_effect_ratio"),
                        "env_near_zero_ratio": seed_payload.get("env_near_zero_ratio"),
                    }
                )
    return pd.DataFrame(rows)


def policy_behavior_frame(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return pd.DataFrame()
    for algo, algo_payload in payload.items():
        if not isinstance(algo_payload, dict):
            continue
        overall = algo_payload.get("_overall", algo_payload)
        if not isinstance(overall, dict):
            continue
        rows.append(
            {
                "algorithm": algo,
                "near_bound_ratio_mean": overall.get("near_bound_ratio_mean")
                or overall.get("near_actor_ratio_mean"),
                "action_entropy_mean": overall.get("action_entropy_mean"),
                "avg_daily_portfolio_weight_change_mean": overall.get("avg_daily_portfolio_weight_change_mean"),
                "actor_collapse_detected": overall.get("actor_collapse_detected"),
            }
        )
    return pd.DataFrame(rows)


def action_saturation_frame(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return pd.DataFrame()
    for algo, algo_payload in payload.items():
        if not isinstance(algo_payload, dict):
            continue
        overall = algo_payload.get("_overall", algo_payload.get("summary", algo_payload))
        if not isinstance(overall, dict):
            continue
        rows.append(
            {
                "algorithm": algo,
                "near_actor_ratio_mean": overall.get("near_actor_ratio_mean"),
                "unique_action_count_mean": overall.get("unique_action_count_mean"),
                "sign_flip_rate_mean": overall.get("sign_flip_rate_mean"),
                "actor_collapse_detected": overall.get("actor_collapse_detected"),
            }
        )
    return pd.DataFrame(rows)


def render_window_detail(window: Any) -> None:
    st.markdown(status_badge(window.state, window.state), unsafe_allow_html=True)
    detail_cols = st.columns(4)
    detail_cols[0].metric("Window", window.display_name)
    detail_cols[1].metric("Last Update", format_timestamp(window.last_updated_ts))
    detail_cols[2].metric("Has Metrics", "yes" if window.file_status.get("metrics") else "no")
    detail_cols[3].metric("Has Trace", "yes" if window.file_status.get("llm_iter_trace") else "no")

    st.json(
        {
            "path": str(window.path),
            "split": window.split or {},
            "files": window.file_status,
        },
        expanded=False,
    )
    if window.load_warnings:
        for warning in window.load_warnings:
            st.warning(warning)

    if window.run_summary_path:
        run_summary_text, run_summary_error = read_text_safe(window.run_summary_path)
        if run_summary_error:
            st.warning(run_summary_error)
        elif run_summary_text:
            with st.expander("Run Summary", expanded=False):
                st.markdown(run_summary_text)

    metrics_df, metrics_error = read_csv_safe(window.metrics_table_path)
    if metrics_error:
        st.warning(metrics_error)
    elif metrics_df is not None and not metrics_df.empty:
        st.subheader("Metrics Table")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        metric_options = [column for column in ["Sharpe_mean", "CR_mean", "MDD_mean", "AV_mean", "intrinsic_mean"] if column in metrics_df.columns]
        if metric_options:
            metric_name = st.selectbox(
                "Metrics Chart",
                options=metric_options,
                key=f"metric_chart_{window.name}",
            )
            fig = px.bar(
                metrics_df,
                x="group",
                y=metric_name,
                color="algorithm" if "algorithm" in metrics_df.columns else None,
                barmode="group",
                title=f"{metric_name} by Group",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Metrics table not yet available / partial write.")

    reward_trace_payload, reward_trace_error = read_json_safe(window.reward_trace_path)
    if reward_trace_error:
        st.warning(reward_trace_error)
    reward_df = reward_trace_frame(reward_trace_payload)
    if not reward_df.empty:
        st.subheader("Reward Trace Summary")
        reward_metric = st.selectbox(
            "Reward Metric",
            options=["reward_total_mean", "reward_env_mean", "intrinsic_mean", "intrinsic_ratio_mean", "env_near_zero_ratio"],
            key=f"reward_metric_{window.name}",
        )
        fig = px.bar(
            reward_df,
            x="group",
            y=reward_metric,
            color="algorithm",
            barmode="group",
            title=f"{reward_metric} by Group",
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Reward Trace Table", expanded=False):
            st.dataframe(reward_df, use_container_width=True, hide_index=True)

    policy_payload, policy_error = read_json_safe(window.policy_behavior_path)
    if policy_error:
        st.warning(policy_error)
    policy_df = policy_behavior_frame(policy_payload)
    if not policy_df.empty:
        st.subheader("Policy Behavior Summary")
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
        fig = px.bar(
            policy_df,
            x="algorithm",
            y="near_bound_ratio_mean",
            color="actor_collapse_detected",
            title="Near-Bound Ratio by Algorithm",
        )
        st.plotly_chart(fig, use_container_width=True)

    action_payload, action_error = read_json_safe(window.action_saturation_path)
    if action_error:
        st.warning(action_error)
    action_df = action_saturation_frame(action_payload)
    if not action_df.empty:
        st.subheader("Action Saturation Summary")
        st.dataframe(action_df, use_container_width=True, hide_index=True)

    scenario_payload, scenario_error = read_json_safe(window.scenario_profile_path)
    if scenario_error:
        st.warning(scenario_error)
    elif scenario_payload is not None:
        with st.expander("Scenario Profile", expanded=False):
            st.json(scenario_payload, expanded=False)

    manifest_payload = window.metadata.get("run_manifest")
    if manifest_payload:
        with st.expander("Window Manifest", expanded=False):
            st.json(manifest_payload, expanded=False)


def render_iteration_detail(window: Any) -> None:
    has_any_iterations = any(window.iterations_by_algo.get(algo) for algo in window.algorithms)
    if not has_any_iterations:
        st.info("Iteration-level artifacts are not yet available for this window.")
        return

    for algo in window.algorithms:
        iterations = window.iterations_by_algo.get(algo, [])
        latest_iteration = window.latest_iteration_by_algo.get(algo)
        title = f"{algo.upper()} | latest={'-' if latest_iteration is None else f'it{latest_iteration}'} | {len(iterations)} iterations"
        with st.expander(title, expanded=False):
            if not iterations:
                st.caption("No per-iteration artifacts found for this algorithm yet.")
                continue
            summary_df = pd.DataFrame(
                [
                    {
                        "iteration": f"it{item.iteration}",
                        "candidate_count": item.candidate_count,
                        "response_count": item.response_count,
                        "error_count": item.error_count,
                        "has_dialog": item.has_dialog,
                        "has_finalized_trace": item.has_finalized_trace,
                        "last_update": format_timestamp(item.last_updated_ts),
                    }
                    for item in iterations
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            sorted_iterations = sorted(iterations, key=lambda current: current.iteration, reverse=True)
            selected_iteration_label = st.selectbox(
                "Iteration",
                options=[f"it{item.iteration}" for item in sorted_iterations],
                key=f"iteration_select_{window.name}_{algo}",
            )
            item = next(
                current
                for current in sorted_iterations
                if f"it{current.iteration}" == selected_iteration_label
            )

            if item.trace_entry:
                feedback = item.trace_entry.get("feedback")
                if feedback:
                    st.subheader("Feedback")
                    st.code(str(feedback))
                candidates = item.trace_entry.get("candidates", [])
                if isinstance(candidates, list) and candidates:
                    candidate_rows = [
                        {
                            "rank": candidate.get("rank"),
                            "name": candidate.get("name"),
                            "family": candidate.get("family"),
                            "design_mode": candidate.get("design_mode"),
                            "score": candidate.get("score"),
                            "valid": candidate.get("valid"),
                            "error": candidate.get("error"),
                        }
                        for candidate in candidates
                    ]
                    st.subheader("Candidates")
                    st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)
                    selected_candidate = choose_candidate(candidates)
                    if selected_candidate is not None and selected_candidate.get("code"):
                        st.subheader("Selected Candidate Code")
                        st.caption(selected_candidate.get("name", "candidate"))
                        st.code(str(selected_candidate["code"]), language="python")
                else:
                    st.info("Finalized candidate trace not yet available / partial write.")
            else:
                st.info("Finalized candidate trace not yet available / partial write.")

            st.subheader("Raw API Responses")
            if item.response_items:
                for index, response in enumerate(item.response_items, start=1):
                    label = (
                        f"Response {index} | attempt={response.get('attempt')} | index={response.get('index')} | "
                        f"family={response.get('family')} | mode={response.get('design_mode')}"
                    )
                    with st.container():
                        st.caption(label)
                        st.code(str(response.get("content", "")))
            elif window.llm_responses_path is None:
                st.info("Raw API responses not yet available / partial write.")
            else:
                st.caption("No API responses recorded for this iteration.")

            st.subheader("Errors")
            if item.error_items:
                st.dataframe(pd.DataFrame(item.error_items), use_container_width=True, hide_index=True)
            elif window.llm_errors_path is None:
                st.info("LLM error log not yet available / partial write.")
            else:
                st.caption("No errors recorded for this iteration.")

            st.subheader("Dialog")
            if item.dialog_path:
                dialog_text, dialog_error = read_text_safe(item.dialog_path)
                if dialog_error:
                    st.warning(dialog_error)
                elif dialog_text:
                    st.text_area(
                        f"dialog_it{item.iteration}",
                        value=dialog_text,
                        height=320,
                        key=f"dialog_{window.name}_{algo}_{item.iteration}",
                    )
            else:
                st.info("Dialog text not yet available / partial write.")


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = [candidate for candidate in candidates if isinstance(candidate.get("rank"), int)]
    if ranked:
        return sorted(ranked, key=lambda candidate: candidate["rank"])[0]
    scored = [candidate for candidate in candidates if isinstance(candidate.get("score"), (int, float))]
    if scored:
        return sorted(scored, key=lambda candidate: float(candidate["score"]), reverse=True)[0]
    return candidates[0] if candidates else None


def _nested_mean(payload: dict[str, Any], key: str) -> Any:
    nested = payload.get(key)
    if isinstance(nested, dict):
        return nested.get("mean")
    return None


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()

    st.set_page_config(page_title="LESR Run Dashboard", layout="wide")
    st.title("LESR Run Dashboard")
    st.caption(f"Runs root: `{runs_root}`")

    catalog = sort_runs(discover_runs(runs_root=runs_root, stale_minutes=args.stale_minutes))
    if not catalog:
        st.warning("No run directories found under the configured runs root.")
        return

    run_options = [run.run_id for run in catalog]
    default_run_id = st.session_state.get("selected_run_id", run_options[0])
    if default_run_id not in run_options:
        default_run_id = run_options[0]
    selected_run_id = st.sidebar.selectbox(
        "Run",
        options=run_options,
        index=run_options.index(default_run_id),
        format_func=lambda run_id: build_run_label(next(run for run in catalog if run.run_id == run_id)),
        key="selected_run_id",
    )

    selected_summary = next(run for run in catalog if run.run_id == selected_run_id)
    selected_run_summary = load_run_snapshot(
        run_dir=selected_summary.path,
        runs_root=runs_root,
        stale_minutes=args.stale_minutes,
        detailed=False,
    )
    window_names = [window.name for window in selected_run_summary.windows]
    default_window = st.session_state.get("selected_window_name", window_names[0] if window_names else None)
    if default_window not in window_names:
        default_window = window_names[0] if window_names else None
    if default_window is None:
        st.warning("Selected run has no window structure.")
        return

    selected_window_name = st.sidebar.selectbox(
        "Window",
        options=window_names,
        index=window_names.index(default_window),
        format_func=lambda name: build_window_label(next(window for window in selected_run_summary.windows if window.name == name)),
        key="selected_window_name",
    )

    if st.sidebar.button("Refresh now"):
        st.experimental_rerun()
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
    st.sidebar.caption(
        f"Polling every {args.refresh_seconds}s. Running windows update from files; API/trace JSON appears after later stages."
    )

    run = load_run_snapshot(
        run_dir=selected_summary.path,
        runs_root=runs_root,
        stale_minutes=args.stale_minutes,
        detailed=True,
        detailed_window_names={selected_window_name},
    )
    window = next(current for current in run.windows if current.name == selected_window_name)

    overview_tab, windows_tab, detail_tab, iteration_tab = st.tabs(
        ["Overview", "Windows", "Window Detail", "Iteration Detail"]
    )
    with overview_tab:
        render_overview(run)
    with windows_tab:
        render_windows(run)
    with detail_tab:
        render_window_detail(window)
    with iteration_tab:
        render_iteration_detail(window)

    if auto_refresh and args.refresh_seconds > 0:
        time.sleep(args.refresh_seconds)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
