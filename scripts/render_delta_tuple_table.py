import argparse
from pathlib import Path

import pandas as pd


ALGOS = ["a2c", "ppo", "sac", "td3"]
GROUPS = [
    ("G1", "G1_revise_only"),
    ("G2", "G2_intrinsic_only"),
    ("G3", "G3_revise_intrinsic"),
]
METRIC_LABELS = {
    "Sharpe_mean": "Sharpe",
    "CR_mean": "CR",
    "MDD_mean": "MDD",
    "AV_mean": "AV",
    "Score_mean": "Score (Sharpe + CR)",
    "intrinsic_mean": "Intrinsic",
    "intrinsic_w_effective_mean": "Intrinsic (w_effective)",
}


def _format_float(value: float) -> str:
    return f"{float(value):+0.4f}"


def _tuple_str(values: tuple[float, float, float]) -> str:
    return f"({_format_float(values[0])},{_format_float(values[1])},{_format_float(values[2])})"


def _resolve_run_dir(run_dir: str) -> Path:
    path = Path(run_dir)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _metric_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "Score_mean":
        return frame["Sharpe_mean"] + frame["CR_mean"]
    return frame[metric]


def _metric_delta_table(df: pd.DataFrame, metric: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    windows = (
        df.loc[df["window_name"] != "aggregate", ["window_index", "window_name"]]
        .drop_duplicates()
        .sort_values(["window_index", "window_name"])
    )
    ordered_windows = list(windows["window_name"].astype(str)) + ["aggregate"]
    table: dict[str, dict[str, tuple[float, float, float]]] = {}
    for window_name in ordered_windows:
        table[window_name] = {}
        window_df = df[df["window_name"] == window_name]
        for algo in ALGOS:
            algo_df = window_df[window_df["algorithm"] == algo]
            baseline = algo_df[algo_df["group"] == "G0_baseline"]
            if baseline.empty:
                deltas = (0.0, 0.0, 0.0)
            else:
                g0 = float(_metric_series(baseline, metric).iloc[0])
                vals = []
                for _short_name, group_name in GROUPS:
                    group_row = algo_df[algo_df["group"] == group_name]
                    if group_row.empty:
                        vals.append(0.0)
                    else:
                        vals.append(float(_metric_series(group_row, metric).iloc[0]) - g0)
                deltas = tuple(vals)
            table[window_name][algo] = deltas
    return table


def _render_metric_section(metric: str, delta_table: dict[str, dict[str, tuple[float, float, float]]]) -> list[str]:
    label = METRIC_LABELS.get(metric, metric)
    lines = [
        f"## {label}",
        "",
        "| window | a2c | ppo | sac | td3 |",
        "|---|---|---|---|---|",
    ]
    ordered_windows = [name for name in delta_table.keys() if name != "aggregate"] + ["aggregate"]
    for window_name in ordered_windows:
        row = [window_name]
        for algo in ALGOS:
            row.append(_tuple_str(delta_table[window_name][algo]))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_markdown(
    run_dir: Path,
    metric_tables: list[tuple[str, dict[str, dict[str, tuple[float, float, float]]]]],
) -> str:
    try:
        display_run = run_dir.relative_to(Path.cwd()).as_posix()
    except ValueError:
        display_run = run_dir.as_posix()
    lines = [
        f"run:",
        f"`{display_run}`",
        "",
        "Cell format: `(G1-G0,G2-G0,G3-G0)`.",
        "",
        "Interpretation note: for `MDD`, more negative delta means improvement.",
        "",
    ]
    for idx, (metric, delta_table) in enumerate(metric_tables):
        if idx:
            lines.append("")
        lines.extend(_render_metric_section(metric, delta_table))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a walk-forward delta tuple Markdown table for a run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing walk_forward_metrics_table.csv")
    parser.add_argument("--metric", default="Sharpe_mean", help="Metric column to compare against G0")
    parser.add_argument("--metrics", nargs="+", default=None, help="Optional metric list for a multi-section report")
    parser.add_argument("--out", help="Optional markdown output path")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    table_path = run_dir / "walk_forward_metrics_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing metrics table: {table_path}")

    df = pd.read_csv(table_path)
    selected_metrics = args.metrics or [args.metric]
    metric_tables = [(metric, _metric_delta_table(df, metric)) for metric in selected_metrics]
    markdown = _render_markdown(run_dir, metric_tables)

    if args.out:
        out_path = _resolve_run_dir(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
