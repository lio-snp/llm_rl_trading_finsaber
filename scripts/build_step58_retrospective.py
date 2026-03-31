#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/liuyanlinsnp/Desktop/LLM升级路/llm_rl_trading')
DOCS_DIR = ROOT / 'docs/steps/step_58_all_algo_positive_consensus'
OUTPUT_JSON = DOCS_DIR / 'retrospective_summary.json'
OUTPUT_MD = DOCS_DIR / 'results.md'

RUNS = {
    'selected4': {
        'run_dir': ROOT / 'runs/20260310_081640_131_508c_demo',
        'old_run_dir': ROOT / 'runs/20260305_163250_845_c008_demo',
        'label': 'selected4',
    },
    'composite': {
        'run_dir': ROOT / 'runs/20260311_010504_184_70a1_demo',
        'old_run_dir': ROOT / 'runs/20260302_140511_375_b1b3_demo',
        'label': 'composite',
    },
}

GROUPS = ['G1_revise_only', 'G2_intrinsic_only', 'G3_revise_intrinsic']
GROUP_SHORT = {
    'G1_revise_only': 'G1',
    'G2_intrinsic_only': 'G2',
    'G3_revise_intrinsic': 'G3',
}
ALGOS = ['a2c', 'ppo', 'sac', 'td3']
NETWORK_TOKENS = (
    'urlerror',
    'timeout',
    'timed out',
    'connection reset',
    'remote disconnected',
    'proxyerror',
    'connection aborted',
    'apiconnectionerror',
    'ssl',
)
LITERATURE = [
    {
        'title': 'A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem',
        'url': 'https://arxiv.org/abs/1706.10059',
        'lever': 'portfolio memory / holdings vector memory',
        'note': 'Supports explicit portfolio-memory features so revise_state can react to current exposure instead of only price signals.',
    },
    {
        'title': 'DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding',
        'url': 'https://doi.org/10.1609/aaai.v35i1.16144',
        'lever': 'market-condition embedding / regime conditioning',
        'note': 'Supports keeping regime variables explicit instead of collapsing everything into one routed family label.',
    },
    {
        'title': 'Towards Representation Learning for Cross-Sectional Portfolio Construction',
        'url': 'https://openreview.net/forum?id=1ZqJZNuVDY',
        'lever': 'cross-sectional representation / dispersion',
        'note': 'Supports adding dispersion, breadth, and cross-asset context as semantic feature groups instead of raw opaque indices only.',
    },
    {
        'title': 'Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools',
        'url': 'https://arxiv.org/abs/2311.10801',
        'lever': 'representation portability across changing universes',
        'note': 'Supports keeping shared state libraries pool-aware rather than rebuilding algo-specific state branches.',
    },
    {
        'title': 'Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization',
        'url': 'https://arxiv.org/abs/2602.17098',
        'lever': 'risk-adjusted intrinsic / running risk state',
        'note': 'Supports Step56-style running risk proxies and turnover-aware stability rather than raw return-only shaping.',
    },
]


def _safe_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _format_float(val: float) -> str:
    return f'{val:+0.4f}'


def _metric_delta_table(df: pd.DataFrame, metric: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for window_name, window_df in df.groupby('window_name'):
        per_algo: dict[str, tuple[float, float, float]] = {}
        for algo in ALGOS:
            algo_df = window_df[window_df['algorithm'] == algo]
            if algo_df.empty:
                continue
            sub = algo_df.set_index('group')
            baseline = float(sub.loc['G0_baseline', metric])
            per_algo[algo] = tuple(float(sub.loc[group, metric]) - baseline for group in GROUPS)
        out[str(window_name)] = per_algo
    return out


def _positive_window_count(delta_table: dict[str, dict[str, tuple[float, float, float]]]) -> dict[str, dict[str, int]]:
    counts = {algo: {'G1': 0, 'G2': 0, 'G3': 0} for algo in ALGOS}
    for window_name, per_algo in delta_table.items():
        if window_name == 'aggregate':
            continue
        for algo, vals in per_algo.items():
            for idx, short in enumerate(['G1', 'G2', 'G3']):
                if float(vals[idx]) > 0.0:
                    counts[algo][short] += 1
    return counts


def _zero_valid_stats(run_dir: Path) -> tuple[int, int, dict[str, int]]:
    total = 0
    zero = 0
    by_window: dict[str, int] = {}
    for window_dir in sorted(run_dir.glob('wf_window_*')):
        trace_path = window_dir / 'llm_iter_trace.json'
        if not trace_path.exists():
            continue
        rows = _safe_json(trace_path)
        cur_zero = 0
        for row in rows:
            total += 1
            valid = sum(1 for cand in row.get('candidates', []) if cand.get('valid'))
            if valid == 0:
                zero += 1
                cur_zero += 1
        by_window[window_dir.name] = cur_zero
    return zero, total, by_window


def _error_stats(run_dir: Path) -> dict[str, Any]:
    total = 0
    network = 0
    top_error_types = Counter()
    network_examples: list[dict[str, Any]] = []
    for window_dir in sorted(run_dir.glob('wf_window_*')):
        err_path = window_dir / 'llm_errors.json'
        if not err_path.exists():
            continue
        rows = _safe_json(err_path)
        total += len(rows)
        for row in rows:
            err_type = str(row.get('error_type', ''))
            top_error_types[err_type] += 1
            hay = f"{err_type} {row.get('message', '')}".lower()
            if any(token in hay for token in NETWORK_TOKENS):
                network += 1
                if len(network_examples) < 5:
                    network_examples.append(
                        {
                            'window': window_dir.name,
                            'iteration': row.get('iteration'),
                            'error_type': err_type,
                            'message': str(row.get('message', '')),
                        }
                    )
    return {
        'total_errors': total,
        'network_like_errors': network,
        'top_error_types': dict(top_error_types.most_common(10)),
        'network_examples': network_examples,
    }


def _fallback_stats(run_dir: Path) -> dict[str, Any]:
    total = 0
    by_algo = Counter()
    by_window: dict[str, dict[str, str]] = {}
    for window_dir in sorted(run_dir.glob('wf_window_*')):
        manifest_path = window_dir / 'run_manifest.json'
        if not manifest_path.exists():
            continue
        manifest = _safe_json(manifest_path)
        bad: dict[str, str] = {}
        for algo, name in (manifest.get('best_candidate_by_algo') or {}).items():
            candidate_name = str(name)
            if any(tag in candidate_name for tag in ('fallback', 'static', 'fixed')):
                total += 1
                by_algo[str(algo)] += 1
                bad[str(algo)] = candidate_name
        if bad:
            by_window[window_dir.name] = bad
    return {
        'fallback_best_total': total,
        'fallback_best_by_algo': dict(by_algo),
        'fallback_best_by_window': by_window,
    }


def _family_router_stats(run_dir: Path) -> dict[str, int]:
    counter = Counter()
    for window_dir in sorted(run_dir.glob('wf_window_*')):
        path = window_dir / 'scenario_profile.json'
        if path.exists():
            counter[str(_safe_json(path).get('family', ''))] += 1
    return dict(counter)


def _distillation_stats(run_dir: Path) -> dict[str, Any]:
    path = run_dir / 'cross_window_distillation.json'
    if not path.exists():
        return {'algorithms': {}, 'official_shared_cores': None}
    payload = _safe_json(path)
    return {
        'algorithms': payload.get('algorithms', {}),
        'official_shared_cores': payload.get('official_shared_cores'),
    }


@dataclass
class RunSummary:
    label: str
    run_dir: Path
    old_run_dir: Path
    dsharpe: dict[str, dict[str, tuple[float, float, float]]]
    dscore: dict[str, dict[str, tuple[float, float, float]]]
    dcr: dict[str, dict[str, tuple[float, float, float]]]
    dmdd: dict[str, dict[str, tuple[float, float, float]]]
    dav: dict[str, dict[str, tuple[float, float, float]]]
    positive_window_count: dict[str, dict[str, int]]
    zero_valid_total: int
    zero_valid_denom: int
    zero_valid_by_window: dict[str, int]
    error_stats: dict[str, Any]
    fallback_stats: dict[str, Any]
    family_router_stats: dict[str, int]
    distillation_stats: dict[str, Any]
    old_aggregate_dsharpe: dict[str, tuple[float, float, float]]


def _load_run_summary(label: str, run_dir: Path, old_run_dir: Path) -> RunSummary:
    df = pd.read_csv(run_dir / 'walk_forward_metrics_table.csv')
    old_df = pd.read_csv(old_run_dir / 'walk_forward_metrics_table.csv')
    dsharpe = _metric_delta_table(df, 'Sharpe_mean')
    dcr = _metric_delta_table(df, 'CR_mean')
    dmdd = _metric_delta_table(df, 'MDD_mean')
    dav = _metric_delta_table(df, 'AV_mean')
    dscore = {}
    for window_name in dsharpe:
        dscore[window_name] = {}
        for algo in ALGOS:
            dscore[window_name][algo] = tuple(
                float(dsharpe[window_name][algo][i]) + float(dcr[window_name][algo][i])
                for i in range(3)
            )
    old_dsharpe = _metric_delta_table(old_df, 'Sharpe_mean').get('aggregate', {})
    zero_valid_total, zero_valid_denom, zero_valid_by_window = _zero_valid_stats(run_dir)
    return RunSummary(
        label=label,
        run_dir=run_dir,
        old_run_dir=old_run_dir,
        dsharpe=dsharpe,
        dscore=dscore,
        dcr=dcr,
        dmdd=dmdd,
        dav=dav,
        positive_window_count=_positive_window_count(dsharpe),
        zero_valid_total=zero_valid_total,
        zero_valid_denom=zero_valid_denom,
        zero_valid_by_window=zero_valid_by_window,
        error_stats=_error_stats(run_dir),
        fallback_stats=_fallback_stats(run_dir),
        family_router_stats=_family_router_stats(run_dir),
        distillation_stats=_distillation_stats(run_dir),
        old_aggregate_dsharpe=old_dsharpe,
    )


def _tuple_str(vals: tuple[float, float, float]) -> str:
    return f"({_format_float(vals[0])},{_format_float(vals[1])},{_format_float(vals[2])})"


def _markdown_delta_table(title: str, delta_table: dict[str, dict[str, tuple[float, float, float]]]) -> list[str]:
    lines = [f'### {title}', '| window | a2c | ppo | sac | td3 |', '|---|---|---|---|---|']
    order = [w for w in delta_table.keys() if w != 'aggregate'] + ['aggregate']
    for window_name in order:
        per_algo = delta_table[window_name]
        lines.append(
            '| ' + window_name + ' | ' + ' | '.join(_tuple_str(per_algo[algo]) for algo in ALGOS) + ' |'
        )
    return lines


def _best_group_success(delta_table: dict[str, dict[str, tuple[float, float, float]]], counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    out = {}
    aggregate = delta_table['aggregate']
    for algo in ALGOS:
        best_idx = max(range(3), key=lambda idx: float(aggregate[algo][idx]))
        best_group = ['G1', 'G2', 'G3'][best_idx]
        out[algo] = {
            'best_group': best_group,
            'aggregate_dsharpe': float(aggregate[algo][best_idx]),
            'positive_windows': int(counts[algo][best_group]),
            'aggregate_positive': bool(float(aggregate[algo][best_idx]) > 0.0),
            'stability_pass_3of5': bool(int(counts[algo][best_group]) >= 3),
        }
    return out


def _retrospective_json(summaries: dict[str, RunSummary]) -> dict[str, Any]:
    payload: dict[str, Any] = {'runs': {}, 'literature': LITERATURE}
    for label, summary in summaries.items():
        payload['runs'][label] = {
            'run_dir': str(summary.run_dir),
            'old_run_dir': str(summary.old_run_dir),
            'dsharpe': summary.dsharpe,
            'dscore': summary.dscore,
            'dcr': summary.dcr,
            'dmdd': summary.dmdd,
            'dav': summary.dav,
            'positive_window_count': summary.positive_window_count,
            'zero_valid_total': summary.zero_valid_total,
            'zero_valid_denom': summary.zero_valid_denom,
            'zero_valid_by_window': summary.zero_valid_by_window,
            'error_stats': summary.error_stats,
            'fallback_stats': summary.fallback_stats,
            'family_router_stats': summary.family_router_stats,
            'distillation_stats': summary.distillation_stats,
            'old_aggregate_dsharpe': summary.old_aggregate_dsharpe,
            'best_group_success': _best_group_success(summary.dsharpe, summary.positive_window_count),
        }
    return payload


def _diagnosis_lines(summary: RunSummary) -> list[str]:
    lines: list[str] = []
    aggregate = summary.dsharpe['aggregate']
    best_success = _best_group_success(summary.dsharpe, summary.positive_window_count)
    if summary.label == 'selected4':
        lines.append('- Relative to the older selected4 full baseline, `sac` and `td3` improved sharply at aggregate dSharpe level, but `a2c` still fails the true goal because its best LESR group is only mildly positive and unstable across windows.')
        lines.append('- The real blocker is promotion quality, not connectivity: `zero-valid` stayed high and fallback bests still entered official window-level winners for `ppo` and `td3`.')
    else:
        lines.append('- Composite is less negative overall because `a2c`, `sac`, and `td3` each have at least one positive aggregate LESR group, but it is still not consensus-positive because `ppo` remains negative on `G1` and `G2`, with only a marginally positive `G3`.')
        lines.append('- Composite also shows more network-like noise than selected4, but validation failure and duplicate-candidate errors still dominate the error mass, so connectivity is not the only explanation.')
    if summary.family_router_stats == {'risk_shield': 5}:
        lines.append('- All windows were routed to `risk_shield`, so the router did not create real mechanism diversity. The later promotion layer had to work with a narrow family distribution from the start.')
    if summary.fallback_stats['fallback_best_total']:
        lines.append(
            f"- Fallback/static contamination is non-trivial: total fallback bests = `{summary.fallback_stats['fallback_best_total']}` with windows {summary.fallback_stats['fallback_best_by_window']}."
        )
    weak = [algo for algo, row in best_success.items() if not row['aggregate_positive'] or not row['stability_pass_3of5']]
    if weak:
        lines.append(f"- The current acceptance target still fails for: `{', '.join(weak)}`.")
    return lines


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = ['# Step58 Retrospective And All-Four-Positive Baseline Report', '']
    lines.append('This report is built from the completed Step57 full runs and their older full-run comparators. It is the fixed baseline for the next consensus-promotion iteration.')
    lines.append('')
    lines.append('## Baselines')
    for label, run in payload['runs'].items():
        lines.append(f"- `{label}`: `{run['run_dir']}`")
        lines.append(f"- `{label}` old comparator: `{run['old_run_dir']}`")
    lines.append('')
    lines.append('## Run-Level Summary')
    lines.append('| run | zero-valid | fallback bests | network-like errors | router families |')
    lines.append('|---|---:|---:|---:|---|')
    for label, run in payload['runs'].items():
        families = ', '.join(f'{k}:{v}' for k, v in run['family_router_stats'].items())
        lines.append(
            f"| {label} | {run['zero_valid_total']}/{run['zero_valid_denom']} | {run['fallback_stats']['fallback_best_total']} | {run['error_stats']['network_like_errors']} | {families} |"
        )
    for label, run in payload['runs'].items():
        lines.append('')
        lines.append(f'## {label}')
        summary = RunSummary(
            label=label,
            run_dir=Path(run['run_dir']),
            old_run_dir=Path(run['old_run_dir']),
            dsharpe=run['dsharpe'],
            dscore=run['dscore'],
            dcr=run['dcr'],
            dmdd=run['dmdd'],
            dav=run['dav'],
            positive_window_count=run['positive_window_count'],
            zero_valid_total=run['zero_valid_total'],
            zero_valid_denom=run['zero_valid_denom'],
            zero_valid_by_window=run['zero_valid_by_window'],
            error_stats=run['error_stats'],
            fallback_stats=run['fallback_stats'],
            family_router_stats=run['family_router_stats'],
            distillation_stats=run['distillation_stats'],
            old_aggregate_dsharpe=run['old_aggregate_dsharpe'],
        )
        lines.extend(_markdown_delta_table(f'{label} dSharpe', summary.dsharpe))
        lines.append('')
        lines.extend(_markdown_delta_table(f'{label} dScore (Sharpe + CR)', summary.dscore))
        lines.append('')
        lines.extend(_markdown_delta_table(f'{label} dCR', summary.dcr))
        lines.append('')
        lines.extend(_markdown_delta_table(f'{label} dMDD', summary.dmdd))
        lines.append('')
        lines.extend(_markdown_delta_table(f'{label} dAV', summary.dav))
        lines.append('')
        lines.append('### Positive Window Count (dSharpe)')
        lines.append('| algo | G1 | G2 | G3 | best_group | best_group_dSharpe | best_group_positive_windows | |')
        lines.append('|---|---:|---:|---:|---|---:|---:|---|')
        best_success = run['best_group_success']
        for algo in ALGOS:
            cnt = run['positive_window_count'][algo]
            best = best_success[algo]
            status = 'pass' if best['aggregate_positive'] and best['stability_pass_3of5'] else 'fail'
            lines.append(
                f"| {algo} | {cnt['G1']} | {cnt['G2']} | {cnt['G3']} | {best['best_group']} | {best['aggregate_dsharpe']:+0.4f} | {best['positive_windows']}/5 | {status} |"
            )
        lines.append('')
        lines.append('### Search Quality Diagnostics')
        lines.append(f"- zero-valid by window: `{run['zero_valid_by_window']}`")
        lines.append(f"- fallback bests by algo: `{run['fallback_stats']['fallback_best_by_algo']}`")
        lines.append(f"- fallback best windows: `{run['fallback_stats']['fallback_best_by_window']}`")
        lines.append(f"- top error types: `{run['error_stats']['top_error_types']}`")
        lines.append(f"- network-like error examples: `{run['error_stats']['network_examples']}`")
        lines.append('')
        lines.append('### Best Candidate Family / Design Frequencies')
        lines.append('| algo | family frequency | design frequency |')
        lines.append('|---|---|---|')
        for algo in ALGOS:
            algo_stats = run['distillation_stats']['algorithms'].get(algo, {})
            lines.append(
                f"| {algo} | `{algo_stats.get('best_candidate_family_frequency', {})}` | `{algo_stats.get('best_candidate_design_mode_frequency', {})}` |"
            )
        if run['distillation_stats'].get('official_shared_cores'):
            lines.append('')
            lines.append(f"- official shared cores: `{run['distillation_stats']['official_shared_cores']}`")
        else:
            lines.append('')
            lines.append('- official shared cores: not available on the Step57 baselines; these runs predate the new consensus-promotion path.')
        lines.append('')
        lines.append('### Diagnosis')
        lines.extend(_diagnosis_lines(summary))
        lines.append('')
        lines.append('### Old vs Current Aggregate dSharpe')
        lines.append('| algo | old `(G1,G2,G3)` | current `(G1,G2,G3)` |')
        lines.append('|---|---|---|')
        for algo in ALGOS:
            old_vals = tuple(run['old_aggregate_dsharpe'][algo])
            cur_vals = tuple(run['dsharpe']['aggregate'][algo])
            lines.append(f"| {algo} | `{_tuple_str(old_vals)}` | `{_tuple_str(cur_vals)}` |")
    lines.append('')
    lines.append('## Literature Appendix')
    lines.append('| paper | LESR lever | why it matters here |')
    lines.append('|---|---|---|')
    for item in payload['literature']:
        lines.append(f"| [{item['title']}]({item['url']}) | {item['lever']} | {item['note']} |")
    lines.append('')
    lines.append('## Decision')
    lines.append('- Keep `per-algo / per-window` search as exploration only.')
    lines.append('- Promote official candidates only through shared `state-core`, `intrinsic-core`, and `joint-pair` consensus.')
    lines.append('- Treat fallback/static outputs as debugging continuity, not official winners.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {
        label: _load_run_summary(label, config['run_dir'], config['old_run_dir'])
        for label, config in RUNS.items()
    }
    payload = _retrospective_json(summaries)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    OUTPUT_MD.write_text(_markdown_report(payload), encoding='utf-8')
    print(f'wrote {OUTPUT_JSON}')
    print(f'wrote {OUTPUT_MD}')


if __name__ == '__main__':
    main()
