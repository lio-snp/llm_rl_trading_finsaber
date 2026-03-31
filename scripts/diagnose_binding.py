from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml

from src.data.features import add_indicators
from src.data.finsaber_data import load_finsaber_prices
from src.env.state_schema import StateSchema
from src.env.trading_env import EnvConfig, TradingEnv
from src.utils.code_loader import load_functions_from_code

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _state_signature(state: np.ndarray) -> str:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.round(arr, 6)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _sanitize_scalar(value) -> float:
    try:
        out = float(value) if value is not None else 0.0
    except Exception:
        out = 0.0
    if not np.isfinite(out):
        out = 0.0
    return out


def _scale_intrinsic(value: float, mode: str) -> float:
    mode = (mode or "raw").lower()
    if mode == "bounded_100":
        return float(np.clip(value, -100.0, 100.0))
    if mode == "normalized":
        return float(np.tanh(value) * 100.0)
    return float(value)


def _align_dates_all_assets(df, assets: List[str]):
    out = df[df["asset"].isin(assets)].copy()
    if out.empty:
        return out
    counts = out.groupby("date")["asset"].nunique()
    valid_dates = counts[counts == len(assets)].index
    return out[out["date"].isin(valid_dates)].copy()


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-12)
    return float(np.sum(x * y) / denom)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    return float(1.0 - sse / sst)


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    if HAS_SKLEARN:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xts = scaler.transform(X_test)
        model = Ridge(alpha=float(alpha))
        model.fit(Xs, y)
        return model.predict(Xts)
    # numpy fallback ridge
    X0 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)
    Xt0 = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=float)], axis=1)
    eye = np.eye(X0.shape[1], dtype=float)
    beta = np.linalg.pinv(X0.T @ X0 + float(alpha) * eye) @ X0.T @ y
    return Xt0 @ beta


def _cv_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, seed: int = 42) -> float:
    n = int(X.shape[0])
    if n < 8:
        pred = _ridge_fit_predict(X, y, X, alpha=alpha)
        return _r2_score(y, pred)

    if HAS_SKLEARN:
        k = min(5, max(2, n // 20))
        cv = KFold(n_splits=k, shuffle=True, random_state=seed)
        scores = []
        for tr, te in cv.split(X):
            pred = _ridge_fit_predict(X[tr], y[tr], X[te], alpha=alpha)
            scores.append(_r2_score(y[te], pred))
        return float(np.mean(scores))

    # fallback simple holdout
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(max(1, round(0.8 * n)))
    tr = idx[:cut]
    te = idx[cut:]
    if len(te) < 2:
        te = tr
    pred = _ridge_fit_predict(X[tr], y[tr], X[te], alpha=alpha)
    return _r2_score(y[te], pred)


def _residualize(y: np.ndarray, Z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    pred = _ridge_fit_predict(Z, y, Z, alpha=alpha)
    return np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)


def _quantile_bin(x: np.ndarray, bins: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=int)
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)


def _discrete_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    xb = _quantile_bin(x, bins=bins)
    yb = _quantile_bin(y, bins=bins)
    n = len(xb)
    if n == 0:
        return 0.0
    mi = 0.0
    x_vals = np.unique(xb)
    y_vals = np.unique(yb)
    for xv in x_vals:
        px = float(np.mean(xb == xv))
        if px <= 0:
            continue
        for yv in y_vals:
            py = float(np.mean(yb == yv))
            pxy = float(np.mean((xb == xv) & (yb == yv)))
            if pxy > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))
    return float(max(mi, 0.0))


def _mi_per_feature(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    if X.size == 0:
        return np.array([], dtype=float)
    if HAS_SKLEARN:
        try:
            return mutual_info_regression(X, y, random_state=seed)
        except Exception:
            pass
    return np.array([_discrete_mi(X[:, j], y) for j in range(X.shape[1])], dtype=float)


def _parse_seed_list(seed_str: str | None) -> set[int] | None:
    if not seed_str:
        return None
    out = set()
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _parse_windows(run_dir: Path, window_arg: str) -> List[Path]:
    if (run_dir / "td3_seed_trace.json").exists():
        return [run_dir]
    windows = sorted([p for p in run_dir.glob("wf_window_*") if p.is_dir()])
    if not windows:
        raise ValueError(f"No wf_window_* dirs in {run_dir}")
    if window_arg.lower() == "all":
        return windows
    wanted = {w.strip() for w in window_arg.split(",") if w.strip()}
    selected = [p for p in windows if p.name in wanted]
    if not selected:
        raise ValueError(f"Requested windows not found: {sorted(wanted)}")
    return selected


def _load_root_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists() and run_dir.name.startswith("wf_window_"):
        cfg_path = run_dir.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found under {run_dir}")
    return yaml.safe_load(cfg_path.read_text())


def _load_candidate_fns(cfg: dict, repo_root: Path):
    candidate_path = cfg.get("fixed_candidate_path") or cfg.get("candidate_path")
    if not candidate_path:
        raise ValueError("No fixed_candidate_path in config.")
    path = Path(candidate_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    code = path.read_text(encoding="utf-8")
    revise_state, intrinsic_reward = load_functions_from_code(code)
    return revise_state, intrinsic_reward, path


def _load_window_test_df(
    cfg: dict,
    window_manifest: dict,
    repo_root: Path,
) -> tuple[StateSchema, np.ndarray, TradingEnv]:
    data_source = str(cfg.get("data_source", ""))
    if data_source != "finsaber":
        raise ValueError(f"Only finsaber data_source supported for now, got: {data_source}")

    split = window_manifest.get("split", {})
    test = split.get("test", {})
    selected_assets = list(window_manifest.get("selected_assets") or cfg.get("assets") or [])
    if not selected_assets:
        raise ValueError("selected_assets missing in run_manifest")
    start = str(test.get("start"))
    end = str(test.get("end"))
    if not start or not end:
        raise ValueError("test split start/end missing in run_manifest")

    price_path = cfg.get("finsaber_price_path")
    if not price_path:
        raise ValueError("finsaber_price_path missing in config")
    price_path = (repo_root / price_path).resolve()

    df = load_finsaber_prices(price_path, selected_assets, start, end)
    indicators = list(cfg.get("indicators") or [])
    df_feat = add_indicators(df, indicators)
    df_feat = _align_dates_all_assets(df_feat, selected_assets)
    if df_feat.empty:
        raise ValueError("test dataframe is empty after date/asset alignment")

    schema = StateSchema(
        assets=selected_assets,
        indicators=indicators,
        global_features=list(cfg.get("global_features") or []),
    )

    env_cfg = EnvConfig(
        initial_cash=float(cfg.get("initial_cash", 100000.0)),
        max_trade=int(cfg.get("max_trade", 100)),
        fee_rate=float(cfg.get("fee_rate", 0.0)),
        decision_ts_rule=str(window_manifest.get("decision_ts_rule", "close_t_to_open_t1")),
        action_quantization_mode=str(window_manifest.get("action_quantization_mode", "integer")),
    )
    env = TradingEnv(df_feat, selected_assets, schema, env_cfg)
    return schema, df_feat, env


@dataclass
class SeedDiagnostics:
    window: str
    seed: int
    n_steps: int
    phi_dim: int
    action_dim: int
    action_nonint_ratio: float
    state_signature_match_ratio: float
    r2_phi: float
    r2_phi_a: float
    delta_r2_action: float
    r2_lag_phi: float
    r2_lag_phi_a: float
    delta_r2_lag_action: float
    max_abs_partial_corr: float
    max_abs_partial_corr_lag1: float
    cmi_resid_mean: float
    cmi_resid_lag1_mean: float
    binding_class: str
    top_mi_features: List[int]
    top_mi_values: List[float]


def _classify_binding(
    r2_phi: float,
    delta_r2_action: float,
    max_abs_pc: float,
    delta_r2_lag: float,
) -> str:
    if r2_phi > 0.90 and delta_r2_action < 0.01 and max_abs_pc < 0.05 and delta_r2_lag < 0.01:
        return "BINDING_STRONG_ACTION_WEAK"
    if r2_phi > 0.75 and delta_r2_action < 0.03 and max_abs_pc < 0.10:
        return "BINDING_STRONG_ACTION_MEDIUM"
    if delta_r2_action > 0.05 or max_abs_pc > 0.15 or delta_r2_lag > 0.05:
        return "BINDING_WEAK_ACTION_STRONG"
    return "BINDING_MIXED"


def _collect_seed_dataset(
    window_dir: Path,
    cfg: dict,
    revise_state,
    intrinsic_reward,
    algo: str,
    group: str,
    seed: int,
) -> dict:
    repo_root = _repo_root()
    manifest = json.loads((window_dir / "run_manifest.json").read_text())
    td3_trace = json.loads((window_dir / "td3_seed_trace.json").read_text())
    rows = td3_trace.get(algo, {}).get(group, [])
    row = next((r for r in rows if int(r.get("seed")) == int(seed)), None)
    if row is None:
        raise ValueError(f"seed={seed} not found in {window_dir.name}/{algo}/{group}")

    _, _, env = _load_window_test_df(cfg, manifest, repo_root)
    intrinsic_scale_mode = str(cfg.get("intrinsic_scale_mode", "raw"))
    actions = [np.asarray(a, dtype=float) for a in (row.get("eval_actions_final") or [])]
    state_sigs_ref = list(row.get("eval_states_final") or [])
    state = env.reset()

    phi_rows: List[np.ndarray] = []
    action_rows: List[np.ndarray] = []
    intrinsic_rows: List[float] = []
    sig_matches = 0
    sig_total = 0
    nonint_count = 0
    action_count = 0

    for step, action in enumerate(actions):
        cur_sig = _state_signature(state)
        if step < len(state_sigs_ref):
            sig_total += 1
            if cur_sig == state_sigs_ref[step]:
                sig_matches += 1

        revised = revise_state(state) if revise_state is not None else state
        revised = np.asarray(revised, dtype=float).reshape(-1)
        r_int = _sanitize_scalar(intrinsic_reward(revised))
        r_int = _scale_intrinsic(r_int, intrinsic_scale_mode)

        phi_rows.append(revised)
        action_rows.append(np.asarray(action, dtype=float).reshape(-1))
        intrinsic_rows.append(float(r_int))

        action_count += int(np.asarray(action).size)
        nonint_count += int(np.sum(np.abs(np.asarray(action) - np.round(action)) > 1e-6))

        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break

    if len(phi_rows) < 8:
        raise ValueError(f"Too few aligned steps for diagnostics in {window_dir.name}, seed={seed}")

    phi = np.stack(phi_rows, axis=0)
    actions_arr = np.stack(action_rows, axis=0)
    r_int_arr = np.asarray(intrinsic_rows, dtype=float)
    return {
        "phi": phi,
        "actions": actions_arr,
        "intrinsic": r_int_arr,
        "action_nonint_ratio": float(nonint_count / max(1, action_count)),
        "state_signature_match_ratio": float(sig_matches / max(1, sig_total)),
        "n_steps": int(phi.shape[0]),
        "phi_dim": int(phi.shape[1]),
        "action_dim": int(actions_arr.shape[1]),
    }


def _diagnose_seed(dataset: dict, alpha: float, seed: int, window: str) -> SeedDiagnostics:
    phi = dataset["phi"]
    actions = dataset["actions"]
    y = dataset["intrinsic"]

    r2_phi = _cv_r2(phi, y, alpha=alpha, seed=42)
    r2_phi_a = _cv_r2(np.concatenate([phi, actions], axis=1), y, alpha=alpha, seed=42)
    delta_r2 = float(r2_phi_a - r2_phi)

    if len(y) > 2:
        y1 = y[1:]
        phi0 = phi[:-1]
        a0 = actions[:-1]
        r2_lag_phi = _cv_r2(phi0, y1, alpha=alpha, seed=42)
        r2_lag_phi_a = _cv_r2(np.concatenate([phi0, a0], axis=1), y1, alpha=alpha, seed=42)
        delta_r2_lag = float(r2_lag_phi_a - r2_lag_phi)
    else:
        r2_lag_phi = 0.0
        r2_lag_phi_a = 0.0
        delta_r2_lag = 0.0

    pcs = []
    cmi_resid = []
    y_resid = _residualize(y, phi, alpha=alpha)
    for j in range(actions.shape[1]):
        a_resid = _residualize(actions[:, j], phi, alpha=alpha)
        pcs.append(_pearson_corr(y_resid, a_resid))
        cmi_resid.append(_discrete_mi(y_resid, a_resid))
    max_abs_pc = float(np.max(np.abs(pcs))) if pcs else 0.0
    cmi_resid_mean = float(np.mean(cmi_resid)) if cmi_resid else 0.0

    pcs_lag = []
    cmi_resid_lag = []
    if len(y) > 2:
        y1 = y[1:]
        phi0 = phi[:-1]
        a0 = actions[:-1]
        y1_resid = _residualize(y1, phi0, alpha=alpha)
        for j in range(a0.shape[1]):
            a0_resid = _residualize(a0[:, j], phi0, alpha=alpha)
            pcs_lag.append(_pearson_corr(y1_resid, a0_resid))
            cmi_resid_lag.append(_discrete_mi(y1_resid, a0_resid))
    max_abs_pc_lag = float(np.max(np.abs(pcs_lag))) if pcs_lag else 0.0
    cmi_resid_lag_mean = float(np.mean(cmi_resid_lag)) if cmi_resid_lag else 0.0

    mi_vals = _mi_per_feature(phi, y)
    order = np.argsort(mi_vals)[::-1] if len(mi_vals) > 0 else np.array([], dtype=int)
    top_idx = [int(i) for i in order[:5]]
    top_vals = [float(mi_vals[i]) for i in order[:5]]

    binding_class = _classify_binding(r2_phi, delta_r2, max_abs_pc, delta_r2_lag)

    return SeedDiagnostics(
        window=window,
        seed=int(seed),
        n_steps=int(dataset["n_steps"]),
        phi_dim=int(dataset["phi_dim"]),
        action_dim=int(dataset["action_dim"]),
        action_nonint_ratio=float(dataset["action_nonint_ratio"]),
        state_signature_match_ratio=float(dataset["state_signature_match_ratio"]),
        r2_phi=float(r2_phi),
        r2_phi_a=float(r2_phi_a),
        delta_r2_action=float(delta_r2),
        r2_lag_phi=float(r2_lag_phi),
        r2_lag_phi_a=float(r2_lag_phi_a),
        delta_r2_lag_action=float(delta_r2_lag),
        max_abs_partial_corr=float(max_abs_pc),
        max_abs_partial_corr_lag1=float(max_abs_pc_lag),
        cmi_resid_mean=float(cmi_resid_mean),
        cmi_resid_lag1_mean=float(cmi_resid_lag_mean),
        binding_class=binding_class,
        top_mi_features=top_idx,
        top_mi_values=top_vals,
    )


def _to_dict(row: SeedDiagnostics) -> dict:
    return {
        "window": row.window,
        "seed": row.seed,
        "n_steps": row.n_steps,
        "phi_dim": row.phi_dim,
        "action_dim": row.action_dim,
        "action_nonint_ratio": row.action_nonint_ratio,
        "state_signature_match_ratio": row.state_signature_match_ratio,
        "r2_phi": row.r2_phi,
        "r2_phi_a": row.r2_phi_a,
        "delta_r2_action": row.delta_r2_action,
        "r2_lag_phi": row.r2_lag_phi,
        "r2_lag_phi_a": row.r2_lag_phi_a,
        "delta_r2_lag_action": row.delta_r2_lag_action,
        "max_abs_partial_corr": row.max_abs_partial_corr,
        "max_abs_partial_corr_lag1": row.max_abs_partial_corr_lag1,
        "cmi_resid_mean": row.cmi_resid_mean,
        "cmi_resid_lag1_mean": row.cmi_resid_lag1_mean,
        "binding_class": row.binding_class,
        "top_mi_features": row.top_mi_features,
        "top_mi_values": row.top_mi_values,
    }


def _mean(values: Iterable[float]) -> float:
    arr = [float(v) for v in values]
    return float(np.mean(arr)) if arr else 0.0


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = [
        "window",
        "seed",
        "n_steps",
        "phi_dim",
        "action_dim",
        "action_nonint_ratio",
        "state_signature_match_ratio",
        "r2_phi",
        "r2_phi_a",
        "delta_r2_action",
        "r2_lag_phi",
        "r2_lag_phi_a",
        "delta_r2_lag_action",
        "max_abs_partial_corr",
        "max_abs_partial_corr_lag1",
        "cmi_resid_mean",
        "cmi_resid_lag1_mean",
        "binding_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in cols}
            writer.writerow(out)


def _write_md(path: Path, summary: dict, rows: List[dict], candidate_path: str, group: str, algo: str) -> None:
    lines = [
        "# Binding Diagnosis Summary",
        "",
        f"- algo: {algo}",
        f"- group: {group}",
        f"- candidate: {candidate_path}",
        f"- windows: {', '.join(sorted({r['window'] for r in rows}))}",
        f"- seeds: {', '.join(str(s) for s in sorted({int(r['seed']) for r in rows}))}",
        "",
        "## Aggregate",
        f"- r2_phi_mean: {summary['r2_phi_mean']:.6f}",
        f"- delta_r2_action_mean: {summary['delta_r2_action_mean']:.6f}",
        f"- delta_r2_lag_action_mean: {summary['delta_r2_lag_action_mean']:.6f}",
        f"- max_abs_partial_corr_mean: {summary['max_abs_partial_corr_mean']:.6f}",
        f"- max_abs_partial_corr_lag1_mean: {summary['max_abs_partial_corr_lag1_mean']:.6f}",
        f"- action_nonint_ratio_mean: {summary['action_nonint_ratio_mean']:.6f}",
        f"- state_signature_match_ratio_mean: {summary['state_signature_match_ratio_mean']:.6f}",
        f"- dominant_binding_class: {summary['dominant_binding_class']}",
        "",
        "## Seed Rows",
    ]
    for r in rows:
        lines.append(
            f"- {r['window']} seed={r['seed']} class={r['binding_class']} "
            f"r2_phi={r['r2_phi']:.4f} delta_r2={r['delta_r2_action']:.4f} "
            f"delta_r2_lag={r['delta_r2_lag_action']:.4f} "
            f"pc={r['max_abs_partial_corr']:.4f} pc_lag={r['max_abs_partial_corr_lag1']:.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose intrinsic/revised binding strength.")
    parser.add_argument("--run-dir", required=True, help="Run dir (root run or wf_window dir).")
    parser.add_argument("--window", default="all", help="Window name(s): all or wf_window_00,wf_window_01")
    parser.add_argument("--algo", default="td3")
    parser.add_argument("--group", default="G3_revise_intrinsic")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seeds.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha for residualization/regression.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    root_cfg = _load_root_config(run_dir)
    repo_root = _repo_root()
    revise_state, intrinsic_reward, candidate_path = _load_candidate_fns(root_cfg, repo_root)
    seed_filter = _parse_seed_list(args.seeds)

    windows = _parse_windows(run_dir, args.window)
    all_rows: List[dict] = []

    for window_dir in windows:
        trace_path = window_dir / "td3_seed_trace.json"
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        group_rows = trace.get(args.algo, {}).get(args.group, [])
        for row in group_rows:
            seed = int(row.get("seed"))
            if seed_filter is not None and seed not in seed_filter:
                continue
            dataset = _collect_seed_dataset(
                window_dir=window_dir,
                cfg=root_cfg,
                revise_state=revise_state,
                intrinsic_reward=intrinsic_reward,
                algo=args.algo,
                group=args.group,
                seed=seed,
            )
            diag = _diagnose_seed(dataset=dataset, alpha=float(args.alpha), seed=seed, window=window_dir.name)
            all_rows.append(_to_dict(diag))

    if not all_rows:
        raise ValueError("No diagnostics rows produced; check algo/group/window/seeds.")

    class_counts: Dict[str, int] = {}
    for r in all_rows:
        cls = str(r["binding_class"])
        class_counts[cls] = class_counts.get(cls, 0) + 1
    dominant_cls = max(class_counts.items(), key=lambda kv: kv[1])[0]

    summary = {
        "run_dir": str(run_dir),
        "algo": args.algo,
        "group": args.group,
        "candidate_path": str(candidate_path),
        "n_rows": len(all_rows),
        "r2_phi_mean": _mean(r["r2_phi"] for r in all_rows),
        "delta_r2_action_mean": _mean(r["delta_r2_action"] for r in all_rows),
        "delta_r2_lag_action_mean": _mean(r["delta_r2_lag_action"] for r in all_rows),
        "max_abs_partial_corr_mean": _mean(r["max_abs_partial_corr"] for r in all_rows),
        "max_abs_partial_corr_lag1_mean": _mean(r["max_abs_partial_corr_lag1"] for r in all_rows),
        "cmi_resid_mean": _mean(r["cmi_resid_mean"] for r in all_rows),
        "cmi_resid_lag1_mean": _mean(r["cmi_resid_lag1_mean"] for r in all_rows),
        "action_nonint_ratio_mean": _mean(r["action_nonint_ratio"] for r in all_rows),
        "state_signature_match_ratio_mean": _mean(r["state_signature_match_ratio"] for r in all_rows),
        "binding_class_counts": class_counts,
        "dominant_binding_class": dominant_cls,
        "rows": all_rows,
    }

    out_base = run_dir
    if run_dir.name.startswith("wf_window_"):
        out_base = run_dir
    json_path = out_base / "binding_diagnosis.json"
    csv_path = out_base / "binding_probe_table.csv"
    md_path = out_base / "binding_summary.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, all_rows)
    _write_md(md_path, summary, all_rows, str(candidate_path), args.group, args.algo)

    print(f"[OK] rows={len(all_rows)}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(
        "[SUMMARY] "
        f"r2_phi_mean={summary['r2_phi_mean']:.6f} "
        f"delta_r2_action_mean={summary['delta_r2_action_mean']:.6f} "
        f"delta_r2_lag_action_mean={summary['delta_r2_lag_action_mean']:.6f} "
        f"dominant={summary['dominant_binding_class']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
