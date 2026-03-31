from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.demo import (
    DemoConfig,
    FinAgentStub,
    FinAgentStubConfig,
    _action_space_type,
    _build_policy_state_fn_for_selection,
    _candidate_origin_from_name,
    _coerce_bool,
    _extract_candidate_component_hashes,
    _identity_revise_state,
    _infer_candidate_feature_groups,
    _json_safe,
    _llm_chat_with_retries,
    _prepare_intrinsic_for_selection,
    _rank_candidate_rows,
    _score_candidate_payload_for_algo_external,
    _set_windows_safe_worker_limits,
    _sha256_text,
    _validate_candidate_code_for_backend,
    _validate_candidate_pair_for_backend,
    _write_final_selection_progress,
    build_initial_prompt,
    build_cot_prompt,
    build_next_iteration_prompt,
    deepseek_from_env,
    extract_lesr_code,
    generate_candidate_codes,
    load_functions_from_code,
)


def _run_branch_iteration(payload: dict, progress_path: Path | None = None) -> dict:
    cfg = payload["cfg"]
    if isinstance(cfg, dict):
        cfg = DemoConfig(**cfg)
    llm_cfg = dict(payload["llm_cfg"])
    branch_algo = str(payload["branch_algo"])
    it = int(payload["iteration"])
    branch_state = dict(payload["branch_state"])
    system_prompt = str(payload["system_prompt"])
    raw_state_desc = payload["state_desc"]
    state_contract_note = str(payload.get("state_contract_note", "") or "")
    drl_backend = str(payload.get("drl_backend", "current") or "current")
    native_validation_states = payload.get("native_validation_states")
    native_raw_dim = payload.get("native_raw_dim")
    if isinstance(raw_state_desc, list):
        state_desc = [str(x) for x in raw_state_desc]
    elif isinstance(raw_state_desc, str):
        state_desc = [line for line in raw_state_desc.splitlines() if line.strip()]
        if not state_desc:
            state_desc = [raw_state_desc]
    else:
        state_desc = [str(raw_state_desc)]
    scenario_profile = dict(payload.get("scenario_profile", {}) or {})
    scenario_enabled = bool(payload.get("scenario_enabled", False))
    scenario_priority = [str(x) for x in (payload.get("scenario_priority", []) or [])]
    candidates_per_family = int(payload.get("candidates_per_family", 1))
    candidate_scoring_cfg = dict(payload["candidate_scoring_cfg"])
    llm_iteration_mode = str(payload["llm_iteration_mode"])
    generation_target = str(payload.get("generation_target", "global_best"))
    max_iterations = int(payload.get("max_iterations", 1))
    selection_seeds = [int(x) for x in payload["selection_seeds"]]
    runtime = payload["runtime"]
    train_df = payload["train_df"]
    val_df = payload["val_df"]
    schema = payload["schema"]
    env_cfg = payload["env_cfg"]
    reference_states = np.asarray(payload["reference_states"], dtype=np.float32)

    def _heartbeat(status: str, **extra) -> None:
        if progress_path is None:
            return
        _write_final_selection_progress(
            progress_path,
            {
                "status": status,
                "algorithm": branch_algo,
                "iteration": int(it),
                **extra,
            },
        )

    def _append_prompt_sections(base_prompt: str, sections: list[str]) -> str:
        non_empty = [str(section).strip() for section in sections if str(section).strip()]
        if not non_empty:
            return base_prompt
        return f"{base_prompt}\n\n" + "\n\n".join(non_empty)

    def _truncate_text(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max(0, int(max_chars * 0.65))
        tail = max(0, max_chars - head - len("\n...\n[truncated]\n...\n"))
        if tail <= 0:
            return text[:max_chars]
        return f"{text[:head]}\n...\n[truncated]\n...\n{text[-tail:]}"

    def _fresh_iteration_prompt(extra_sections: list[str] | None = None) -> str:
        if all_it_func_results or all_it_cot_suggestions:
            base_prompt = build_next_iteration_prompt(
                cfg.task_description,
                state_desc,
                all_it_func_results,
                all_it_cot_suggestions,
                state_contract_note=state_contract_note,
            )
        else:
            base_prompt = build_initial_prompt(
                cfg.task_description,
                state_desc,
                state_contract_note=state_contract_note,
            )
        return _append_prompt_sections(base_prompt, extra_sections or [])

    def _validate_candidate_pair(revise_state_fn, intrinsic_reward_fn):
        return _validate_candidate_pair_for_backend(
            revise_state_fn,
            intrinsic_reward_fn,
            drl_backend=drl_backend,
            schema=schema,
            native_validation_states=native_validation_states,
            native_raw_dim=native_raw_dim,
        )

    def _algo_branch_instruction() -> str:
        if llm_iteration_mode != "per_algorithm_branches":
            return ""
        action_space = _action_space_type(branch_algo, drl_backend)
        mechanism_hint = {
            "a2c": (
                "Continuous-control bias under finsaber_native: prefer smooth, bounded revise_state features that improve "
                "ranking, separability, and regime clarity without pushing actions to constant extremes."
                if drl_backend.strip().lower() == "finsaber_native"
                else "Discrete-control bias: prioritize revise_state features that improve ranking, separability, trend-confidence, and regime clarity. Keep intrinsic smooth and secondary."
            ),
            "ppo": (
                "Continuous-control bias under finsaber_native: favor robust revise_state signal shaping first; intrinsic reward "
                "should stabilize confidence without collapsing action diversity."
                if drl_backend.strip().lower() == "finsaber_native"
                else "Discrete-control bias: favor robust revise_state signal shaping first; intrinsic reward should stabilize confidence rather than dominate behavior."
            ),
            "sac": "Continuous-control bias: prefer balanced revise+intrinsic designs with confidence-gated risk penalties. Avoid aggressive monotonic suppression that can flatten useful exploration.",
            "td3": (
                "Continuous-control bias: prefer smooth, bounded, action-sensitive intrinsic terms. "
                "Bind the intrinsic mechanism to portfolio footprint variables such as holdings concentration, exposure imbalance, "
                "rebalancing pressure, or near-bound saturation proxies. Avoid pure market-regime bonuses/penalties that change totals "
                "but leave policy behavior or action diversity unchanged."
            ),
        }.get(
            branch_algo,
            "Prefer smooth, bounded, action-sensitive candidates with explicit confidence or risk gating.",
        )
        return (
            "Current LESR search branch target RL algorithm: "
            f"`{branch_algo}` (action_space=`{action_space}`).\n"
            + (
                "This branch is using backend-specific state semantics; follow the authoritative state contract note "
                "instead of assuming the generic LESR state layout. "
                if state_contract_note
                else "Use the same generic LESR mechanism as every other branch. "
            )
            + "Do not rely on algorithm-specific hacks; prefer robust, action-sensitive, smooth candidates that "
            "score well under this branch's short-train feedback.\n"
            f"{mechanism_hint}\n"
            "The candidate scorer now checks whether intrinsic reward still helps under raw-policy control; "
            "avoid designs where revise_state carries all useful signal and intrinsic_reward becomes negligible.\n"
            "Shared objective: search for structures that could remain plausible across other branches and market windows, not only this branch."
        )

    def _branch_prompt(base_prompt: str) -> str:
        extra = _algo_branch_instruction()
        return f"{base_prompt}\n\n{extra}" if extra else base_prompt

    def _summarize_error_types(errors: list[dict], top_k: int = 4) -> str:
        sample_errors = [str(row.get("error_type", "unknown")) for row in errors or [] if row.get("phase") == "sample"]
        if not sample_errors:
            return ""
        counts = {}
        for name in sample_errors:
            counts[name] = counts.get(name, 0) + 1
        top_rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return "Recent sample failure mix: " + ", ".join(f"{name}={count}" for name, count in top_rows)

    def _build_branch_iteration_feedback(candidate_stats: list[dict], errors: list[dict]) -> str:
        lines: list[str] = []
        if candidate_stats:
            ranked = _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)
            best = ranked[0] if ranked else max(candidate_stats, key=lambda row: float(row.get("score", 0.0)))
            behavior = best.get("behavior", {}) or {}
            lines.append(
                "Latest branch best: "
                f"name={best.get('name')}, family={best.get('family')}, design_mode={best.get('design_mode')}, "
                f"perf_delta_sharpe={float(best.get('performance_delta_sharpe', best.get('performance_score_delta', 0.0))):.4f}, "
                f"state_probe_delta_sharpe={float(best.get('state_probe_delta_sharpe', best.get('state_probe_score_delta', 0.0))):.4f}, "
                f"intrinsic_probe_delta_sharpe={float(best.get('intrinsic_probe_delta_sharpe', best.get('intrinsic_probe_score_delta', 0.0))):.4f}, "
                f"lipschitz={float(best.get('lipschitz_raw') or 0.0):.4f}, "
                f"behavior_score={float(best.get('behavior_score', 0.0)):.4f}, "
                f"near_bound={float(behavior.get('near_bound_ratio_mean', 0.0)):.4f}, "
                f"avg_weight_change={float(behavior.get('avg_daily_portfolio_weight_change_mean', 0.0)):.4f}, "
                f"turnover_score={float(best.get('turnover_score', 0.0)):.4f}"
            )
        error_summary = _summarize_error_types(errors)
        if error_summary:
            lines.append(error_summary)
        return "Structured branch feedback:\n" + "\n".join(f"- {line}" for line in lines) if lines else ""

    def _build_design_schedule(count: int) -> list[str]:
        base = ["intrinsic_first", "balanced", "state_first"]
        return [base[i % len(base)] for i in range(count)] if count > 0 else []

    def _design_mode_instruction(design_mode: str) -> str:
        if design_mode == "intrinsic_first":
            return (
                "Design mode: intrinsic_first.\n"
                f"Target RL branch: `{branch_algo}`.\n"
                "Build a candidate where intrinsic_reward is a primary signal even when use_revised=False.\n"
                "intrinsic_reward must depend directly on original holdings/price/risk exposure terms, not only on revised proxy dims.\n"
                "Use revised dims mainly as confidence gates or context scalars; avoid making revise_state carry the whole edge.\n"
                "Prefer using global running-risk dims such as return EMA, return-squared EMA, drawdown, and turnover EMA when available.\n"
                "Target a genuinely useful G2 path: avoid intrinsic functions that are mostly one-sided negative offsets with near-zero standalone Sharpe impact.\n"
                "Prefer a bounded opportunity-minus-risk structure whose standalone effect can improve at least some raw-policy branches.\n"
                + (
                    "TD3-specific rule: make the raw-state intrinsic branch action-sensitive via portfolio footprint proxies such as "
                    "concentration, exposure imbalance, holdings saturation, and rebalance pressure. Avoid regime-only gates or "
                    "constant risk offsets that do not move the deterministic actor.\n"
                    if str(branch_algo).lower() == "td3"
                    else ""
                )
            )
        if design_mode == "state_first":
            return (
                "Design mode: state_first.\n"
                f"Target RL branch: `{branch_algo}`.\n"
                "Build a candidate where revise_state does most of the work, while intrinsic_reward is a secondary stabilizer.\n"
                "Keep intrinsic smooth and bounded; do not let it dominate behavior.\n"
                "Portfolio-memory and running-risk dims may be used only as stabilizers, not as the sole edge."
            )
        return (
            "Design mode: balanced.\n"
            f"Target RL branch: `{branch_algo}`.\n"
            "Build a candidate where revise_state and intrinsic_reward both contribute meaningfully.\n"
            "Avoid candidates where either path is numerically non-trivial but behaviorally irrelevant.\n"
            "Running-risk and portfolio-memory dims may be used, but neither path should collapse into a pure turnover suppressor."
        )

    def _algo_family_generation_instruction(family: str, design_mode: str) -> str:
        algo_key = str(branch_algo).lower()
        family_key = str(family or "").lower()
        mode_key = str(design_mode or "").lower()

        lines: list[str] = []
        if algo_key == "td3" and mode_key in {"intrinsic_first", "balanced"}:
            lines.extend(
                [
                    "TD3 generation rule:",
                    "- Raw-state intrinsic must include at least one action-sensitive opportunity term and one action-sensitive risk term.",
                    "- Opportunity term should respond to holdings/exposure footprints such as concentration, exposure imbalance, holdings-weighted spread/rank, or rebalancing pressure.",
                    "- Risk term should penalize saturation/bound pressure smoothly using concentration, entropy drop, cash depletion, or near-bound portfolio footprint proxies.",
                    "- Do not submit regime-only, drawdown-only, or confidence-only offsets if they can stay unchanged when holdings/actions stay unchanged.",
                    "- Prefer formulas where the intrinsic magnitude changes when portfolio weights or holdings composition change, even under similar market states.",
                ]
            )
            if family_key == "trend_follow":
                lines.append(
                    "- For TD3 trend_follow, tie the positive term to holdings-aware trend capture or exposure-aligned rank spread, not just market trend level."
                )
            elif family_key == "mean_revert":
                lines.append(
                    "- For TD3 mean_revert, tie the positive term to position-gap repair or overshoot reversal scaled by current holdings imbalance."
                )
            elif family_key == "risk_shield":
                lines.append(
                    "- For TD3 risk_shield, keep a positive opportunity leg; do not emit a pure shield that only subtracts risk regardless of action change."
                )

        if algo_key == "sac" and mode_key in {"intrinsic_first", "balanced"}:
            lines.extend(
                [
                    "SAC generation rule:",
                    "- Preserve a standalone raw-state G2 signal: intrinsic_reward should not collapse into a pure negative penalty or near-zero centered offset.",
                    "- Use an opportunity-minus-risk decomposition where the opportunity leg can be positive on some raw states without requiring revised extra dims.",
                    "- Keep the raw-state branch non-trivial across the whole window; avoid designs that are almost always zero or always negative.",
                ]
            )
            if family_key == "risk_shield":
                lines.append(
                    "- For SAC risk_shield, include a bounded positive preservation/recovery term alongside the shield; do not return a pure suppression penalty."
                )
        return "\n".join(lines)

    def _sampling_instruction(family: str, design_mode: str) -> str:
        family_block = (
            "Target family:\n"
            f"- family={family}\n"
            "Family semantics:\n"
            "- trend_follow: action-sensitive trend/rank positive term + smooth risk budget.\n"
            "- mean_revert: deviation-repair/position-gap repair with bounded control.\n"
            "- risk_shield: volatility/drawdown-aware guard that avoids over-clipping.\n"
        ) if family else ""
        return (
            "Generate ONE candidate code pair.\n"
            f"{family_block}"
            "Current window profile:\n"
            f"- inferred_family={scenario_profile.get('family')}\n"
            f"- mu_ann={float(scenario_profile.get('mu_ann', 0.0)):.4f}\n"
            f"- vol_ann={float(scenario_profile.get('vol_ann', 0.0)):.4f}\n"
            f"- max_dd={float(scenario_profile.get('max_dd', 0.0)):.4f}\n"
            f"- vol_short_ann={float(scenario_profile.get('vol_short_ann', 0.0)):.4f}\n"
            f"- vol_long_ann={float(scenario_profile.get('vol_long_ann', 0.0)):.4f}\n"
            f"- vol_ratio_20_60={float(scenario_profile.get('vol_ratio_20_60', 0.0)):.4f}\n"
            f"- trend_strength_20={float(scenario_profile.get('trend_strength_20', 0.0)):.4f}\n"
            f"- dispersion_20={float(scenario_profile.get('dispersion_20', 0.0)):.6f}\n"
            f"- market_stress_score={float(scenario_profile.get('market_stress_score', 0.0)):.4f}\n"
            f"{_design_mode_instruction(design_mode)}\n"
            + (
                f"{_algo_family_generation_instruction(family, design_mode)}\n"
                if _algo_family_generation_instruction(family, design_mode)
                else ""
            )
            + "\n"
            "Mechanism targeting rule:\n"
            "- Use the window profile above explicitly. If inferred_family is trend_follow, prefer persistent trend/rank/confidence structures. If mean_revert, prefer deviation-repair / position-gap repair / snapback logic. If risk_shield, prefer volatility, drawdown, and concentration-aware protection.\n"
            "- Primary optimization target is delta_sharpe under branch evaluation, not absolute Sharpe or raw reward scale.\n"
            "- The candidate must be materially different from recent history. Do not submit the same cash_ratio + concentration + trend_strength + dispersion scaffold with only renamed variables or coefficient tweaks.\n"
            "- Change at least one of: core feature family, normalization rule, gating logic, or intrinsic opportunity/risk decomposition.\n"
            "Numeric safety rule:\n"
            "- Every ratio or normalization must be division-safe: use denominator floors and explicit fallback branches for tiny denominators.\n"
            "- Do not use raw mean/std/volume/position ratios without a safe branch when the scale is near zero.\n"
            "If possible, declare FEATURE_GROUPS = [...] using any of: portfolio_memory, regime, dispersion, running_risk_state.\n"
            "Prefer revise_state features that include portfolio-memory terms such as cash ratio, concentration, entropy, exposure ratio, or rebalancing pressure.\n"
            "Keep constraints valid and keep intrinsic_reward informative under raw-policy control while avoiding unstable portfolio-weight jumps."
        )

    dialogs = list(branch_state["dialogs"])
    all_it_func_results = list(branch_state["all_it_func_results"])
    all_it_cot_suggestions = list(branch_state["all_it_cot_suggestions"])
    global_seen_candidate_hashes = set(branch_state["seen_candidate_hashes"])
    iteration_seen_candidate_hashes: set[str] = set()
    local_llm_responses: list[dict] = []
    local_llm_errors: list[dict] = []
    family_schedule: list[str] = []
    target_valid_count = int(max(1, llm_cfg.get("k", 2)))
    if scenario_enabled:
        for fam in scenario_priority:
            for _ in range(candidates_per_family):
                family_schedule.append(fam)
        if family_schedule:
            target_valid_count = len(family_schedule)
    design_schedule = _build_design_schedule(target_valid_count)
    valid_sample_count = 0
    revise_code_buffer: list[str] = []
    revise_dim_buffer: list[int] = []
    assistant_reply_buffer: list[str] = []
    candidate_hash_map: dict[str, str] = {}
    trying_count = 0
    failed_sample_calls = 0
    max_failed_calls = int(max(1, llm_cfg.get("max_failed_calls", 10)))
    max_empty_response_calls = int(max(1, llm_cfg.get("max_empty_response_calls", max_failed_calls * 2)))
    max_invalid_code_calls = int(max(1, llm_cfg.get("max_invalid_code_calls", max_failed_calls * 2)))
    max_duplicate_calls = int(max(max_failed_calls, llm_cfg.get("max_duplicate_calls", max_failed_calls * 4)))
    max_validation_failed_calls = int(
        max(max_failed_calls, llm_cfg.get("max_validation_failed_calls", max_failed_calls * 3))
    )
    sample_failure_counters = {
        "hard": 0,
        "empty_response": 0,
        "invalid_code": 0,
        "duplicate_candidate": 0,
        "validation_failed": 0,
    }
    stop_sampling = False
    candidates_it: list[tuple[str, str]] = []
    candidate_family_map: dict[str, str] = {}
    candidate_design_map: dict[str, str] = {}
    iter_log = {
        "algorithm": branch_algo,
        "iteration": it,
        "prompt": dialogs[-1]["content"],
        "prompt_length": int(len(dialogs[-1]["content"])),
        "candidates": [],
        "feedback": None,
        "generation_target": generation_target,
        "scenario_enabled": bool(scenario_enabled),
        "scenario_profile": scenario_profile,
        "llm_iteration_mode": llm_iteration_mode,
        "design_schedule": list(design_schedule),
    }
    if scenario_enabled:
        iter_log["family_schedule"] = list(family_schedule)

    _heartbeat("running", phase="bootstrap", step="client_init")
    client = deepseek_from_env(
        llm_cfg["base_url"],
        timeout_s=int(llm_cfg.get("request_timeout_s", 60)),
        use_env_proxy=_coerce_bool(llm_cfg.get("use_env_proxy"), default=False),
    )
    finagent = FinAgentStub(FinAgentStubConfig()) if cfg.use_finagent_signal else None
    _heartbeat("running", phase="bootstrap", step="state_fn_init")
    state_fn_raw = _build_policy_state_fn_for_selection(
        _identity_revise_state,
        cfg=cfg,
        schema=schema,
        reference_states=reference_states,
        drl_backend=drl_backend,
        native_raw_dim=native_raw_dim,
    )
    _heartbeat("running", phase="bootstrap", step="ready")

    def _record_sample_failure(error_type: str, message: str, attempt: int, extra: dict | None = None) -> bool:
        nonlocal failed_sample_calls
        counter_key = "hard"
        failure_limit = max_failed_calls
        limit_error_type = "max_failed_calls_reached"
        if error_type == "sample_empty_response":
            counter_key = "empty_response"
            failure_limit = max_empty_response_calls
            limit_error_type = "max_empty_response_calls_reached"
        elif error_type == "invalid_code":
            counter_key = "invalid_code"
            failure_limit = max_invalid_code_calls
            limit_error_type = "max_invalid_code_calls_reached"
        elif error_type == "duplicate_candidate":
            counter_key = "duplicate_candidate"
            failure_limit = max_duplicate_calls
            limit_error_type = "max_duplicate_calls_reached"
        elif error_type == "validation_failed":
            counter_key = "validation_failed"
            failure_limit = max_validation_failed_calls
            limit_error_type = "max_validation_failed_calls_reached"
        sample_failure_counters[counter_key] += 1
        failed_sample_calls = int(sum(sample_failure_counters.values()))
        row = {
            "algorithm": branch_algo,
            "iteration": it,
            "phase": "sample",
            "attempt": attempt,
            "error_type": error_type,
            "message": message,
            "failure_counter_key": counter_key,
            "failure_counter_value": int(sample_failure_counters[counter_key]),
            "failure_counter_limit": int(failure_limit),
        }
        if extra:
            row.update(extra)
        local_llm_errors.append(row)
        if sample_failure_counters[counter_key] >= failure_limit:
            local_llm_errors.append(
                {
                    "algorithm": branch_algo,
                    "iteration": it,
                    "phase": "sample",
                    "attempt": attempt,
                    "error_type": limit_error_type,
                    "message": f"{counter_key} reached {failure_limit}, fallback to static candidates",
                    "failure_counter_key": counter_key,
                    "failure_counter_value": int(sample_failure_counters[counter_key]),
                    "failure_counter_limit": int(failure_limit),
                }
            )
            return True
        return False

    def _compress_iteration_memory(candidate_rows: list[dict], errors: list[dict]) -> str:
        if not candidate_rows and not errors:
            return ""
        lines: list[str] = []
        if candidate_rows:
            ranked_rows = _rank_candidate_rows(candidate_rows, candidate_scoring_cfg)
            best_row = ranked_rows[0] if ranked_rows else max(candidate_rows, key=lambda row: float(row.get("score", 0.0)))
            behavior = best_row.get("behavior", {}) or {}
            lines.append(
                "best="
                f"{best_row.get('name')} family={best_row.get('family')} design={best_row.get('design_mode')} "
                f"score={float(best_row.get('score', 0.0)):.4f} "
                f"perf_delta={float(best_row.get('performance_score_delta', 0.0)):.4f} "
                f"intrinsic_delta={float(best_row.get('intrinsic_probe_score_delta', 0.0)):.4f} "
                f"lipschitz={float(best_row.get('lipschitz_raw') or 0.0):.4f} "
                f"behavior={float(best_row.get('behavior_score', 0.0)):.4f} "
                f"near_bound={float(behavior.get('near_bound_ratio_mean', 0.0)):.4f}"
            )
        error_summary = _summarize_error_types(errors)
        if error_summary:
            lines.append(error_summary)
        return "\n".join(lines)

    def _add_fallback_candidates(start_slot: int) -> None:
        if drl_backend.strip().lower() == "finsaber_native":
            return
        fallback_candidates = generate_candidate_codes(schema)
        if not fallback_candidates:
            return
        fallback_schedule = family_schedule if scenario_enabled and family_schedule else ["global_best"]
        for slot_idx in range(start_slot, int(target_valid_count)):
            fam = fallback_schedule[slot_idx] if slot_idx < len(fallback_schedule) else fallback_schedule[-1]
            design_mode = design_schedule[slot_idx] if slot_idx < len(design_schedule) else "balanced"
            added = False
            for offset in range(len(fallback_candidates)):
                fallback_base_name, fallback_code = fallback_candidates[(slot_idx + offset) % len(fallback_candidates)]
                fallback_hash = _sha256_text(fallback_code)
                if fallback_hash in iteration_seen_candidate_hashes or fallback_hash in global_seen_candidate_hashes:
                    continue
                fallback_prefix = f"{branch_algo}_" if llm_iteration_mode == "per_algorithm_branches" else ""
                fallback_name = f"{fallback_prefix}fallback_it{it}_{fam}_{fallback_base_name}_{slot_idx}"
                try:
                    revise_state_fb, intrinsic_reward_fb = load_functions_from_code(fallback_code)
                    fallback_dim, _ = _validate_candidate_pair(revise_state_fb, intrinsic_reward_fb)
                    iteration_seen_candidate_hashes.add(fallback_hash)
                    candidate_hash_map[fallback_name] = fallback_hash
                    candidates_it.append((fallback_name, fallback_code))
                    candidate_family_map[fallback_name] = fam
                    candidate_design_map[fallback_name] = design_mode
                    revise_code_buffer.append(fallback_code)
                    revise_dim_buffer.append(fallback_dim)
                    assistant_reply_buffer.append(fallback_code)
                    local_llm_errors.append(
                        {
                            "algorithm": branch_algo,
                            "iteration": it,
                            "phase": "sample",
                            "attempt": trying_count,
                            "error_type": "fallback_static_candidate",
                            "message": "use_static_candidate_to_fill_missing_slots",
                            "fallback_name": fallback_name,
                            "slot": int(slot_idx),
                        }
                    )
                    added = True
                    break
                except Exception as exc:
                    local_llm_errors.append(
                        {
                            "algorithm": branch_algo,
                            "iteration": it,
                            "phase": "sample",
                            "attempt": trying_count,
                            "error_type": "fallback_static_candidate_failed",
                            "message": str(exc),
                            "fallback_name": fallback_name,
                            "slot": int(slot_idx),
                        }
                    )
            if not added:
                local_llm_errors.append(
                    {
                        "algorithm": branch_algo,
                        "iteration": it,
                        "phase": "sample",
                        "attempt": trying_count,
                        "error_type": "fallback_static_candidate_exhausted",
                        "message": "no additional unseen fallback candidate available",
                        "slot": int(slot_idx),
                    }
                )
                break

    _heartbeat("running", phase="sample", valid_sample_count=0, target_valid_count=int(target_valid_count))
    while valid_sample_count < target_valid_count:
        trying_count += 1
        if trying_count > 50:
            break
        requested_family = family_schedule[valid_sample_count] if scenario_enabled and valid_sample_count < len(family_schedule) else ""
        requested_design_mode = design_schedule[valid_sample_count] if valid_sample_count < len(design_schedule) else "balanced"
        sample_messages = dialogs + [{"role": "user", "content": _sampling_instruction(requested_family, requested_design_mode)}]
        _heartbeat("running", phase="sample", attempt=int(trying_count), valid_sample_count=int(valid_sample_count), requested_family=requested_family, requested_design_mode=requested_design_mode)
        content = _llm_chat_with_retries(client=client, llm_cfg=llm_cfg, messages=sample_messages, llm_errors=local_llm_errors, iteration=it, phase="sample")
        if content is None:
            stop_sampling = _record_sample_failure("sample_empty_response", "llm_response_empty_or_failed_after_retries", trying_count)
            if stop_sampling:
                break
            continue
        local_llm_responses.append({"algorithm": branch_algo, "iteration": it, "attempt": trying_count, "index": valid_sample_count, "family": requested_family or "global_best", "design_mode": requested_design_mode, "content": content})
        code = extract_lesr_code(content)
        if "def revise_state" not in code or "def intrinsic_reward" not in code:
            stop_sampling = _record_sample_failure(
                "invalid_code",
                f"missing function definitions:revise_state={'def revise_state' in code},intrinsic_reward={'def intrinsic_reward' in code}",
                trying_count,
                {"code_len": int(len(code)), "response_len": int(len(content))},
            )
            if stop_sampling:
                break
            continue
        try:
            _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
            revise_state, intrinsic_reward = load_functions_from_code(code)
            revised_dim, _ = _validate_candidate_pair(revise_state, intrinsic_reward)
        except Exception as exc:
            stop_sampling = _record_sample_failure("validation_failed", str(exc), trying_count)
            if stop_sampling:
                break
            continue
        code_hash = _sha256_text(code)
        if code_hash in iteration_seen_candidate_hashes or code_hash in global_seen_candidate_hashes:
            stop_sampling = _record_sample_failure("duplicate_candidate", "duplicate candidate code hash within this algorithm branch", trying_count)
            if stop_sampling:
                break
            continue
        iteration_seen_candidate_hashes.add(code_hash)
        prefix = f"{branch_algo}_" if llm_iteration_mode == "per_algorithm_branches" else ""
        name = f"{prefix}llm_it{it}_{requested_family}_k{valid_sample_count}" if requested_family else f"{prefix}llm_it{it}_k{valid_sample_count}"
        candidate_family_map[name] = requested_family or "global_best"
        candidate_design_map[name] = requested_design_mode
        candidate_hash_map[name] = code_hash
        candidates_it.append((name, code))
        revise_code_buffer.append(code)
        revise_dim_buffer.append(revised_dim)
        assistant_reply_buffer.append(content)
        valid_sample_count += 1

    if len(candidates_it) < int(target_valid_count):
        _add_fallback_candidates(len(candidates_it))

    candidate_stats = []
    every_score = []
    results_corr = []
    for idx, (name, code) in enumerate(candidates_it):
        feature_groups = _infer_candidate_feature_groups(code)
        component_hashes = _extract_candidate_component_hashes(code)
        cand_entry = {
            "algorithm": branch_algo,
            "name": name,
            "family": candidate_family_map.get(name, "global_best"),
            "design_mode": candidate_design_map.get(name, "balanced"),
            "origin": _candidate_origin_from_name(name),
            "feature_groups": feature_groups,
            "revise_hash": component_hashes.get("revise_hash", ""),
            "intrinsic_hash": component_hashes.get("intrinsic_hash", ""),
            "code": code,
            "valid": False,
            "score": None,
            "corrs": None,
            "error": None,
            "revised_dim": revise_dim_buffer[idx] if idx < len(revise_dim_buffer) else None,
        }
        try:
            _validate_candidate_code_for_backend(code, drl_backend=drl_backend)
            revise_state, intrinsic_reward = load_functions_from_code(code)
            checked_dim, _ = _validate_candidate_pair(revise_state, intrinsic_reward)
            cand_entry["revised_dim"] = checked_dim
        except Exception as exc:
            cand_entry["error"] = str(exc)
            iter_log["candidates"].append(cand_entry)
            continue
        _heartbeat("running", phase="score_candidates", candidate_index=int(idx + 1), candidate_count=int(len(candidates_it)), candidate=name)
        intrinsic_reward_eval = _prepare_intrinsic_for_selection(revise_state, intrinsic_reward, cfg=cfg, reference_states=reference_states)
        intrinsic_reward_probe_eval = _prepare_intrinsic_for_selection(
            revise_state,
            intrinsic_reward,
            cfg=cfg,
            reference_states=reference_states,
            input_mode="raw",
        )
        policy_state_fn_candidate = _build_policy_state_fn_for_selection(
            revise_state,
            cfg=cfg,
            schema=schema,
            reference_states=reference_states,
            drl_backend=drl_backend,
            native_raw_dim=native_raw_dim,
        )
        score_payload = _score_candidate_payload_for_algo_external(
            cfg=cfg,
            algo=branch_algo,
            runtime=runtime,
            revise_state=revise_state,
            intrinsic_reward_eval=intrinsic_reward_eval,
            intrinsic_reward_probe_eval=intrinsic_reward_probe_eval,
            policy_state_fn_candidate=policy_state_fn_candidate,
            seeds=selection_seeds,
            train_df=train_df,
            val_df=val_df,
            schema=schema,
            env_cfg=env_cfg,
            state_fn_raw=state_fn_raw,
            finagent=finagent,
            candidate_scoring_cfg=candidate_scoring_cfg,
            reference_states=reference_states,
            drl_backend=drl_backend,
        )
        candidate_stats.append(
            {
                "name": name,
                "family": candidate_family_map.get(name, "global_best"),
                "design_mode": candidate_design_map.get(name, "balanced"),
                "origin": _candidate_origin_from_name(name),
                "feature_groups": feature_groups,
                "revise_hash": component_hashes.get("revise_hash", ""),
                "intrinsic_hash": component_hashes.get("intrinsic_hash", ""),
                "score": float(score_payload["score"]),
                "performance_mode": str(score_payload["performance_mode"]),
                "performance_score": float(score_payload["performance_score"]),
                "performance_score_absolute": float(score_payload["performance_score_absolute"]),
                "performance_score_baseline": float(score_payload["performance_score_baseline"]),
                "performance_score_delta": float(score_payload["performance_score_delta"]),
                "intrinsic_probe_score": float(score_payload["intrinsic_probe_score"]),
                "intrinsic_probe_score_absolute": float(score_payload["intrinsic_probe_score_absolute"]),
                "intrinsic_probe_score_baseline": float(score_payload["intrinsic_probe_score_baseline"]),
                "intrinsic_probe_score_delta": float(score_payload["intrinsic_probe_score_delta"]),
                "lipschitz_raw": score_payload["lipschitz_raw"],
                "lipschitz_score": float(score_payload["lipschitz_score"]),
                "behavior_score": float(score_payload["behavior_score"]),
                "turnover_score": float(score_payload["turnover_score"]),
                "behavior": score_payload["behavior"],
                "corrs": score_payload["corrs"],
            }
        )
        every_score.append(float(score_payload["score"]))
        results_corr.append(score_payload["corrs"])
        cand_entry["valid"] = True
        cand_entry["score"] = float(score_payload["score"])
        cand_entry["performance_mode"] = str(score_payload["performance_mode"])
        cand_entry["performance_score"] = float(score_payload["performance_score"])
        cand_entry["performance_score_absolute"] = float(score_payload["performance_score_absolute"])
        cand_entry["performance_score_baseline"] = float(score_payload["performance_score_baseline"])
        cand_entry["performance_score_delta"] = float(score_payload["performance_score_delta"])
        cand_entry["intrinsic_probe_score"] = float(score_payload["intrinsic_probe_score"])
        cand_entry["intrinsic_probe_score_absolute"] = float(score_payload["intrinsic_probe_score_absolute"])
        cand_entry["intrinsic_probe_score_baseline"] = float(score_payload["intrinsic_probe_score_baseline"])
        cand_entry["intrinsic_probe_score_delta"] = float(score_payload["intrinsic_probe_score_delta"])
        cand_entry["lipschitz_raw"] = score_payload["lipschitz_raw"]
        cand_entry["lipschitz_score"] = float(score_payload["lipschitz_score"])
        cand_entry["behavior_score"] = float(score_payload["behavior_score"])
        cand_entry["turnover_score"] = float(score_payload["turnover_score"])
        cand_entry["behavior"] = score_payload["behavior"]
        cand_entry["seed_behavior"] = score_payload["seed_behavior"]
        cand_entry["corrs"] = score_payload["corrs"]
        cand_entry["seed_metrics"] = score_payload["seed_metrics"]
        cand_entry["revised_dim_delta"] = int(cand_entry["revised_dim"] - schema.dim()) if cand_entry["revised_dim"] is not None else None
        iter_log["candidates"].append(cand_entry)
        _heartbeat(
            "running",
            phase="score_candidates",
            step="candidate_done",
            candidate=name,
            candidate_index=int(idx + 1),
            candidate_count=int(len(candidates_it)),
        )

    if iter_log["candidates"]:
        ranked_names = [s["name"] for s in _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)]
        rank_map = {name: idx + 1 for idx, name in enumerate(ranked_names)}
        for c in iter_log["candidates"]:
            c["rank"] = rank_map.get(c["name"])
        if scenario_enabled:
            family_counts = {}
            for c in iter_log["candidates"]:
                if bool(c.get("valid")):
                    fam = str(c.get("family", "unknown"))
                    family_counts[fam] = family_counts.get(fam, 0) + 1
            iter_log["family_counts"] = family_counts

    next_dialogs = dialogs
    dialog_text = ""
    if candidate_stats:
        ranked_candidates = _rank_candidate_rows(candidate_stats, candidate_scoring_cfg)
        promoted_count = int(max(1, llm_cfg.get("global_seen_promote_top_n", min(2, max(1, target_valid_count)))))
        for promoted_row in ranked_candidates[:promoted_count]:
            promoted_hash = candidate_hash_map.get(str(promoted_row.get("name", "")))
            if promoted_hash:
                global_seen_candidate_hashes.add(promoted_hash)
        best_candidate_name = str(ranked_candidates[0]["name"]) if ranked_candidates else str(candidate_stats[int(np.argmax(np.array(every_score)))]["name"])
        max_id = next((idx for idx, (cand_name, _code) in enumerate(candidates_it) if cand_name == best_candidate_name), int(np.argmax(np.array(every_score))))
        cot_prompt, cur_it_func_results = build_cot_prompt(
            revise_code_buffer,
            every_score,
            max_id,
            results_corr,
            revise_dim_buffer,
            schema.dim(),
            task_name=f"{branch_algo.upper()} short-train score",
        )
        all_it_func_results.append(_compress_iteration_memory(candidate_stats, local_llm_errors) or cur_it_func_results)
        cot_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_reply_buffer[max_id]},
            {"role": "user", "content": cot_prompt},
        ]
        _heartbeat("running", phase="cot", candidate=best_candidate_name)
        cot_suggestion = _llm_chat_with_retries(client=client, llm_cfg=llm_cfg, messages=cot_messages, llm_errors=local_llm_errors, iteration=it, phase="cot") or ""
        all_it_cot_suggestions.append(
            _truncate_text(cot_suggestion, int(max(300, llm_cfg.get("history_suggestion_max_chars", 700))))
        )
        iter_log["feedback"] = cot_suggestion
        iter_log["feedback_summary"] = cot_suggestion[:500]
        iter_log["cot_prompt"] = cot_prompt
        dialogs_log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": iter_log["prompt"]},
            {"role": "assistant", "content": assistant_reply_buffer[max_id]},
            {"role": "user", "content": cot_prompt},
            {"role": "assistant", "content": cot_suggestion},
        ]
        for dialog in dialogs_log:
            dialog_text += "*" * 50 + "\n"
            dialog_text += "*" * 20 + f"role:{dialog['role']}" + "*" * 20 + "\n"
            dialog_text += "*" * 50 + "\n"
            dialog_text += f"{dialog['content']}\n\n"
        if it < max_iterations - 1:
            branch_feedback = _build_branch_iteration_feedback(candidate_stats, local_llm_errors)
            next_prompt = _fresh_iteration_prompt([branch_feedback])
            iter_log["next_prompt_length"] = int(len(next_prompt))
            next_dialogs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _branch_prompt(next_prompt)},
            ]
    elif it < max_iterations - 1:
        retry_feedback = _truncate_text(
            _build_branch_iteration_feedback(candidate_stats, local_llm_errors),
            int(max(400, llm_cfg.get("retry_feedback_max_chars", 2000))),
        )
        retry_prompt = _fresh_iteration_prompt(
            [
                "Retry guidance:",
                "The previous iteration produced no valid candidate. Avoid repeating duplicate, invalid, or numerically unstable patterns.",
                retry_feedback,
            ]
        )
        iter_log["next_prompt_length"] = int(len(retry_prompt))
        next_dialogs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _branch_prompt(retry_prompt)},
        ]

    iter_log["sample_attempts"] = int(trying_count)
    iter_log["sample_valid_count"] = int(len(candidates_it))
    iter_log["sample_failed_calls"] = int(failed_sample_calls)
    iter_log["sample_failure_counters"] = {k: int(v) for k, v in sample_failure_counters.items()}
    iter_log["sample_failure_limits"] = {
        "hard": int(max_failed_calls),
        "empty_response": int(max_empty_response_calls),
        "invalid_code": int(max_invalid_code_calls),
        "duplicate_candidate": int(max_duplicate_calls),
        "validation_failed": int(max_validation_failed_calls),
    }
    iter_log["sample_stop_by_failure_limit"] = bool(stop_sampling)
    _heartbeat("done", phase="done", sample_valid_count=int(len(candidates_it)), llm_error_count=int(len(local_llm_errors)), llm_response_count=int(len(local_llm_responses)))
    return {
        "algorithm": branch_algo,
        "candidate_entries": list(candidates_it),
        "iter_log": iter_log,
        "llm_responses": local_llm_responses,
        "llm_errors": local_llm_errors,
        "dialog_text": dialog_text,
        "next_dialogs": next_dialogs,
        "all_it_func_results": all_it_func_results,
        "all_it_cot_suggestions": all_it_cot_suggestions,
        "seen_candidate_hashes": list(global_seen_candidate_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    output_path = Path(args.output)
    progress_path = Path(args.progress)
    try:
        _set_windows_safe_worker_limits()
        with payload_path.open("rb") as f:
            payload = pickle.load(f)
        result = _run_branch_iteration(payload, progress_path=progress_path)
        output_path.write_text(json.dumps({"ok": True, "result": _json_safe(result)}, indent=2), encoding="utf-8")
        return 0
    except Exception:
        output_path.write_text(json.dumps({"ok": False, "error": traceback.format_exc()}, indent=2), encoding="utf-8")
        _write_final_selection_progress(
            progress_path,
            {
                "status": "failed",
                "algorithm": str(payload.get("branch_algo", "")) if "payload" in locals() else "",
                "iteration": int(payload.get("iteration", -1)) if "payload" in locals() else -1,
                "error": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
