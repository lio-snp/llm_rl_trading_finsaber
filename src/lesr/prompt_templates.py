from __future__ import annotations

import re
from typing import List


_PROMPT_HISTORY_MAX_ITERS = 3
_PROMPT_HISTORY_MAX_RESULT_CHARS = 1200
_PROMPT_HISTORY_MAX_SUGGESTION_CHARS = 700
_PROMPT_HISTORY_TOTAL_CHAR_BUDGET = 5000


def _normalize_state_desc(state_desc) -> List[str]:
    if isinstance(state_desc, list):
        return [str(x) for x in state_desc]
    if isinstance(state_desc, tuple):
        return [str(x) for x in state_desc]
    if isinstance(state_desc, str):
        rows = [line for line in state_desc.splitlines() if line.strip()]
        return rows or [state_desc]
    if state_desc is None:
        return []
    return [str(state_desc)]


def _infer_state_dim_from_desc(state_desc_rows: List[str]) -> int:
    max_dim = 0
    for row in state_desc_rows or []:
        text = str(row)
        for start_s, end_s in re.findall(r"s\[(\d+):(\d+)\]", text):
            try:
                max_dim = max(max_dim, int(end_s))
            except ValueError:
                continue
        for idx_s in re.findall(r"s\[(\d+)\]", text):
            try:
                max_dim = max(max_dim, int(idx_s) + 1)
            except ValueError:
                continue
    if max_dim > 0:
        return max_dim
    return len(state_desc_rows or [])


def _trim_block(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max(0, int(max_chars * 0.65))
    tail = max(0, max_chars - head - len("\n...\n[truncated]\n...\n"))
    if tail <= 0:
        return text[:max_chars]
    return f"{text[:head]}\n...\n[truncated]\n...\n{text[-tail:]}"


def _compact_history_text(text: str, max_lines: int = 12) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = text.replace("```python", "").replace("```", "")
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines: List[str] = []
    for line in raw_lines:
        if len(line) > 240:
            line = _trim_block(line, 240)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _format_history_block(history_results: List[str], history_suggestions: List[str]) -> str:
    pairs = list(zip(history_results or [], history_suggestions or []))
    if not pairs:
        return "No former iterations are available yet."

    if len(pairs) > _PROMPT_HISTORY_MAX_ITERS:
        omitted = len(pairs) - _PROMPT_HISTORY_MAX_ITERS
        pairs = pairs[-_PROMPT_HISTORY_MAX_ITERS:]
        start_idx = omitted + 1
        header = f"\n[Only the latest {_PROMPT_HISTORY_MAX_ITERS} former iterations are shown; {omitted} earlier iterations omitted.]\n"
    else:
        start_idx = 1
        header = ""

    remaining_budget = _PROMPT_HISTORY_TOTAL_CHAR_BUDGET
    blocks: List[str] = [header] if header else []
    for offset, (result_text, suggestion_text) in enumerate(pairs):
        iter_idx = start_idx + offset
        result_trimmed = _trim_block(
            _compact_history_text(result_text, max_lines=10),
            _PROMPT_HISTORY_MAX_RESULT_CHARS,
        )
        suggestion_trimmed = _trim_block(
            _compact_history_text(suggestion_text, max_lines=8),
            _PROMPT_HISTORY_MAX_SUGGESTION_CHARS,
        )
        block = (
            f"\n\nFormer Iteration:{iter_idx} Summary\n"
            f"{result_trimmed}"
            f"\n\nKeep/avoid guidance from Iteration:{iter_idx}\n"
            f"{suggestion_trimmed}"
        )
        if remaining_budget <= 0:
            break
        if len(block) > remaining_budget:
            block = _trim_block(block, remaining_budget)
            blocks.append(block)
            remaining_budget = 0
            break
        blocks.append(block)
        remaining_budget -= len(block)
    return "".join(blocks).strip()


def build_system_prompt(llm_cfg: dict | None = None) -> str:
    cfg = llm_cfg or {}
    mode = str(cfg.get("system_prompt_mode", "trading_lesr_prior_v1")).strip().lower()
    extra = str(cfg.get("system_prompt_extra", "")).strip()
    disable_priors = bool(cfg.get("system_prompt_disable_priors", False))

    if mode != "trading_lesr_prior_v1":
        base = (
            "You are an expert in reinforcement-learning feature revision and intrinsic reward design. "
            "Return executable Python code only."
        )
        return base if not extra else f"{base}\n\n{extra}"

    base = """
You are designing `revise_state(s)` and `intrinsic_reward(updated_s)` for LESR-style trading RL.

Primary objective:
- Produce code that improves sample efficiency and risk-adjusted performance while preserving policy-action sensitivity.

Output contract:
- Return ONLY executable Python code.
- Must contain:
  - `import numpy as np`
  - `def revise_state(s): ...`
  - `def intrinsic_reward(updated_s): ...`
- No markdown, no explanations, no comments.

Hard constraints:
- No future information leakage.
- NumPy-only operations.
- Keep all outputs finite and bounded.
- Keep intrinsic reward informative; avoid trivial tiny clipping.
- Intrinsic reward must stay in [-100, 100].
- `intrinsic_reward` must remain meaningful when only the original source dims are available.
- If appended revised dims are available, `intrinsic_reward` should use them as extra context rather than as a hard dependency.
- Do not use hard-coded out-of-range indices; all index access must be valid for the returned vector length.
- If you can identify the mechanism, declare `FEATURE_GROUPS = [...]` at module scope using any of:
  - `portfolio_memory`
  - `regime`
  - `dispersion`
  - `running_risk_state`

Failure patterns to avoid:
- Action-insensitive intrinsic design dominated by state-only bias terms.
- Large monotonic concentration push that drives near-bound action saturation.
- Overly sparse trigger logic that makes reward mostly zero.
- Unstable sign-flip terms without confidence gating.
"""

    if not disable_priors:
        base += """

Empirical priors from recent experiments (use as design bias):
1) Positive/robust families:
- `action_sensitive_spread_rank_v4` style (spread-rank + risk-budget + bounded normalization) showed repeated positive deltas across A2C/SAC cells.
- `action_sensitive_spread_rank_v6_penalty_clip` and `action_sensitive_spread_rank_v7_balanced` style intrinsic shaping improved SAC/TD3 directional gains in selected4-like runs.
- `action_sensitive_step42_bull_preserve_boundclip_v17` and `step41 bull_safe_asym` style were useful in index-like protocols.
- For SAC-like intrinsic-only wins, the raw-state branch usually kept an explicit bounded positive opportunity term instead of a pure negative shield.

2) Mechanism requirements from TD3 diagnostics:
- Keep concentration/bound penalties smooth and state-dependent (not hard discontinuities).
- Prefer confidence-gated penalties over unconditional suppression.
- Encourage action-relevant ranking terms, not only level shifts.
- Avoid designs that change reward totals but leave action behavior nearly unchanged.
- For native continuous control, prefer portfolio-footprint-aware terms driven by holdings concentration, exposure imbalance, saturation pressure, and rebalancing pressure rather than market-state-only offsets.
- Treat revised extra dims as context or denoising aids; the raw-state intrinsic branch should already move behavior on its own.
- For TD3 specifically, prefer intrinsic terms whose value changes when holdings composition or portfolio footprint changes, even if market-state levels remain similar.

3) Prefer these structural motifs:
- revise_state:
  - robust trend proxy + volatility proxy + concentration/risk-budget proxy + confidence proxy
  - portfolio-memory terms such as cash ratio, exposure ratio, concentration, entropy, and rebalancing pressure
  - bounded transforms (`tanh`, safe ratio, normalized spread) to control scale
- intrinsic_reward:
  - positive term: action-relevant spread/rank/trend-confidence interaction
  - negative term: concentration/bound risk with confidence-aware clipping
  - prefer risk-adjusted and turnover-aware shaping over raw return amplification
  - final bounded aggregation in stable numeric range
"""

    if extra:
        base += f"\n\nAdditional run-specific instruction:\n{extra}\n"
    return base.strip()


def build_initial_prompt(
    task_description: str,
    state_desc: List[str],
    state_contract_note: str = "",
) -> str:
    state_desc = _normalize_state_desc(state_desc)
    total_dim = _infer_state_dim_from_desc(state_desc)
    detail_content = "\n".join([f"- {d}" for d in state_desc])
    state_contract_note = str(state_contract_note or "").strip()
    extra_note = f"\nAuthoritative state contract note:\n{state_contract_note}\n" if state_contract_note else ""

    return f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
{extra_note}

You should design a task-related state representation based on the source {total_dim} dims.
Use the details above to compute new features, then concatenate them to the original state.

Besides, we want you to design an intrinsic reward function based on the revise_state function.

That is to say, we will evaluate the intrinsic reward in two modes:
1. G2-like intrinsic-only mode: `r_int = intrinsic_reward(s)` using the original source state.
2. G3-like joint mode: `updated_s = revise_state(s)` then `r_int = intrinsic_reward(updated_s)`.
3. Therefore intrinsic_reward must be valid and informative on the original dims alone.
4. If revised extra dims are present, intrinsic_reward should use them to refine the signal rather than requiring them to exist.

Constraints:
- Do NOT use any future data.
- Only use NumPy operations.
- Keep outputs numeric and bounded.
- Do NOT trivially clip intrinsic reward to a tiny range like [-1, 1].
- Keep intrinsic reward informative and roughly comparable to environment reward scale, and bounded in [-100, 100].
- Intrinsic reward must have a raw-state fallback path when only the source dims are available.
- The primary intrinsic signal must come from the raw/source dims; revised extra dims may only refine, gate, or denoise that same signal.
- Reject near-constant or almost-zero intrinsic designs on raw states; the raw-state branch should remain non-trivial by itself.
- When revised extra dims are present, use them as additional context to improve the same signal instead of carrying the sole predictive content.
- Prefer revise_state features that expose portfolio memory, regime, and risk-budget context.
- Prefer intrinsic_reward designs that improve risk-adjusted behavior and avoid unstable portfolio-weight jumps.
- Feature-group semantics:
  - `portfolio_memory`: cash ratio, holdings, exposure, concentration, entropy, rebalancing pressure
  - `regime`: volatility level, volatility ratio, drawdown, market stress, trend-strength regime
  - `dispersion`: spread, rank, breadth, cross-asset disagreement, winner-minus-loser structure
  - `running_risk_state`: return EMA, return-squared EMA, drawdown EMA, turnover EMA
- Output ONLY code. No comments. No markdown.
- Do not use hard-coded out-of-range indices; all index access must be valid for the returned vector length.
- If possible, declare `FEATURE_GROUPS = [...]` before the functions to indicate which semantic feature groups the candidate is using.

Your task is to create executable `revise_state` and `intrinsic_reward` functions.
"""


def build_cot_prompt(
    codes: List[str],
    scores: List[float],
    max_id: int,
    factors: List[List[float]],
    dims: List[int],
    source_dim: int,
    task_name: str = "Final Policy Performance",
) -> str:
    s_feedback = ""
    for i, code in enumerate(codes):
        s_feedback += f"========== State Revise and Intrinsic Reward Code -- {i + 1} ==========\n"
        s_feedback += code + "\n"
        s_feedback += (
            f"========== State Revise and Intrinsic Reward Code -- {i + 1}'s {task_name}: "
            f"{round(scores[i], 4)} ==========\n"
        )
        try:
            s_feedback += (
                f"In this State Revise Code {i + 1}, the source state dim is from s[0] to s[{source_dim - 1}], "
                "the Lipschitz constant between them and the reward are(Note: The Lipschitz constant is always "
                "greater than or equal to 0, and a lower Lipschitz constant implies better smoothness.):\n"
            )
            cur_dim_corr = ""
            for k in range(0, source_dim):
                cur_dim_corr += f"s[{k}] lipschitz constant with reward = {round(factors[i][k], 4)}\n"
            s_feedback += cur_dim_corr + "\n"

            extra_dims = max(dims[i] - source_dim, 0)
            if extra_dims > 0:
                s_feedback += (
                    f"In this State Revise Code {i + 1}, you give {extra_dims} extra dim from s[{source_dim}] "
                    f"to s[{dims[i] - 1}], the lipschitz constant between them and the reward are:\n"
                )
                cur_dim_corr = ""
                for k in range(source_dim, dims[i]):
                    cur_dim_corr += f"s[{k}] lipschitz constant with reward = {round(factors[i][k], 4)}\n"
        except Exception:
            cur_dim_corr = ""
        s_feedback += cur_dim_corr + "\n======================================================================\n\n"

    cot_prompt = f"""
We have successfully trained Reinforcement Learning (RL) policy using {len(codes)} different state revision codes and intrinsic reward function codes sampled by you, and each pair of code is associated with the training of a policy.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Most importantly, every state revise dim's Lipschitz constant with the reward. That is to say, you can see which state revise dim is more related to the reward and which dim can contribute to enhancing the continuity of the reward function mapping. Lower Lipschitz constant means better continuity and smoother of the mapping. Note: Lower Lipschitz constant is better.

Here are the results:
{s_feedback}

You should analyze the results mentioned above and give suggestions about how to improve the performance of the state revision code.

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to figure out why it fails
(b) if you find some dims are more related to the final performance, then you should analyze what makes it succeed
(c) you should also analyze how to improve the performance of the state revision code and intrinsic reward code later

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy.
"""
    return cot_prompt, s_feedback


def build_next_iteration_prompt(
    task_description: str,
    state_desc: List[str],
    history_results: List[str],
    history_suggestions: List[str],
    state_contract_note: str = "",
) -> str:
    state_desc = _normalize_state_desc(state_desc)
    total_dim = _infer_state_dim_from_desc(state_desc)
    detail_content = "\n".join([f"- {d}" for d in state_desc])
    former_history = _format_history_block(history_results, history_suggestions)
    state_contract_note = str(state_contract_note or "").strip()
    extra_note = f"\nAuthoritative state contract note:\n{state_contract_note}\n" if state_contract_note else ""

    return f"""
Revise the state representation for a reinforcement learning agent.
=========================================================
The agent's task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
{extra_note}

You should design a task-related state representation based on the source {total_dim} dims to better for reinforcement training, using the detailed information mentioned above to do some calculations, and feel free to do complex calculations, and then concatenate them to the source state.

Recent compressed history from former iterations:
{former_history}

Based on the history above, seek an improved state revision code and an improved intrinsic reward code. Do not repeat the same feature family, normalization, gating rule, or intrinsic mechanism unless you are making a clear structural change.

That is to say, we will evaluate the intrinsic reward in two modes:
1. G2-like intrinsic-only mode: `r_int = intrinsic_reward(s)` using the original source state.
2. G3-like joint mode: `updated_s = revise_state(s)` then `r_int = intrinsic_reward(updated_s)`.
3. intrinsic_reward must therefore work on the original dims alone.
4. If revised extra dims are present, use them to enrich the same mechanism instead of making them a hard dependency.

Constraints:
- Do NOT use any future data.
- Only use NumPy operations.
- Keep outputs numeric and bounded.
- Any division or normalization must be division-safe: use a positive denominator floor and a fallback branch when the denominator is too small.
- Do not use raw ratios such as `x / mean(y)` or `x / std(y)` without an explicit denominator guard and a safe fallback value.
- Prefer numerically stable transforms: clipped z-score, bounded spread/rank, tanh-normalized proxy, or guarded difference-over-scale.
- Do NOT trivially clip intrinsic reward to a tiny range like [-1, 1].
- Keep intrinsic reward informative and roughly comparable to environment reward scale, and bounded in [-100, 100].
- Intrinsic reward must have a raw-state fallback path and should remain useful even without revised extra dims.
- The primary intrinsic signal must come from raw/source dims; revised extra dims may only refine, gate, or denoise that same signal.
- Reject near-constant or almost-zero intrinsic designs on raw states; the raw-state branch should stay non-trivial on its own.
- If revised extra dims are available, use them as extra evidence or gating, not as the sole source of signal.
- Prefer revise_state features that expose portfolio memory, regime, and risk-budget context.
- Prefer intrinsic_reward designs that improve risk-adjusted behavior and avoid unstable portfolio-weight jumps.
- Output ONLY code. No comments. No markdown.
- Prefer a materially new candidate over a small cosmetic rewrite of prior code.
- Mechanism diversity requirement: do not reuse the same revise_state scaffold with only renamed variables or minor coefficient changes. Change at least one of:
  1. the main feature family,
  2. the normalization strategy,
  3. the gating logic,
  4. the intrinsic decomposition into opportunity/risk terms.

Your task is to create executable `revise_state` and `intrinsic_reward` functions ready for integration into the RL environment.
"""
