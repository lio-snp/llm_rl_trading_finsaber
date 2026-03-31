# Step 66 Design: FINSABER Native Compatibility Audit and Research Memo

## Goal
Produce a repo-native audit package that answers three questions without changing runtime behavior:

1. What is the real `state/action/reward` contract of the native DRL backend?
2. Which old LESR assumptions from `llm_rl_trading` still remain in `llm_rl_trading_finsaber`?
3. Do those residual assumptions already show up in the current run as invalid intrinsic paths, prompt bias, or distorted candidate diagnostics?

## Scope
- Read-only code and artifact audit.
- Chinese deliverables only.
- No training rerun in this round.
- No code-path mutation in this round.

## Planned Deliverables
- `01_compatibility_audit.md`
- `02_research_transfer_memo.md`
- `03_priority_backlog.md`
- `inputs.md`
- `outputs.md`

## Stop Condition
This step is complete when the audit package contains:
- one contract-level comparison table;
- one mismatch list ordered by severity;
- one evidence table that binds claims to concrete code or run artifacts;
- one transfer matrix that maps research ideas to concrete hook points in this repo;
- one priority backlog split into `baseline 先跑`, `轻改可试`, `中期重构`.
