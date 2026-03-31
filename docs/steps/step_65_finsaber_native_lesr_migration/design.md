# Step 65 Design: LESR Migration onto `finsaber_native`

## Goal
Enable `G1/G2/G3` LESR flows on top of the existing `finsaber_native` backend without changing the current outer walk-forward protocol.

## Hypothesis
If LESR is migrated with a native state contract that exactly matches FINSABER-style state ordering, then prompt semantics, candidate validation, and intrinsic scoring will remain aligned with the actual DRL environment and avoid generic-schema mismatch.

## Evaluation Protocol
- Phase label: `E0` for planning and code review only in this round.
- No training run in this round.
- Review scope:
  - backend state contract
  - prompt/state description
  - candidate validation
  - G2/G3 intrinsic input semantics
  - run-manifest auditability

## Falsification Plan
- Reject the migration design if any part still depends on generic `StateSchema.describe()` for `finsaber_native`.
- Reject the migration design if `intrinsic_reward(s)` on raw native state and `intrinsic_reward(revise_state(s))` do not share the same state contract.
- Reject the migration design if official runs would bypass `scripts/run.py`.

## Expected Outputs
- One planning note with migration phases and minimal code-touch list.
- One copied `agent.md` at repo root.
- No official run output in this round.

## Stop Condition
This round is complete when:
- the repo-level working rules are present in `llm_rl_trading_finsaber/agent.md`;
- a minimal migration plan is documented;
- code review findings are summarized into the plan before implementation starts.
