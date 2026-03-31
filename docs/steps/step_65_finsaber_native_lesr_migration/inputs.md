# Inputs

## Files Reviewed
- `agent.md`
- `src/pipeline/demo.py`
- `src/drl/finsaber_native_runner.py`
- `src/finsaber_native/env_stocktrading.py`
- `src/lesr/prompt_templates.py`
- `src/env/state_schema.py`

## Working Assumptions
- `finsaber_native` remains on the current outer 5-window protocol in the first LESR migration round.
- FINSABER-native state ordering is:
  - `cash`
  - close-price block
  - holdings block
  - flattened indicator blocks
- The first implementation target is feature/state alignment, not hyperparameter tuning.

## Exclusions
- No LLM execution in this planning round.
- No change to walk-forward window policy in this planning round.
- No new official experiment config yet.
