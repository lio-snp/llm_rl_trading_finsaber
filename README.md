# llm_rl_trading_finsaber

Standalone collaboration snapshot of the `finsaber`-migrated LESR trading project.

## Main experiment path

For collaborator review, the primary entry and config are:

- Entry: `python scripts/run.py demo --config configs/current_baseline/finsaber_native_composite_peralgo_full.yaml`
- Main pipeline: `src/pipeline/demo.py`
- Native backend: `src/drl/finsaber_native_runner.py` and `src/finsaber_native/*`
- LESR generation: `src/pipeline/branch_iteration_worker.py`, `src/lesr/*`, `src/llm/deepseek_client.py`

The repository has been trimmed to keep the current `finsaber_native` experiment path and a small number of supporting smoke/baseline configs.

## Kept in repo

- `src/`: main experiment code, native backend, LESR pipeline
- `configs/current_baseline/`: current native baseline/composite/smoke configs
- `scripts/run.py`: unified experiment entry
- `scripts/resume_walk_forward.py`: window-level resume/rebuild helper
- `scripts/run_bull_regime_long_window.py`: secondary regime entry retained because `scripts/run.py` still exposes it
- `tests/`: lightweight data/compat tests
- `docs/steps/step_65_*` and `docs/steps/step_66_*`: migration and compatibility audit notes

## Excluded from git

- `runs/`: experiment outputs and logs
- `data/`: local datasets and generated files
- Python caches and local editor caches

## Notes

- This snapshot was prepared from the local migration workspace for collaborator review.
- Some scripts expect runtime secrets such as `DEEPSEEK_API_KEY` from environment variables rather than committed files.
