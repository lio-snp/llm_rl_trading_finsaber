# llm_rl_trading_finsaber

Standalone collaboration snapshot of the `finsaber`-migrated LESR trading project.

## Main experiment path

For collaborator review, the primary entry and config are:

- Entry: `python scripts/run.py demo --config configs/current_baseline/finsaber_native_composite_peralgo_full.yaml`
- Main pipeline: `src/pipeline/demo.py`
- Native backend: `src/drl/finsaber_native_runner.py` and `src/finsaber_native/*`
- LESR generation: `src/pipeline/branch_iteration_worker.py`, `src/lesr/*`, `src/llm/deepseek_client.py`

The repository has been trimmed to keep the current `finsaber_native` experiment path and a small number of supporting smoke/baseline configs.

## Collaborator clone-and-run configs

The original FINSABER price CSV used during migration is about 265 MB, so it is not bundled in the repository.

For collaborator verification, the repo now includes a tracked smoke subset:

- Data bundle: `data/collab/finsaber_sp12_2010_2024.csv`
- Coverage: 12 symbols from 2010-01-04 to 2024-12-31

Recommended configs:

- Full composite: `configs/current_baseline/finsaber_native_composite_collab_full.yaml`
- Smoke composite: `configs/current_baseline/finsaber_native_composite_collab_smoke.yaml`

Data expectations:

- Full composite: place the original `all_sp500_prices_2000_2024_delisted_include.csv` at `data/full/all_sp500_prices_2000_2024_delisted_include.csv`
- Smoke composite: uses the tracked subset already included in this repo

To place the full CSV into the expected repo location:

- `python scripts/prepare_finsaber_full_data.py /path/to/all_sp500_prices_2000_2024_delisted_include.csv`

Example commands:

- `python scripts/run.py demo --config configs/current_baseline/finsaber_native_composite_collab_full.yaml`
- `python scripts/run.py demo --config configs/current_baseline/finsaber_native_composite_collab_smoke.yaml`

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
