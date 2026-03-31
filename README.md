# llm_rl_trading_finsaber

Standalone collaboration snapshot of the `finsaber`-migrated LESR trading project.

## Included

- `src/`: trading environment, DRL runners, LESR pipeline, prompt logic
- `configs/`: experiment configs and focused diagnostics
- `scripts/`: launch, analysis, and reporting utilities
- `tests/`: lightweight compatibility/data tests
- `docs/`: migration notes and experiment step records
- `ref/`: local reference materials copied from the working project

## Excluded From Git

- `runs/`: experiment outputs and logs
- `data/`: local datasets and generated files
- Python caches and local editor caches

## Notes

- This snapshot was prepared from the local migration workspace for collaborator review.
- Some scripts expect runtime secrets such as `DEEPSEEK_API_KEY` from environment variables rather than committed files.
