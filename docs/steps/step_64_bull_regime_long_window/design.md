# Step 64 Design

## Goal
Test whether a bull-only regime specialist trained on causally labeled bull dates can deliver positive LESR deltas on bull-only test windows.

## Hypothesis
Compared with shared full-market training, bull-only windows reduce regime contamination and improve at least one of `G1-G0`, `G2-G0`, or `G3-G0` on bull test windows.

## Protocol
- Label full history with causal bull/bear/sideways labels.
- Keep only `bull` dates.
- Build bull-only walk-forward windows with:
  - train: 504 bull bars
  - val: 126 bull bars
  - test: 126 bull bars
  - step: 126 bull bars
- Algorithms: `a2c, ppo, sac, td3`
- LESR: `iterations=6`, `k=2`, `selection_seed_count=1`, `intrinsic_w=0.02`, `seeds=[1,2,3]`

## Success Criteria
- The run completes with bull-window `metrics_table.csv` files and an aggregate `walk_forward_metrics_table.csv`.
- At least one algorithm shows positive aggregate delta on bull windows for `G1-G0` or `G3-G0`.

## Stop Condition
- Stop if bull windows cannot be generated with sufficient counts.
- Stop if runtime errors show the bull-only split pipeline is invalid.
