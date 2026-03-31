# Inputs

- Base config: `configs/current_baseline/lesr_short_window_refresh_full_2014_2023_5level.yaml`
- Bull specialist config: `configs/current_baseline/bull_regime_long_window_5level.yaml`
- Price source: FINSABER full-history SP500 price file referenced by base config
- Regime labeling:
  - lookback: 63 days
  - persistence: 5 days
  - target regime: bull
- Bull windowing:
  - train: 504 bull bars
  - val: 126 bull bars
  - test: 126 bull bars
  - step: 126 bull bars
