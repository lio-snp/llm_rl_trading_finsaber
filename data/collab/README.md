# Collaborator Data Bundle

This directory contains a small repo-tracked subset of the original FINSABER price file so the project can be cloned and run directly from GitHub.

- Source file: `all_sp500_prices_2000_2024_delisted_include.csv`
- Source size: about 265 MB
- Bundled subset: `finsaber_sp12_2010_2024.csv`
- Bundled subset size: about 2.7 MB
- Coverage: 12 symbols, 2010-01-04 to 2024-12-31

The bundled symbols are:

- `AAPL`
- `ADBE`
- `AMZN`
- `COST`
- `CRM`
- `INTC`
- `MSFT`
- `NFLX`
- `NVDA`
- `PEP`
- `QCOM`
- `TSLA`

The collaborator configs in `configs/current_baseline/` point to this file via a repo-relative path.
