# Full Data Placement

The collaborator full composite config expects the original FINSABER price file at:

- `data/full/all_sp500_prices_2000_2024_delisted_include.csv`

This file is not committed because the original CSV is about 265 MB.

Use this file with:

- `configs/current_baseline/finsaber_native_composite_collab_full.yaml`

The smoke config does not need this file and instead uses:

- `data/collab/finsaber_sp12_2010_2024.csv`
