# Quant Crypto Kaggle

Walk-forward crypto return forecasting experiments for Kaggle's
[G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting)
competition.

This repository explores short-horizon return prediction for 14 cryptocurrencies
using the competition's weighted Pearson correlation metric. The emphasis is on
clean validation, readable feature engineering, and simple models that are easy
to inspect.

## Results

Validation uses chronological splits. The ridge runs were completed on CPU; the
MLP run was completed on an RTX 4090 GPU.

| Model | Mean weighted correlation | Notes |
| --- | ---: | --- |
| Naive `-ret_15m` baseline | `0.023366` | 15-minute mean reversion |
| Rolling global ridge | `0.031154` | Expanding walk-forward folds |
| Enhanced global ridge | `0.044532` | Market, BTC/ETH, and asset-interaction factors |
| Global MLP | `0.051753` | RTX 4090 GPU run, folds 6-9 |

Best completed model: **global MLP**, with a mean lift of `0.031801` over its
same-fold naive baseline.

## The MLP

The neural model lives in [`rolling_mlp_global.py`](rolling_mlp_global.py). It is
a global PyTorch tabular MLP that uses:

- engineered return, volume, volatility, market, BTC, ETH, and time features
- a learned embedding for `Asset_ID`
- chronological walk-forward validation
- a configurable gap between train and validation chunks (`GAP_CHUNKS = 3`)

The script automatically uses CUDA when available. The latest full run used an
NVIDIA GeForce RTX 4090 and completed folds 6-9:

| Fold | Best MLP corr | Naive corr | Best epoch |
| ---: | ---: | ---: | ---: |
| 6 | `0.045439` | `0.018026` | 2 |
| 7 | `0.047253` | `0.027357` | 4 |
| 8 | `0.053706` | `0.004554` | 6 |
| 9 | `0.060615` | `0.029873` | 2 |

Summary: mean `0.051753`, std `0.005968`, min `0.045439`, max `0.060615`.

## Repository Map

```text
rolling_mlp_global.py            # Global MLP with asset embeddings
rolling_ridge_global_enhanced.py # Strong linear baseline
rolling_ridge_baseline.py        # Simple global ridge baseline
rolling_ridge_per_asset.py       # Separate ridge model per asset
analyze_baseline.py              # Naive baseline diagnostics
inspect_dataset.py               # Dataset schema and row counts
src/metrics.py                   # Weighted correlation metric
docs/                            # GitHub Pages summary
DATA.md                          # Kaggle data setup notes
results/                         # Concise tracked run summaries
```

## Setup

Download the competition files from Kaggle and place them in `raw/`:

```text
raw/asset_details.csv
raw/train.csv
```

Then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python inspect_dataset.py
python analyze_baseline.py
python rolling_ridge_global_enhanced.py
python rolling_mlp_global.py
```

The Kaggle data and local virtual environments are intentionally excluded from
Git. See [`DATA.md`](DATA.md) for the data layout.

## Notes

This is an educational research project, not financial advice or a production
trading system.
