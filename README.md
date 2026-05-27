# Quant Crypto Kaggle

Readable, reproducible experiments for Kaggle's
[G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting)
competition.

The goal is to forecast short-term returns for 14 cryptocurrencies using
minute-level market data. The competition metric is a weighted Pearson
correlation between predictions and the provided target, where each asset has
a fixed importance weight.

## Project Highlights

- Chronological, walk-forward validation instead of random train/test splits.
- A transparent mean-reversion baseline using the negative 15-minute return.
- Ridge regression baselines with rolling windows and asset-aware features.
- An enhanced global ridge model with market, BTC/ETH, and asset interaction
  factors.
- A global PyTorch MLP that combines engineered market features with learned
  asset embeddings.
- Polars-based feature engineering for large CSV files.

## Repository Layout

```text
.
├── analyze_baseline.py              # Chunk and per-asset baseline diagnostics
├── baseline_lagged_return.py        # Minimal lagged-return sanity check
├── inspect_dataset.py               # Dataset schema and row-count inspection
├── rolling_ridge_baseline.py        # Simple expanding-window ridge model
├── rolling_ridge_global_enhanced.py # Cross-asset and interaction ridge model
├── rolling_ridge_per_asset.py       # Separate ridge model per asset
├── rolling_mlp_global.py            # Global MLP with asset embeddings
├── src/metrics.py                   # Competition metric implementation
├── docs/                            # GitHub Pages landing page
└── DATA.md                          # Data download and placement notes
```

Kaggle data is intentionally ignored by Git. See [DATA.md](DATA.md) for setup.

## Data

Download the competition data from Kaggle:

https://www.kaggle.com/competitions/g-research-crypto-forecasting

Extract it into `raw/` so the project has:

```text
raw/asset_details.csv
raw/train.csv
raw/supplemental_train.csv
raw/example_test.csv
raw/example_sample_submission.csv
```

The offline validation scripts documented here require only
`raw/train.csv` and `raw/asset_details.csv`.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MLP automatically uses CUDA when available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

On CPU, the ridge scripts are the most practical full-data experiments. The MLP
can run on CPU, but the full walk-forward setup is much better suited to a GPU.

## Validation Strategy

Financial time series are strongly time-dependent, so the experiments use
chronological validation:

1. Sort observations by `timestamp`.
2. Split the full time range into 10 contiguous chunks.
3. Train on earlier chunks.
4. Validate on a later chunk.
5. Report the weighted correlation on held-out future data.

The MLP additionally leaves a configurable gap between train and validation
chunks (`GAP_CHUNKS = 3`) to reduce near-boundary leakage risk.

## Feature Engineering

The main feature families are:

- Lagged log returns at multiple horizons: 1, 2, 3, 5, 10, 15, 30, 60, 120,
  and 240 minutes.
- Volume features: `log1p(volume)`, 15-minute volume change, and 60-minute
  relative volume.
- Volatility estimates from rolling standard deviations of 1-minute returns.
- Market-wide weighted returns using the competition asset weights.
- Bitcoin and Ethereum cross-asset lag features.
- Relative returns versus market, Bitcoin, and Ethereum factors.
- Cyclical hour-of-day and week-of-year features.
- Learned asset embeddings in the MLP.

All raw price/volume features used for prediction are shifted by one row within
each asset before modeling, so a row's prediction does not use same-row
information that would be unavailable at prediction time.

## Results From This Local Run

Environment:

- Python 3.12.3
- Polars 1.41.0
- scikit-learn 1.8.0
- PyTorch 2.12.0
- CUDA available: no

Dataset inspection:

- 14 assets.
- About 24.24 million training rows before feature drops.
- Time range: Unix `1514764860` to `1632182400`.

### Naive Mean-Reversion Baseline

Prediction: `-ret_15m`.

| Chunk | Weighted correlation |
| ---: | ---: |
| 0 | 0.003983 |
| 1 | 0.011143 |
| 2 | 0.033610 |
| 3 | 0.033953 |
| 4 | 0.027515 |
| 5 | 0.025714 |
| 6 | 0.020759 |
| 7 | 0.038952 |
| 8 | 0.006393 |
| 9 | 0.031636 |

Summary:

- Mean correlation: `0.023366`
- Standard deviation: `0.011709`
- Minimum: `0.003983`
- Maximum: `0.038952`

### Rolling Global Ridge Baseline

The simple rolling ridge model uses return, volume, volatility, and numeric
asset-id features.

| Fold | Ridge corr | Naive corr | Train rows | Valid rows |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.033098 | 0.033929 | 6,005,465 | 2,431,204 |
| 4 | 0.024293 | 0.027524 | 8,436,669 | 2,265,253 |
| 5 | 0.034102 | 0.025711 | 10,701,922 | 2,325,216 |
| 6 | 0.033109 | 0.020755 | 13,027,138 | 2,328,105 |
| 7 | 0.046484 | 0.038952 | 15,355,243 | 2,654,476 |
| 8 | 0.015654 | 0.006378 | 18,009,719 | 2,739,022 |
| 9 | 0.031336 | 0.031664 | 20,748,741 | 2,737,041 |

Summary:

- Ridge mean correlation: `0.031154`
- Ridge standard deviation: `0.008772`
- Ridge minimum: `0.015654`
- Ridge maximum: `0.046484`

The ridge model improves mean walk-forward correlation by roughly `0.0078`
against the same-fold naive baseline.

### Enhanced Global Ridge

The enhanced ridge model adds market-wide returns, Bitcoin/Ethereum cross-asset
factors, one-hot asset indicators, and asset-return interactions.

| Fold | Enhanced ridge corr | Naive corr |
| ---: | ---: | ---: |
| 3 | 0.053682 | 0.033929 |
| 4 | 0.058863 | 0.027544 |
| 5 | 0.065452 | 0.025715 |
| 6 | 0.022471 | 0.020755 |
| 7 | 0.054724 | 0.038901 |
| 8 | 0.015227 | 0.006378 |
| 9 | 0.041305 | 0.031657 |

Summary:

- Enhanced ridge mean correlation: `0.044532`
- Enhanced ridge standard deviation: `0.017674`
- Enhanced ridge minimum: `0.015227`
- Enhanced ridge maximum: `0.065452`
- Same-fold naive mean correlation: `0.026411`
- Mean improvement: `0.018121`

## Run The Experiments

```bash
python inspect_dataset.py
python analyze_baseline.py
python rolling_ridge_baseline.py
python rolling_ridge_per_asset.py
python rolling_ridge_global_enhanced.py
python rolling_mlp_global.py
```

For the MLP, tune the constants near the top of `rolling_mlp_global.py`:

```python
BATCH_SIZE = 2**13
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
GAP_CHUNKS = 3
```

## GitHub Pages

This repository includes a static project page in `docs/index.md`. In GitHub:

1. Open repository settings.
2. Go to **Pages**.
3. Choose **Deploy from a branch**.
4. Select branch `main` and folder `/docs`.

The page summarizes the competition, modeling approach, validation protocol,
and current offline results.

## Notes

This project is for research and education. It is not financial advice, and the
models here are not production trading systems.
