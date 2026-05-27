# Quant Crypto Kaggle

Walk-forward crypto return forecasting experiments for Kaggle's
[G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting)
competition.

## What This Project Shows

This project studies short-horizon cryptocurrency return prediction using the
competition's official weighted-correlation objective. The repository starts
with an intentionally simple 15-minute mean-reversion baseline, then evaluates
rolling ridge regressions and a global neural network with asset embeddings.

The validation protocol is chronological. Each fold trains on earlier market
history and evaluates on a later time block, avoiding random splits that would
leak future market regimes into training.

## Current Offline Results

| Model | Validation protocol | Mean weighted correlation |
| --- | --- | ---: |
| Naive `-ret_15m` baseline | 10 chronological chunks | 0.023366 |
| Rolling global ridge | Expanding walk-forward folds 3-9 | 0.031154 |
| Enhanced global ridge | Cross-asset expanding walk-forward folds 3-9 | 0.044532 |
| Global MLP with asset embeddings | Verified startup; full scoring needs GPU | not fully scored |

The enhanced ridge model improves mean correlation by `0.018121` over its
same-fold naive baseline. The signal is small, which is normal for noisy
financial forecasting tasks, but the walk-forward lift is consistent enough to
be worth further modeling.

The MLP implementation is included in `rolling_mlp_global.py`. It builds the
full feature matrix and starts training on CPU, but the complete 24-million-row
walk-forward run is intended for a CUDA/GPU environment.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put Kaggle files in raw/ first.
python inspect_dataset.py
python analyze_baseline.py
python rolling_ridge_baseline.py
python rolling_mlp_global.py
```

The MLP is intended for GPU use. It will run on CPU, but the full 24-million-row
walk-forward experiment can take a long time.
