# MLP GPU Results

Full `rolling_mlp_global.py` run on an NVIDIA GeForce RTX 4090.

Run configuration:

- Device: CUDA
- Rows after feature engineering: 23,482,291
- Numeric features: 86
- Train/validation gap: 3 chronological chunks
- Epochs per fold: 10
- Trainable parameters: 23,089

## Fold Results

| Fold | Best MLP corr | Naive corr | Best epoch | Train rows | Valid rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.045439 | 0.018026 | 2 | 6,005,158 | 2,328,024 |
| 7 | 0.047253 | 0.027357 | 4 | 8,436,079 | 2,654,072 |
| 8 | 0.053706 | 0.004554 | 6 | 10,699,712 | 2,738,755 |
| 9 | 0.060615 | 0.029873 | 2 | 13,024,648 | 2,736,778 |

## Summary

- MLP mean correlation: 0.051753
- MLP std correlation: 0.005968
- MLP min correlation: 0.045439
- MLP max correlation: 0.060615
- Naive mean correlation: 0.019952
- Mean improvement over naive: 0.031801
