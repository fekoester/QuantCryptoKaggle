"""Walk-forward global MLP for the G-Research Crypto Forecasting dataset.

The script builds leakage-aware, asset-level and market-level features from the
Kaggle minute bars, then trains one global neural network across all assets.
Asset identity is represented with a learned embedding, which lets the model
share statistical strength across coins while still learning asset-specific
behavior.

The validation scheme is chronological: train on earlier chunks, leave a gap,
and validate on a later chunk. This is intentionally stricter than a random
split because financial time series are regime-dependent.
"""

import polars as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training knobs. The defaults are conservative enough for GPU experimentation
# while still fitting in memory on many machines. CPU runs are possible, but
# full-data walk-forward training can be slow.
BATCH_SIZE = 2**13
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
GAP_CHUNKS = 3


def weighted_correlation(y_true, y_pred, weights):
    """Compute the Kaggle weighted Pearson correlation metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    if len(y_true) < 10:
        return np.nan

    y_mean = np.sum(weights * y_true) / np.sum(weights)
    p_mean = np.sum(weights * y_pred) / np.sum(weights)

    yc = y_true - y_mean
    pc = y_pred - p_mean

    cov = np.sum(weights * yc * pc)
    var_y = np.sum(weights * yc * yc)
    var_p = np.sum(weights * pc * pc)

    if var_y <= 0 or var_p <= 0:
        return np.nan

    return cov / np.sqrt(var_y * var_p)


class MLP(nn.Module):
    """Small tabular neural net with a learned cryptocurrency embedding."""

    def __init__(self, n_numeric_features, n_assets=14, asset_emb_dim=8):
        super().__init__()

        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(n_numeric_features + asset_emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x_num, asset_id):
        emb = self.asset_emb(asset_id)
        x = torch.cat([x_num, emb], dim=1)
        return self.net(x).squeeze(-1)


def build_features():
    """Create lagged, cross-asset, volatility, volume, and time features.

    Every raw price/volume feature is shifted by one observation within each
    asset before use. That keeps the feature matrix aligned with information
    that would have been known before the target row.
    """
    assets = pl.read_csv("raw/asset_details.csv")

    RETURN_LAGS = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]
    VOL_WINDOWS = [5, 15, 30, 60, 120]
    EPS = 1e-8

    base = (
        pl.scan_csv("raw/train.csv")
        .join(assets.lazy(), on="Asset_ID", how="left")
        .sort(["Asset_ID", "timestamp"])
        .with_columns(
            [
                (
                    pl.col("Close").log()
                    - pl.col("Close").shift(lag).over("Asset_ID").log()
                ).alias(f"ret_{lag}m_raw")
                for lag in RETURN_LAGS
            ]
            + [
                pl.col("Volume").log1p().alias("log_volume_raw"),

                (
                    pl.col("Volume").log1p()
                    - pl.col("Volume").shift(15).over("Asset_ID").log1p()
                ).alias("volume_change_15m_raw"),

                (pl.col("High").log() - pl.col("Low").log()).alias("hl_range_raw"),
                (pl.col("Close").log() - pl.col("Open").log()).alias("co_return_raw"),
            ]
        )
        .with_columns(
            [
                pl.col("ret_1m_raw")
                .rolling_std(window_size=w)
                .over("Asset_ID")
                .alias(f"volatility_{w}m_raw")
                for w in VOL_WINDOWS
            ]
            + [
                pl.col("log_volume_raw")
                .rolling_mean(window_size=60)
                .over("Asset_ID")
                .alias("log_volume_mean_60m_raw"),
            ]
        )
        .with_columns(
            [
                pl.col(f"ret_{lag}m_raw")
                .shift(1)
                .over("Asset_ID")
                .alias(f"ret_{lag}m")
                for lag in RETURN_LAGS
            ]
            + [
                pl.col("log_volume_raw").shift(1).over("Asset_ID").alias("log_volume"),
                pl.col("volume_change_15m_raw").shift(1).over("Asset_ID").alias("volume_change_15m"),
                pl.col("hl_range_raw").shift(1).over("Asset_ID").alias("hl_range"),
                pl.col("co_return_raw").shift(1).over("Asset_ID").alias("co_return"),

                (
                    pl.col("log_volume_raw")
                    - pl.col("log_volume_mean_60m_raw")
                ).shift(1).over("Asset_ID").alias("volume_z_60m"),
            ]
            + [
                pl.col(f"volatility_{w}m_raw")
                .shift(1)
                .over("Asset_ID")
                .alias(f"volatility_{w}m")
                for w in VOL_WINDOWS
            ]
        )
    )

    market = (
        base
        .group_by("timestamp")
        .agg([
            (
                (pl.col(f"ret_{lag}m") * pl.col("Weight")).sum()
                / pl.col("Weight").sum()
            ).alias(f"mkt_ret_{lag}m")
            for lag in RETURN_LAGS
        ])
    )

    df = base.join(market, on="timestamp", how="left")

    btc = (
        df
        .filter(pl.col("Asset_ID") == 1)
        .sort("timestamp")
        .select(
            ["timestamp"]
            + [
                pl.col(f"ret_{lag}m").alias(f"btc_ret_{lag}m")
                for lag in RETURN_LAGS
            ]
        )
    )

    eth = (
        df
        .filter(pl.col("Asset_ID") == 6)
        .sort("timestamp")
        .select(
            ["timestamp"]
            + [
                pl.col(f"ret_{lag}m").alias(f"eth_ret_{lag}m")
                for lag in RETURN_LAGS
            ]
        )
    )

    df = df.join(btc, on="timestamp", how="left").join(eth, on="timestamp", how="left")

    # Relative and normalized features.
    df = df.with_columns(
        [
            (pl.col(f"ret_{lag}m") - pl.col(f"mkt_ret_{lag}m")).alias(f"rel_mkt_ret_{lag}m")
            for lag in RETURN_LAGS
        ]
        + [
            (pl.col(f"ret_{lag}m") - pl.col(f"btc_ret_{lag}m")).alias(f"rel_btc_ret_{lag}m")
            for lag in RETURN_LAGS
        ]
        + [
            (pl.col(f"ret_{lag}m") - pl.col(f"eth_ret_{lag}m")).alias(f"rel_eth_ret_{lag}m")
            for lag in RETURN_LAGS
        ]
        + [
            (pl.col("ret_15m") / (pl.col("volatility_60m") + EPS)).alias("ret_15m_over_vol_60m"),
            (pl.col("ret_60m") / (pl.col("volatility_120m") + EPS)).alias("ret_60m_over_vol_120m"),
        ]
    )

    # Time features. Timestamp is seconds since epoch.
    seconds_per_day = 24 * 60 * 60
    seconds_per_week = 7 * seconds_per_day

    df = df.with_columns([
        (2 * np.pi * (pl.col("timestamp") % seconds_per_day) / seconds_per_day).sin().alias("hour_sin"),
        (2 * np.pi * (pl.col("timestamp") % seconds_per_day) / seconds_per_day).cos().alias("hour_cos"),
        (2 * np.pi * (pl.col("timestamp") % seconds_per_week) / seconds_per_week).sin().alias("week_sin"),
        (2 * np.pi * (pl.col("timestamp") % seconds_per_week) / seconds_per_week).cos().alias("week_cos"),
    ])

    own_return_features = [f"ret_{lag}m" for lag in RETURN_LAGS]
    market_features = [f"mkt_ret_{lag}m" for lag in RETURN_LAGS]
    btc_features = [f"btc_ret_{lag}m" for lag in RETURN_LAGS]
    eth_features = [f"eth_ret_{lag}m" for lag in RETURN_LAGS]

    rel_mkt_features = [f"rel_mkt_ret_{lag}m" for lag in RETURN_LAGS]
    rel_btc_features = [f"rel_btc_ret_{lag}m" for lag in RETURN_LAGS]
    rel_eth_features = [f"rel_eth_ret_{lag}m" for lag in RETURN_LAGS]

    volatility_features = [f"volatility_{w}m" for w in VOL_WINDOWS]

    feature_cols = (
        own_return_features
        + [
            "log_volume",
            "volume_change_15m",
            "volume_z_60m",
            "hl_range",
            "co_return",
        ]
        + volatility_features
        + [
            "ret_15m_over_vol_60m",
            "ret_60m_over_vol_120m",
        ]
        + market_features
        + btc_features
        + eth_features
        + rel_mkt_features
        + rel_btc_features
        + rel_eth_features
        + [
            "hour_sin",
            "hour_cos",
            "week_sin",
            "week_cos",
        ]
    )

    df = (
        df
        .select(["timestamp", "Asset_ID", "Weight", "Target"] + feature_cols)
        .drop_nulls()
        .collect()
    )

    return df, feature_cols


def standardize_train_valid(X_train, X_valid):
    """Fit standardization on training rows and apply it to validation rows."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0

    return (X_train - mean) / std, (X_valid - mean) / std


def train_one_fold(train, valid, feature_cols, fold_id):
    """Train one chronological fold and return best MLP and naive scores."""
    X_train = train.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train["Target"].to_numpy().astype(np.float32)

    X_valid = valid.select(feature_cols).to_numpy().astype(np.float32)
    y_valid = valid["Target"].to_numpy().astype(np.float32)
    w_valid = valid["Weight"].to_numpy().astype(np.float32)

    asset_train = train["Asset_ID"].to_numpy().astype(np.int64)
    asset_valid = valid["Asset_ID"].to_numpy().astype(np.int64)

    X_train, X_valid = standardize_train_valid(X_train, X_valid)

    # Standardize target for easier neural-net optimization.
    y_mean = np.nanmean(y_train)
    y_std = np.nanstd(y_train)
    if y_std < 1e-12:
        y_std = 1.0

    y_train_std = (y_train - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(asset_train),
        torch.from_numpy(y_train_std),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # Multiprocessing workers can be blocked in restricted notebook/sandbox
        # environments. CUDA runs benefit from workers and pinned memory; CPU
        # runs stay single-process for portability.
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=DEVICE == "cuda",
    )

    model = MLP(len(feature_cols)).to(DEVICE)
    asset_valid_t = torch.from_numpy(asset_valid).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP trainable parameters: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    Xv = torch.from_numpy(X_valid).to(DEVICE)

    best_score = -999.0
    best_epoch = -1

    print(f"\nfold {fold_id:02d}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"fold {fold_id:02d} epoch {epoch:02d}",
            leave=False,
        )

        for xb, asset_b, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            asset_b = asset_b.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, asset_b)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}"
            })

            total_loss += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]

        model.eval()
        preds = []

        with torch.no_grad():
            for start in range(0, X_valid.shape[0], BATCH_SIZE):
                pv = model(
                    Xv[start:start + BATCH_SIZE],
                    asset_valid_t[start:start + BATCH_SIZE],
                )
                preds.append(pv.detach().cpu().numpy())

        pred_valid_std = np.concatenate(preds)
        pred_valid = pred_valid_std * y_std + y_mean

        score = weighted_correlation(y_valid, pred_valid, w_valid)

        if score > best_score:
            best_score = score
            best_epoch = epoch

        print(
            f"epoch {epoch:02d}: "
            f"train_mse = {total_loss / n_seen:.6f}, "
            f"valid_corr = {score:.6f}, "
            f"best = {best_score:.6f} @ epoch {best_epoch}"
        )

    naive_score = weighted_correlation(
        y_valid,
        -valid["ret_15m"].to_numpy(),
        w_valid,
    )

    return best_score, naive_score


def main():
    print(f"Using device: {DEVICE}")

    df, feature_cols = build_features()

    print(f"Rows after features/drop_nulls: {df.height:,}")
    print(f"Number of features: {len(feature_cols)}")
    

    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    n_chunks = 10
    edges = np.linspace(t_min, t_max, n_chunks + 1).astype(int)
    print(f"Gap chunks: {GAP_CHUNKS}")
    print(f"Each chunk length: {(edges[1] - edges[0]) / 86400:.1f} days")

    scores = []
    naive_scores = []

    for k in range(3 + GAP_CHUNKS, n_chunks):
        train = df.filter(
            (pl.col("timestamp") >= edges[0]) &
            (pl.col("timestamp") < edges[k - GAP_CHUNKS])
        )

        valid = df.filter(
            (pl.col("timestamp") >= edges[k]) &
            (pl.col("timestamp") < edges[k + 1])
        )

        score, naive_score = train_one_fold(train, valid, feature_cols, k)

        scores.append(score)
        naive_scores.append(naive_score)

        print(
            f"fold {k:02d} result: "
            f"best MLP corr = {score:.6f}, "
            f"naive corr = {naive_score:.6f}, "
            f"train rows = {train.height:,}, "
            f"valid rows = {valid.height:,}"
        )

    print("\nSummary:")
    print(f"MLP mean corr = {np.nanmean(scores): .6f}")
    print(f"MLP std  corr = {np.nanstd(scores): .6f}")
    print(f"MLP min  corr = {np.nanmin(scores): .6f}")
    print(f"MLP max  corr = {np.nanmax(scores): .6f}")
    print()
    print(f"naive mean corr = {np.nanmean(naive_scores): .6f}")
    print(f"mean improvement over naive = {np.nanmean(scores) - np.nanmean(naive_scores): .6f}")


if __name__ == "__main__":
    main()
