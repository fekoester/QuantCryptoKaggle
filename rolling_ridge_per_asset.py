import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def weighted_correlation(y_true, y_pred, weights):
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


assets = pl.read_csv("raw/asset_details.csv")

df = (
    pl.scan_csv("raw/train.csv")
    .join(assets.lazy(), on="Asset_ID", how="left")
    .sort(["Asset_ID", "timestamp"])
    .with_columns([
        (pl.col("Close").log() - pl.col("Close").shift(1).over("Asset_ID").log()).alias("ret_1m"),
        (pl.col("Close").log() - pl.col("Close").shift(5).over("Asset_ID").log()).alias("ret_5m"),
        (pl.col("Close").log() - pl.col("Close").shift(15).over("Asset_ID").log()).alias("ret_15m"),
        (pl.col("Close").log() - pl.col("Close").shift(60).over("Asset_ID").log()).alias("ret_60m"),
        pl.col("Volume").log1p().alias("log_volume"),
        (
            pl.col("Volume").log1p()
            - pl.col("Volume").shift(15).over("Asset_ID").log1p()
        ).alias("volume_change_15m"),
        (
            pl.col("Close").log()
            - pl.col("Close").shift(1).over("Asset_ID").log()
        ).rolling_std(window_size=15).over("Asset_ID").alias("volatility_15m"),
    ])
    .select([
        "timestamp",
        "Asset_ID",
        "Asset_Name",
        "Weight",
        "Target",
        "ret_1m",
        "ret_5m",
        "ret_15m",
        "ret_60m",
        "log_volume",
        "volume_change_15m",
        "volatility_15m",
    ])
    .drop_nulls()
    .collect()
)

feature_cols = [
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "log_volume",
    "volume_change_15m",
    "volatility_15m",
]

t_min = df["timestamp"].min()
t_max = df["timestamp"].max()

n_chunks = 10
edges = np.linspace(t_min, t_max, n_chunks + 1).astype(int)

print(f"Rows after features/drop_nulls: {df.height:,}")
print(f"Time range: {t_min} -> {t_max}")
print()
print("Walk-forward per-asset ridge scores:")

scores = []
naive_scores = []

for k in range(3, n_chunks):
    train_lo = edges[0]
    train_hi = edges[k]
    valid_lo = edges[k]
    valid_hi = edges[k + 1]

    train_all = df.filter(
        (pl.col("timestamp") >= train_lo) &
        (pl.col("timestamp") < train_hi)
    )

    valid_all = df.filter(
        (pl.col("timestamp") >= valid_lo) &
        (pl.col("timestamp") < valid_hi)
    )

    preds = np.full(valid_all.height, np.nan)

    valid_asset_ids = valid_all["Asset_ID"].to_numpy()

    for asset_id in sorted(df["Asset_ID"].unique().to_list()):
        train = train_all.filter(pl.col("Asset_ID") == asset_id)
        valid = valid_all.filter(pl.col("Asset_ID") == asset_id)

        if train.height < 1000 or valid.height == 0:
            continue

        X_train = train.select(feature_cols).to_numpy()
        y_train = train["Target"].to_numpy()

        X_valid = valid.select(feature_cols).to_numpy()

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )

        model.fit(X_train, y_train)

        asset_pred = model.predict(X_valid)

        mask = valid_asset_ids == asset_id
        preds[mask] = asset_pred

    y_valid = valid_all["Target"].to_numpy()
    w_valid = valid_all["Weight"].to_numpy()

    score = weighted_correlation(y_valid, preds, w_valid)
    scores.append(score)

    naive_pred = -valid_all["ret_15m"].to_numpy()
    naive_score = weighted_correlation(y_valid, naive_pred, w_valid)
    naive_scores.append(naive_score)

    print(
        f"fold {k:02d}: "
        f"per-asset ridge corr = {score: .6f}, "
        f"naive corr = {naive_score: .6f}, "
        f"train rows = {train_all.height:,}, "
        f"valid rows = {valid_all.height:,}"
    )

print()
print("Summary:")
print(f"per-asset ridge mean corr = {np.nanmean(scores): .6f}")
print(f"per-asset ridge std  corr = {np.nanstd(scores): .6f}")
print(f"per-asset ridge min  corr = {np.nanmin(scores): .6f}")
print(f"per-asset ridge max  corr = {np.nanmax(scores): .6f}")
print()
print(f"naive mean corr = {np.nanmean(naive_scores): .6f}")
print(f"mean improvement = {np.nanmean(scores) - np.nanmean(naive_scores): .6f}")
