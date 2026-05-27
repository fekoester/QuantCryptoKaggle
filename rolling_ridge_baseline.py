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

    w_sum = np.sum(weights)
    y_mean = np.sum(weights * y_true) / w_sum
    p_mean = np.sum(weights * y_pred) / w_sum

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
    .with_columns([
        pl.col("Asset_ID").cast(pl.Float64).alias("asset_id_float"),
    ])
    .select([
        "timestamp",
        "Asset_ID",
        "Weight",
        "Target",
        "ret_1m",
        "ret_5m",
        "ret_15m",
        "ret_60m",
        "log_volume",
        "volume_change_15m",
        "volatility_15m",
        "asset_id_float",
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
    "asset_id_float",
]

t_min = df["timestamp"].min()
t_max = df["timestamp"].max()

n_chunks = 10
edges = np.linspace(t_min, t_max, n_chunks + 1).astype(int)

print(f"Rows after features/drop_nulls: {df.height:,}")
print(f"Time range: {t_min} -> {t_max}")
print()
print("Walk-forward ridge scores:")

scores = []

# Train on chunks [0 ... k-1], validate on chunk k.
# Start at k=3 so the model has enough history.
for k in range(3, n_chunks):
    train_lo = edges[0]
    train_hi = edges[k]
    valid_lo = edges[k]
    valid_hi = edges[k + 1]

    train = df.filter(
        (pl.col("timestamp") >= train_lo) &
        (pl.col("timestamp") < train_hi)
    )

    valid = df.filter(
        (pl.col("timestamp") >= valid_lo) &
        (pl.col("timestamp") < valid_hi)
    )

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["Target"].to_numpy()

    X_valid = valid.select(feature_cols).to_numpy()
    y_valid = valid["Target"].to_numpy()
    w_valid = valid["Weight"].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    score = weighted_correlation(y_valid, pred, w_valid)
    scores.append(score)

    # Also compare directly against naive -ret_15m on same fold
    naive_pred = -valid["ret_15m"].to_numpy()
    naive_score = weighted_correlation(y_valid, naive_pred, w_valid)

    print(
        f"fold {k:02d}: "
        f"ridge corr = {score: .6f}, "
        f"naive corr = {naive_score: .6f}, "
        f"train rows = {train.height:,}, "
        f"valid rows = {valid.height:,}"
    )

print()
print("Summary:")
print(f"ridge mean corr = {np.nanmean(scores): .6f}")
print(f"ridge std  corr = {np.nanstd(scores): .6f}")
print(f"ridge min  corr = {np.nanmin(scores): .6f}")
print(f"ridge max  corr = {np.nanmax(scores): .6f}")
