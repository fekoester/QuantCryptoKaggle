import polars as pl
import numpy as np


def weighted_correlation(y_true, y_pred, weights):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights)

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    y_mean = np.sum(weights * y_true) / np.sum(weights)
    p_mean = np.sum(weights * y_pred) / np.sum(weights)

    yc = y_true - y_mean
    pc = y_pred - p_mean

    cov = np.sum(weights * yc * pc)
    var_y = np.sum(weights * yc * yc)
    var_p = np.sum(weights * pc * pc)

    return cov / np.sqrt(var_y * var_p)


# Load asset weights
assets = pl.read_csv("raw/asset_details.csv")

# Lazy-load train data
df = (
    pl.scan_csv("raw/train.csv")
    .join(assets.lazy(), on="Asset_ID", how="left")
    .sort(["Asset_ID", "timestamp"])
    .with_columns([
        # 1-minute log return
        (pl.col("Close").log() - pl.col("Close").shift(1).over("Asset_ID").log())
        .alias("ret_1m"),

        # 15-minute lagged log return
        (pl.col("Close").log() - pl.col("Close").shift(15).over("Asset_ID").log())
        .alias("ret_15m"),
    ])
    .select([
        "timestamp",
        "Asset_ID",
        "Asset_Name",
        "Weight",
        "Target",
        "ret_1m",
        "ret_15m",
    ])
)

# Use last 20% of time as validation
time_bounds = df.select([
    pl.col("timestamp").min().alias("t_min"),
    pl.col("timestamp").max().alias("t_max"),
]).collect()

t_min = time_bounds["t_min"][0]
t_max = time_bounds["t_max"][0]
t_split = int(t_min + 0.8 * (t_max - t_min))

valid = (
    df
    .filter(pl.col("timestamp") >= t_split)
    .collect()
)

print(f"Validation rows: {valid.height:,}")
print(f"Validation time split: {t_split}")

for pred_col in ["ret_1m", "ret_15m"]:
    score = weighted_correlation(
        valid["Target"].to_numpy(),
        valid[pred_col].to_numpy(),
        valid["Weight"].to_numpy(),
    )
    print(f"{pred_col:>10s} weighted corr = {score:.6f}")

# Also check sign-flipped version.
# Finance often has either momentum or mean reversion.
for pred_col in ["ret_1m", "ret_15m"]:
    score = weighted_correlation(
        valid["Target"].to_numpy(),
        -valid[pred_col].to_numpy(),
        valid["Weight"].to_numpy(),
    )
    print(f"{'-' + pred_col:>10s} weighted corr = {score:.6f}")
