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

    return cov / np.sqrt(var_y * var_p)


assets = pl.read_csv("raw/asset_details.csv")

df = (
    pl.scan_csv("raw/train.csv")
    .join(assets.lazy(), on="Asset_ID", how="left")
    .sort(["Asset_ID", "timestamp"])
    .with_columns([
        (
            -(
                pl.col("Close").log()
                - pl.col("Close").shift(15).over("Asset_ID").log()
            )
        ).alias("pred_reversal_15m")
    ])
    .select([
        "timestamp",
        "Asset_ID",
        "Asset_Name",
        "Weight",
        "Target",
        "pred_reversal_15m",
    ])
)

time_bounds = df.select([
    pl.col("timestamp").min().alias("t_min"),
    pl.col("timestamp").max().alias("t_max"),
]).collect()

t_min = time_bounds["t_min"][0]
t_max = time_bounds["t_max"][0]

print(f"Full time range: {t_min} -> {t_max}")

# Use 10 chronological chunks over the full dataset
n_chunks = 10
edges = np.linspace(t_min, t_max, n_chunks + 1).astype(int)

print("\nChunk scores over full dataset:")
chunk_scores = []

for i in range(n_chunks):
    lo = edges[i]
    hi = edges[i + 1]

    sub = (
        df
        .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        .collect()
    )

    score = weighted_correlation(
        sub["Target"].to_numpy(),
        sub["pred_reversal_15m"].to_numpy(),
        sub["Weight"].to_numpy(),
    )

    chunk_scores.append(score)
    print(f"chunk {i:02d}: corr = {score: .6f}, rows = {sub.height:,}")

print("\nChunk summary:")
print(f"mean corr = {np.nanmean(chunk_scores): .6f}")
print(f"std  corr = {np.nanstd(chunk_scores): .6f}")
print(f"min  corr = {np.nanmin(chunk_scores): .6f}")
print(f"max  corr = {np.nanmax(chunk_scores): .6f}")


# Per-asset score on last 20% validation period
t_split = int(t_min + 0.8 * (t_max - t_min))

valid = df.filter(pl.col("timestamp") >= t_split).collect()

print("\nPer-asset scores on last 20%:")
for row in assets.sort("Asset_ID").iter_rows(named=True):
    asset_id = row["Asset_ID"]
    name = row["Asset_Name"]

    sub = valid.filter(pl.col("Asset_ID") == asset_id)

    score = weighted_correlation(
        sub["Target"].to_numpy(),
        sub["pred_reversal_15m"].to_numpy(),
        sub["Weight"].to_numpy(),
    )

    print(f"{asset_id:2d} {name:20s} corr = {score: .6f}, rows = {sub.height:,}")
