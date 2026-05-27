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

    y_mean = np.sum(weights * y_true) / np.sum(weights)
    p_mean = np.sum(weights * y_pred) / np.sum(weights)

    yc = y_true - y_mean
    pc = y_pred - p_mean

    cov = np.sum(weights * yc * pc)
    var_y = np.sum(weights * yc * yc)
    var_p = np.sum(weights * pc * pc)

    return cov / np.sqrt(var_y * var_p)


assets = pl.read_csv("raw/asset_details.csv")

base = (
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
)

# Market-wide features at each timestamp.
market = (
    base
    .group_by("timestamp")
    .agg([
        ((pl.col("ret_1m") * pl.col("Weight")).sum() / pl.col("Weight").sum()).alias("mkt_ret_1m"),
        ((pl.col("ret_5m") * pl.col("Weight")).sum() / pl.col("Weight").sum()).alias("mkt_ret_5m"),
        ((pl.col("ret_15m") * pl.col("Weight")).sum() / pl.col("Weight").sum()).alias("mkt_ret_15m"),
        ((pl.col("ret_60m") * pl.col("Weight")).sum() / pl.col("Weight").sum()).alias("mkt_ret_60m"),
    ])
)

df = (
    base
    .join(market, on="timestamp", how="left")
)

# Add BTC/ETH lag features as cross-asset factors.
btc = (
    df
    .filter(pl.col("Asset_ID") == 1)
    .select([
        "timestamp",
        pl.col("ret_15m").alias("btc_ret_15m"),
        pl.col("ret_60m").alias("btc_ret_60m"),
    ])
)

eth = (
    df
    .filter(pl.col("Asset_ID") == 6)
    .select([
        "timestamp",
        pl.col("ret_15m").alias("eth_ret_15m"),
        pl.col("ret_60m").alias("eth_ret_60m"),
    ])
)

df = (
    df
    .join(btc, on="timestamp", how="left")
    .join(eth, on="timestamp", how="left")
)

# One-hot asset features and interactions.
for asset_id in range(14):
    is_asset = (pl.col("Asset_ID") == asset_id).cast(pl.Float64)

    df = df.with_columns([
        is_asset.alias(f"asset_{asset_id}"),

        (pl.col("ret_1m") * is_asset).alias(f"asset_{asset_id}_ret_1m"),
        (pl.col("ret_5m") * is_asset).alias(f"asset_{asset_id}_ret_5m"),
        (pl.col("ret_15m") * is_asset).alias(f"asset_{asset_id}_ret_15m"),
        (pl.col("ret_60m") * is_asset).alias(f"asset_{asset_id}_ret_60m"),
    ])

base_features = [
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "log_volume",
    "volume_change_15m",
    "volatility_15m",
    "mkt_ret_1m",
    "mkt_ret_5m",
    "mkt_ret_15m",
    "mkt_ret_60m",
    "btc_ret_15m",
    "btc_ret_60m",
    "eth_ret_15m",
    "eth_ret_60m",
]

asset_features = [f"asset_{i}" for i in range(14)]

interaction_features = []
for i in range(14):
    interaction_features += [
        f"asset_{i}_ret_1m",
        f"asset_{i}_ret_5m",
        f"asset_{i}_ret_15m",
        f"asset_{i}_ret_60m",
    ]

feature_cols = base_features + asset_features + interaction_features

df = (
    df
    .select(["timestamp", "Asset_ID", "Weight", "Target"] + feature_cols)
    .drop_nulls()
    .collect()
)

print(f"Rows after features/drop_nulls: {df.height:,}")
print(f"Number of features: {len(feature_cols)}")
print(f"Linear weights including bias: {len(feature_cols) + 1}")

t_min = df["timestamp"].min()
t_max = df["timestamp"].max()

n_chunks = 10
edges = np.linspace(t_min, t_max, n_chunks + 1).astype(int)

scores = []
naive_scores = []

print("\nWalk-forward enhanced global ridge scores:")

for k in range(3, n_chunks):
    train = df.filter(
        (pl.col("timestamp") >= edges[0]) &
        (pl.col("timestamp") < edges[k])
    )

    valid = df.filter(
        (pl.col("timestamp") >= edges[k]) &
        (pl.col("timestamp") < edges[k + 1])
    )

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["Target"].to_numpy()

    X_valid = valid.select(feature_cols).to_numpy()
    y_valid = valid["Target"].to_numpy()
    w_valid = valid["Weight"].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    )

    model.fit(X_train, y_train)
    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["standardscaler"]

    coef = ridge.coef_ / scaler.scale_

    coef_table = sorted(
        zip(feature_cols, coef),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    print("\nTop coefficients for this fold:")
    for name, value in coef_table[:20]:
        print(f"{name:30s} {value: .8f}")
    pred = model.predict(X_valid)

    score = weighted_correlation(y_valid, pred, w_valid)
    scores.append(score)

    naive_score = weighted_correlation(
        y_valid,
        -valid["ret_15m"].to_numpy(),
        w_valid,
    )
    naive_scores.append(naive_score)

    print(
        f"fold {k:02d}: "
        f"enhanced ridge corr = {score: .6f}, "
        f"naive corr = {naive_score: .6f}, "
        f"train rows = {train.height:,}, "
        f"valid rows = {valid.height:,}"
    )

print("\nSummary:")
print(f"enhanced ridge mean corr = {np.nanmean(scores): .6f}")
print(f"enhanced ridge std  corr = {np.nanstd(scores): .6f}")
print(f"enhanced ridge min  corr = {np.nanmin(scores): .6f}")
print(f"enhanced ridge max  corr = {np.nanmax(scores): .6f}")
print()
print(f"naive mean corr = {np.nanmean(naive_scores): .6f}")
print(f"mean improvement = {np.nanmean(scores) - np.nanmean(naive_scores): .6f}")
