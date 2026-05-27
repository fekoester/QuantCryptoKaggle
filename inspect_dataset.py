import polars as pl

print("\nAsset details:")
assets = pl.read_csv("raw/asset_details.csv")
print(assets)

print("\nTrain schema:")
train = pl.scan_csv("raw/train.csv")
print(train.collect_schema())

print("\nFirst rows:")
print(train.head(5).collect())

print("\nRows per asset:")
counts = (
    train
    .group_by("Asset_ID")
    .len()
    .sort("Asset_ID")
    .collect()
)
print(counts)
