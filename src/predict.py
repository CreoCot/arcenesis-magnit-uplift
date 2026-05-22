import argparse
import numpy as np
import polars as pl

from src.config import ID, SEED

p = argparse.ArgumentParser()
p.add_argument(
    "--test",
    default="data/test.parquet",
    help="Path to input test parquet file (default: data/test.parquet)",
)
p.add_argument(
    "--out",
    default="artifacts/predictions.csv",
    help="Path to save predictions CSV (default: artifacts/predictions.csv)",
)
p.add_argument(
    "--stub", action="store_true", help="Flag to run with random stub predictions"
)
args = p.parse_args()

df = pl.read_parquet(args.test)

if args.stub:
    np.random.seed(SEED)
    scores = np.random.uniform(-67, 67, size=len(df))
else:
    raise NotImplementedError("real predict позже")

out = pl.DataFrame({ID: df[ID], "uplift_score": scores})
out.write_csv(args.out)
print(f"saved {len(out)} rows")
