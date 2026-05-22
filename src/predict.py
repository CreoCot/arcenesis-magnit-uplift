import argparse
import numpy as np
import polars as pl

from src.config import ID, SEED

p = argparse.ArgumentParser()
p.add_argument("--test", required=True)
p.add_argument("--out", required=True)
p.add_argument("--stub", action="store_true")
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
