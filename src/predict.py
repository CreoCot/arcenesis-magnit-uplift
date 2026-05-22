import argparse
import numpy as np
import polars as pl
from pathlib import Path

from src.config import ID, SEED
from src.models import load_models, predict_uplift

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
    "--models",
    default="artifacts/",
    help="Path to directory with saved models",
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
    models = load_models(args.models)
    X_test = df.to_pandas()
    scores = predict_uplift(models, X_test)

out = pl.DataFrame({ID: df[ID], "uplift_score": scores})
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out.write_csv(out_path)

print(f"✅  Saved {len(out)} rows")
