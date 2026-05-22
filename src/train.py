import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from src.config import FOLDS, SEED, T, Y
from src.models import predict_uplift, save_models, train_hurdle
from src.validation import make_folds, uplift_at_k, uplift_at_k_lb

def load_train(path):
    print("loading train")
    df = pl.read_parquet(path)
    print(df.shape)
    return df.to_pandas()

def run_cv(df, n_folds):
    folds = make_folds(df[T].values, df[Y].values, n_folds, SEED)
    metrics = []
    for i, fold in enumerate(folds):
        print()
        print("fold", i + 1)
        start = time.time()
        train_idx = fold[0]
        val_idx = fold[1]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        models = train_hurdle(df_train)
        score = predict_uplift(models, df_val)
        y_val = df_val[Y].values
        t_val = df_val[T].values
        u10 = uplift_at_k(y_val, t_val, score, 0.1)
        u10_mean, u10_lb = uplift_at_k_lb(y_val, t_val, score, 0.1)
        duration = round(time.time() - start, 1)
        result = {
            "fold": i + 1,
            "uplift_10": float(u10),
            "uplift_10_lb": float(u10_lb),
            "duration_sec": duration,
        }
        metrics.append(result)
        print("uplift@10:", round(u10, 4))
        print("uplift@10 lb:", round(u10_lb, 4))
        print("time:", duration)
    return metrics

def print_report(metrics):
    if len(metrics) == 0:
        return
    uplift_values = []
    lb_values = []
    for m in metrics:
        uplift_values.append(m["uplift_10"])
        lb_values.append(m["uplift_10_lb"])
    print()
    print("report")
    print("mean uplift@10:", round(np.mean(uplift_values), 4))
    print("mean lb:", round(np.mean(lb_values), 4))
    print("std uplift@10:", round(np.std(uplift_values), 4))

def make_metric_json(df, metrics, n_folds, train_time, skip_cv):
    uplift_values = []
    lb_values = []
    for m in metrics:
        uplift_values.append(m["uplift_10"])
        lb_values.append(m["uplift_10_lb"])
    result = {
        "n_rows": int(len(df)),
        "n_folds": n_folds,
        "fold_metrics": metrics,
        "cv_mean_uplift_10": None,
        "cv_mean_uplift_10_lb": None,
        "cv_std_uplift_10": None,
        "final_train_time_sec": train_time,
        "seed": SEED,
    }
    if skip_cv:
        result["n_folds"] = 0
    if len(metrics) > 0:
        result["cv_mean_uplift_10"] = float(np.mean(uplift_values))
        result["cv_mean_uplift_10_lb"] = float(np.mean(lb_values))
        result["cv_std_uplift_10"] = float(np.std(uplift_values))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.parquet")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--n-folds", type=int, default=FOLDS)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    train_path = Path(args.train)
    out_dir = Path(args.out)
    df = load_train(train_path)
    if args.sample is not None:
        parts = []
        for value, group in df.groupby(T):
            n = args.sample // 2
            n = min(n, len(group))
            part = group.sample(n, random_state=SEED)
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        print("sample shape")
        print(df.shape)
    metrics = []
    if not args.skip_cv:
        print()
        print("cross validation")
        metrics = run_cv(df, args.n_folds)
        print_report(metrics)
    print()
    print("final train")
    start = time.time()
    models = train_hurdle(df)
    train_time = round(time.time() - start, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_models(models, out_dir)
    print("models saved")
    print(out_dir)
    result = make_metric_json(df, metrics, args.n_folds, train_time, args.skip_cv)
    metric_path = out_dir / "baseline_metric.json"
    with open(metric_path, "w") as f:
        json.dump(result, f, indent=2)
    print("metrics saved")
    print(metric_path)

if __name__ == "__main__":
    main()
