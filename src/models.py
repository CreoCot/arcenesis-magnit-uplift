import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

from src.config import CATS, DEPTH, DROP, ES, ITERS, LR, SEED, T, Y


def fit_one(model, X, y, stratify=None):
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=stratify)
    model.fit(Xt, yt, eval_set=(Xv, yv))
    return model


def train_hurdle(df):
    features = []
    for c in df.columns:
        if c not in DROP:
            features.append(c)

    cats = []
    for c in CATS:
        if c in features:
            cats.append(c)

    X = df[features]
    y = df[Y].values
    t = df[T].values
    buy = (y > 0).astype(int)

    params = {
        "iterations": ITERS,
        "learning_rate": LR,
        "depth": DEPTH,
        "random_seed": SEED,
        "verbose": False,
        "early_stopping_rounds": ES,
        "cat_features": cats,
    }

    mask = t == 1
    p_treat = fit_one(CatBoostClassifier(**params), X[mask], buy[mask], stratify=buy[mask])

    mask = t == 0
    p_ctrl = fit_one(CatBoostClassifier(**params), X[mask], buy[mask], stratify=buy[mask])

    mask = (t == 1) & (y > 0)
    e_treat = fit_one(CatBoostRegressor(**params), X[mask], np.log1p(y[mask]))

    mask = (t == 0) & (y > 0)
    e_ctrl = fit_one(CatBoostRegressor(**params), X[mask], np.log1p(y[mask]))

    return {
        "features": features,
        "cats": cats,
        "p_treat": p_treat,
        "p_ctrl": p_ctrl,
        "e_treat": e_treat,
        "e_ctrl": e_ctrl,
    }


def predict_uplift(models, X):
    X = X[models["features"]]
    p_treat = models["p_treat"].predict_proba(X)[:, 1]
    p_ctrl = models["p_ctrl"].predict_proba(X)[:, 1]
    e_treat = np.expm1(models["e_treat"].predict(X))
    e_ctrl = np.expm1(models["e_ctrl"].predict(X))
    return p_treat * e_treat - p_ctrl * e_ctrl


def save_models(models, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["p_treat", "p_ctrl", "e_treat", "e_ctrl"]:
        models[name].save_model(str(out_dir/f"{name}.cbm"))

    meta = {"features": models["features"], "cats": models["cats"]}
    with open(out_dir/"meta.json", "w") as f:
        json.dump(meta, f)


def load_models(in_dir):
    in_dir = Path(in_dir)
    with open(in_dir/"meta.json") as f:
        meta = json.load(f)

    models = {"features": meta["features"], "cats": meta["cats"]}
    for name in ["p_treat", "p_ctrl"]:
        m = CatBoostClassifier()
        m.load_model(str(in_dir/f"{name}.cbm"))
        models[name] = m
    for name in ["e_treat", "e_ctrl"]:
        m = CatBoostRegressor()
        m.load_model(str(in_dir/f"{name}.cbm"))
        models[name] = m
    return models
