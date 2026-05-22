import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import ALPHA, BOOT, FOLDS, K, SEED


def make_folds(treatment, target, n_splits=FOLDS, seed=SEED):
    groups = treatment * 2 + (target > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(groups, groups))


def uplift_at_k(y, t, score, k=K):
    n_top = int(np.ceil(k * len(y)))
    top = np.argsort(score)[::-1][:n_top]
    y_top = y[top]
    t_top = t[top]
    treat = y_top[t_top == 1].mean()
    ctrl = y_top[t_top == 0].mean()
    return treat - ctrl


def uplift_at_k_lb(y, t, score, k=K, n_boot=BOOT, alpha=ALPHA, seed=SEED):
    n_top = int(np.ceil(k * len(y)))
    top = np.argsort(score)[::-1][:n_top]
    y_top = y[top]
    t_top = t[top]

    np.random.seed(seed)
    results = []
    for _ in range(n_boot):
        i = np.random.choice(n_top, size=n_top, replace=True)
        yb = y_top[i]
        tb = t_top[i]
        treat = yb[tb == 1].mean()
        ctrl = yb[tb == 0].mean()
        results.append(treat - ctrl)

    return np.mean(results), np.quantile(results, alpha/2)

def print_folds(folds):
    for m in folds:
        print(m)
