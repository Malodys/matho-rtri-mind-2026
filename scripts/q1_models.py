"""Q1 model definitions, tuning grids, and model-specific fitting helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MOUTHFEEL = {"body", "bitternes", "co2", "aftertaste", "alcohol", "sweetness", "acidity"}
FAULTS = {"dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"}
SPICES = {"orange", "coriander", "clove"}
WILD = {"lactic", "acetic", "barnyard"}
ALL_MODELS = ["global_mean", "ridge", "pls2", "multitask_elastic_net",
              "global_extra_trees", "grouped_extra_trees", "independent_gbr"]


def semantic_group(label):
    """Map a sensory label to one of eight grouped-tree modeling groups."""
    if label.startswith(("A_malt_", "F_malt_")): return "malt"
    if label.startswith(("A_hops_", "F_hops_")): return "hops"
    if label.startswith(("A_esters_", "F_esters_")): return "esters"
    if label in MOUTHFEEL: return "mouthfeel"
    if label in FAULTS: return "faults"
    if label in SPICES: return "spices"
    if label in WILD: return "wild_fermentation"
    if label == "overall": return "overall"
    raise ValueError(f"Unassigned label: {label}")


def model_registry(inner):
    """Return global estimators with preprocessing nested inside inner CV."""
    return {
        "ridge": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-3, 4, 40)}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=-1, refit=True),
        "multitask_elastic_net": GridSearchCV(
            Pipeline([("scale", StandardScaler()),
                      ("model", MultiTaskElasticNet(max_iter=100000, tol=1e-4,
                                                     selection="random", random_state=20261026))]),
            {"model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
             "model__alpha": np.logspace(-3, 0.5, 20)}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=-1, refit=True),
        "pls2": GridSearchCV(
            Pipeline([("scale", StandardScaler()),
                      ("model", PLSRegression(scale=False, max_iter=2000))]),
            {"model__n_components": [2, 5, 10, 15, 20]}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=-1, refit=True),
        "global_extra_trees": GridSearchCV(
            ExtraTreesRegressor(n_estimators=500, random_state=20261026, n_jobs=1),
            {"max_features": [0.3, 0.6], "min_samples_leaf": [3, 6],
             "max_depth": [None, 8]}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=-1, refit=True),
    }


def fit_grouped_extra_trees(x_train, y_train, x_test, groups, inner):
    """Tune and predict one shared-structure Extra Trees model per target group."""
    pred = pd.DataFrame(index=x_test.index, columns=y_train.columns, dtype=float)
    params = []
    for group, members in groups.groupby(groups):
        members = list(members.index)
        target = y_train[members].iloc[:, 0] if len(members) == 1 else y_train[members]
        search = GridSearchCV(
            ExtraTreesRegressor(n_estimators=500, random_state=20261026, n_jobs=1),
            {"max_features": [0.3, 0.6], "min_samples_leaf": [3, 6],
             "max_depth": [None, 8]}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=-1, refit=True).fit(x_train, target)
        pred.loc[:, members] = np.asarray(search.predict(x_test)).reshape(len(x_test), len(members))
        params.append({"model_group": group, **search.best_params_,
                       "inner_best_neg_MSE": search.best_score_})
    return pred, pd.DataFrame(params)


def fit_independent_gbr(x_train, y_train, x_test, inner):
    """Tune one regularized GBR per sensory attribute and join its predictions."""
    pred = pd.DataFrame(index=x_test.index, columns=y_train.columns, dtype=float)
    params = []
    grid = {"n_estimators": [100, 300], "learning_rate": [0.03, 0.05],
            "max_depth": [1, 2], "min_samples_leaf": [8, 15],
            "max_features": ["sqrt"], "subsample": [0.8]}
    for target in y_train.columns:
        search = GridSearchCV(
            GradientBoostingRegressor(loss="squared_error", random_state=20261026),
            grid, cv=inner, scoring="r2", n_jobs=-1, refit=True,
        ).fit(x_train, y_train[target])
        pred[target] = search.predict(x_test)
        params.append({"target": target, **search.best_params_,
                       "inner_best_R2": search.best_score_})
    return pred, pd.DataFrame(params)
