#!/usr/bin/env python3
"""Question 1: chemical profile -> sensory scores -> calibrated 0-6 grades.

The published sensory targets are continuous panel-standardized values.  The
0-6 outputs produced here are therefore explicitly named proxy grades: their
cut points are learned from each outer training fold and are not presented as
recovered individual-panelist raw scores.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RunConfig:
    repeats: int = 10
    outer_splits: int = 3
    inner_splits: int = 2
    random_state: int = 20261026
    models: tuple[str, ...] = ("global_mean", "ridge", "pls2", "extra_trees")
    input_variant: str = "X0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=list(RunConfig.models))
    parser.add_argument("--input-variant", choices=["X0", "X1"], default="X0")
    return parser.parse_args()


def load_inputs(root: Path, input_variant: str):
    """直接读取 02/03 的落盘产物与原始标签,不经过 manifest/哈希契约。"""
    interim = root / "data" / "interim"
    raw = root / "data/raw"

    # X:02 的主矩阵;派生 _sum 变量按 02 的原则单独管理,主建模排除
    x = pd.read_csv(interim / "chem_log10_matrix_v1.csv", index_col=0)
    x.index = x.index.astype(int)
    x = x.drop(columns=[c for c in x.columns if c.endswith("_sum")])

    if input_variant == "X1":
        # 左删失位置显式作为 0/1 指示列交给模型
        status = pd.read_csv(interim / "chem_observation_status_v1.csv", index_col=0)
        status.index = status.index.astype(int)
        status = status.reindex(index=x.index, columns=x.columns)
        lod = status.eq(1)
        lod = lod.loc[:, lod.any(axis=0)].astype("uint8")
        lod.columns = [f"LOD__{c}" for c in lod.columns]
        x = pd.concat([x, lod], axis=1)

    # y 和风格:直接来自原始 4.csv,按 03 的标签字典取列
    raw_y = pd.read_csv(raw / "4.csv", encoding="utf-8-sig").set_index("beer")
    raw_y.index = raw_y.index.astype(int)
    raw_y = raw_y.reindex(x.index)
    label_dictionary = pd.read_csv(interim / "label_dictionary_v1.csv", index_col=0)
    y = raw_y[list(label_dictionary.index)].apply(pd.to_numeric, errors="coerce")
    assert not raw_y["tasting_category_fine"].isna().any()
    styles = raw_y["tasting_category_fine"].astype(str)
    label_dictionary = label_dictionary.reindex(y.columns)

    assert x.shape[0] == 250
    assert x.shape[1] >= 227
    assert y.shape == (250, 50), y.shape
    assert not x.isna().any().any()
    assert not y.isna().any().any()
    assert list(y.columns) == list(label_dictionary.index)
    return x, y, styles, label_dictionary


def make_inner_splits(n_train: int, n_splits: int, seed: int):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(n_train)))


def model_registry(inner_splits, seed: int):
    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(
            alphas=np.logspace(-3, 4, 40),
            cv=inner_splits,
            scoring="neg_mean_squared_error",
        )),
    ])

    elastic = Pipeline([
        ("scale", StandardScaler()),
        ("model", MultiTaskElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-3, 0.5, 15),
            cv=inner_splits,
            max_iter=10000,
            tol=1e-4,
            n_jobs=1,
            random_state=0,
        )),
    ])

    pls = GridSearchCV(
        Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(scale=False, max_iter=2000)),
        ]),
        param_grid={"model__n_components": [2, 5, 10, 15, 20]},
        scoring="neg_mean_squared_error",
        cv=inner_splits,
        n_jobs=1,
        refit=True,
    )
    extra_trees = GridSearchCV(
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=1,
        ),
        param_grid={
            "max_features": [0.5, 1.0],
            "min_samples_leaf": [1, 3],
        },
        scoring="neg_mean_squared_error",
        cv=inner_splits,
        n_jobs=1,
        refit=True,
    )
    kernel_ridge = GridSearchCV(
        Pipeline([
            ("scale", StandardScaler()),
            ("model", KernelRidge(kernel="rbf")),
        ]),
        param_grid={
            "model__alpha": [0.1, 1.0, 10.0],
            "model__gamma": [1e-4, 1e-3],
        },
        scoring="neg_mean_squared_error",
        cv=inner_splits,
        n_jobs=1,
        refit=True,
    )
    return {"ridge": ridge, "elastic_net": elastic, "pls2": pls,
            "extra_trees": extra_trees, "kernel_ridge_rbf": kernel_ridge}


def training_quantile_cutpoints(y_train: np.ndarray) -> np.ndarray:
    """Six cut points producing seven ordered proxy grades per target."""
    return np.quantile(y_train, np.arange(1, 7) / 7, axis=0).T


def apply_cutpoints(values: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    grades = np.empty_like(values, dtype=np.int8)
    for j in range(values.shape[1]):
        grades[:, j] = np.digitize(values[:, j], cutpoints[j], right=True)
    return grades


def continuous_metric_rows(y_true, y_pred, labels, label_dictionary, metadata):
    rows = []
    for j, label in enumerate(labels):
        if np.std(y_true[:, j]) <= 1e-12 or np.std(y_pred[:, j]) <= 1e-12:
            rho = np.nan
        else:
            rho = spearmanr(y_true[:, j], y_pred[:, j]).statistic
        rows.append({
            **metadata,
            "label": label,
            "sensory_dimension": label_dictionary.loc[label, "sensory_dimension"],
            "family": label_dictionary.loc[label, "chemical_sensory_family"],
            "RMSE": mean_squared_error(y_true[:, j], y_pred[:, j]) ** 0.5,
            "MAE": mean_absolute_error(y_true[:, j], y_pred[:, j]),
            "R2": r2_score(y_true[:, j], y_pred[:, j]),
            "Spearman": rho,
        })
    return rows


def grade_metric_rows(true_grade, pred_grade, labels, label_dictionary, metadata):
    rows = []
    for j, label in enumerate(labels):
        truth = true_grade[:, j]
        pred = pred_grade[:, j]
        rows.append({
            **metadata,
            "label": label,
            "sensory_dimension": label_dictionary.loc[label, "sensory_dimension"],
            "family": label_dictionary.loc[label, "chemical_sensory_family"],
            "grade_MAE": mean_absolute_error(truth, pred),
            "exact_accuracy": float(np.mean(truth == pred)),
            "within_one_accuracy": float(np.mean(np.abs(truth - pred) <= 1)),
            "quadratic_weighted_kappa": cohen_kappa_score(truth, pred, weights="quadratic"),
        })
    return rows


def summarize(continuous: pd.DataFrame, grades: pd.DataFrame):
    cont_summary = continuous.groupby("model").agg(
        RMSE_mean=("RMSE", "mean"),
        MAE_mean=("MAE", "mean"),
        R2_mean=("R2", "mean"),
        R2_median=("R2", "median"),
        Spearman_mean=("Spearman", "mean"),
        Spearman_median=("Spearman", "median"),
    )
    grade_summary = grades.groupby("model").agg(
        proxy_grade_MAE=("grade_MAE", "mean"),
        proxy_exact_accuracy=("exact_accuracy", "mean"),
        proxy_within_one_accuracy=("within_one_accuracy", "mean"),
        proxy_quadratic_kappa=("quadratic_weighted_kappa", "mean"),
    )
    return cont_summary.join(grade_summary).sort_values("RMSE_mean")


def main():
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "reports" / "modeling" / "q1_baselines_v1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(repeats=args.repeats, models=tuple(args.models), input_variant=args.input_variant)
    allowed = {"global_mean", "ridge", "elastic_net", "pls2",
               "extra_trees", "kernel_ridge_rbf"}
    unknown = set(config.models) - allowed
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    x, y, styles, label_dictionary = load_inputs(root, config.input_variant)
    labels = list(y.columns)
    splitter = RepeatedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.repeats,
        random_state=config.random_state,
    )

    continuous_rows = []
    grade_rows = []
    prediction_rows = []
    cutpoint_rows = []
    fold_rows = []

    for split_id, (train_pos, test_pos) in enumerate(splitter.split(x)):
        repeat = split_id // config.outer_splits
        fold = split_id % config.outer_splits
        train_ids = x.index[train_pos]
        test_ids = x.index[test_pos]
        x_train, x_test = x.loc[train_ids], x.loc[test_ids]
        y_train, y_test = y.loc[train_ids], y.loc[test_ids]
        inner = make_inner_splits(len(train_ids), config.inner_splits, config.random_state + split_id)
        estimators = model_registry(inner, config.random_state + split_id)

        cutpoints = training_quantile_cutpoints(y_train.to_numpy())
        true_proxy_grade = apply_cutpoints(y_test.to_numpy(), cutpoints)
        for j, label in enumerate(labels):
            for boundary, value in enumerate(cutpoints[j], start=1):
                cutpoint_rows.append({"repeat": repeat, "fold": fold, "label": label,
                                      "boundary": boundary, "value": value})

        predictions = {}
        if "global_mean" in config.models:
            predictions["global_mean"] = np.tile(y_train.mean(axis=0).to_numpy(), (len(test_ids), 1))

        for model_name in ["ridge", "elastic_net", "pls2", "extra_trees", "kernel_ridge_rbf"]:
            if model_name not in config.models:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = clone(estimators[model_name]).fit(x_train, y_train)
            predictions[model_name] = np.asarray(fitted.predict(x_test)).reshape(len(test_ids), len(labels))

        for beer in train_ids:
            fold_rows.append({"repeat": repeat, "fold": fold, "beer": beer,
                              "style": styles.loc[beer], "role": "train"})
        for beer in test_ids:
            fold_rows.append({"repeat": repeat, "fold": fold, "beer": beer,
                              "style": styles.loc[beer], "role": "test"})

        for model_name, pred in predictions.items():
            metadata = {"repeat": repeat, "fold": fold, "model": model_name}
            continuous_rows.extend(continuous_metric_rows(
                y_test.to_numpy(), pred, labels, label_dictionary, metadata))
            pred_proxy_grade = apply_cutpoints(pred, cutpoints)
            grade_rows.extend(grade_metric_rows(
                true_proxy_grade, pred_proxy_grade, labels, label_dictionary, metadata))
            for i, beer in enumerate(test_ids):
                for j, label in enumerate(labels):
                    prediction_rows.append({
                        **metadata,
                        "beer": beer,
                        "style": styles.loc[beer],
                        "label": label,
                        "observed_continuous": y_test.iloc[i, j],
                        "predicted_continuous": pred[i, j],
                        "observed_proxy_grade": int(true_proxy_grade[i, j]),
                        "predicted_proxy_grade": int(pred_proxy_grade[i, j]),
                    })
        print(f"completed repeat={repeat + 1}/{config.repeats}, fold={fold + 1}/{config.outer_splits}", flush=True)

    continuous = pd.DataFrame(continuous_rows)
    grades = pd.DataFrame(grade_rows)
    predictions = pd.DataFrame(prediction_rows)
    cutpoints_df = pd.DataFrame(cutpoint_rows)
    folds = pd.DataFrame(fold_rows)
    model_summary = summarize(continuous, grades)
    family_summary = continuous.groupby(["model", "sensory_dimension"], dropna=False).agg(
        RMSE_mean=("RMSE", "mean"), R2_median=("R2", "median"),
        Spearman_median=("Spearman", "median")).reset_index()

    continuous.to_csv(output_dir / "continuous_metrics_by_fold_label.csv", index=False)
    grades.to_csv(output_dir / "proxy_grade_metrics_by_fold_label.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    cutpoints_df.to_csv(output_dir / "proxy_grade_cutpoints.csv", index=False)
    folds.to_csv(output_dir / "fold_assignments.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv")
    family_summary.to_csv(output_dir / "sensory_dimension_summary.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    ax = model_summary[["R2_median", "Spearman_median"]].plot.bar(figsize=(10, 5), rot=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Question 1 baseline models: repeated random outer CV")
    ax.set_ylabel("Mean/median across labels and outer folds")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nMODEL SUMMARY")
    print(model_summary.to_string())
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
