"""Q1 data contract and model-independent evaluation functions."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score

from q1_models import MOUTHFEEL


def official_family(label):
    """Return the five-family reporting classification for one sensory label."""
    if label.startswith("A_"): return "aroma"
    if label.startswith("F_"): return "flavour"
    if label in MOUTHFEEL: return "mouthfeel"
    if label == "overall": return "overall"
    return "fault_special"


def load_inputs(root):
    """Load the locked 227-feature matrix, 50 targets, styles, and dictionary."""
    root = Path(root)
    x = pd.read_csv(root / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    x.index = x.index.astype(int)
    x = x.drop(columns=[c for c in x if c.endswith("_sum")])
    source = pd.read_csv(root / "data/raw/4.csv", encoding="utf-8-sig").set_index("beer")
    source.index = source.index.astype(int)
    source = source.reindex(x.index)
    dictionary = pd.read_csv(root / "data/interim/label_dictionary_v1.csv", index_col=0)
    y = source[list(dictionary.index)].apply(pd.to_numeric, errors="coerce")
    styles = source["tasting_category_fine"].astype(str)
    assert x.shape == (250, 227) and y.shape == (250, 50)
    assert not x.isna().any().any() and not y.isna().any().any()
    return x, y, styles, dictionary.reindex(y.columns)


def training_quantile_cutpoints(y_train):
    """Learn six training-only quantiles defining seven auxiliary proxy grades."""
    return np.quantile(y_train, np.arange(1, 7) / 7, axis=0).T


def apply_cutpoints(values, cutpoints):
    """Map continuous values to proxy grades 0 through 6."""
    grades = np.empty_like(values, dtype=np.int8)
    for j in range(values.shape[1]):
        grades[:, j] = np.digitize(values[:, j], cutpoints[j], right=True)
    return grades


def evaluate_predictions(y_test, predictions, labels, dictionary, test_ids, styles, cutpoints):
    """Evaluate every 75x50 prediction matrix with the same per-target rules."""
    true = y_test.to_numpy()
    true_grade = apply_cutpoints(true, cutpoints)
    continuous_rows, grade_rows, prediction_rows = [], [], []
    for model, pred in predictions.items():
        pred = np.asarray(pred).reshape(len(test_ids), len(labels))
        pred_grade = apply_cutpoints(pred, cutpoints)
        for j, label in enumerate(labels):
            rho = np.nan if np.std(pred[:, j]) <= 1e-12 else spearmanr(true[:, j], pred[:, j]).statistic
            common = {"model": model, "split_seed": 0, "label": label,
                      "sensory_dimension": dictionary.loc[label, "sensory_dimension"],
                      "family": official_family(label),
                      "chemical_sensory_family": dictionary.loc[label, "chemical_sensory_family"]}
            continuous_rows.append({**common,
                "RMSE": mean_squared_error(true[:, j], pred[:, j]) ** 0.5,
                "MAE": mean_absolute_error(true[:, j], pred[:, j]),
                "R2": r2_score(true[:, j], pred[:, j]), "Spearman": rho})
            grade_rows.append({**common,
                "grade_MAE": mean_absolute_error(true_grade[:, j], pred_grade[:, j]),
                "exact_accuracy": float(np.mean(true_grade[:, j] == pred_grade[:, j])),
                "within_one_accuracy": float(np.mean(np.abs(true_grade[:, j] - pred_grade[:, j]) <= 1)),
                "quadratic_weighted_kappa": cohen_kappa_score(
                    true_grade[:, j], pred_grade[:, j], weights="quadratic")})
        for i, beer in enumerate(test_ids):
            for j, label in enumerate(labels):
                prediction_rows.append({"model": model, "split_seed": 0, "beer": beer,
                    "style": styles.loc[beer], "label": label,
                    "observed_continuous": y_test.iloc[i, j], "predicted_continuous": pred[i, j],
                    "observed_proxy_grade": int(true_grade[i, j]),
                    "predicted_proxy_grade": int(pred_grade[i, j])})
    return pd.DataFrame(continuous_rows), pd.DataFrame(grade_rows), pd.DataFrame(prediction_rows)


def summarize(continuous, grades):
    """Aggregate equal-target-weight continuous and auxiliary grade metrics."""
    a = continuous.groupby("model").agg(
        Macro_RMSE=("RMSE", "mean"), Macro_MAE=("MAE", "mean"),
        R2_mean=("R2", "mean"), R2_median=("R2", "median"),
        Spearman_mean=("Spearman", "mean"), Spearman_median=("Spearman", "median"))
    b = grades.groupby("model").agg(
        proxy_grade_MAE=("grade_MAE", "mean"),
        proxy_exact_accuracy=("exact_accuracy", "mean"),
        proxy_within_one_accuracy=("within_one_accuracy", "mean"),
        proxy_quadratic_kappa=("quadratic_weighted_kappa", "mean"))
    return a.join(b).sort_values("Macro_RMSE")


def load_completed_runs(runs_dir):
    """Load, deduplicate, and validate completed independent model runs."""
    run_dirs = sorted({p.parent for p in Path(runs_dir).rglob("test_predictions.csv.gz")})
    parts = {"continuous": [], "grades": [], "predictions": []}
    for run_dir in run_dirs:
        paths = [run_dir / "continuous_metrics_by_label.csv",
                 run_dir / "proxy_grade_metrics_by_label.csv",
                 run_dir / "test_predictions.csv.gz"]
        if all(p.exists() for p in paths):
            parts["continuous"].append(pd.read_csv(paths[0]))
            parts["grades"].append(pd.read_csv(paths[1]))
            parts["predictions"].append(pd.read_csv(paths[2]))
    if not parts["predictions"]:
        raise FileNotFoundError(f"No complete model runs found under {runs_dir}")
    continuous = pd.concat(parts["continuous"], ignore_index=True).drop_duplicates(["model", "label"], keep="last")
    grades = pd.concat(parts["grades"], ignore_index=True).drop_duplicates(["model", "label"], keep="last")
    predictions = pd.concat(parts["predictions"], ignore_index=True).drop_duplicates(["model", "beer", "label"], keep="last")
    incomplete = predictions.groupby("model").size().loc[lambda s: s.ne(75 * 50)]
    if not incomplete.empty:
        raise ValueError(f"Incomplete 75 x 50 predictions: {incomplete.to_dict()}")
    return continuous, grades, predictions, run_dirs
