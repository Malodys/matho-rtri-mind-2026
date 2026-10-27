#!/usr/bin/env python3
"""Thin Q1 entry point: fit selected models or collect completed runs."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from q1_evaluation import (evaluate_predictions, load_completed_runs, load_inputs,
                           official_family, summarize, training_quantile_cutpoints)
from q1_models import (ALL_MODELS, fit_grouped_extra_trees, fit_independent_gbr,
                       model_registry, semantic_group)
from q1_visualize import make_evaluation_figures, make_linear_explanation_figures


def write_common_outputs(out, continuous, grades, predictions):
    """Save common metrics/predictions and generate evaluation figures."""
    summary = summarize(continuous, grades)
    continuous.to_csv(out / "continuous_metrics_by_label.csv", index=False)
    grades.to_csv(out / "proxy_grade_metrics_by_label.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "model_summary.csv")
    family = make_evaluation_figures(continuous, predictions, summary, out)
    family.to_csv(out / "family_rmse_summary.csv")
    return summary


def collect_runs(runs_dir, out):
    """Collect independent runs and recompute their common comparison."""
    continuous, grades, predictions, run_dirs = load_completed_runs(runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = write_common_outputs(out, continuous, grades, predictions)
    for filename in ["global_model_best_params.csv", "grouped_model_best_params.csv",
                     "gbr_best_params_by_target.csv"]:
        parts = [pd.read_csv(d / filename) for d in run_dirs
                 if (d / filename).exists() and (d / filename).stat().st_size > 1]
        if parts:
            pd.concat(parts, ignore_index=True).drop_duplicates().to_csv(out / filename, index=False)
    for filename in ["proxy_grade_cutpoints.csv", "target_groups.csv"]:
        source = next((d / filename for d in run_dirs if (d / filename).exists()), None)
        if source is not None: shutil.copy2(source, out / filename)
    config = {"mode": "collected_independent_runs", "runs_dir": str(runs_dir),
              "models": list(summary.index), "primary_metric": "Macro_RMSE"}
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string()); print(f"Collected {len(summary)} models into {out}")


def run_models(root, split_dir, out, selected):
    """Fit selected models on the locked split and save standardized artifacts."""
    out.mkdir(parents=True, exist_ok=True)
    x, y, styles, dictionary = load_inputs(root)
    plan = pd.read_csv(split_dir / "split_manifest.csv", encoding="utf-8-sig")
    plan["beer"] = plan["beer"].astype(int)
    assert set(plan.beer) == set(x.index) and plan.beer.is_unique
    train_ids = plan.loc[plan.split.eq("train"), "beer"]
    test_ids = plan.loc[plan.split.eq("test"), "beer"]
    assert len(train_ids) == 175 and len(test_ids) == 75
    x_train, x_test, y_train, y_test = x.loc[train_ids], x.loc[test_ids], y.loc[train_ids], y.loc[test_ids]
    labels = list(y.columns)
    inner = list(KFold(n_splits=5, shuffle=True, random_state=20261026).split(x_train))
    predictions, fitted_models, best_params = {}, {}, []
    if "global_mean" in selected:
        predictions["global_mean"] = np.tile(y_train.mean().to_numpy(), (len(test_ids), 1))
    for name, estimator in model_registry(inner).items():
        if name not in selected: continue
        fitted = clone(estimator).fit(x_train, y_train)
        fitted_models[name] = fitted
        predictions[name] = np.asarray(fitted.predict(x_test)).reshape(len(test_ids), len(labels))
        best_params.append({"model": name, **fitted.best_params_,
                            "inner_best_neg_MSE": fitted.best_score_})
    groups = pd.Series({label: semantic_group(label) for label in labels}, name="model_group")
    grouped_params, gbr_params = pd.DataFrame(), pd.DataFrame()
    if "grouped_extra_trees" in selected:
        pred, grouped_params = fit_grouped_extra_trees(x_train, y_train, x_test, groups, inner)
        predictions["grouped_extra_trees"] = pred[labels].to_numpy()
    if "independent_gbr" in selected:
        pred, gbr_params = fit_independent_gbr(x_train, y_train, x_test, inner)
        predictions["independent_gbr"] = pred[labels].to_numpy()
    if not predictions: raise ValueError("No model was selected")
    cutpoints = training_quantile_cutpoints(y_train.to_numpy())
    continuous, grades, prediction_table = evaluate_predictions(
        y_test, predictions, labels, dictionary, test_ids, styles, cutpoints)
    summary = write_common_outputs(out, continuous, grades, prediction_table)
    make_linear_explanation_figures(fitted_models, list(x.columns), labels, out)
    if best_params: pd.DataFrame(best_params).to_csv(out / "global_model_best_params.csv", index=False)
    if not grouped_params.empty: grouped_params.to_csv(out / "grouped_model_best_params.csv", index=False)
    if not gbr_params.empty: gbr_params.to_csv(out / "gbr_best_params_by_target.csv", index=False)
    pd.DataFrame(cutpoints, index=labels, columns=[f"boundary_{i}" for i in range(1, 7)]).to_csv(out / "proxy_grade_cutpoints.csv")
    pd.DataFrame({"family": [official_family(z) for z in labels],
                  "model_group": groups.reindex(labels).to_numpy()}, index=labels).to_csv(out / "target_groups.csv")
    config = {"outer_evaluation": "fixed 70/30 holdout", "train_n": 175, "test_n": 75,
              "style_used_as_model_feature": False, "models": list(predictions),
              "primary_metric": "Macro_RMSE",
              "tuning": {"ridge_alpha": "logspace(-3,4,40)",
                  "pls2_n_components": [2, 5, 10, 15, 20],
                  "multitask_elastic_net_l1_ratio": [0.2, 0.5, 0.8, 0.95, 1.0],
                  "multitask_elastic_net_alpha": "logspace(-2.5,0.5,16)",
                  "extra_trees": {"n_estimators": 500, "max_features": [0.3, 0.6],
                                  "min_samples_leaf": [3, 6], "max_depth": [None, 8]},
                  "independent_gbr_selection_metric": "R2"}}
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string()); print(f"Saved to {out}")


def main():
    """Dispatch fitting or collection from the command line."""
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--split-dir", type=Path); p.add_argument("--output-dir", type=Path)
    p.add_argument("--models", nargs="+", choices=ALL_MODELS)
    p.add_argument("--collect", action="store_true"); p.add_argument("--runs-dir", type=Path)
    args = p.parse_args(); root = args.project_root.resolve()
    if args.collect:
        collect_runs((args.runs_dir or root / "reports/modeling/q1_runs").resolve(),
                     (args.output_dir or root / "reports/modeling/q1_entry_v1").resolve()); return
    selected = args.models or ALL_MODELS
    default_out = (root / "reports/modeling/q1_runs" / selected[0]
                   if len(selected) == 1 else root / "reports/modeling/q1_entry_v1")
    run_models(root, (args.split_dir or root / "data/splits/seed_0").resolve(),
               (args.output_dir or default_out).resolve(), selected)


if __name__ == "__main__": main()
