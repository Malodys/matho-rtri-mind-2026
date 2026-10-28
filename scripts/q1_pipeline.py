# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 取数全部经 scripts/data_contract.py（load_modeling_matrix / load_targets / load_split /
#      load_label_dictionary），未直接 read_csv 原始附件，也未对全表做 10**x；
#      已核对特征矩阵 250×227（剔除 4 个 *_sum 派生列）、目标 250×50、切分 175/75。
#   2) 七个模型族的调参与拟合复用 scripts/q1_models.py（model_registry / fit_grouped_extra_trees /
#      fit_independent_gbr）与 scripts/q1_evaluation.py（training_quantile_cutpoints /
#      apply_cutpoints / evaluate_predictions / summarize），未改写队友脚本。
#   3) 分级切点严格只由 175 个训练样本的七分位数产生（test 数据从未参与切点拟合），
#      并由 scripts/q1_leakage_check.py 与 tests_q1/ 做断言复核。
#   4) 交付物中的 <dim>_pred 为「5 折 OOF（250 样本，seed 0）」预测，避免用留出集拟合后再自评；
#      所有对外报告的指标一律来自锁定的 175/75 留出集。
#   5) 绝对锚定副轨设有效性护栏（scripts/q1_absolute_anchor.py 的 VALIDITY_RULES）：
#      未通过的维度 valid=False，其 <dim>_abs 在交付表与所有派生表中一律留空（NaN），
#      绝不以代理分级回填，避免读者误以为 50 个维度都有绝对分。
"""问题一 · 主建模管线。

一条命令完成：七模型族真实重跑 → 双轨分级 → bootstrap 比较 → 250 行 0–6 交付物落盘。

用法：
    python scripts/q1_pipeline.py --output-dir data/processed/q1_final
    python scripts/q1_pipeline.py --models ridge pls2      # 只跑部分模型（冒烟）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_contract import (load_label_dictionary, load_modeling_matrix, load_raw,  # noqa: E402
                           load_split, load_styles, load_targets)
from q1_evaluation import (apply_cutpoints, evaluate_predictions, load_completed_runs,  # noqa: E402
                           official_family, summarize, training_quantile_cutpoints)
from q1_models import (ALL_MODELS, fit_grouped_extra_trees, fit_independent_gbr,  # noqa: E402
                       model_registry, semantic_group)
import q1_absolute_anchor as anchor_mod  # noqa: E402
import q1_bootstrap as boot  # noqa: E402

INNER_SEED = 20261026
OOF_SEED = 0


# --------------------------------------------------------------------------- 取数


def load_q1_inputs() -> Dict[str, object]:
    """经 data_contract 取统一矩阵、目标、风格、字典与锁定切分。"""
    x = load_modeling_matrix()
    y = load_targets()
    styles = load_styles()
    dictionary = load_label_dictionary().reindex(y.columns)
    split = load_split()
    assert x.shape == (250, 227), x.shape
    assert y.shape == (250, 50), y.shape
    assert not x.isna().any().any() and not y.isna().any().any()
    train_ids = split.loc[split["split"] == "train", "beer"].to_numpy(dtype=int)
    test_ids = split.loc[split["split"] == "test", "beer"].to_numpy(dtype=int)
    assert len(train_ids) == 175 and len(test_ids) == 75
    assert not set(train_ids) & set(test_ids)
    return {"x": x, "y": y, "styles": styles, "dictionary": dictionary,
            "train_ids": train_ids, "test_ids": test_ids}


# --------------------------------------------------------------------------- 建模


def _clean_params(params: pd.DataFrame) -> List[dict]:
    """把 best_params 表里的 NaN（代表 None，如 max_depth=None）还原成 None。"""
    rows = []
    for row in params.to_dict("records"):
        rows.append({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                     for k, v in row.items()})
    return rows


def fit_one_model(name: str, x_train: pd.DataFrame, y_train: pd.DataFrame,
                  x_test: pd.DataFrame, inner, groups: pd.Series
                  ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    """拟合单个模型族，返回 (75×50 预测, 最佳参数表, 供 OOF/工件复用的模板)。"""
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

    labels = list(y_train.columns)
    templates: Dict[str, object] = {}
    if name == "global_mean":
        pred = np.tile(y_train.mean().to_numpy(), (len(x_test), 1))
        templates["constant"] = y_train.mean().to_numpy()
        return pred, pd.DataFrame([{"model": name, "note": "训练集目标均值"}]), templates

    if name in model_registry(inner):
        fitted = clone(model_registry(inner)[name]).fit(x_train, y_train)
        pred = np.asarray(fitted.predict(x_test)).reshape(len(x_test), len(labels))
        params = pd.DataFrame([{"model": name, **fitted.best_params_,
                                "inner_best_neg_MSE": fitted.best_score_}])
        templates["estimator"] = clone(fitted.best_estimator_)
        return pred, params, templates

    if name == "grouped_extra_trees":
        pred_df, params = fit_grouped_extra_trees(x_train, y_train, x_test, groups, inner)
        ests = {}
        for row in _clean_params(params):
            ests[row["model_group"]] = ExtraTreesRegressor(
                n_estimators=500, random_state=INNER_SEED, n_jobs=1,
                max_features=row["max_features"], min_samples_leaf=row["min_samples_leaf"],
                max_depth=row["max_depth"])
        templates["grouped"] = ests
        return pred_df[labels].to_numpy(), params, templates

    if name == "independent_gbr":
        pred_df, params = fit_independent_gbr(x_train, y_train, x_test, inner)
        ests = {}
        for row in _clean_params(params):
            ests[row["target"]] = GradientBoostingRegressor(
                loss="squared_error", random_state=INNER_SEED,
                n_estimators=int(row["n_estimators"]), learning_rate=float(row["learning_rate"]),
                max_depth=int(row["max_depth"]), min_samples_leaf=int(row["min_samples_leaf"]),
                max_features=row["max_features"], subsample=float(row["subsample"]))
        templates["independent"] = ests
        return pred_df[labels].to_numpy(), params, templates

    raise ValueError(f"未知模型：{name}")


def predict_with_templates(templates: Dict[str, object], kind: str, x: pd.DataFrame,
                           labels: List[str], groups: pd.Series) -> np.ndarray:
    """用固定超参的模板在任意 X 上预测 50 维（供 OOF 与全量重训复用）。"""
    out = np.zeros((len(x), len(labels)))
    if kind == "constant":
        out[:, :] = templates["constant"]
        return out
    if kind == "global":
        est = templates["estimator"]
        return np.asarray(est.predict(x)).reshape(len(x), len(labels))
    if kind == "grouped":
        for group, members in groups.groupby(groups):
            members = list(members.index)
            est = templates["grouped"][group]
            out[:, [labels.index(m) for m in members]] = np.asarray(
                est.predict(x)).reshape(len(x), len(members))
        return out
    if kind == "independent":
        for j, label in enumerate(labels):
            est = templates["independent"][label]
            out[:, j] = np.asarray(est.predict(x)).reshape(-1)
        return out
    raise ValueError(f"未知模板类型：{kind}")


def template_kind(name: str) -> str:
    return {"global_mean": "constant", "grouped_extra_trees": "grouped",
            "independent_gbr": "independent"}.get(name, "global")


# --------------------------------------------------------------------------- 交叉验证


def build_templates_from_params(name: str, params: pd.DataFrame, inner, groups: pd.Series,
                                y: pd.DataFrame) -> Dict[str, object]:
    """从已保存的最佳参数重建「固定超参」的模板（用于 --skip-fit 快速重算交付物）。"""
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

    if name == "global_mean":
        return {"constant": y.mean().to_numpy()}
    if name in model_registry(inner):
        base = clone(model_registry(inner)[name].estimator)
        best = _clean_params(params)[0]
        overrides = {k[len("model__"):]: v for k, v in best.items() if k.startswith("model__")}
        base.set_params(**overrides)
        return {"estimator": base}
    if name == "grouped_extra_trees":
        ests = {}
        for row in _clean_params(params):
            ests[row["model_group"]] = ExtraTreesRegressor(
                n_estimators=500, random_state=INNER_SEED, n_jobs=1,
                max_features=row["max_features"], min_samples_leaf=row["min_samples_leaf"],
                max_depth=row["max_depth"])
        return {"grouped": ests}
    if name == "independent_gbr":
        ests = {}
        for row in _clean_params(params):
            ests[row["target"]] = GradientBoostingRegressor(
                loss="squared_error", random_state=INNER_SEED,
                n_estimators=int(row["n_estimators"]), learning_rate=float(row["learning_rate"]),
                max_depth=int(row["max_depth"]), min_samples_leaf=int(row["min_samples_leaf"]),
                max_features=row["max_features"], subsample=float(row["subsample"]))
        return {"independent": ests}
    raise ValueError(f"未知模型：{name}")


def out_of_fold_predictions(name: str, templates: Dict[str, object], x: pd.DataFrame,
                            y: pd.DataFrame, groups: pd.Series, n_splits: int = 5,
                            seed: int = OOF_SEED) -> np.ndarray:
    """在全部 250 样本上做 K 折 OOF 预测（每个样本的预测都来自未见过它的模型）。

    人工核验点：折划分 KFold(n_splits=5, shuffle=True, random_state=0)，与留出集评估互不影响；
    目的仅是给交付物提供「每个啤酒都有非自证预测」的分值，避免用留出集拟合后再自评。
    """
    labels = list(y.columns)
    kind = template_kind(name)
    out = np.zeros((len(x), len(labels)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(x), 1):
        x_tr, x_va = x.iloc[tr], x.iloc[va]
        y_tr = y.iloc[tr]
        if kind == "constant":
            fitted_templates = {"constant": y_tr.mean().to_numpy()}
        elif kind == "global":
            fitted_templates = {"estimator": clone(templates["estimator"]).fit(x_tr, y_tr)}
        elif kind == "grouped":
            fitted_templates = {"grouped": {g: clone(e).fit(x_tr, y_tr[[m for m in groups[groups == g].index]])
                                            for g, e in templates["grouped"].items()}}
        else:
            fitted_templates = {"independent": {t: clone(e).fit(x_tr, y_tr[[t]])
                                                for t, e in templates["independent"].items()}}
        out[va] = predict_with_templates(fitted_templates, kind, x_va, labels, groups)
        print(f"  OOF fold {fold}/{n_splits} 完成", flush=True)
    return out


def fit_full_model(name: str, templates: Dict[str, object], x: pd.DataFrame, y: pd.DataFrame,
                   groups: pd.Series) -> Dict[str, object]:
    """在全部 250 样本上重训（用于交付给问题三的工件）。"""
    labels = list(y.columns)
    kind = template_kind(name)
    if kind == "constant":
        return {"kind": kind, "estimators": {"constant": y.mean().to_numpy()}}
    if kind == "global":
        return {"kind": kind, "estimators": {"estimator": clone(templates["estimator"]).fit(x, y)}}
    if kind == "grouped":
        ests = {g: clone(e).fit(x, y[[m for m in groups[groups == g].index]])
                for g, e in templates["grouped"].items()}
        return {"kind": kind, "estimators": ests}
    ests = {t: clone(e).fit(x, y[[t]]) for t, e in templates["independent"].items()}
    return {"kind": kind, "estimators": ests}


# --------------------------------------------------------------------------- 主流程


def run_models(runs_dir: Path, selected: List[str], data: Dict[str, object]) -> Dict[str, object]:
    """在锁定切分上真实重跑七个模型族，逐模型写入 data/processed/q1_runs/<model>/。"""
    x, y = data["x"], data["y"]
    styles, dictionary = data["styles"], data["dictionary"]
    train_ids, test_ids = data["train_ids"], data["test_ids"]
    labels = list(y.columns)
    x_train, x_test = x.loc[train_ids], x.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    groups = pd.Series({label: semantic_group(label) for label in labels}, name="model_group")
    inner = list(KFold(n_splits=5, shuffle=True, random_state=INNER_SEED).split(x_train))
    cutpoints = training_quantile_cutpoints(y_train.to_numpy())

    templates_by_model: Dict[str, Dict[str, object]] = {}
    for name in selected:
        print(f"[fit] {name}", flush=True)
        out = runs_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pred, params, templates = fit_one_model(name, x_train, y_train, x_test, inner, groups)
        templates_by_model[name] = templates
        continuous, grades, predictions = evaluate_predictions(
            y_test, {name: pred}, labels, dictionary, test_ids, styles, cutpoints)
        continuous.to_csv(out / "continuous_metrics_by_label.csv", index=False)
        grades.to_csv(out / "proxy_grade_metrics_by_label.csv", index=False)
        predictions.to_csv(out / "test_predictions.csv.gz", index=False, compression="gzip")
        summarize(continuous, grades).to_csv(out / "model_summary.csv")
        pd.DataFrame(cutpoints, index=labels,
                     columns=[f"boundary_{i}" for i in range(1, 7)]).to_csv(out / "proxy_grade_cutpoints.csv")
        pd.DataFrame({"family": [official_family(z) for z in labels],
                      "model_group": groups.reindex(labels).to_numpy()}, index=labels).to_csv(out / "target_groups.csv")
        params.to_csv(out / "model_best_params.csv", index=False)
        (out / "run_config.json").write_text(json.dumps({
            "outer_evaluation": "fixed 70/30 holdout (seed 0, style-stratified)",
            "train_n": int(len(train_ids)), "test_n": int(len(test_ids)),
            "model": name, "primary_metric": "Macro_RMSE",
            "cutpoint_source": "train-only 6 septile quantiles (175 samples)",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {out}", flush=True)
    return {"templates": templates_by_model, "cutpoints": cutpoints, "groups": groups}


def absolute_track_detail(params: pd.DataFrame, y_true: pd.DataFrame, pred_frame: pd.DataFrame,
                          dims: List[str]) -> pd.DataFrame:
    """绝对轨逐维指标（真值与预测各自过同一套锚定映射后比较）。"""
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for dim in dims:
        row = params.loc[params["dimension"] == dim].iloc[0]
        a, b = float(row["a"]), float(row["b"])
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        true_abs = np.clip(a + b * y_true[dim].to_numpy(dtype=float), 0, 6)
        pred_abs = np.clip(a + b * pred_frame[dim].to_numpy(dtype=float), 0, 6)
        t_grade = np.rint(true_abs).astype(int)
        p_grade = np.rint(pred_abs).astype(int)
        rows.append({
            "dimension": dim,
            "anchor_type": row["anchor_type"],
            "MAE": float(np.mean(np.abs(true_abs - pred_abs))),
            "exact_accuracy": float(np.mean(t_grade == p_grade)),
            "within_one_accuracy": float(np.mean(np.abs(t_grade - p_grade) <= 1)),
            "QWK": float(cohen_kappa_score(t_grade, p_grade, weights="quadratic")),
        })
    return pd.DataFrame(rows)


def absolute_track_metrics(params: pd.DataFrame, y_true: pd.DataFrame, pred_frame: pd.DataFrame,
                           dims: List[str]) -> Dict[str, float]:
    """在给定维度上评估绝对轨（真值与预测各自过同一套锚定映射）。"""
    frame = absolute_track_detail(params, y_true, pred_frame, dims)
    if frame.empty:
        return {}
    return {f"abs_{k}": float(frame[k].mean()) for k in ("MAE", "exact_accuracy",
                                                        "within_one_accuracy", "QWK")}


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 · 主建模管线（双轨分级 + bootstrap + 交付物）")
    ap.add_argument("--runs-dir", default="data/processed/q1_runs")
    ap.add_argument("--output-dir", default="data/processed/q1_final")
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-fit", action="store_true", help="复用已有 runs 目录，不重跑模型")
    ap.add_argument("--figures", action="store_true", help="同时生成图表")
    args = ap.parse_args()

    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] 取数（data_contract）", flush=True)
    data = load_q1_inputs()
    x, y = data["x"], data["y"]
    labels = list(y.columns)
    train_ids, test_ids = data["train_ids"], data["test_ids"]

    if args.skip_fit:
        print("[2/7] 复用已有模型运行", flush=True)
        continuous, grades, predictions, _ = load_completed_runs(runs_dir)
        summary = summarize(continuous, grades)
        templates_by_model = {}
        cutpoints = training_quantile_cutpoints(y.loc[train_ids].to_numpy())
        groups = pd.Series({label: semantic_group(label) for label in labels}, name="model_group")
    else:
        print("[2/7] 重跑七个模型族", flush=True)
        fit_info = run_models(runs_dir, list(args.models), data)
        templates_by_model = fit_info["templates"]
        cutpoints = fit_info["cutpoints"]
        groups = fit_info["groups"]
        continuous, grades, predictions, _ = load_completed_runs(runs_dir)
        summary = summarize(continuous, grades)

    print("[3/7] 留出集汇总", flush=True)
    summary = summarize(continuous, grades).reset_index().rename(columns={"index": "model"})
    baseline = float(summary.loc[summary["model"] == "global_mean", "Macro_RMSE"].iloc[0])
    summary["beats_global_mean"] = summary["Macro_RMSE"] < baseline
    print(summary[["model", "Macro_RMSE", "R2_median", "Spearman_median",
                   "proxy_grade_MAE", "proxy_exact_accuracy", "proxy_quadratic_kappa",
                   "beats_global_mean"]].to_string(index=False), flush=True)

    print("[4/7] 配对 bootstrap（%d 次重采样）" % args.n_boot, flush=True)
    _, cont, pred_grade, truth = boot._matrices(predictions, labels)
    boot_res = boot.bootstrap_models(truth["true_cont"], cont, truth["true_grade"], pred_grade,
                                      n_boot=args.n_boot, seed=args.seed)
    boot_res["summary"].to_csv(out_dir / "bootstrap_macro_rmse.csv", index=False)
    boot_res["diffs"].to_csv(out_dir / "bootstrap_model_differences.csv", index=False)
    boot_res["overlaps"].to_csv(out_dir / "bootstrap_ci_overlap.csv", index=False)

    best_model = str(summary.sort_values("Macro_RMSE").iloc[0]["model"])
    print(f"  最优模型：{best_model}（Macro-RMSE "
          f"{float(summary.sort_values('Macro_RMSE').iloc[0]['Macro_RMSE']):.4f}）", flush=True)

    print("[5/7] 250 样本 5 折 OOF 预测", flush=True)
    if not templates_by_model:
        # --skip-fit：用已保存的最佳参数重建固定超参模板，避免重跑网格搜索
        inner = list(KFold(n_splits=5, shuffle=True, random_state=INNER_SEED).split(x.loc[train_ids]))
        groups = pd.Series({label: semantic_group(label) for label in labels}, name="model_group")
        params = pd.read_csv(runs_dir / best_model / "model_best_params.csv")
        templates_by_model = {best_model: build_templates_from_params(best_model, params, inner,
                                                                     groups, y)}
        print(f"  已从 {runs_dir / best_model / 'model_best_params.csv'} 重建模板", flush=True)
    oof = out_of_fold_predictions(best_model, templates_by_model[best_model], x, y, groups)
    oof_frame = pd.DataFrame(oof, index=x.index, columns=labels)

    print("[6/7] 绝对锚定（副轨）", flush=True)
    specs = anchor_mod.build_anchor_specs()
    anchor_params = anchor_mod.fit_anchor_params(x, y, train_ids, specs)
    anchor_params.to_csv(out_dir / "absolute_anchor_params.csv", index=False)
    abs_all = anchor_mod.apply_absolute(anchor_params, oof_frame)
    coverage = anchor_mod.anchor_coverage(anchor_params, oof_frame)
    anchor_params = anchor_params.merge(coverage, on="dimension", how="left")
    anchor_params.to_csv(out_dir / "absolute_anchor_params.csv", index=False)
    valid_dims = anchor_params.loc[anchor_params["valid"].astype(bool), "dimension"].tolist()
    thr_dims = anchor_params.loc[anchor_params["valid"].astype(bool)
                                 & anchor_params["anchor_type"].eq("threshold_anchored"), "dimension"].tolist()
    fb_dims = anchor_params.loc[anchor_params["valid"].astype(bool)
                                & anchor_params["anchor_type"].eq("relative_fallback"), "dimension"].tolist()
    invalid_dims = anchor_params.loc[~anchor_params["valid"].astype(bool), "dimension"].tolist()
    print(f"  锚定通过 {len(valid_dims)}/{len(anchor_params)} 维：{valid_dims}", flush=True)
    for _, row in anchor_params.loc[~anchor_params["valid"].astype(bool)].iterrows():
        print(f"    [不合格] {row['dimension']}：{row['invalid_reason']}", flush=True)

    # 留出集上的绝对轨评估：与连续指标同口径，统一使用锁定切分下该模型的留出集预测
    y_test = y.loc[test_ids]
    holdout_table = pd.read_csv(runs_dir / best_model / "test_predictions.csv.gz")
    holdout_table = holdout_table.loc[holdout_table["model"] == best_model]
    holdout_pred = holdout_table.pivot(index="beer", columns="label",
                                       values="predicted_continuous").loc[test_ids, labels]
    abs_metrics = absolute_track_metrics(anchor_params, y_test, holdout_pred, thr_dims)
    fb_metrics = absolute_track_metrics(anchor_params, y_test, holdout_pred, fb_dims)
    abs_detail = absolute_track_detail(anchor_params, y_test, holdout_pred, thr_dims + fb_dims)
    abs_detail.to_csv(out_dir / "absolute_grade_metrics_by_label.csv", index=False)

    print("[7/7] 组装交付物", flush=True)
    grades_all = apply_cutpoints(oof, cutpoints)
    grade_frame = pd.DataFrame(grades_all, index=x.index, columns=labels)
    raw = load_raw().sensory.set_index("beer")
    raw.index = raw.index.astype(int)
    keys = raw.reindex(x.index)[["beer_id", "tasting_category_fine"]]
    deliverable = pd.DataFrame(index=x.index)
    deliverable.insert(0, "beer", x.index)
    deliverable["beer_id"] = keys["beer_id"].to_numpy()
    deliverable["tasting_category_fine"] = keys["tasting_category_fine"].to_numpy()
    for dim in labels:
        deliverable[f"{dim}_pred"] = oof_frame[dim].to_numpy()
        deliverable[f"{dim}_grade"] = grade_frame[dim].to_numpy(dtype=int)
        if dim in valid_dims:
            deliverable[f"{dim}_abs"] = abs_all[dim].to_numpy()
        else:
            # 锚定不合格或不具备化学载体：绝对轨不适用，<dim>_abs 留空（NaN）。
            # 该维度仅提供主轨的相对分级（<dim>_grade），适用性见 absolute_anchor_params.csv。
            deliverable[f"{dim}_abs"] = np.nan
    deliverable = deliverable[["beer", "beer_id", "tasting_category_fine"] +
                               [f"{d}_{k}" for d in labels for k in ("pred", "grade", "abs")]]
    assert deliverable.shape == (250, 153), deliverable.shape
    grade_cols = [c for c in deliverable.columns if c.endswith("_grade")]
    abs_cols = [c for c in deliverable.columns if c.endswith("_abs")]
    pred_cols = [c for c in deliverable.columns if c.endswith("_pred")]
    assert not deliverable[pred_cols + grade_cols].isna().any().any(), "预测/分级列存在缺失值"
    assert deliverable[grade_cols].to_numpy().min() >= 0 and deliverable[grade_cols].to_numpy().max() <= 6
    for dim in valid_dims:
        values = deliverable[f"{dim}_abs"].to_numpy(dtype=float)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 6
    for dim in labels:
        if dim not in valid_dims:
            assert deliverable[f"{dim}_abs"].isna().all(), f"{dim}_abs 应留空"
    deliverable.to_csv(out_dir / "sensory_scores_0_6.csv", index=False)

    predictions.to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cutpoints, index=labels,
                 columns=[f"boundary_{i}" for i in range(1, 7)]).to_csv(out_dir / "proxy_grade_cutpoints.csv")

    model_summary = summary.merge(boot_res["summary"][["model", "bootstrap_ci_lo", "bootstrap_ci_hi",
                                                         "bootstrap_se"]], on="model", how="left")
    model_summary["global_mean_Macro_RMSE"] = baseline
    model_summary["delta_vs_global_mean"] = model_summary["Macro_RMSE"] - baseline
    model_summary["beats_global_mean"] = model_summary["Macro_RMSE"] < baseline
    for key, value in {**abs_metrics, **fb_metrics}.items():
        model_summary[key] = value
    model_summary["selected_as_final_model"] = model_summary["model"] == best_model
    model_summary["final_predictions_source"] = "5-fold OOF over all 250 samples (seed 0)"
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)

    # 模型工件（供问题三调用 scripts/q1_predict.py）
    import joblib

    full = fit_full_model(best_model, templates_by_model[best_model], x, y, groups)
    artifact = {
        "model_name": best_model,
        "kind": full["kind"],
        "estimators": full["estimators"],
        "group_members": {g: list(members.index) for g, members in groups.groupby(groups)},
        "feature_columns": list(x.columns),
        "target_columns": labels,
        "n_train_rows": int(len(x)),
    }
    joblib.dump(artifact, out_dir / f"model_{best_model}.joblib")
    (out_dir / f"model_{best_model}_metadata.json").write_text(json.dumps({
        "model_name": best_model,
        "kind": full["kind"],
        "feature_columns": list(x.columns),
        "target_columns": labels,
        "holdout_macro_rmse": float(summary.loc[summary["model"] == best_model, "Macro_RMSE"].iloc[0]),
        "holdout_baseline_macro_rmse": baseline,
        "cutpoints_file": "proxy_grade_cutpoints.csv",
        "anchor_params_file": "absolute_anchor_params.csv",
        "note": "工件在全部 250 样本上重训；<dim>_pred 交付值来自 5 折 OOF，二者不可混用。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "run_config.json").write_text(json.dumps({
        "models": list(summary["model"]),
        "best_model": best_model,
        "bootstrap_resamples": args.n_boot,
        "cutpoints_source": "train-only septiles (175 samples)",
        "absolute_track": {
            "valid_dimensions": valid_dims,
            "threshold_anchored": thr_dims,
            "relative_fallback": fb_dims,
            "invalid_dimensions": invalid_dims,
            "validity_rules": {k: v for k, v in anchor_mod.VALIDITY_RULES.items()},
        },
        "deliverable_shape": list(deliverable.shape),
        "deliverable_abs_policy": "仅 valid=True 的维度给出 <dim>_abs；其余留空（NaN）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== 最终模型对比 ==")
    print(model_summary[["model", "Macro_RMSE", "bootstrap_ci_lo", "bootstrap_ci_hi", "R2_median",
                         "Spearman_median", "proxy_grade_MAE", "proxy_exact_accuracy",
                         "proxy_quadratic_kappa", "beats_global_mean"]].to_string(index=False))
    print(f"\n绝对锚定：{len(valid_dims)}/{len(anchor_params)} 个候选维度通过有效性护栏 → {valid_dims}")
    print(f"  其中阈值锚定 {len(thr_dims)} 维 {thr_dims}；相对约束替代规则 {len(fb_dims)} 维 {fb_dims}")
    print(f"  未通过 {len(invalid_dims)} 维（<dim>_abs 留空）：{invalid_dims}")
    print(f"交付物：{out_dir / 'sensory_scores_0_6.csv'}  shape={deliverable.shape}")

    if args.figures:
        import q1_figures

        q1_figures.make_all(out_dir, runs_dir, (REPO_ROOT / "data/processed/q1_figures").resolve())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
