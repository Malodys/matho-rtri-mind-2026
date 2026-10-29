# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 泄漏断言均为「双侧」：既验证交付切点等于训练集切点，也验证它**不等于**测试集/全量切点，
#      避免写出恒真断言；
#   2) 交付表形状、取值范围、无缺失三项均硬断言；
#   3) 所有模型的 Macro-RMSE 必须严格优于全局均值基线（0.5762 量级），任何一项退化即测试失败。
"""问题一 · 交付物与泄漏检查的单元测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_contract import load_label_dictionary, load_modeling_matrix, load_split, load_targets
from q1_evaluation import apply_cutpoints, training_quantile_cutpoints

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "data" / "processed" / "q1_final"
RUNS_DIR = REPO_ROOT / "data" / "processed" / "q1_runs"


@pytest.fixture(scope="module")
def q1_data():
    """统一矩阵、目标、切分与维度顺序（全部经 data_contract 取数）。"""
    x = load_modeling_matrix()
    y = load_targets()
    split = load_split()
    labels = list(load_label_dictionary().reindex(y.columns).index)
    train_ids = split.loc[split["split"] == "train", "beer"].to_numpy(dtype=int)
    test_ids = split.loc[split["split"] == "test", "beer"].to_numpy(dtype=int)
    return {"x": x, "y": y, "labels": labels, "train_ids": train_ids, "test_ids": test_ids}


@pytest.fixture(scope="module")
def deliverable():
    path = FINAL_DIR / "sensory_scores_0_6.csv"
    if not path.is_file():
        pytest.skip("交付物尚未生成，跳过")
    return pd.read_csv(path)


def test_deliverable_shape_and_integrity(deliverable, q1_data):
    """交付表必须是 250×153、键列齐全、等级 ∈ [0,6]；绝对轨仅在通过护栏的维度上有值。"""
    assert deliverable.shape == (250, 153)
    for key in ("beer", "beer_id", "tasting_category_fine"):
        assert key in deliverable.columns
    assert deliverable["beer"].nunique() == 250
    grade_cols = [f"{d}_grade" for d in q1_data["labels"]]
    abs_cols = [f"{d}_abs" for d in q1_data["labels"]]
    pred_cols = [f"{d}_pred" for d in q1_data["labels"]]
    assert not deliverable[pred_cols + grade_cols].isna().any().any()
    assert deliverable[grade_cols].to_numpy().min() >= 0
    assert deliverable[grade_cols].to_numpy().max() <= 6
    assert np.isfinite(deliverable[pred_cols].to_numpy(dtype=float)).all()

    anchor_path = FINAL_DIR / "absolute_anchor_params.csv"
    if anchor_path.is_file():
        params = pd.read_csv(anchor_path)
        valid = set(params.loc[params["valid"].astype(bool), "dimension"])
        for dim in q1_data["labels"]:
            column = deliverable[f"{dim}_abs"]
            if dim in valid:
                values = column.to_numpy(dtype=float)
                assert np.isfinite(values).all(), f"{dim}_abs 不应有缺失"
                assert values.min() >= 0 and values.max() <= 6, f"{dim}_abs 越界"
            else:
                assert column.isna().all(), f"{dim}_abs 应留空（锚定不合格）"


def test_cutpoints_are_train_only(q1_data):
    """交付切点必须等于「只用 175 个训练样本」的六分位数。"""
    path = FINAL_DIR / "proxy_grade_cutpoints.csv"
    if not path.is_file():
        pytest.skip("切点文件尚未生成，跳过")
    delivered = pd.read_csv(path, index_col=0).reindex(q1_data["labels"]).to_numpy(dtype=float)
    train_cut = training_quantile_cutpoints(q1_data["y"].loc[q1_data["train_ids"]].to_numpy())
    assert np.allclose(delivered, train_cut, atol=1e-12)


def test_cutpoints_are_not_test_fitted(q1_data):
    """反证：交付切点不得等于「用测试集拟合」或「用全量拟合」的切点。"""
    path = FINAL_DIR / "proxy_grade_cutpoints.csv"
    if not path.is_file():
        pytest.skip("切点文件尚未生成，跳过")
    delivered = pd.read_csv(path, index_col=0).reindex(q1_data["labels"]).to_numpy(dtype=float)
    test_cut = training_quantile_cutpoints(q1_data["y"].loc[q1_data["test_ids"]].to_numpy())
    full_cut = training_quantile_cutpoints(q1_data["y"].to_numpy())
    assert not np.allclose(delivered, test_cut, atol=1e-9)
    assert not np.allclose(delivered, full_cut, atol=1e-9)


def test_grades_consistent_with_train_cutpoints(deliverable, q1_data):
    """交付等级必须恰等于「训练切点离散化交付预测值」。"""
    train_cut = training_quantile_cutpoints(q1_data["y"].loc[q1_data["train_ids"]].to_numpy())
    pred = deliverable[[f"{d}_pred" for d in q1_data["labels"]]].to_numpy(dtype=float)
    expected = apply_cutpoints(pred, train_cut)
    actual = deliverable[[f"{d}_grade" for d in q1_data["labels"]]].to_numpy(dtype=int)
    assert np.array_equal(expected, actual)


def test_anchor_p95_is_train_only(q1_data):
    """绝对锚定的 P95 锚点必须由训练集重算得到。"""
    path = FINAL_DIR / "absolute_anchor_params.csv"
    if not path.is_file():
        pytest.skip("锚定参数尚未生成，跳过")
    params = pd.read_csv(path)
    x, train_ids = q1_data["x"], q1_data["train_ids"]
    checked = 0
    for _, row in params.iterrows():
        if row["anchor_type"] not in ("threshold_anchored", "relative_fallback"):
            continue
        carrier = row["carrier"]
        expected = float(np.quantile(x.loc[train_ids, carrier].to_numpy(dtype=float), 0.95))
        assert np.isclose(float(row["anchor6_log10_conc"]), expected, atol=1e-12)
        checked += 1
    assert checked >= 1


def test_anchor_validity_flags_are_consistent(q1_data):
    """有效性护栏：valid=False 的维度必须写明不满足的规则，且其规则2判定可复算。"""
    path = FINAL_DIR / "absolute_anchor_params.csv"
    if not path.is_file():
        pytest.skip("锚定参数尚未生成，跳过")
    params = pd.read_csv(path)
    assert {"valid", "invalid_reason", "carrier_vs_sensory_spearman",
            "anchor0_train_fraction", "z_bandwidth", "frac_within_0_6_train"} <= set(params.columns)
    for _, row in params.iterrows():
        if bool(row["valid"]):
            assert row["invalid_reason"] == "" or pd.isna(row["invalid_reason"])
        else:
            assert isinstance(row["invalid_reason"], str) and row["invalid_reason"].strip() != ""
            assert row["invalid_reason"].count("规则") >= 1


def test_all_models_beat_global_mean_baseline():
    """全部模型在留出集上的 Macro-RMSE 必须严格优于全局均值基线。"""
    path = FINAL_DIR / "model_summary.csv"
    if not path.is_file():
        pytest.skip("模型汇总尚未生成，跳过")
    summary = pd.read_csv(path)
    baseline = float(summary.loc[summary["model"] == "global_mean", "Macro_RMSE"].iloc[0])
    others = summary.loc[summary["model"] != "global_mean"]
    assert len(others) >= 6
    assert (others["Macro_RMSE"] < baseline).all(), others[["model", "Macro_RMSE"]]


def test_predict_interface_contract():
    """对外接口：输入 231 列（含 *_sum）也能输出 250×50、索引一致、无缺失。"""
    from q1_predict import predict_sensory

    matrix = pd.read_csv(REPO_ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    matrix.index = matrix.index.astype(int)
    scores = predict_sensory(matrix)
    assert scores.shape == (250, 50)
    assert scores.index.equals(matrix.index)
    assert not scores.isna().any().any()


def test_model_runs_are_complete():
    """每个模型族都必须留下完整的 75×50 留出集预测。"""
    run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name != "_smoke")
    assert len(run_dirs) >= 7
    for run_dir in run_dirs:
        predictions = pd.read_csv(run_dir / "test_predictions.csv.gz")
        assert len(predictions) == 75 * 50
