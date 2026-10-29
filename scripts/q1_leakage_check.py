# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 断言 1：交付的 proxy_grade_cutpoints.csv 与「只用 175 个训练样本的六分位数」逐元素相等
#      （np.allclose，误差 < 1e-12）；
#   2) 断言 2（带牙齿的反证）：用 75 个留出样本单独拟合的切点与交付切点**不相等**，
#      用全部 250 样本拟合的切点也**不相等**——说明断言 1 不是恒真；
#   3) 断言 3：交付表 <dim>_grade 恰等于「训练切点离散化 <dim>_pred」的结果；
#   4) 断言 4：绝对锚定的 P95 锚点由训练集重算得到，与交付的 absolute_anchor_params.csv 一致；
#      且用全部 250 样本重算 P95 会得到不同的锚点（再次证明锚点无测试集信息）。
"""问题一 · 泄漏检查（分级切点与绝对锚定锚点必须只由训练集产生）。

用法：
    python scripts/q1_leakage_check.py
    python scripts/q1_leakage_check.py --final-dir data/processed/q1_final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_contract import load_label_dictionary, load_modeling_matrix, load_split, load_targets  # noqa: E402
from q1_evaluation import apply_cutpoints, training_quantile_cutpoints  # noqa: E402


def check(final_dir: Path) -> dict:
    """执行全部泄漏断言，返回逐项结果。"""
    x = load_modeling_matrix()
    y = load_targets()
    split = load_split()
    train_ids = split.loc[split["split"] == "train", "beer"].to_numpy(dtype=int)
    test_ids = split.loc[split["split"] == "test", "beer"].to_numpy(dtype=int)
    labels = list(load_label_dictionary().reindex(y.columns).index)

    results = {}

    # ---- 断言 1：交付切点 == 训练集切点
    delivered = pd.read_csv(final_dir / "proxy_grade_cutpoints.csv", index_col=0)
    delivered = delivered.reindex(labels)
    train_cut = training_quantile_cutpoints(y.loc[train_ids].to_numpy())
    results["cutpoints_equal_train_only"] = bool(
        np.allclose(delivered.to_numpy(dtype=float), train_cut, atol=1e-12))
    results["cutpoints_max_abs_diff"] = float(np.max(np.abs(delivered.to_numpy(dtype=float) - train_cut)))

    # ---- 断言 2（反证）：测试集/全量切点与交付切点不同
    test_cut = training_quantile_cutpoints(y.loc[test_ids].to_numpy())
    full_cut = training_quantile_cutpoints(y.to_numpy())
    results["cutpoints_differ_from_test_fitted"] = bool(
        not np.allclose(delivered.to_numpy(dtype=float), test_cut, atol=1e-9))
    results["cutpoints_differ_from_full_fitted"] = bool(
        not np.allclose(delivered.to_numpy(dtype=float), full_cut, atol=1e-9))
    results["test_fitted_max_abs_diff"] = float(np.max(np.abs(delivered.to_numpy(dtype=float) - test_cut)))

    # ---- 断言 3：交付等级 == 用训练切点离散化交付预测值
    scores = pd.read_csv(final_dir / "sensory_scores_0_6.csv")
    pred = scores[[f"{d}_pred" for d in labels]].to_numpy(dtype=float)
    expected = apply_cutpoints(pred, train_cut)
    actual = scores[[f"{d}_grade" for d in labels]].to_numpy(dtype=int)
    results["grades_match_train_cutpoints"] = bool(np.array_equal(expected, actual))
    results["grades_in_range"] = bool(actual.min() >= 0 and actual.max() <= 6)
    results["deliverable_shape_ok"] = tuple(scores.shape) == (250, 153)
    results["deliverable_pred_grade_no_na"] = bool(
        not scores[[f"{d}_pred" for d in labels] + [f"{d}_grade" for d in labels]].isna().any().any())
    # 绝对轨：仅通过有效性护栏的维度有值，其余必须留空（NaN），绝不以代理分级回填
    anchor_params = pd.read_csv(final_dir / "absolute_anchor_params.csv")
    valid_dims = set(anchor_params.loc[anchor_params["valid"].astype(bool), "dimension"])
    abs_ok = True
    for dim in labels:
        column = scores[f"{dim}_abs"]
        if dim in valid_dims:
            if column.isna().any() or column.min() < 0 or column.max() > 6:
                abs_ok = False
        elif not column.isna().all():
            abs_ok = False
    results["deliverable_abs_matches_validity"] = bool(abs_ok)

    # ---- 断言 4：绝对锚定的 P95 锚点只由训练集产生
    anchor_path = final_dir / "absolute_anchor_params.csv"
    if anchor_path.is_file():
        params = anchor_params
        train_p95 = {c: float(np.quantile(x.loc[train_ids, c].to_numpy(dtype=float), 0.95))
                     for c in x.columns}
        full_p95 = {c: float(np.quantile(x[c].to_numpy(dtype=float), 0.95)) for c in x.columns}
        ok_train, matches_full, changed = True, False, False
        for _, row in params.iterrows():
            if row["anchor_type"] not in ("threshold_anchored", "relative_fallback"):
                continue
            carrier = row["carrier"]
            if not np.isclose(float(row["anchor6_log10_conc"]), train_p95[carrier], atol=1e-12):
                ok_train = False
            if np.isclose(float(row["anchor6_log10_conc"]), full_p95[carrier], atol=1e-12):
                matches_full = True
            if not np.isclose(train_p95[carrier], full_p95[carrier], atol=1e-12):
                changed = True
        results["anchor_p95_equals_train_p95"] = bool(ok_train)
        results["anchor_p95_differs_from_full_p95"] = bool(changed and not matches_full)

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 · 泄漏检查")
    ap.add_argument("--final-dir", default="data/processed/q1_final")
    args = ap.parse_args()
    final_dir = (REPO_ROOT / args.final_dir).resolve()
    results = check(final_dir)
    width = max(len(k) for k in results)
    failed = []
    for key, value in results.items():
        flag = "PASS" if value else "FAIL"
        print(f"[{flag}] {key.ljust(width)} : {value}")
        if value is False:
            failed.append(key)
    print("\n结论：" + ("全部通过，无测试集信息泄漏" if not failed else f"存在未通过项 {failed}"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
