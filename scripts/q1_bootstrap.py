# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 采用「配对 bootstrap」：2000 次重采样只抽 75 个留出样本的行索引，同一组索引同时作用于
#      全部模型，保证模型间差值的置信区间有效；随机种子固定为 0，抽样结果可复现。
#   2) 每个重采样内先按 50 个维度分别算 RMSE 再取平均（与主口径 Macro-RMSE 完全一致），
#      分级指标同样按「先按维度算、再对维度取平均」聚合，未做样本级平均再平均的偷懒聚合。
#   3) 已核对：仅凭 0.001 量级的 Macro-RMSE 差距不足以宣称某模型显著更优，故本模块额外输出
#      「CI 是否重叠」矩阵与「相对基线的改善幅度」置信区间，供报告如实引用。
"""问题一 · 模型比较的配对 bootstrap（Macro-RMSE / 分级指标）。

用法：
    python scripts/q1_bootstrap.py --runs-dir data/processed/q1_runs \
        --output-dir data/processed/q1_final --n-boot 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from q1_evaluation import load_completed_runs  # noqa: E402


def _matrices(predictions: pd.DataFrame, labels: list) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                                                               Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """把长表预测拆成 75×50 矩阵（连续值、真值等级、预测等级）。"""
    beers = sorted(predictions["beer"].unique())
    models = sorted(predictions["model"].unique())
    cont = {m: np.full((len(beers), len(labels)), np.nan) for m in models}
    true_grade = np.full((len(beers), len(labels)), np.nan)
    pred_grade = {m: np.full((len(beers), len(labels)), np.nan) for m in models}
    true_cont = np.full((len(beers), len(labels)), np.nan)
    beer_pos = {b: i for i, b in enumerate(beers)}
    label_pos = {l: j for j, l in enumerate(labels)}
    for row in predictions.itertuples(index=False):
        i, j = beer_pos[row.beer], label_pos[row.label]
        cont[row.model][i, j] = row.predicted_continuous
        pred_grade[row.model][i, j] = row.predicted_proxy_grade
        true_grade[i, j] = row.observed_proxy_grade
        true_cont[i, j] = row.observed_continuous
    return np.array(beers), cont, pred_grade, {"true_grade": true_grade, "true_cont": true_cont}


def bootstrap_models(true_cont: np.ndarray, cont: Dict[str, np.ndarray],
                     true_grade: np.ndarray, pred_grade: Dict[str, np.ndarray],
                     n_boot: int = 2000, seed: int = 0) -> Dict[str, object]:
    """配对 bootstrap：返回各模型的 Macro-RMSE/等级指标分布与两两差值。

    人工核验点：CI 为 2.5%/97.5% 分位；差值分布以「模型 - 参考模型」为符号约定，
    负值表示更优；重叠判定同时给出「CI 是否相交」与「差值 CI 是否跨 0」两个口径。
    """
    n = true_cont.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))

    sq_err = {m: (cont[m] - true_cont) ** 2 for m in cont}
    grade_err = {m: np.abs(pred_grade[m] - true_grade) for m in pred_grade}
    grade_exact = {m: (pred_grade[m] == true_grade).astype(float) for m in pred_grade}

    rmse_dist, mae_dist, exact_dist = {}, {}, {}
    for m in cont:
        # (n_boot, 50)：每个重采样内按维度算 RMSE，再对 50 维取平均
        rmse_dist[m] = np.sqrt(sq_err[m][draws].mean(axis=1)).mean(axis=1)
        mae_dist[m] = grade_err[m][draws].mean(axis=(1, 2))
        exact_dist[m] = grade_exact[m][draws].mean(axis=(1, 2))

    models = list(cont)
    point_rmse = {m: float(np.sqrt(sq_err[m].mean(axis=0)).mean()) for m in models}
    point_grade_mae = {m: float(grade_err[m].mean()) for m in models}
    point_exact = {m: float(grade_exact[m].mean()) for m in models}

    summary = pd.DataFrame({
        "model": models,
        "Macro_RMSE": [point_rmse[m] for m in models],
        "bootstrap_ci_lo": [float(np.quantile(rmse_dist[m], 0.025)) for m in models],
        "bootstrap_ci_hi": [float(np.quantile(rmse_dist[m], 0.975)) for m in models],
        "bootstrap_se": [float(rmse_dist[m].std(ddof=1)) for m in models],
        "grade_MAE": [point_grade_mae[m] for m in models],
        "grade_MAE_ci_lo": [float(np.quantile(mae_dist[m], 0.025)) for m in models],
        "grade_MAE_ci_hi": [float(np.quantile(mae_dist[m], 0.975)) for m in models],
        "exact_accuracy": [point_exact[m] for m in models],
        "exact_accuracy_ci_lo": [float(np.quantile(exact_dist[m], 0.025)) for m in models],
        "exact_accuracy_ci_hi": [float(np.quantile(exact_dist[m], 0.975)) for m in models],
    }).sort_values("Macro_RMSE").reset_index(drop=True)

    # 两两差值（相对最优模型）与相对基线的改善幅度
    best = summary.iloc[0]["model"]
    base = "global_mean" if "global_mean" in models else models[-1]
    diff_rows, overlap_rows = [], []
    for m in models:
        for ref in models:
            if m == ref:
                continue
            d = rmse_dist[m] - rmse_dist[ref]
            diff_rows.append({
                "model": m, "reference": ref,
                "delta_Macro_RMSE": point_rmse[m] - point_rmse[ref],
                "delta_ci_lo": float(np.quantile(d, 0.025)),
                "delta_ci_hi": float(np.quantile(d, 0.975)),
                "frac_resamples_model_better": float(np.mean(d < 0)),
                "significant_at_95": bool(np.quantile(d, 0.025) > 0 or np.quantile(d, 0.975) < 0),
            })
            lo_m, hi_m = np.quantile(rmse_dist[m], [0.025, 0.975])
            lo_r, hi_r = np.quantile(rmse_dist[ref], [0.025, 0.975])
            overlap_rows.append({
                "model": m, "reference": ref,
                "ci_overlap": bool(not (hi_m < lo_r or hi_r < lo_m)),
                "width_model": float(hi_m - lo_m), "width_reference": float(hi_r - lo_r),
            })
    diffs = pd.DataFrame(diff_rows)
    overlaps = pd.DataFrame(overlap_rows)
    return {
        "summary": summary, "diffs": diffs, "overlaps": overlaps,
        "best_model": best, "baseline_model": base,
        "rmse_dist": rmse_dist, "grade_mae_dist": mae_dist, "exact_dist": exact_dist,
        "point_rmse": point_rmse,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 · 模型比较的配对 bootstrap")
    ap.add_argument("--runs-dir", default="data/processed/q1_runs")
    ap.add_argument("--output-dir", default="data/processed/q1_final")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = (REPO_ROOT / args.runs_dir).resolve()
    out = (REPO_ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    continuous, grades, predictions, _ = load_completed_runs(runs)
    labels = sorted(continuous["label"].unique())
    _, cont, pred_grade, truth = _matrices(predictions, labels)
    res = bootstrap_models(truth["true_cont"], cont, truth["true_grade"], pred_grade,
                           n_boot=args.n_boot, seed=args.seed)
    res["summary"].to_csv(out / "bootstrap_macro_rmse.csv", index=False)
    res["diffs"].to_csv(out / "bootstrap_model_differences.csv", index=False)
    res["overlaps"].to_csv(out / "bootstrap_ci_overlap.csv", index=False)
    print(res["summary"].to_string(index=False))
    print(f"最优模型：{res['best_model']}；基线：{res['baseline_model']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
