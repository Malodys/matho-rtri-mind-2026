# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 迁移后四项缺陷浓度逐样本 ≤ 迁移前：脚本内置 assert（11 个 Pils/Lager 样本 × 4 项 = 44 项
#      逐条核验），并输出 worsened 标记列供人工逐行抽查。
#   2) 迁移后 Spearman ρ 相对迁移前的平均绝对偏差 < 0.05：脚本内置 assert。
#      原理：秩-分位映射对每一列都是严格单调替换，故每列的秩向量不变 ⇒ Spearman ρ 矩阵不变。
#   3) 迁移后有阈值的 3 项超标数 = 0：脚本内置 assert。
#   4) 源样本、目标边际均取自统一 log10(mg/L) 矩阵，经 data_contract 取数，无 10**x 逆变换。
"""问题 3 Step 7：跨风格迁移 Pils/Lager → Tripel。

方法：秩-分位映射（Iman-Conover 边际替换）+ 缺陷约束

1. 取真实 Pils/Lager 样本（n=11）的 231 维统一 log10(mg/L) 谱作为源矩阵 X_before。
2. 对每一化合物 j：把源样本的（平均）秩转成概率 u=(rank-0.5)/n，用 **Tripel 风格条件边际**
   F_Tripel^{-1} 替换边际，得到 X_after。由于每列替换严格单调，列内秩不变 ⇒ 相关结构保持。
3. 施加缺陷约束（同一套阈值口径）：
   - 有阈值的 DMS/Diacetyl/4VG：after ≤ min(before, 阈值-margin)；
   - 无阈值的 trans-2-nonenal：after ≤ before（「不恶化」相对约束兜底）。
   约束为逐样本的单调压缩，故仍保持列内序 ⇒ Spearman ρ 矩阵保持。

产出
----
* ``data/processed/q3_runs/transfer_pils_to_tripel.csv``（逐样本逐化合物前后对照 + 核验列）
* ``data/processed/q3_runs/transfer_pils_to_tripel_defects.csv``（逐样本四项缺陷前后值 + 核验）
* ``data/processed/q3_runs/transfer_pils_to_tripel_summary.csv``（迁移核验摘要）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402
import q3_copula as q3  # noqa: E402

RUN_DIR = REPO_ROOT / "data" / "processed" / "q3_runs"
SOURCE_STYLE = "Pils/Lager"
TARGET_STYLE = "Tripel"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题3 Step 7：跨风格迁移 Pils/Lager → Tripel（秩-分位映射 + 缺陷约束）")
    parser.add_argument("--source", type=str, default=SOURCE_STYLE, help="源风格")
    parser.add_argument("--target", type=str, default=TARGET_STYLE, help="目标风格")
    parser.add_argument("--ridge", type=float, default=1e-3, help="相关矩阵岭正则")
    parser.add_argument("--margin", type=float, default=q3.DEFECT_MARGIN,
                        help="阈值安全余量（log10 口径）")
    parser.add_argument("--max-rho-dev", type=float, default=0.05,
                        help="迁移前后 Spearman ρ 平均绝对偏差上限")
    args = parser.parse_args(argv)

    data = q3.load_dataset()
    specs = q3.defect_specs(data.defects)
    values = data.matrix.to_numpy(dtype=float)

    src_mask = (data.styles == args.source).to_numpy()
    tgt_mask = (data.styles == args.target).to_numpy()
    before = values[src_mask]
    target_real = values[tgt_mask]
    beers = data.matrix.index[src_mask].to_numpy()

    print(f"[取数] 源风格 {args.source} n={before.shape[0]}；"
          f"目标风格 {args.target} n={target_real.shape[0]}；维度 {before.shape[1]}")

    copula = q3.fit_global_copula(data.matrix, ridge=args.ridge)
    marginal = q3.build_marginal(copula, target_real, args.target, mode="cdf")
    print(f"[边际] λ = {marginal.lam:.4f}（= n/(n+10)），"
          f"目标风格常量列 {int(marginal.const_mask.sum())} 个")

    # Step 7.1 秩-分位映射：替换边际，保留原样本相关结构
    after = q3.transfer_rank_quantile(before, marginal, copula.columns)
    # Step 7.2 缺陷约束：阈值 + 「不恶化」
    after, cinfo = q3.apply_transfer_constraints(before, after, specs, copula.columns,
                                                 margin=args.margin)

    # ---------------- 核验 ----------------
    index = {c: i for i, c in enumerate(copula.columns)}
    defect_rows: List[dict] = []
    n_worsened = 0
    for spec in specs:
        j = index[spec["column"]]
        b = before[:, j]
        a = after[:, j]
        worse = a > b + 1e-12
        n_worsened += int(worse.sum())
        thr = spec["log10_threshold"]
        over = (a > thr) if spec["has_threshold"] else np.zeros(a.shape, dtype=bool)
        for k in range(a.shape[0]):
            defect_rows.append({
                "beer": int(beers[k]),
                "transferred_id": f"TRF_{args.source.replace('/', '-')}_to_"
                                  f"{args.target.replace(' ', '_')}_{int(beers[k])}",
                "defect_key": spec["key"],
                "compound": spec["column"],
                "value_before_log10_mgL": round(float(b[k]), 6),
                "value_after_log10_mgL": round(float(a[k]), 6),
                "delta_after_minus_before": round(float(a[k] - b[k]), 6),
                "improvement_before_minus_after": round(float(b[k] - a[k]), 6),
                "log10_threshold": (round(float(thr), 6) if spec["has_threshold"]
                                    else None),
                "has_threshold": spec["has_threshold"],
                "worsened": bool(worse[k]),
                "exceeds_threshold_after": bool(over[k]),
            })

    rho_mean, rho_max = q3.spearman_mean_abs_dev(before, after)
    pearson_dev = q3.pearson_mean_abs_dev(before, after)
    n_over_total = int(sum(r["exceeds_threshold_after"] for r in defect_rows))

    # 断言：逐样本四项缺陷均不恶化；有阈值三项零超标；相关结构保持
    assert n_worsened == 0, f"迁移后有 {n_worsened} 项缺陷浓度恶化，违反「不恶化」约束"
    assert n_over_total == 0, f"迁移后有 {n_over_total} 项超出阈值"
    assert rho_mean < args.max_rho_dev, (
        f"迁移前后 Spearman ρ 平均绝对偏差 {rho_mean:.4f} ≥ {args.max_rho_dev}")
    assert np.isfinite(after).all(), "迁移结果含非有限值"

    # ---------------- 落盘 ----------------
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    long_rows: List[dict] = []
    for k in range(before.shape[0]):
        for j, col in enumerate(copula.columns):
            spec = next((s for s in specs if s["column"] == col), None)
            b, a = float(before[k, j]), float(after[k, j])
            row = {
                "beer": int(beers[k]),
                "transferred_id": f"TRF_{args.source.replace('/', '-')}_to_"
                                  f"{args.target.replace(' ', '_')}_{int(beers[k])}",
                "source_style": args.source,
                "target_style": args.target,
                "compound": col,
                "value_before_log10_mgL": round(b, 6),
                "value_after_log10_mgL": round(a, 6),
                "delta_after_minus_before": round(a - b, 6),
                "is_defect": spec is not None,
                "defect_key": spec["key"] if spec else "",
                "worsened": bool(a > b + 1e-12),
                "log10_threshold": (round(float(spec["log10_threshold"]), 6)
                                    if spec and spec["has_threshold"] else None),
                "exceeds_threshold_after": (bool(a > spec["log10_threshold"])
                                            if spec and spec["has_threshold"] else None),
            }
            long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(RUN_DIR / "transfer_pils_to_tripel.csv", index=False,
                   encoding="utf-8-sig")

    defect_df = pd.DataFrame(defect_rows)
    defect_df.to_csv(RUN_DIR / "transfer_pils_to_tripel_defects.csv", index=False,
                      encoding="utf-8-sig")

    summary = {
        "source_style": args.source,
        "target_style": args.target,
        "n_source_samples": int(before.shape[0]),
        "n_compounds": int(before.shape[1]),
        "lambda_target": round(float(marginal.lam), 6),
        "n_defect_items_checked": len(defect_rows),
        "n_worsened_items": n_worsened,
        "all_defects_not_worsened": bool(n_worsened == 0),
        "n_exceeds_threshold_after": n_over_total,
        "spearman_mean_abs_dev": round(float(rho_mean), 6),
        "spearman_max_abs_dev": round(float(rho_max), 6),
        "spearman_dev_threshold": args.max_rho_dev,
        "spearman_dev_pass": bool(rho_mean < args.max_rho_dev),
        "pearson_mean_abs_dev": round(float(pearson_dev), 6),
        "n_columns_touched_by_constraint": int(sum(cinfo["n_touched_per_defect"].values())),
        "constraint_touched_per_defect": cinfo["n_touched_per_defect"],
    }
    for key in ("dms", "diacetyl", "4vg", "t2n"):
        sub = defect_df[defect_df.defect_key == key]
        summary[f"mean_before_{key}"] = round(float(sub["value_before_log10_mgL"].mean()), 6)
        summary[f"mean_after_{key}"] = round(float(sub["value_after_log10_mgL"].mean()), 6)
        summary[f"median_before_{key}"] = round(float(sub["value_before_log10_mgL"].median()), 6)
        summary[f"median_after_{key}"] = round(float(sub["value_after_log10_mgL"].median()), 6)
        summary[f"n_improved_{key}"] = int((sub["improvement_before_minus_after"] > 0).sum())
        summary[f"mean_improvement_{key}"] = round(
            float(sub["improvement_before_minus_after"].mean()), 6)
    pd.DataFrame([summary]).to_csv(RUN_DIR / "transfer_pils_to_tripel_summary.csv",
                                    index=False, encoding="utf-8-sig")

    # 追加到总可行性报告
    feas_path = RUN_DIR / "feasibility_report.csv"
    feas = pd.read_csv(feas_path, encoding="utf-8-sig") if feas_path.exists() \
        else pd.DataFrame(columns=["scope", "metric", "value", "note"])
    feas = feas[~feas["scope"].astype(str).str.startswith("transfer:")]
    extra = [
        {"scope": "transfer:pils_to_tripel", "metric": "n_defect_items_checked",
         "value": len(defect_rows), "note": "11 样本 × 4 项缺陷，逐样本核验"},
        {"scope": "transfer:pils_to_tripel", "metric": "n_worsened_items",
         "value": n_worsened, "note": "迁移后浓度 > 迁移前的项数，必须为 0"},
        {"scope": "transfer:pils_to_tripel", "metric": "n_exceeds_threshold_after",
         "value": n_over_total, "note": "迁移后超阈值项数，必须为 0"},
        {"scope": "transfer:pils_to_tripel", "metric": "spearman_mean_abs_dev",
         "value": round(float(rho_mean), 6), "note": "迁移前后 Spearman ρ 平均绝对偏差"},
        {"scope": "transfer:pils_to_tripel", "metric": "spearman_max_abs_dev",
         "value": round(float(rho_max), 6), "note": "最大单对偏差"},
        {"scope": "transfer:pils_to_tripel", "metric": "pearson_mean_abs_dev",
         "value": round(float(pearson_dev), 6), "note": "Pearson 口径参考"},
    ]
    feas = pd.concat([feas, pd.DataFrame(extra)], ignore_index=True)
    feas.to_csv(feas_path, index=False, encoding="utf-8-sig")

    (RUN_DIR / "transfer_run_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[核验] 缺陷项 {len(defect_rows)} 条，恶化 {n_worsened} 条（必须 0），"
          f"超标 {n_over_total} 条（必须 0）")
    print(f"[核验] Spearman ρ 平均绝对偏差 {rho_mean:.6f}（上限 {args.max_rho_dev}），"
          f"最大 {rho_max:.6f}；Pearson 平均偏差 {pearson_dev:.6f}")
    for key in ["dms", "diacetyl", "4vg", "t2n"]:
        d = defect_df[defect_df.defect_key == key]
        print(f"  {key:<9} 前中位={d['value_before_log10_mgL'].median():+.4f} "
              f"后中位={d['value_after_log10_mgL'].median():+.4f} "
              f"平均改善={d['improvement_before_minus_after'].mean():+.4f} "
              f"被约束样本数={cinfo['n_touched_per_defect'][key]}")
    print(f"[落盘] {RUN_DIR/'transfer_pils_to_tripel.csv'} 等 3 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
