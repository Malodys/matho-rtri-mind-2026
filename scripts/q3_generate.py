# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 每个风格生成 250 条（≥200），全部有限值、无 NaN —— 脚本内置 assert 并打印核对。
#   2) 有阈值的 3 项缺陷（DMS/Diacetyl/4VG）违规样本数必须为 0 —— 脚本内置 assert。
#   3) 边际 KS 距离逐化合物计算并落盘 ks_per_compound.csv；汇总中位数写入 generation_summary.csv。
#   4) 边缘覆盖率：生成值落在该风格真实样本 [min,max] 内的比例，逐化合物计算后汇总。
"""问题 3 步骤 1–5：按风格生成化合物谱（全局高斯 Copula + 风格条件边际 + 缺陷约束）。

产出
----
* ``data/processed/q3_runs/generation/<style>_synthetic.csv``
* ``data/processed/q3_runs/generation_summary.csv``
* ``data/processed/q3_runs/ks_per_compound.csv``
* ``data/processed/q3_runs/feasibility_report.csv``（生成部分；迁移部分由 q3_transfer.py 追加）

人工核验点见文件头；所有断言失败即中止，不会写出半成品。
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
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
GEN_DIR = RUN_DIR / "generation"

#: 参与 Copula 的全部缺陷键（3 项有阈值 + t2n 无阈值）
DEFECT_KEYS = ["dms", "diacetyl", "4vg", "t2n"]


def slug(style: str) -> str:
    return (style.replace(" ", "_").replace("/", "-")).strip("_")


def synthetic_id(style: str, i: int) -> str:
    return f"SYN_{slug(style)}_{i:04d}"


def generate_style(data: q3.Dataset, copula: q3.GlobalCopula, style: str, n: int,
                   seed: int, mode: str, specs: List[dict],
                   marginal_mode: str = "cdf", lam_scale: float = 1.0) -> Dict:
    """生成一个风格的化合物谱并返回统计信息。"""
    values_all = data.matrix.to_numpy(dtype=float)
    mask = (data.styles == style).to_numpy()
    real = values_all[mask]
    marginal = q3.build_marginal(copula, real, style, mode=marginal_mode,
                                 lam_scale=lam_scale)

    # 用 CRC32 而非内置 hash()：后者按进程随机化，会破坏种子可复现性
    style_seed = zlib.crc32(style.encode("utf-8"))
    rng = np.random.default_rng([seed, style_seed])
    gen, info, uncon = q3.enforce_defect_constraints(
        copula, marginal, n, specs, rng, mode=mode, return_unconstrained=True)
    assert uncon is not None and uncon.shape == gen.shape

    assert np.isfinite(gen).all(), f"{style}: 生成了非有限值"
    assert not np.isnan(gen).any(), f"{style}: 生成了 NaN"
    assert np.isfinite(uncon).all(), f"{style}: 无约束参照谱含非有限值"

    viol = q3.threshold_violations(gen, copula.columns, specs, 0.0)
    assert int(viol.sum()) == 0, f"{style}: 仍有 {int(viol.sum())} 条阈值违规"

    ks_style = q3.ks_per_column_fast(gen, real)
    ks_global = q3.ks_per_column_fast(gen, values_all)
    cov_style = q3.coverage_per_column(gen, real)
    cov_global = q3.coverage_per_column(gen, values_all)
    ks_target = q3.ks_vs_target_marginal(gen, marginal)

    # 无约束参照谱（同一候选池，仅去掉缺陷约束）
    ks_uncon = q3.ks_per_column_fast(uncon, real)
    cov_uncon = q3.coverage_per_column(uncon, real)
    viol_uncon = q3.threshold_violations(uncon, copula.columns, specs, 0.0)

    # 相关结构差异（生成的 vs 真实的同风格样本）
    pearson_dev = q3.pearson_mean_abs_dev(gen, real)
    spearman_dev, _ = q3.spearman_mean_abs_dev(gen, real)
    # 更公平的结构检验：生成 vs 全局 250 行真实相关矩阵、生成 vs Copula 结构 R
    pearson_dev_global = q3.pearson_mean_abs_dev(gen, values_all)
    pearson_dev_copula = float(np.abs(
        q3.safe_corr(gen)[np.triu_indices(copula.p, 1)] -
        copula.corr[np.triu_indices(copula.p, 1)]).mean())

    # 收缩带来的系统性偏差：KS(F_global, F_style)（真实样本 vs 全局样本）
    ks_global_style = q3.ks_per_column_fast(values_all, real)

    constrained_cols = [s["column"] for s in specs if s["has_threshold"]]
    free_mask = np.array([c not in constrained_cols for c in copula.columns])
    ks_free = ks_style[free_mask]

    defect_viol = {}
    defect_viol_uncon = {}
    defect_viol_real = {}
    defect_median_con = {}
    defect_median_uncon = {}
    defect_median_real = {}
    defect_target_exceed = {}
    for spec in specs:
        j = copula.columns.index(spec["column"])
        defect_median_con[spec["key"]] = float(np.median(gen[:, j]))
        defect_median_uncon[spec["key"]] = float(np.median(uncon[:, j]))
        defect_median_real[spec["key"]] = float(np.median(real[:, j]))
        if spec["has_threshold"]:
            defect_viol[spec["key"]] = int((gen[:, j] > spec["log10_threshold"]).sum())
            defect_viol_uncon[spec["key"]] = int((uncon[:, j] > spec["log10_threshold"]).sum())
            defect_viol_real[spec["key"]] = int((real[:, j] > spec["log10_threshold"]).sum())
            # 目标条件边际的理论超标率：1 − M(thr)
            defect_target_exceed[spec["key"]] = q3.marginal_exceed_prob(
                marginal, j, spec["log10_threshold"])
        else:
            defect_viol[spec["key"]] = -1     # 无阈值，-1 表示不适用
            defect_viol_uncon[spec["key"]] = -1
            defect_viol_real[spec["key"]] = -1
            defect_target_exceed[spec["key"]] = float("nan")

    return {
        "style": style,
        "n_real": int(mask.sum()),
        "n_generated": int(gen.shape[0]),
        "lambda": round(marginal.lam, 4),
        "lambda_base": round(q3.lambda_for(int(mask.sum())), 4),
        "marginal_mode": marginal_mode,
        "n_style_const_cols": int(marginal.const_mask.sum()),
        "values": gen,
        "unconstrained": uncon,
        "real": real,
        "marginal": marginal,
        "ks_style": ks_style,
        "ks_uncon": ks_uncon,
        "ks_global": ks_global,
        "ks_global_style": ks_global_style,
        "ks_target": ks_target,
        "cov_style": cov_style,
        "cov_uncon": cov_uncon,
        "cov_global": cov_global,
        "ks_median": float(np.median(ks_style)),
        "ks_mean": float(np.mean(ks_style)),
        "ks_p90": float(np.quantile(ks_style, 0.9)),
        "ks_max": float(ks_style.max()),
        "ks_median_free": float(np.median(ks_free)),
        "ks_mean_free": float(np.mean(ks_free)),
        "ks_null_expected": q3.ks_null_expectation(gen.shape[0], int(mask.sum())),
        "ks_null_median": q3.ks_null_median(gen.shape[0], int(mask.sum())),
        "ks_empirical_match_ref": q3.ks_oracle_floor(real, gen.shape[0], seed=seed),
        "ks_median_target": float(np.median(ks_target)) if np.isfinite(ks_target).any() else float("nan"),
        "ks_median_global": float(np.median(ks_global)),
        "ks_median_global_style": float(np.median(ks_global_style)),
        "cov_style_mean": float(np.mean(cov_style)),
        "cov_style_median": float(np.median(cov_style)),
        "cov_global_mean": float(np.mean(cov_global)),
        "cov_global_median": float(np.median(cov_global)),
        "cov_style_min": float(np.min(cov_style)),
        "n_violations_total": int(viol.sum()),
        "n_violations_unconstrained_total": int(viol_uncon.sum()),
        "defect_violations": defect_viol,
        "defect_violations_unconstrained": defect_viol_uncon,
        "defect_violations_real": defect_viol_real,
        "defect_median_constrained": defect_median_con,
        "defect_median_unconstrained": defect_median_uncon,
        "defect_median_real": defect_median_real,
        "defect_target_exceed": defect_target_exceed,
        "ks_median_unconstrained": float(np.median(ks_uncon)),
        "ks_mean_unconstrained": float(np.mean(ks_uncon)),
        "cov_uncon_mean": float(np.mean(cov_uncon)),
        "pearson_mean_abs_dev": pearson_dev,
        "pearson_mean_abs_dev_vs_global": pearson_dev_global,
        "pearson_mean_abs_dev_vs_copula": pearson_dev_copula,
        "spearman_mean_abs_dev": spearman_dev,
        "sampler_info": info,
    }


def sensitivity_study(data: q3.Dataset, copula: q3.GlobalCopula, styles: List[str],
                      specs: List[dict], seed: int, mode: str,
                      lam_scales: Sequence[float], n_list: Sequence[int],
                      marginal_mode: str) -> pd.DataFrame:
    """λ 与生成量敏感性研究：量化「收紧 λ」与「增加采样量」对 KS 中位数的影响。

    验收标准 5 要求：KS 中位数超过 0.15 时须说明原因并尝试收紧 λ 或增加采样量，
    本函数即该尝试的可复现证据。
    """
    rows: List[dict] = []
    for style in styles:
        for scale in lam_scales:
            for n in n_list:
                res = generate_style(data, copula, style, n, seed, mode, specs,
                                     marginal_mode=marginal_mode, lam_scale=scale)
                rows.append({
                    "style": style, "n_real": res["n_real"], "n_generated": n,
                    "lam_scale": scale, "lambda": res["lambda"],
                    "marginal_mode": marginal_mode,
                    "ks_median": round(res["ks_median"], 6),
                    "ks_median_vs_target": round(res["ks_median_target"], 6),
                    "ks_median_excl_constrained": round(res["ks_median_free"], 6),
                    "ks_empirical_match_ref": round(res["ks_empirical_match_ref"], 6),
                    "coverage_style_range_mean": round(res["cov_style_mean"], 6),
                    "n_threshold_violations": res["n_violations_total"],
                    "corr_pearson_mean_abs_dev": round(res["pearson_mean_abs_dev"], 6),
                })
                print(f"  [敏感性] {style:<11} λ×{scale:<5} n={n:<4} "
                      f"KS(风格)={res['ks_median']:.4f} "
                      f"KS(目标)={res['ks_median_target']:.4f} "
                      f"覆盖={res['cov_style_mean']:.4f}")
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题3 风格条件化合物谱生成（全局高斯 Copula + 风格条件边际 + 缺陷约束）")
    parser.add_argument("--styles", type=str, default=",".join(q3.DEFAULT_TARGET_STYLES),
                        help="逗号分隔的风格列表")
    parser.add_argument("--n", type=int, default=250, help="每个风格生成的样本数（验收要求 ≥200）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mode", choices=["reject", "compress"], default="reject",
                        help="缺陷约束机制：reject=拒绝-重抽，compress=单调压缩修复")
    parser.add_argument("--marginal-mode", choices=["cdf", "quantile"], default="cdf",
                        help="条件边际实现：cdf=题面口径 F_s=(1-λ)F_global+λF_s^emp 的广义逆；"
                             "quantile=分位空间线性混合（对照）")
    parser.add_argument("--lam-scale", type=float, default=1.0,
                        help="λ 乘子（1.0 = 题面口径 λ=n/(n+10)），用于敏感性研究")
    parser.add_argument("--sensitivity", action="store_true",
                        help="运行 λ 与样本量敏感性研究并落盘 lambda_sensitivity.csv")
    parser.add_argument("--sensitivity-styles", type=str,
                        default="Tripel,Kriek,Lambic,Saison",
                        help="敏感性研究的风格列表（取核心三风格 + KS 最差风格）")
    parser.add_argument("--ridge", type=float, default=1e-3, help="相关矩阵岭正则")
    args = parser.parse_args(argv)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    data = q3.load_dataset()
    specs = q3.defect_specs(data.defects)

    print("[拟合] 全局高斯 Copula …")
    copula = q3.fit_global_copula(data.matrix, ridge=args.ridge)
    print(f"  维度 {copula.p}；原始最小特征值 {copula.eig_raw.min():.3e}；"
          f"裁剪特征值数 {copula.n_clipped}；裁剪后最小 {copula.eig_clipped.min():.3e}")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    ks_rows: List[dict] = []
    feasibility: List[dict] = []
    results: Dict[str, Dict] = {}

    for style in styles:
        if style not in set(data.styles.astype(str)):
            print(f"[跳过] 风格 {style} 不在数据中")
            continue
        res = generate_style(data, copula, style, args.n, args.seed, args.mode, specs,
                             marginal_mode=args.marginal_mode, lam_scale=args.lam_scale)
        results[style] = res

        frame = pd.DataFrame(res["values"], columns=copula.columns)
        frame.insert(0, "tasting_category_fine", style)
        frame.insert(0, "beer", [synthetic_id(style, i) for i in range(frame.shape[0])])
        out_path = GEN_DIR / f"{slug(style)}_synthetic.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 无约束参照谱：同一候选池、仅去掉缺陷约束
        uframe = pd.DataFrame(res["unconstrained"], columns=copula.columns)
        uframe.insert(0, "tasting_category_fine", style)
        uframe.insert(0, "beer", [f"SYNU_{slug(style)}_{i:04d}"
                                   for i in range(uframe.shape[0])])
        uframe.to_csv(GEN_DIR / f"{slug(style)}_unconstrained.csv", index=False,
                      encoding="utf-8-sig")

        # 逐化合物 KS / 覆盖率落盘
        real_vals = res["real"]
        for j, col in enumerate(copula.columns):
            ks_rows.append({
                "style": style,
                "compound": col,
                "is_defect_with_threshold": col in [s["column"] for s in specs
                                                    if s["has_threshold"]],
                "is_style_const": bool(res["marginal"].const_mask[j]),
                "ks_generated_vs_style": round(float(res["ks_style"][j]), 6),
                "ks_generated_unconstrained_vs_style": round(float(res["ks_uncon"][j]), 6),
                "ks_generated_vs_global": round(float(res["ks_global"][j]), 6),
                "coverage_within_style_range": round(float(res["cov_style"][j]), 6),
                "coverage_within_global_range": round(float(res["cov_global"][j]), 6),
                "real_min": round(float(real_vals[:, j].min()), 6),
                "real_max": round(float(real_vals[:, j].max()), 6),
                "gen_min": round(float(res["values"][:, j].min()), 6),
                "gen_max": round(float(res["values"][:, j].max()), 6),
            })

        summary_rows.append({
            "style": style,
            "n_real": res["n_real"],
            "n_generated": res["n_generated"],
            "lambda": res["lambda"],
            "lambda_base": res["lambda_base"],
            "marginal_mode": res["marginal_mode"],
            "n_style_const_cols": res["n_style_const_cols"],
            "ks_median": round(res["ks_median"], 6),
            "ks_median_unconstrained": round(res["ks_median_unconstrained"], 6),
            "ks_mean_unconstrained": round(res["ks_mean_unconstrained"], 6),
            "coverage_style_range_mean_unconstrained": round(res["cov_uncon_mean"], 6),
            "ks_mean": round(res["ks_mean"], 6),
            "ks_p90": round(res["ks_p90"], 6),
            "ks_max": round(res["ks_max"], 6),
            "ks_median_excl_constrained": round(res["ks_median_free"], 6),
            "ks_mean_excl_constrained": round(res["ks_mean_free"], 6),
            "ks_null_expected": round(res["ks_null_expected"], 6),
            "ks_null_median": round(res["ks_null_median"], 6),
            "ks_empirical_match_ref": round(res["ks_empirical_match_ref"], 6),
            "ks_median_vs_target_marginal": round(res["ks_median_target"], 6),
            "ks_median_vs_global": round(res["ks_median_global"], 6),
            "ks_median_global_vs_style": round(res["ks_median_global_style"], 6),
            "coverage_style_range_mean": round(res["cov_style_mean"], 6),
            "coverage_style_range_median": round(res["cov_style_median"], 6),
            "coverage_style_range_min": round(res["cov_style_min"], 6),
            "coverage_global_range_mean": round(res["cov_global_mean"], 6),
            "n_threshold_violations_total": res["n_violations_total"],
            "n_violations_dms": res["defect_violations"]["dms"],
            "n_violations_diacetyl": res["defect_violations"]["diacetyl"],
            "n_violations_4vg": res["defect_violations"]["4vg"],
            "n_violations_t2n": res["defect_violations"]["t2n"],
            "n_violations_unconstrained_total": res["n_violations_unconstrained_total"],
            "n_violations_unconstrained_dms": res["defect_violations_unconstrained"]["dms"],
            "n_violations_unconstrained_diacetyl": res["defect_violations_unconstrained"]["diacetyl"],
            "n_violations_unconstrained_4vg": res["defect_violations_unconstrained"]["4vg"],
            "n_violations_real_dms": res["defect_violations_real"]["dms"],
            "n_violations_real_diacetyl": res["defect_violations_real"]["diacetyl"],
            "n_violations_real_4vg": res["defect_violations_real"]["4vg"],
            "corr_pearson_mean_abs_dev": round(res["pearson_mean_abs_dev"], 6),
            "corr_pearson_mean_abs_dev_vs_global": round(res["pearson_mean_abs_dev_vs_global"], 6),
            "corr_pearson_mean_abs_dev_vs_copula_R": round(res["pearson_mean_abs_dev_vs_copula"], 6),
            "corr_spearman_mean_abs_dev": round(res["spearman_mean_abs_dev"], 6),
            "constraint_mode": args.mode,
            "sensory_profile_status": "pending_q1",
        })

        # 缺陷列三态超标率（无约束 / 有约束 / 真实样本）与边际中位数
        for key in ["dms", "diacetyl", "4vg", "t2n"]:
            has_thr = key != "t2n"
            n_gen = res["n_generated"]
            n_real = res["n_real"]
            if has_thr:
                summary_rows[-1][f"exceed_pct_{key}_unconstrained"] = round(
                    100.0 * res["defect_violations_unconstrained"][key] / n_gen, 2)
                summary_rows[-1][f"exceed_pct_{key}_constrained"] = round(
                    100.0 * res["defect_violations"][key] / n_gen, 2)
                summary_rows[-1][f"exceed_pct_{key}_real"] = round(
                    100.0 * res["defect_violations_real"][key] / n_real, 2)
            else:
                # 文件 8 未提供 t2n 阈值 ⇒ 相对参照：该风格真实样本中位数
                for tag, vals in (("unconstrained", res["unconstrained"]),
                                  ("constrained", res["values"]),
                                  ("real", res["real"])):
                    j = copula.columns.index(
                        next(s["column"] for s in specs if s["key"] == key))
                    summary_rows[-1][f"t2n_above_style_median_pct_{tag}"] = round(
                        100.0 * float((vals[:, j] > res["defect_median_real"][key]).mean()), 2)
            summary_rows[-1][f"target_exceed_pct_{key}"] = round(
                100.0 * res["defect_target_exceed"][key], 2)
            summary_rows[-1][f"median_{key}_constrained"] = round(
                res["defect_median_constrained"][key], 6)
            summary_rows[-1][f"median_{key}_unconstrained"] = round(
                res["defect_median_unconstrained"][key], 6)
            summary_rows[-1][f"median_{key}_real"] = round(res["defect_median_real"][key], 6)
            summary_rows[-1][f"constraint_marginal_shift_{key}"] = round(
                res["defect_median_constrained"][key] - res["defect_median_unconstrained"][key], 6)

        feasibility.append({
            "scope": f"generation:{style}", "metric": "n_generated",
            "value": res["n_generated"], "note": "验收要求 ≥200"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "ks_median",
            "value": round(res["ks_median"], 6),
            "note": f"原假设期望 KS≈{res['ks_null_expected']:.4f}"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "ks_median_oracle_floor",
            "value": round(res["ks_empirical_match_ref"], 6),
            "note": "完美复现风格经验边际时的 KS 中位数参考（自助重采样）"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "ks_median_vs_target_marginal",
            "value": round(res["ks_median_target"], 6),
            "note": "生成样本 vs 其目标条件边际（自洽性）"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "ks_median_excl_constrained",
            "value": round(res["ks_median_free"], 6), "note": "剔除 3 个受阈值约束的缺陷列"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "coverage_within_style_range_mean",
            "value": round(res["cov_style_mean"], 6), "note": "生成值落在真实风格 [min,max] 内的比例"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "n_threshold_violations",
            "value": res["n_violations_total"], "note": "DMS/Diacetyl/4VG 合计，必须为 0"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "n_threshold_violations_unconstrained",
            "value": res["n_violations_unconstrained_total"],
            "note": "无约束参照谱的同口径超标数（用于分离「约束压缩」与「生成器还原度」）"})
        for key in ["dms", "diacetyl", "4vg", "t2n"]:
            feasibility.append({
                "scope": f"generation:{style}",
                "metric": f"constraint_marginal_shift_median_{key}",
                "value": round(res["defect_median_constrained"][key]
                               - res["defect_median_unconstrained"][key], 6),
                "note": f"约束前后该风格 {key} 列边际中位数之差（log10 mg/L；负=被压低）"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "corr_pearson_mean_abs_dev",
            "value": round(res["pearson_mean_abs_dev"], 6), "note": "生成 vs 真实同风格样本相关矩阵偏差"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "corr_pearson_mean_abs_dev_vs_global",
            "value": round(res["pearson_mean_abs_dev_vs_global"], 6),
            "note": "生成 vs 全局 250 行真实相关矩阵偏差（不受小样本噪声污染）"})
        feasibility.append({
            "scope": f"generation:{style}", "metric": "corr_pearson_mean_abs_dev_vs_copula_R",
            "value": round(res["pearson_mean_abs_dev_vs_copula"], 6),
            "note": "生成 vs Copula 结构矩阵 R 的偏差"})

        print(f"[生成] {style:<12} n={res['n_generated']} λ={res['lambda']:.4f} "
              f"KS(风格)={res['ks_median']:.4f} KS(目标)={res['ks_median_target']:.4f} "
              f"覆盖(风格域)={res['cov_style_mean']:.4f} 违规={res['n_violations_total']} "
              f"ρ偏差={res['pearson_mean_abs_dev']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RUN_DIR / "generation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ks_rows).to_csv(RUN_DIR / "ks_per_compound.csv", index=False,
                                  encoding="utf-8-sig")

    if args.sensitivity:
        print("[敏感性] λ 与生成量扫描 …")
        sens_styles = [s.strip() for s in args.sensitivity_styles.split(",") if s.strip()]
        sens_styles = [s for s in sens_styles if s in styles]
        sens = sensitivity_study(
            data, copula, sens_styles, specs, args.seed, args.mode,
            lam_scales=[0.5, 0.75, 1.0, 1.25, 1.5], n_list=[250, 500],
            marginal_mode=args.marginal_mode)
        sens.to_csv(RUN_DIR / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
        print(f"[落盘] {RUN_DIR/'lambda_sensitivity.csv'}")

    feasibility_path = RUN_DIR / "feasibility_report.csv"
    if feasibility_path.exists():
        old = pd.read_csv(feasibility_path, encoding="utf-8-sig")
        old = old[~old["scope"].astype(str).str.startswith("generation:")]
        feas = pd.concat([old, pd.DataFrame(feasibility)], ignore_index=True)
    else:
        feas = pd.DataFrame(feasibility)
    feas.to_csv(feasibility_path, index=False, encoding="utf-8-sig")

    meta = {
        "seed": args.seed, "n_per_style": args.n, "mode": args.mode,
        "marginal_mode": args.marginal_mode, "lam_scale": args.lam_scale,
        "ridge": args.ridge,
        "styles": [s for s in styles if s in results],
        "copula_min_eigenvalue_raw": float(copula.eig_raw.min()),
        "copula_min_eigenvalue_clipped": float(copula.eig_clipped.min()),
        "copula_n_eigen_clipped": int(copula.n_clipped),
        "sensory_profile_status": "pending_q1",
    }
    (RUN_DIR / "generation_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[落盘] {GEN_DIR} ; {RUN_DIR/'generation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
