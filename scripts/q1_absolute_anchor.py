# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 单位换算逐条核对：仅 DMS 需要 µg/L→mg/L（17.25 µg/L → 0.01725 mg/L → log10=-1.7629），
#      其余载体的阈值单位与 provenance 的 unit_in_file8 一致，统一矩阵存 log10(原单位)；
#      Diacetyl/4VG 走「逆变换→统一mg/L→log10」，故阈值先统一到 mg/L 再取 log10。
#   2) 阈值取值一律取 8.xlsx『Sensory thresholds』的 threshold_mean（与 data_contract.defect_table 同口径），
#      同时保留 threshold_median 作敏感性列；无阈值的 3 个载体（X.E..2.nonenal / unfermentables /
#      CO2.PSI.）改用训练集 P05 分位作 0 分锚点，anchor_type 标为 relative_fallback。
#   3) 两个锚点（阈值→0 分、训练集 P95→6 分）经「载体浓度经验分布 → 标准化感官值经验分布」的分位匹配
#      换算到 z 空间，再用两点解出仿射映射 raw = a + b·z；a/b 已逐维打印核对（如 bitternes:
#      z_A=-0.778, z_B=1.285 → b=2.908, a=2.262，代回两点误差 < 1e-9）。
#   4) 每个维度的「载体浓度 vs 观测感官值」Spearman 已逐维打印，用于标注锚定可信度
#      （X4vg ρ=0.507、dms ρ=0.088、diacetyl ρ=-0.096 等弱/反向维度在结果中显式标记）。
#   5) 有效性护栏（2026-10-28 主控复核后新增）：不满足护栏的维度一律判定 valid=False、
#      <dim>_abs 留空，并在 invalid_reason 中写明不满足哪一条；护栏阈值集中在本文件
#      VALIDITY_RULES 中，README 与论文必须同步引用同一组阈值。
"""问题一 · 副轨：基于感官阈值的绝对锚定（absolute anchor）。

定义式
------
对每个具备化学载体的感官维度 d（载体列 c，见 data_contract.ABSOLUTE_ANCHOR_CARRIERS）：

  1. x_i = 统一矩阵中载体 c 的 log10 浓度（单位随列而定，已逐条核对）；
  2. 锚点 1（0 分）：x_thr = log10(该维度的感知阈值，来自 8.xlsx『Sensory thresholds』）。
     通过训练集经验分布的分位匹配换算到 z 空间：z_A = Q_z(F_x(x_thr))，
     其中 F_x 为训练集载体浓度的经验分布，Q_z 为训练集标准化感官值的分位函数；
  3. 锚点 2（6 分）：x_95 = 训练集载体浓度的 P95，z_B = Q_z(F_x(x_95))（= 训练集 z 的 P95）；
  4. 由两点解仿射映射 raw_score = a + b·z，b = 6/(z_B - z_A)，a = -b·z_A；
  5. 对任一样本：<dim>_abs = clip(a + b·ẑ, 0, 6)，ẑ 为该维度的标准化感官值
     （交付物中用模型的 5 折 OOF 预测；评估时用留出集预测与观测值分别代入）。

无阈值的载体以「训练集 P05 浓度 → 0 分」作相对约束兜底，anchor_type = relative_fallback。

有效性护栏（VALIDITY_RULES）
---------------------------
锚定只在下列条件全部满足时成立，否则该维度 valid=False、<dim>_abs 留空：

  规则1 载体阈值：载体须在 8.xlsx 有感知阈值；**本文已声明的替代规则**为「训练集 P05 浓度 → 0 分、
        训练集 P95 浓度 → 6 分」的相对约束（anchor_type=relative_fallback），使用该替代规则的维度
        记 rule1_route=替代规则，其 0 分含义是「该载体在训练集中的最低 5%」，**不是**「低于感知阈值」；
  规则2 方向性：载体浓度与该感官维度的 Spearman 相关系数 ≥ 0.30 且为正；
  规则3 0 分锚点位置：锚点 0 对应的训练集分位 ≤ 0.60（避免 0 分锚点已落在分布 60% 以上）；
  规则4 映射带宽：z_at_anchor6 − z_at_anchor0 ≥ 0.50（避免映射退化成阶跃）；
  规则5 有效覆盖：映射后落在 [0,6] 内的样本比例 ≥ 0.30（在训练集观测 z 上评估）。

本轨在论文中必须写作「基于感官阈值的一阶近似，本文提出」，不得宣称为实测 0–6 分。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_contract import (ABSOLUTE_ANCHOR_CARRIERS, RAW_DIR, load_modeling_matrix,
                           load_raw, load_targets)

#: 载体浓度单位 → mg/L 的换算系数（仅当统一矩阵以 mg/L 存储时才需要）
UNIT_TO_MGL: Dict[str, float] = {
    "ug/L": 1e-3, "µg/L": 1e-3, "mg/L": 1.0, "g/L": 1e3, "ppb": 1e-6, "ng/L": 1e-6,
}

#: 绝对锚定的有效性护栏阈值（README / 论文必须引用同一组数值）
VALIDITY_RULES: Dict[str, object] = {
    "rule1_threshold": "载体须有 8.xlsx 感知阈值；否则可用本文已声明的相对约束替代规则"
                        "（训练集 P05→0 分、P95→6 分），其余规则须全部满足",
    "rule2_min_spearman": 0.30,      # 载体浓度 vs 标准化感官值的 Spearman 下限（且为正）
    "rule3_max_anchor0_fraction": 0.60,  # 0 分锚点对应的训练集分位上限
    "rule4_min_z_bandwidth": 0.50,   # z 锚点带宽下限（避免映射退化为阶跃）
    "rule5_min_within_fraction": 0.30,   # 映射后落在 [0,6] 内的样本比例下限（训练集观测 z）
}


@dataclass
class AnchorSpec:
    """一个感官维度的绝对锚定规格。"""
    dimension: str
    carrier: str
    anchor_type: str          # threshold_anchored | relative_fallback
    threshold_value: float    # 阈值原值（无阈值为 NaN）
    threshold_unit: str
    threshold_median: float
    matrix_unit: str          # 统一矩阵中该列的浓度单位
    log10_threshold: float    # 统一矩阵口径下的 log10 阈值（无阈值为 NaN）
    notes: str


def _load_provenance() -> pd.DataFrame:
    """读取队友产出的逐列溯源表（单位 / 标度动作）。"""
    path = RAW_DIR.parent / "interim" / "chem_feature_provenance_v1.csv"
    prov = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    prov.index = prov.index.astype(str)
    return prov


def _threshold_rows() -> pd.DataFrame:
    """读取 8.xlsx 的感官阈值表（按 field_name_R 对齐统一矩阵列名）。"""
    thresholds = load_raw().thresholds
    thresholds = thresholds.copy()
    thresholds["field_name_R"] = thresholds["field_name_R"].astype(str).str.strip()
    return thresholds


def build_anchor_specs() -> pd.DataFrame:
    """构造 11 个可锚定维度的规格表（单位、阈值、log10 阈值、锚定类型）。

    人工核验点：8 个维度有阈值（bitternes/alcohol/dms/X4vg/diacetyl/metallic/acidity/body），
    3 个维度无阈值（t2n/sweetness/co2）走相对约束兜底；每个维度的矩阵单位与阈值单位已比对一致。
    """
    prov = _load_provenance()
    thresholds = _threshold_rows()
    rows: List[dict] = []
    for dim, carriers in ABSOLUTE_ANCHOR_CARRIERS.items():
        carrier = carriers[0]
        secondary = ";".join(carriers[1:])
        action = str(prov.loc[carrier, "scale_action"]) if carrier in prov.index else ""
        file1_unit = str(prov.loc[carrier, "unit_in_file8"]) if carrier in prov.index else ""
        matrix_unit = "mg/L" if "统一mg/L" in action else file1_unit
        hit = thresholds[thresholds["field_name_R"] == carrier]
        notes = []
        if secondary:
            notes.append(f"次级载体 {secondary}（无阈值，仅作参考）")
        if hit.empty:
            rows.append({
                "dimension": dim, "carrier": carrier, "anchor_type": "relative_fallback",
                "threshold_value": float("nan"), "threshold_unit": "",
                "threshold_median": float("nan"), "matrix_unit": matrix_unit,
                "log10_threshold": float("nan"),
                "notes": "；".join(notes + ["8.xlsx 无该载体阈值，改用训练集 P05 分位作 0 分锚点"]),
            })
            continue
        row = hit.iloc[0]
        thr_unit = str(row["unit"]) if pd.notna(row["unit"]) else file1_unit
        thr_mean = float(row["threshold_mean"]) if pd.notna(row["threshold_mean"]) else float("nan")
        thr_median = float(row["threshold_median"]) if pd.notna(row["threshold_median"]) else float("nan")
        if not np.isfinite(thr_mean):
            # 8.xlsx 有该化合物行但**没有阈值数值**（如 CO2.PSI. / unfermentables / kcalperc），
            # 按「无阈值」处理，改用训练集 P05 分位的相对约束替代规则。
            rows.append({
                "dimension": dim, "carrier": carrier, "anchor_type": "relative_fallback",
                "threshold_value": float("nan"), "threshold_unit": thr_unit,
                "threshold_median": float("nan"), "matrix_unit": matrix_unit,
                "log10_threshold": float("nan"),
                "notes": "；".join(notes + ["8.xlsx 该载体无阈值数值，改用训练集 P05 分位作 0 分锚点"]),
            })
            continue
        if matrix_unit == "mg/L":
            factor = UNIT_TO_MGL.get(thr_unit)
            if factor is None:
                raise ValueError(f"{carrier}: 阈值单位 {thr_unit} 无法换算到 mg/L")
            thr_mgl = thr_mean * factor
            notes.append(f"阈值单位 {thr_unit} ×{factor} → mg/L")
        else:
            if thr_unit != matrix_unit:
                notes.append(f"注意：阈值单位 {thr_unit} 与矩阵单位 {matrix_unit} 不同，按 1:1 处理")
            thr_mgl = thr_mean
        log10_thr = float(np.log10(thr_mgl)) if thr_mgl > 0 else float("nan")
        rows.append({
            "dimension": dim, "carrier": carrier, "anchor_type": "threshold_anchored",
            "threshold_value": thr_mean, "threshold_unit": thr_unit,
            "threshold_median": thr_median, "matrix_unit": matrix_unit,
            "log10_threshold": log10_thr,
            "notes": "；".join(notes),
        })
    return pd.DataFrame(rows)


def fit_anchor_params(x: pd.DataFrame, y: pd.DataFrame, train_ids: np.ndarray,
                      specs: pd.DataFrame) -> pd.DataFrame:
    """在训练集上拟合每个维度的锚定参数（分位匹配 → 仿射映射）并判定锚定是否有效。

    人工核验点：所有分位（x_thr/x_95/P05）均取自 175 个训练样本，绝不触碰 75 个留出样本；
    有效覆盖率（规则5）在训练集观测 z 上评估，与模型无关，故护栏结论不受预测值影响。
    """
    rows = []
    for _, spec in specs.iterrows():
        dim, carrier = spec["dimension"], spec["carrier"]
        x_train = x.loc[train_ids, carrier].to_numpy(dtype=float)
        z_train = y.loc[train_ids, dim].to_numpy(dtype=float)
        x95 = float(np.quantile(x_train, 0.95))
        if spec["anchor_type"] == "threshold_anchored":
            x0 = float(spec["log10_threshold"])
            f0 = float((x_train < x0).mean())
        else:
            x0 = float(np.quantile(x_train, 0.05))
            f0 = 0.05
        z_a = float(np.quantile(z_train, f0))
        z_b = float(np.quantile(z_train, float((x_train < x95).mean())))
        if not np.isfinite(z_b - z_a) or abs(z_b - z_a) < 1e-9:
            b, a = float("nan"), float("nan")
        else:
            b = 6.0 / (z_b - z_a)
            a = -b * z_a
        rho = float(pd.Series(x_train).corr(pd.Series(z_train), method="spearman"))
        bandwidth = z_b - z_a
        if np.isfinite(a) and np.isfinite(b):
            raw_train = a + b * z_train
            within = float(np.mean((raw_train >= 0) & (raw_train <= 6)))
            clip0 = float(np.mean(raw_train < 0))
            clip6 = float(np.mean(raw_train > 6))
        else:
            raw_train = np.full(len(z_train), np.nan)
            within, clip0, clip6 = float("nan"), float("nan"), float("nan")

        # ---- 有效性护栏
        reasons = []
        has_threshold = spec["anchor_type"] == "threshold_anchored"
        rule1_route = ("规则1:阈值锚定（8.xlsx 感知阈值）" if has_threshold
                       else "规则1:相对约束替代规则（训练集 P05→0 分、P95→6 分，本文已声明）")
        if not (np.isfinite(rho) and rho >= float(VALIDITY_RULES["rule2_min_spearman"])):
            reasons.append(f"规则2:Spearman={rho:.4f}<{VALIDITY_RULES['rule2_min_spearman']}")
        if not f0 <= float(VALIDITY_RULES["rule3_max_anchor0_fraction"]):
            reasons.append(f"规则3:0分锚点分位={f0:.3f}>{VALIDITY_RULES['rule3_max_anchor0_fraction']}")
        if not (np.isfinite(bandwidth) and bandwidth >= float(VALIDITY_RULES["rule4_min_z_bandwidth"])):
            reasons.append(f"规则4:z锚点带宽={bandwidth:.3f}<{VALIDITY_RULES['rule4_min_z_bandwidth']}")
        if not (np.isfinite(within) and within >= float(VALIDITY_RULES["rule5_min_within_fraction"])):
            reasons.append(f"规则5:[0,6]内比例={within:.3f}<{VALIDITY_RULES['rule5_min_within_fraction']}")

        rows.append({
            "dimension": dim, "carrier": carrier, "anchor_type": spec["anchor_type"],
            "threshold_value": spec["threshold_value"], "threshold_unit": spec["threshold_unit"],
            "log10_threshold": spec["log10_threshold"], "matrix_unit": spec["matrix_unit"],
            "anchor0_log10_conc": x0, "anchor0_train_fraction": f0,
            "anchor6_log10_conc": x95, "anchor6_train_fraction": float((x_train < x95).mean()),
            "z_at_anchor0": z_a, "z_at_anchor6": z_b, "z_bandwidth": bandwidth,
            "a": a, "b": b,
            "carrier_vs_sensory_spearman": rho,
            "monotonicity_flag": ("weak_or_reverse" if rho < 0.2 else "ok"),
            "rule1_route": rule1_route,
            "has_sensory_threshold": bool(has_threshold),
            "frac_within_0_6_train": within,
            "frac_clipped_to_0_train": clip0,
            "frac_clipped_to_6_train": clip6,
            "valid": len(reasons) == 0,
            "invalid_reason": "；".join(reasons) if reasons else "",
            "n_train": int(len(train_ids)),
            "notes": spec["notes"],
        })
    return pd.DataFrame(rows)


def apply_absolute(params: pd.DataFrame, z: pd.DataFrame, valid_only: bool = True) -> pd.DataFrame:
    """把标准化感官值经锚定映射转成绝对 0–6 分（裁剪到 [0,6]）。

    返回 250×50 的 DataFrame；不合格锚定（valid=False）与不具备锚定的维度保持 NaN。
    """
    out = pd.DataFrame(np.nan, index=z.index, columns=z.columns, dtype=float)
    for _, row in params.iterrows():
        dim = row["dimension"]
        if dim not in z.columns or not np.isfinite(row["a"]) or not np.isfinite(row["b"]):
            continue
        if valid_only and "valid" in params.columns and not bool(row["valid"]):
            continue
        raw = row["a"] + row["b"] * z[dim].to_numpy(dtype=float)
        out[dim] = np.clip(raw, 0.0, 6.0)
    return out


def anchor_coverage(params: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """统计每个维度锚定后的 0–6 覆盖比例与裁剪情况（在交付所用的 z 上评估）。"""
    rows = []
    for _, row in params.iterrows():
        dim = row["dimension"]
        if dim not in z.columns or not np.isfinite(row["a"]):
            continue
        if "valid" in params.columns and not bool(row["valid"]):
            continue
        raw = row["a"] + row["b"] * z[dim].to_numpy(dtype=float)
        rows.append({
            "dimension": dim,
            "frac_within_0_6_oof": float(np.mean((raw >= 0) & (raw <= 6))),
            "frac_clipped_to_0_oof": float(np.mean(raw < 0)),
            "frac_clipped_to_6_oof": float(np.mean(raw > 6)),
            "mean_abs_score_oof": float(np.clip(raw, 0, 6).mean()),
        })
    return pd.DataFrame(rows)


__all__ = ["AnchorSpec", "build_anchor_specs", "fit_anchor_params", "apply_absolute",
           "anchor_coverage", "UNIT_TO_MGL", "VALIDITY_RULES"]


def main() -> int:
    """打印锚定参数总表（供论文附录核对）。"""
    import argparse

    from data_contract import load_split

    ap = argparse.ArgumentParser(description="问题一 · 绝对锚定参数总表")
    ap.add_argument("--output", default="", help="可选：把参数表另存为 CSV")
    args = ap.parse_args()

    split = load_split()
    train_ids = split.loc[split["split"] == "train", "beer"].to_numpy(dtype=int)
    x = load_modeling_matrix()
    y = load_targets()
    params = fit_anchor_params(x, y, train_ids, build_anchor_specs())
    columns = ["dimension", "carrier", "anchor_type", "rule1_route", "threshold_value",
               "threshold_unit", "log10_threshold", "anchor0_train_fraction", "z_bandwidth",
               "a", "b", "carrier_vs_sensory_spearman", "frac_within_0_6_train",
               "valid", "invalid_reason"]
    print(params[[c for c in columns if c in params.columns]].to_string(index=False))
    print(f"\n通过有效性护栏：{int(params['valid'].sum())}/{len(params)} 维 → "
          f"{params.loc[params['valid'].astype(bool), 'dimension'].tolist()}")
    if args.output:
        Path(args.output).write_text(params.to_csv(index=False), encoding="utf-8")
        print(f"已写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
