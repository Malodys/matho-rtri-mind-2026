# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) E1=6.csv（44 化合物）、E2=自算全量 Spearman（227 化合物）、E3=GBDT+SHAP（227）、
#      E4=8.xlsx MDI（231）四路证据的覆盖数已逐一实测打印。
#   2) E2 与 E1 的口径一致性：在 1.csv 原始存储标度上重算 6.csv，2200 个单元中
#      2199 个 max|Δ|=8.88e-16；唯一例外 Diacetyl（1.csv 有 9 个缺失值，统一矩阵做了插补，
#      使其 ρ 偏移 ≤0.047）。E2 使用统一 log10(mg/L) 矩阵，故对 Diacetyl 存在可解释偏差。
#   3) 融合规则：E2、E3 各自按维度做秩归一化后加权求和（0.5/0.5，理由见 q2_common.FUSION_WEIGHTS）；
#      E1 仅作交叉验证，E4 仅作外部印证，均不进入主排序。
"""问题二 Step 2：四路证据 → 每个感官维度的 Top-3 关键化合物。

用法：
    python scripts/q2_evidence.py [--repo-root <仓库根>]
输出：
    data/processed/q2_runs/evidence/*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import q2_common as qc


# --------------------------------------------------------------------------- 证据路


def evidence_e1(data: qc.Q2Data) -> pd.DataFrame:
    """E1：6.csv 给出的交叉相关（Spearman），行=1.csv 列，列=4.csv 目标。"""
    out = pd.DataFrame(np.nan, index=list(data.X.columns), columns=list(data.Y.columns))
    for rname, rvals in data.cross.iterrows():
        col = data.cross_row_to_col.get(rname)
        if not col or col not in out.index:
            continue
        for cname, val in rvals.items():
            target = data.cross_col_to_target.get(cname)
            if target and target in out.columns:
                out.loc[col, target] = float(val)
    return out


def evidence_e2(data: qc.Q2Data) -> pd.DataFrame:
    """E2：自算全量 Spearman（227 化合物 × 50 维度），输入为统一 log10(mg/L) 矩阵。"""
    return qc.spearman_matrix(data.X, data.Y)


def evidence_e3(data: qc.Q2Data) -> pd.DataFrame:
    """E3：GBDT + SHAP 归因（LightGBM，逐维度拟合，mean|SHAP| 作为重要度）。"""
    import lightgbm as lgb

    Xtr, Ytr = data.Xtr, data.Ytr
    out = pd.DataFrame(0.0, index=list(Xtr.columns), columns=list(Ytr.columns))
    for target in Ytr.columns:
        model = lgb.LGBMRegressor(**qc.LGB_PARAMS)
        model.fit(Xtr, Ytr[target].to_numpy(dtype=float))
        contrib = model.predict(Xtr, pred_contrib=True)[:, :-1]
        out[target] = np.abs(contrib).mean(axis=0)
    return out


def evidence_e4(data: qc.Q2Data) -> pd.DataFrame:
    """E4：8.xlsx「GBM importance MDI」——原论文模型的全局重要性先验（无符号）。"""
    m = data.mdi.copy()
    m = m.set_index("col")["mdi"]
    out = pd.DataFrame({"mdi": m})
    out["mdi_rank"] = out["mdi"].rank(ascending=False, method="average")
    out["mdi_percentile"] = 1.0 - (out["mdi_rank"] - 1) / max(len(out) - 1, 1)
    return out


# --------------------------------------------------------------------------- 融合


def build_score_table(data: qc.Q2Data, e1: pd.DataFrame, e2: pd.DataFrame,
                      e3: pd.DataFrame, e4: pd.DataFrame) -> pd.DataFrame:
    """把四路证据拼成一张 (维度 × 化合物) 长表，并给出主排序（E2+E3 融合）。"""
    n_e2 = qc.rank_normalize(e2.abs())
    n_e3 = qc.rank_normalize(e3)
    w2, w3 = qc.FUSION_WEIGHTS["e2"], qc.FUSION_WEIGHTS["e3"]

    # E1 只在 44 个化合物上有值，秩在 44 个内部计算
    e1_abs = e1.abs()
    e1_rank = e1_abs.rank(axis=0, method="average", ascending=False)

    e4_rank = e4["mdi_rank"]
    e4_pct = e4["mdi_percentile"]

    rows = []
    for target in data.Y.columns:
        fused = w2 * n_e2[target] + w3 * n_e3[target]
        fused_rank = fused.rank(ascending=False, method="min")
        for col in data.X.columns:
            rho = float(e2.loc[col, target]) if col in e2.index else np.nan
            e1v = float(e1.loc[col, target]) if col in e1.index and pd.notna(e1.loc[col, target]) else np.nan
            rows.append({
                "sensory_dimension": target,
                "compound": col,
                "e1_rho": e1v,
                "e1_rank_in_44": (float(e1_rank.loc[col, target])
                                  if col in e1.index and pd.notna(e1.loc[col, target]) else np.nan),
                "e2_rho": rho,
                "e2_abs_rho": abs(rho) if pd.notna(rho) else np.nan,
                "e2_rank_in_227": float(e2.abs()[target].rank(ascending=False, method="min")[col]),
                "e3_shap_mean_abs": float(e3.loc[col, target]),
                "e3_rank_in_227": float(e3[target].rank(ascending=False, method="min")[col]),
                "e4_mdi": float(e4.loc[col, "mdi"]) if col in e4.index else np.nan,
                "e4_mdi_rank": float(e4_rank[col]) if col in e4.index else np.nan,
                "e4_mdi_percentile": float(e4_pct[col]) if col in e4.index else np.nan,
                "fused_score": float(fused[col]),
                "fused_rank": float(fused_rank[col]),
                "direction": ("正" if (pd.notna(rho) and rho > 0) else
                              ("负" if (pd.notna(rho) and rho < 0) else "无方向")),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 主流程


def run(root: Path) -> dict:
    qc.ensure_dirs()
    data = qc.load_q2_data(root)
    ev = qc.EVIDENCE_DIR

    e1 = evidence_e1(data)
    e2 = evidence_e2(data)
    e3 = evidence_e3(data)
    e4 = evidence_e4(data)

    e1.to_csv(ev / "e1_cross_rho_6csv.csv", encoding="utf-8")
    e2.to_csv(ev / "e2_spearman_full.csv", encoding="utf-8")
    e3.to_csv(ev / "e3_shap_importance.csv", encoding="utf-8")
    e4.to_csv(ev / "e4_gbm_mdi.csv", encoding="utf-8")

    scores = build_score_table(data, e1, e2, e3, e4)
    scores.to_csv(ev / "evidence_scores.csv", index=False, encoding="utf-8")

    # 供 stability 脚本复用（避免重复拟合 50 个模型）
    np.save(ev / "e3_shap_matrix.npy", e3.to_numpy(dtype=float))
    (ev / "e3_features.txt").write_text("\n".join(e3.index), encoding="utf-8")
    (ev / "e3_targets.txt").write_text("\n".join(e3.columns), encoding="utf-8")

    coverage = {
        "E1_6csv": {"n_compounds": int(e1.notna().any(axis=1).sum()), "n_targets": int(e1.notna().any(axis=0).sum())},
        "E2_spearman_full": {"n_compounds": int(e2.shape[0]), "n_targets": int(e2.shape[1]),
                             "n_finite": int(np.isfinite(e2.to_numpy(dtype=float)).sum())},
        "E3_shap": {"n_compounds": int(e3.shape[0]), "n_targets": int(e3.shape[1])},
        "E4_mdi": {"n_compounds": int(len(e4))},
    }
    qc.write_json(ev / "evidence_coverage.json", coverage)

    print(f"[q2_evidence] E1={coverage['E1_6csv']} E2={coverage['E2_spearman_full']} "
          f"E3={coverage['E3_shap']} E4={coverage['E4_mdi']}")
    for t in ("bitternes", "alcohol", "A_hops_citrus", "A_malt_bread"):
        top = scores[(scores["sensory_dimension"] == t)].nsmallest(3, "fused_rank")
        print(f"  {t}: " + ", ".join(
            f"{r['compound']}(ρ={r['e2_rho']:+.3f}, {r['direction']})" for _, r in top.iterrows()))
    return coverage


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二 Step 2：四路证据与 Top-3 主排序")
    ap.add_argument("--repo-root", default=str(qc.REPO_ROOT))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(Path(args.repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
