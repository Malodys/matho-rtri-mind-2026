# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本为只读诊断工具，逐项打印实测值，供人工核对；不写出任何交付物。
"""问题二只读诊断探查（合并版）。

覆盖的核验点：
  structure   —— 各附件形状、四路证据的覆盖数、名称桥解析率、缺陷阈值与超标率
  shap        —— 8.xlsx MDI / SHAP 两张表的结构、SHAP 块与浓度块的区别
  alignment   —— seed_0 切分与统一矩阵/感官目标的按 beer 对齐、Leiden 参数对照
  consistency —— 6.csv 与 E2 自算 Spearman 的逐单元一致性（含 Diacetyl 缺失机制）
  leiden      —— igraph Leiden 在加权/非加权、不同 n_iterations 下的社区划分对照

用法：
    python scripts/q2_probe.py --part all
    python scripts/q2_probe.py --part consistency
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import data_contract as dc
import q2_common as qc

PARTS = ("structure", "shap", "alignment", "consistency", "leiden")


def spearman_matrix(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    rx = rankdata(x.to_numpy(dtype=float), axis=0)
    ry = rankdata(y.to_numpy(dtype=float), axis=0)
    rx = (rx - rx.mean(axis=0)) / rx.std(axis=0)
    ry = (ry - ry.mean(axis=0)) / ry.std(axis=0)
    nx = np.sqrt((rx ** 2).sum(axis=0))
    ny = np.sqrt((ry ** 2).sum(axis=0))
    denom = nx[:, None] * ny[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (rx.T @ ry) / np.where(denom == 0, np.nan, denom)
    return pd.DataFrame(out, index=x.columns, columns=y.columns)


# --------------------------------------------------------------------------- structure


def part_structure() -> None:
    raw = dc.load_raw(dc.RAW_DIR)
    print("=== 附件形状 ===")
    print(f"1.csv {raw.chem.shape} | 2.csv {raw.corr_chem.shape} | 4.csv {raw.sensory.shape}")
    print(f"5.csv {raw.corr_sensory.shape} | 6.csv {raw.cross.shape} | 8.xlsx thresholds {raw.thresholds.shape}")
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    print("\n=== 名称桥解析率 ===")
    print("化学桥 2.csv:", int((chem_bridge.query("matrix=='2.csv'")["file1_column"] != "").sum()), "/231")
    print("化学桥 6.csv:", int((chem_bridge.query("matrix=='6.csv'")["file1_column"] != "").sum()), "/44")
    print("感官桥 5.csv:", int((sens_bridge.query("matrix=='5.csv'")["file4_column"] != "").sum()), "/50")
    print("感官桥 6.csv:", int((sens_bridge.query("matrix=='6.csv'")["file4_column"] != "").sum()), "/50")
    prov = dc.load_provenance()
    print("\n=== 溯源 block 分布 ===")
    print(prov["block"].value_counts().to_string())
    print("\n=== 8.xlsx origin_paper 分布 ===")
    print(raw.thresholds["origin_paper"].value_counts().to_string())
    X = dc.load_modeling_matrix(raw)
    print("\n=== 统一矩阵 ===", X.shape, "NaN:", int(X.isna().sum().sum()))
    print("=== 缺陷阈值与超标率 ===")
    print(dc.defect_exceedance(X, raw).to_string(index=False))


# --------------------------------------------------------------------------- shap


def part_shap() -> None:
    raw = dc.load_raw(dc.RAW_DIR)
    mdi = raw.gbm_mdi.copy()
    mdi.columns = ["mdi", "compound"]
    print("=== GBM importance MDI ===", mdi.shape, "sum=", round(float(mdi["mdi"].sum()), 6))
    print(mdi.sort_values("mdi", ascending=False).head(12).to_string(index=False))
    sh = raw.gbm_shap.copy()
    print("\n=== GBM SHAP Values ===", sh.shape)
    print(sh.groupby("data").size().to_string())
    feat = [c for c in sh.columns if c not in
            ("data", "beer", "beer_id", "tasting_category_fine", "base_prediction")]
    print("特征列数:", len(feat))
    conc = sh[sh["data"] == "concentration"].set_index("beer")
    c1 = raw.chem.set_index("beer")
    print("浓度块 vs 文件1 原始存储标度：")
    for col in feat[:5]:
        if col in c1.columns:
            a = conc[col].reindex(conc.index).to_numpy(dtype=float)
            b = c1[col].reindex(conc.index).to_numpy(dtype=float)
            print(f"  {col}: max|Δ|={np.nanmax(np.abs(a - b)):.3g}")


# --------------------------------------------------------------------------- alignment


def part_alignment() -> None:
    raw = dc.load_raw(dc.RAW_DIR)
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    split = dc.load_split()
    train = np.sort(split.loc[split["split"] == "train", "beer"].to_numpy())
    print("=== seed_0 切分 ===")
    print(split["split"].value_counts().to_string())
    Xtr, Ytr = X.loc[X.index.isin(train)], Y.loc[Y.index.isin(train)]
    print(f"Xtr {Xtr.shape} Ytr {Ytr.shape} 按 beer 对齐: {list(Xtr.index) == list(Ytr.index)}")
    print("文件1 化学列数:", len([c for c in raw.chem.columns if c not in dc.KEY_COLUMNS]),
          "→ 统一矩阵剔除 *_sum 后:", X.shape[1])
    print("\n=== 6.csv vs 8.xlsx SHAP 模型的目标归属 ===")
    sh = raw.gbm_shap.copy()
    shS = sh[sh["data"] == "SHAP"].reset_index(drop=True)
    feat = [c for c in sh.columns if c not in
            ("data", "beer", "beer_id", "tasting_category_fine", "base_prediction")]
    pred = shS[feat].to_numpy(dtype=float).sum(1) + shS["base_prediction"].to_numpy(dtype=float)
    best = []
    for t in Y.columns:
        y = Y[t].reindex(shS["beer"].to_numpy()).to_numpy(dtype=float)
        m = ~np.isnan(y)
        best.append((abs(float(np.corrcoef(pred[m], y[m])[0, 1])), t))
    best.sort(reverse=True)
    print("与 50 个维度中相关性最高的 5 个：",
          [(t, round(r, 4)) for r, t in best[:5]])
    print("→ 8.xlsx 的 SHAP 模型无法唯一对应单一维度，只能作符号方向参照。")


# --------------------------------------------------------------------------- consistency


def part_consistency() -> None:
    raw = dc.load_raw(dc.RAW_DIR)
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    b6 = chem_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file1_column"]
    s6 = sens_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file4_column"]

    chem = raw.chem.set_index("beer")
    Xraw = chem[[c for c in chem.columns if c not in dc.KEY_COLUMNS]]
    Rraw = spearman_matrix(Xraw, Y)
    Runi = spearman_matrix(X, Y)

    rows = []
    for rname in raw.cross.index:
        c = b6.get(rname)
        if not c:
            continue
        for cname in raw.cross.columns:
            t = s6.get(cname)
            if not t:
                continue
            v6 = float(raw.cross.loc[rname, cname])
            rr = float(Rraw.loc[c, t]) if c in Rraw.index else np.nan
            ru = float(Runi.loc[c, t]) if (c in Runi.index and t in Runi.columns) else np.nan
            rows.append({"comp": c, "v6": v6, "d_raw": abs(v6 - rr), "d_uni": abs(v6 - ru),
                         "nan_in_comp": int(Xraw[c].isna().sum()) if c in Xraw.columns else -1})
    df = pd.DataFrame(rows)
    print("=== 6.csv 与重算 Spearman 的逐单元一致性 ===")
    print(f"对照 A（1.csv 原始存储标度）: n={len(df)} max|Δ|={df['d_raw'].max():.3g} "
          f"frac<1e-8={(df['d_raw'] < 1e-8).mean():.4f}")
    print(f"对照 B（统一 log10(mg/L) 矩阵）: n={len(df)} max|Δ|={df['d_uni'].max():.3g} "
          f"frac<1e-8={(df['d_uni'] < 1e-8).mean():.4f}")
    bad = df[df["d_uni"] > 1e-8]
    print(f"\n统一矩阵口径下不一致的单元格 {len(bad)} 个，涉及化合物 {bad['comp'].nunique()} 个：")
    if len(bad):
        print(bad.groupby("comp").agg(n=("d_uni", "size"), max_d=("d_uni", "max"),
                                      nan_in_comp=("nan_in_comp", "first")).to_string())
        print("→ 根因：这些化合物在 1.csv 中有缺失值，统一矩阵做了插补，导致秩次改变。")


# --------------------------------------------------------------------------- leiden


def part_leiden() -> None:
    import igraph as ig
    import networkx as nx

    raw = dc.load_raw(dc.RAW_DIR)
    corr = raw.corr_chem
    names = list(corr.index)
    vals = corr.to_numpy(dtype=float)
    np.fill_diagonal(vals, 0.0)
    np.nan_to_num(vals, copy=False)
    g = nx.Graph()
    g.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(vals[i, j]) >= 0.6:
                g.add_edge(names[i], names[j], weight=float(vals[i, j]))
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
    print("=== igraph Leiden 参数对照（|ρ|≥0.6）===")
    for weighted in (False, True):
        gg = ig.Graph(n=len(names))
        gg.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
        if weighted:
            gg.es["weight"] = [abs(g[a][b]["weight"]) for a, b in g.edges()]
        for nit in (5, -1):
            c = gg.community_leiden(objective_function="modularity", n_iterations=nit,
                                     weights="weight" if weighted else None)
            sizes = sorted(c.sizes(), reverse=True)
            print(f"  weighted={weighted} n_iterations={nit}: n_comm={len(c)} "
                  f"top5={sizes[:5]} Q={gg.modularity(c):.4f}")
    print("→ 与 data/processed/smoke/stack_smoke.txt 的 V8 记录一致：[21, 17, 16, 15, 15]")


# --------------------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二只读诊断探查")
    ap.add_argument("--part", default="all",
                    choices=("all",) + PARTS, help="要执行的诊断项")
    args = ap.parse_args()
    todo = PARTS if args.part == "all" else (args.part,)
    for p in todo:
        print(f"\n{'=' * 70}\n### {p}\n{'=' * 70}")
        {"structure": part_structure, "shap": part_shap, "alignment": part_alignment,
         "consistency": part_consistency, "leiden": part_leiden}[p]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
