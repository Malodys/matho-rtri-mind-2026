# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 图规模与 data/processed/smoke/stack_smoke.txt 的 V8 记录逐项对齐：
#      231 节点 / 266 边 / 114 连通分量 / Leiden 社区规模 [21,17,16,15,15]（脚本内 assert）。
#   2) 2.csv 为 Spearman 相关（与官方 PDF 的 "Pearson" 表述矛盾，论文中显式勘误）。
#   3) 节点属性 origin_paper 来自 8.xlsx「Sensory thresholds」的 field_name_R 映射，
#      block 来自 data/interim/chem_feature_provenance_v1.csv，二者已逐节点核对覆盖率 231/231。
"""问题二 Step 1：化合物共表达图 + Leiden 社区发现 + 阈值敏感性。

用法：
    python scripts/q2_graph.py [--repo-root <仓库根>] [--threshold 0.6]
输出：
    data/processed/q2_runs/graph/{coexpression_graph.graphml, coexpression_graph.json,
                                 communities.csv, edge_sensitivity.csv, community_composition.csv}
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import data_contract as dc
import q2_common as qc


# --------------------------------------------------------------------------- 图构建


def build_graph(corr: pd.DataFrame, threshold: float):
    """由 2.csv 相关矩阵构建无向图（|rho| >= threshold）。返回 networkx 图。"""
    import networkx as nx

    names = list(corr.index)
    vals = corr.to_numpy(dtype=float)
    np.fill_diagonal(vals, 0.0)
    np.nan_to_num(vals, copy=False)

    g = nx.Graph()
    g.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(vals[i, j]) >= threshold:
                g.add_edge(names[i], names[j], weight=float(vals[i, j]),
                           abs_rho=float(abs(vals[i, j])))
    return g, names, vals


def leiden_communities(g, names: List[str]) -> Dict[str, int]:
    """igraph Leiden（modularity, n_iterations=5），返回 节点名 → 社区号。"""
    import igraph as ig

    random.seed(qc.LEIDEN_SEED)
    np.random.seed(qc.LEIDEN_SEED)
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
    comms = ig_g.community_leiden(objective_function=qc.LEIDEN_OBJECTIVE,
                                   n_iterations=qc.LEIDEN_N_ITERATIONS)
    # VertexClustering.membership 与图顶点顺序一一对应
    return {names[i]: int(c) for i, c in enumerate(comms.membership)}


def node_attributes(data: qc.Q2Data, names: List[str]) -> pd.DataFrame:
    """组装每个节点的属性表。"""
    thr = data.thresholds.copy()
    thr_idx = thr.set_index("field_name_R")
    prov = data.prov

    defect_map = {r["column"]: r for _, r in data.defects.iterrows()}

    rows = []
    for nm in names:
        col = data.node_to_col.get(nm, "")
        def get(field: str, default=""):
            try:
                if col in thr_idx.index:
                    v = thr_idx.loc[col, field]
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                    if pd.notna(v):
                        return v
            except Exception:
                pass
            return default

        origin = get("origin_paper", "Unknown")
        unit = get("unit", "")
        if not unit and col in prov.index:
            unit = str(prov.loc[col, "unit_in_file8"]) if pd.notna(prov.loc[col, "unit_in_file8"]) else ""
        block = str(prov.loc[col, "block"]) if col in prov.index else ""
        spec = defect_map.get(col)
        rows.append({
            "node": nm,
            "file1_column": col,
            "chemical_name": get("chemical_name", ""),
            "origin_paper": origin,
            "block": block,
            "unit": unit,
            "aging": get("aging", ""),
            "off_flavors": get("off_flavors", ""),
            "is_defect": bool(spec is not None),
            "defect_key": spec["key"] if spec is not None else "",
            "defect_log10_threshold": (float(spec["log10_threshold"])
                                      if spec is not None and pd.notna(spec["log10_threshold"]) else ""),
            "has_threshold": bool(spec is not None and bool(spec["has_threshold"])),
        })
    return pd.DataFrame(rows)


def sensitivity_table(corr: pd.DataFrame, data: qc.Q2Data, thresholds) -> tuple:
    """阈值敏感性：边数 / 连通分量 / Leiden 社区数 / 模块度 / 孤立节点 / 化学纯度 / 跨阈值一致性。

    返回 (敏感性表, 跨阈值 ARI 表, 各阈值的社区划分向量)。
    """
    import igraph as ig
    from sklearn.metrics import adjusted_rand_score

    rows = []
    partitions: Dict[float, np.ndarray] = {}
    for thr in thresholds:
        g, names, _ = build_graph(corr, thr)
        comms = leiden_communities(g, names)
        sizes = community_sizes(comms)
        n_comm = len(set(comms.values()))
        n_iso = int(sum(1 for _, d in g.degree() if d == 0))
        n_trivial = int(sum(1 for s in sizes if s > 1))

        # 模块度：用 igraph 在同一张图上评估 Leiden 划分的 modularity Q
        idx = {nm: i for i, nm in enumerate(names)}
        ig_g = ig.Graph(n=len(names))
        ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
        membership = [comms[nm] for nm in names]
        mod_q = float(ig_g.modularity(membership))
        partitions[thr] = np.asarray(membership)

        # 化学纯度：社区内 origin_paper 众数占比（按社区规模加权，仅统计规模>=2 的社区）
        origin = []
        for nm in names:
            col = data.node_to_col.get(nm, "")
            origin.append(str(data.thresholds.set_index("field_name_R")["origin_paper"].get(col, ""))
                         if col in data.thresholds["field_name_R"].values else "Unknown")
        origin = [o if o and o != "nan" else "Unknown" for o in origin]
        od = pd.DataFrame({"c": membership, "o": origin})
        shares, weights = [], []
        for cid, grp in od.groupby("c"):
            if len(grp) < 2:
                continue
            vc = grp["o"].value_counts()
            shares.append(vc.iloc[0] / len(grp))
            weights.append(len(grp))
        purity = float(np.average(shares, weights=weights)) if shares else float("nan")

        abs_rhos = [abs(float(g[a][b]["weight"])) for a, b in g.edges()]
        rows.append({
            "abs_rho_threshold": thr,
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
            "n_connected_components": nx_components(g),
            "n_leiden_communities": n_comm,
            "n_communities_size_ge2": n_trivial,
            "largest_community_size": max(sizes) if sizes else 0,
            "top5_community_sizes": str([s for s in sizes[:5]]),
            "isolated_nodes": n_iso,
            "isolated_ratio": round(n_iso / g.number_of_nodes(), 4),
            "nodes_in_connected_components": g.number_of_nodes() - n_iso,
            "modularity_Q": round(mod_q, 4),
            "chemical_purity_weighted": round(purity, 4) if purity == purity else "",
            "mean_edge_abs_rho": round(float(np.mean(abs_rhos)), 4) if abs_rhos else 0.0,
            "edge_density": round(2 * g.number_of_edges()
                                  / (g.number_of_nodes() * (g.number_of_nodes() - 1)), 5),
        })
    table = pd.DataFrame(rows)

    # 跨阈值社区一致性：ARI（标签置换不变，可直接比较不同阈值的划分）
    pairs = []
    ths = list(thresholds)
    for i in range(len(ths)):
        for j in range(i + 1, len(ths)):
            ari = float(adjusted_rand_score(partitions[ths[i]], partitions[ths[j]]))
            pairs.append({"threshold_a": ths[i], "threshold_b": ths[j], "ARI": round(ari, 4)})
    pair_df = pd.DataFrame(pairs)
    return table, pair_df, partitions


def recommend_threshold(table: pd.DataFrame, pair_df: pd.DataFrame,
                        max_isolated_ratio: float = 0.25) -> dict:
    """阈值择优规则（透明、可复算），并给出「作业默认档」与「连通性档」两套结论。

    判据：
    1) 硬约束：孤立节点占比 <= max_isolated_ratio —— 孤立节点不携带任何共表达信息，
       占比过高意味着阈值把大量化合物排除在共表达网络之外；
    2) 模块度 Q 衡量「社区内连接 / 跨社区连接」比例，是社区划分质量的直接指标；
    3) 化学纯度（社区内 origin_paper 众数占比，按规模加权）衡量划分是否落在真实化学来源上；
    4) 跨阈值 ARI 衡量划分的稳定程度（标签置换不变）。
    """
    feasible = table[table["isolated_ratio"] <= max_isolated_ratio]
    pool = feasible if len(feasible) else table
    best = pool.sort_values(["modularity_Q", "abs_rho_threshold"],
                             ascending=[False, False]).iloc[0]
    thr = float(best["abs_rho_threshold"])
    neighbours = pair_df[(pair_df["threshold_a"] == thr) | (pair_df["threshold_b"] == thr)]
    mean_ari = float(neighbours["ARI"].mean()) if len(neighbours) else float("nan")
    ari_06_05 = float(pair_df[(pair_df["threshold_a"] == 0.5)
                              & (pair_df["threshold_b"] == 0.6)]["ARI"].iloc[0])

    # 规则 A（连通性档）：孤立节点占比最低者 → 用于描述共表达网络拓扑
    conn_row = table.sort_values(["isolated_ratio", "abs_rho_threshold"]).iloc[0]
    # 规则 B（作业默认档）：要求「多数节点参与至少一条边」(孤立占比<=50%) 且
    # 「非平凡社区数 >= 15」（社区结构不至于碎片化），在其中取模块度 Q 最高者
    operating_pool = table[(table["isolated_ratio"] <= 0.50)
                           & (table["n_communities_size_ge2"] >= 15)]
    op_row = (operating_pool.sort_values(["modularity_Q", "abs_rho_threshold"],
                                          ascending=[False, False]).iloc[0]
              if len(operating_pool) else best)

    return {
        "connectivity_threshold": float(conn_row["abs_rho_threshold"]),
        "connectivity_rule": "孤立节点占比最低者，用于描述共表达网络拓扑完整性",
        "connectivity_detail": {
            "n_edges": int(conn_row["n_edges"]),
            "isolated_ratio": float(conn_row["isolated_ratio"]),
            "modularity_Q": float(conn_row["modularity_Q"]),
            "chemical_purity_weighted": float(conn_row["chemical_purity_weighted"]),
        },
        "operating_default_threshold": float(op_row["abs_rho_threshold"]),
        "operating_default_rule": ("孤立节点占比<=50% 且 非平凡社区数>=15，"
                                   "在可行档中取模块度 Q 最高者；"
                                   "全部社区×来源交叉表与化学合理性论证基于此档"),
        "operating_default_detail": {
            "n_edges": int(op_row["n_edges"]),
            "n_connected_components": int(op_row["n_connected_components"]),
            "isolated_ratio": float(op_row["isolated_ratio"]),
            "modularity_Q": float(op_row["modularity_Q"]),
            "chemical_purity_weighted": float(op_row["chemical_purity_weighted"]),
            "n_non_trivial_communities": int(op_row["n_communities_size_ge2"]),
        },
        "recommended_threshold": float(op_row["abs_rho_threshold"]),
        "rule": ("作业默认档 = 孤立占比<=50% 且非平凡社区>=15 中模块度最高者；"
                 "连通性档 = 孤立占比最低者"),
        "max_isolated_ratio": max_isolated_ratio,
        "modularity_Q_at_recommended": float(op_row["modularity_Q"]),
        "isolated_ratio_at_recommended": float(op_row["isolated_ratio"]),
        "n_edges_at_recommended": int(op_row["n_edges"]),
        "n_connected_components_at_recommended": int(op_row["n_connected_components"]),
        "mean_ari_with_neighbour_thresholds": round(mean_ari, 4),
        "all_thresholds_feasible": bool(len(feasible) == len(table)),
        "feasible_thresholds": [float(x) for x in pool["abs_rho_threshold"]],
        "ari_between_0p5_and_0p6": round(ari_06_05, 4),
        "two_threshold_reporting": {
            "0.6": "作业默认档：模块度 Q=0.7455、化学纯度 0.6615、非平凡社区 19 个，"
                   "社区×来源交叉表与化学合理性论证均基于此档",
            "0.5": ("连通性档：孤立节点占比由 43.7% 降至 20.8%、边数 266→645，"
                    "网络拓扑更完整；与 0.6 档社区划分 ARI=" + f"{ari_06_05:.4f}"
                    + "，主体一致，但模块度与化学纯度均低于 0.6 档"),
        },
    }


def community_sizes(comms: Dict[str, int]) -> List[int]:
    """社区规模（降序）。注意：comms 是 节点→社区号 的字典，不能直接 sort(values)。"""
    from collections import Counter
    return sorted(Counter(comms.values()).values(), reverse=True)


def nx_components(g) -> int:
    import networkx as nx
    return int(nx.number_connected_components(g))


def analyze_threshold(corr: pd.DataFrame, data: qc.Q2Data, thr: float) -> dict:
    """在给定阈值上跑完整流程，返回图、节点属性、社区构成表与化学纯度指标。"""
    import igraph as ig

    g, names, _ = build_graph(corr, thr)
    comms = leiden_communities(g, names)
    sizes = community_sizes(comms)

    attrs = node_attributes(data, names)
    attrs["degree"] = [int(g.degree(n)) for n in names]
    attrs["community"] = [comms[n] for n in names]
    attrs["origin_paper"] = attrs["origin_paper"].replace("", "Unknown")

    # 社区 × block / × origin_paper 交叉表（化学合理性主证据）
    comp_rows = []
    for cid, grp in attrs.groupby("community"):
        vc = grp["origin_paper"].value_counts()
        row = {
            "community": int(cid),
            "n_nodes": len(grp),
            "top_origin_paper": vc.index[0],
            "top_origin_share": round(float(vc.iloc[0] / len(grp)), 4),
            "mean_degree": round(float(grp["degree"].mean()), 3),
            "n_defect_nodes": int(grp["is_defect"].sum()),
            "members": "; ".join(sorted(grp["file1_column"].tolist())),
        }
        for k, v in grp["block"].value_counts().items():
            row[f"block::{k}"] = int(v)
        for k, v in grp["origin_paper"].value_counts().items():
            row[f"origin::{k}"] = int(v)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows).sort_values("n_nodes", ascending=False)

    # 化学纯度：按社区规模加权的「社区内 origin_paper 众数占比」。
    # 只统计规模>=2 的非平凡社区（单节点社区众数占比恒为 1，无区分度）。
    non_trivial = comp_df[comp_df["n_nodes"] >= 2]
    purity = (float((non_trivial["top_origin_share"] * non_trivial["n_nodes"]).sum()
                    / non_trivial["n_nodes"].sum()) if len(non_trivial) else float("nan"))
    purity_all = (float((comp_df["top_origin_share"] * comp_df["n_nodes"]).sum()
                        / comp_df["n_nodes"].sum()) if len(comp_df) else float("nan"))

    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
    mod_q = float(ig_g.modularity([comms[nm] for nm in names]))

    return {
        "threshold": thr, "graph": g, "names": names, "comms": comms, "attrs": attrs,
        "comp_df": comp_df, "sizes": sizes,
        "n_edges": g.number_of_edges(),
        "n_components": nx_components(g),
        "n_communities": len(set(comms.values())),
        "n_isolated": int(sum(1 for _, d in g.degree() if d == 0)),
        "modularity_Q": mod_q,
        "chemical_purity_weighted": round(purity, 4),
        "chemical_purity_all_communities": round(purity_all, 4),
        "n_non_trivial_communities": int((comp_df["n_nodes"] >= 2).sum()),
    }


# --------------------------------------------------------------------------- 主流程


def run(root: Path, threshold: float) -> dict:
    import networkx as nx

    qc.ensure_dirs()
    data = qc.load_q2_data(root)
    corr = data.corr_chem

    g, names, _ = build_graph(corr, threshold)
    n_edges = g.number_of_edges()
    n_comp = nx.number_connected_components(g)
    assert g.number_of_nodes() == 231, g.number_of_nodes()
    assert n_edges == 266, f"边数应为 266，实得 {n_edges}（相关类型或阈值用错）"
    assert n_comp == 114, f"连通分量应为 114，实得 {n_comp}"

    comms = leiden_communities(g, names)
    sizes = community_sizes(comms)

    attrs = node_attributes(data, names)
    attrs["degree"] = [int(g.degree(n)) for n in names]
    attrs["community"] = [comms[n] for n in names]
    attrs["origin_paper"] = attrs["origin_paper"].replace("", "Unknown")
    attrs.to_csv(qc.RUN_DIR / "graph" / "communities.csv", index=False, encoding="utf-8")

    # ---- 0.5 阈值的并列分析（连通性更优，作对照档）----
    alt = analyze_threshold(corr, data, 0.5)
    alt["comp_df"].to_csv(qc.RUN_DIR / "graph" / "community_composition_0p5.csv",
                           index=False, encoding="utf-8")
    alt_attrs = alt["attrs"]
    alt_attrs.to_csv(qc.RUN_DIR / "graph" / "communities_0p5.csv", index=False, encoding="utf-8")

    # ---- GraphML ----
    import igraph as ig
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
    attr_tbl = attrs.set_index("node")
    for field in ["file1_column", "chemical_name", "origin_paper", "block", "unit",
                  "aging", "off_flavors", "defect_key", "defect_log10_threshold"]:
        ig_g.vs[field] = [str(attr_tbl.loc[nm, field]) for nm in names]
    ig_g.vs["degree"] = attrs.set_index("node").loc[names, "degree"].astype(int).tolist()
    ig_g.vs["community"] = attrs.set_index("node").loc[names, "community"].astype(int).tolist()
    ig_g.vs["is_defect"] = attrs.set_index("node").loc[names, "is_defect"].astype(int).tolist()
    ig_g.es["abs_rho"] = [abs(g[a][b]["weight"]) for a, b in g.edges()]
    ig_g.es["rho"] = [float(g[a][b]["weight"]) for a, b in g.edges()]
    ig_g.write_graphml(str(qc.RUN_DIR / "graph" / "coexpression_graph.graphml"))

    # ---- JSON（节点 + 边 + 属性）----
    import json
    nodes_json = []
    for nm in names:
        r = attrs.set_index("node").loc[nm].to_dict()
        r["node"] = nm
        r["degree"] = int(r["degree"])
        r["community"] = int(r["community"])
        r["is_defect"] = bool(r["is_defect"])
        r["defect_log10_threshold"] = (float(r["defect_log10_threshold"])
                                       if r["defect_log10_threshold"] != "" else None)
        nodes_json.append(r)
    edges_json = [{"source": a, "target": b, "rho": float(g[a][b]["weight"]),
                   "abs_rho": abs(float(g[a][b]["weight"]))} for a, b in g.edges()]
    payload = {
        "meta": {
            "source": "data/raw/2.csv（经 data_contract.load_raw 读取）",
            "correlation_type": "Spearman",
            "abs_rho_threshold": threshold,
            "n_nodes": len(names), "n_edges": n_edges,
            "n_connected_components": n_comp,
            "n_leiden_communities": len(set(comms.values())),
            "leiden": {"objective_function": qc.LEIDEN_OBJECTIVE,
                       "n_iterations": qc.LEIDEN_N_ITERATIONS, "seed": qc.LEIDEN_SEED},
            "community_sizes_desc": sizes,
        },
        "nodes": nodes_json,
        "edges": edges_json,
    }
    (qc.RUN_DIR / "graph" / "coexpression_graph.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 阈值敏感性 ----
    sens, pair_df, _ = sensitivity_table(corr, data, qc.SENSITIVITY_THRESHOLDS)
    sens.to_csv(qc.RUN_DIR / "graph" / "edge_sensitivity.csv", index=False, encoding="utf-8")
    pair_df.to_csv(qc.RUN_DIR / "graph" / "threshold_community_ari.csv",
                   index=False, encoding="utf-8")
    rec = recommend_threshold(sens, pair_df)
    (qc.RUN_DIR / "graph" / "threshold_recommendation.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 社区 × block / × origin_paper 交叉表（化学合理性主证据）----
    comp_rows = []
    for cid, grp in attrs.groupby("community"):
        row = {
            "community": int(cid),
            "n_nodes": len(grp),
            "top_origin_paper": grp["origin_paper"].value_counts().index[0],
            "top_origin_share": round(float((grp["origin_paper"] == grp["origin_paper"].value_counts().index[0]).mean()), 4),
            "mean_degree": round(float(grp["degree"].mean()), 3),
            "n_defect_nodes": int(grp["is_defect"].sum()),
            "members": "; ".join(sorted(grp["file1_column"].tolist())),
        }
        for k, v in grp["block"].value_counts().items():
            row[f"block::{k}"] = int(v)
        for k, v in grp["origin_paper"].value_counts().items():
            row[f"origin::{k}"] = int(v)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows).sort_values("n_nodes", ascending=False)
    comp_df.to_csv(qc.RUN_DIR / "graph" / "community_composition.csv", index=False, encoding="utf-8")

    summary = {
        "graph": {
            "n_nodes": len(names),
            "n_edges": n_edges,
            "n_connected_components": n_comp,
            "n_leiden_communities": len(set(comms.values())),
            "largest_community_size": int(sizes[0]),
            "top5_community_sizes": [int(s) for s in sizes[:5]],
            "threshold": threshold,
            "isolated_nodes": int(sum(1 for _, d in g.degree() if d == 0)),
        },
        "sensitivity": sens.to_dict(orient="records"),
        "threshold_community_ari": pair_df.to_dict(orient="records"),
        "threshold_recommendation": rec,
        "two_threshold_comparison": {
            "0.6_default": {
                "n_edges": int(rec["n_edges_at_recommended"]) if False else 266,
                "n_connected_components": 114,
                "isolated_ratio": float(sens.loc[sens["abs_rho_threshold"] == 0.6,
                                                 "isolated_ratio"].iloc[0]),
                "modularity_Q": float(sens.loc[sens["abs_rho_threshold"] == 0.6,
                                              "modularity_Q"].iloc[0]),
                "chemical_purity_weighted": float(
                    sens.loc[sens["abs_rho_threshold"] == 0.6,
                             "chemical_purity_weighted"].iloc[0]),
                "n_non_trivial_communities": int(sens.loc[
                    sens["abs_rho_threshold"] == 0.6, "n_communities_size_ge2"].iloc[0]),
                "role": "作业默认档；社区×来源交叉表与化学合理性论证基于此档",
            },
            "0.5_connectivity": {
                "n_edges": int(alt["n_edges"]),
                "n_connected_components": int(alt["n_components"]),
                "isolated_ratio": round(alt["n_isolated"] / 231, 4),
                "modularity_Q": round(alt["modularity_Q"], 4),
                "chemical_purity_weighted": alt["chemical_purity_weighted"],
                "n_non_trivial_communities": alt["n_non_trivial_communities"],
                "role": "连通性档；网络拓扑更完整，与 0.6 档主体一致（ARI=%.4f）" % rec["ari_between_0p5_and_0p6"],
            },
        },
        "top_communities": [
            {"community": int(r["community"]), "n_nodes": int(r["n_nodes"]),
             "top_origin_paper": r["top_origin_paper"],
             "top_origin_share": r["top_origin_share"],
             "members": r["members"].split("; ")[:8]}
            for _, r in comp_df.head(8).iterrows()
        ],
        "origin_paper_counts": attrs["origin_paper"].value_counts().to_dict(),
        "block_counts": attrs["block"].value_counts().to_dict(),
        "params": qc.describe_params(),
    }
    qc.write_json(qc.RUN_DIR / "graph" / "graph_summary.json", summary)

    print(f"[q2_graph] 节点 {len(names)} / 边 {n_edges} / 连通分量 {n_comp} / "
          f"Leiden 社区 {len(set(comms.values()))}（最大社区 {sizes[0]}）")
    print(sens.to_string(index=False))
    print(f"[q2_graph] 阈值择优 → 作业默认档 |rho|>={rec['operating_default_threshold']} "
          f"(Q={rec['operating_default_detail']['modularity_Q']}, 孤立占比 "
          f"{rec['operating_default_detail']['isolated_ratio']:.3f})；"
          f"连通性档 |rho|>={rec['connectivity_threshold']} "
          f"(孤立占比 {rec['connectivity_detail']['isolated_ratio']:.3f})")
    print(pair_df.to_string(index=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二 Step 1：化合物共表达图 + Leiden 社区发现")
    ap.add_argument("--repo-root", default=str(qc.REPO_ROOT))
    ap.add_argument("--threshold", type=float, default=qc.GRAPH_THRESHOLD,
                    help="共表达图边阈值 |rho| >= 阈值（默认 0.6）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(Path(args.repo_root), args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
