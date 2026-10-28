# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：中文字体（Microsoft YaHei）已在本机实测可用（data/processed/cjk_probe.png 已验证）；
#   图中所有数值直接读自 data/processed/q2_runs 下的产物，不做二次加工。
"""问题二出图：共表达图（来源/社区着色）、社区成分热图、bootstrap 频率图、
一致性（Jaccard）热图、阈值择优对比图。

用法：
    python scripts/q2_figures.py
输出：
    data/processed/q2_figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import q2_common as qc

# --------------------------------------------------------------------------- 绘图规范

CJK_FONTS = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DengXian")
INK = "#1f2933"
INK2 = "#52606d"
GRID = "#e4e7eb"
BG = "#ffffff"
PALETTE = ["#2f6f9f", "#c15130", "#3f8f6b", "#8a5fa8", "#b8912f",
           "#6b7280", "#9c6644", "#4361a8", "#7a9e3f", "#a05195"]
CATEGORICAL = ["#2f6f9f", "#c15130", "#3f8f6b", "#8a5fa8", "#b8912f",
               "#6b7280", "#9c6644", "#4361a8", "#7a9e3f", "#a05195",
               "#bf6b8f", "#4c8c8a", "#8c6d4f", "#7f5aa0", "#9b9b3f",
               "#2b6b7f", "#a35a3a", "#5c7a3a", "#7a3f6b", "#3f6b8c"]


def setup_style() -> str:
    """设置中文字体；返回实际使用的字体名。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((f for f in CJK_FONTS if f in available), None)
    if picked is None:
        print("[q2_figures] 警告：未找到中文字体，图中中文可能显示为方块")
        picked = "DejaVu Sans"
    rcParams["font.sans-serif"] = [picked] + list(CJK_FONTS)
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = BG
    rcParams["axes.facecolor"] = BG
    rcParams["savefig.facecolor"] = BG
    rcParams["axes.edgecolor"] = GRID
    rcParams["text.color"] = INK
    rcParams["axes.labelcolor"] = INK2
    rcParams["xtick.color"] = INK2
    rcParams["ytick.color"] = INK2
    return picked


def finish(fig, path: Path, dpi: int = 170) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt_close(fig)
    print(f"[q2_figures] → {path.name}")


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- 图 1/2


def draw_network(ax, nodes: pd.DataFrame, edges: list, pos: dict, color_key: str,
                 color_map: dict, title: str, label_top: int = 8) -> None:
    import networkx as nx

    deg = dict(zip(nodes["node"], nodes["degree"]))
    sizes = [40 + 320 * (deg.get(n, 0) / max(max(deg.values()), 1)) for n in nodes["node"]]
    cols = [color_map.get(nodes.set_index("node").loc[n, color_key], "#9aa5b1")
            for n in nodes["node"]]
    nw = nx.Graph()
    nw.add_nodes_from(nodes["node"])
    nw.add_edges_from([(e["source"], e["target"]) for e in edges])

    # 先画边（按 |rho| 粗细），后画节点
    for e in edges:
        a, b = e["source"], e["target"]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#cbd2d9", lw=0.3 + 1.4 * e["abs_rho"], alpha=0.55, zorder=1)
    ax.scatter([pos[n][0] for n in nodes["node"]], [pos[n][1] for n in nodes["node"]],
               s=sizes, c=cols, edgecolors="white", linewidths=0.35, zorder=2)

    top_nodes = nodes.nlargest(label_top, "degree")["node"].tolist()
    for n in top_nodes:
        x, y = pos[n]
        ax.annotate(n, (x, y), fontsize=5.0, color=INK, zorder=3,
                    xytext=(2.5, 2.5), textcoords="offset points")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_axis_off()


def fig_network(run: Path, out: Path, mode: str) -> None:
    import igraph as ig
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    payload = json.loads((run / "graph" / "coexpression_graph.json").read_text(encoding="utf-8"))
    nodes = pd.DataFrame(payload["nodes"])
    edges = payload["edges"]
    thr = payload["meta"]["abs_rho_threshold"]

    names = nodes["node"].tolist()
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[e["source"]], idx[e["target"]]) for e in edges])

    # 布局策略：把「稠密主连通分量」放大居中，小连通分量排在中环，
    # 孤立节点排在外圈 —— 否则 Kamada-Kawai 的全局范围会被零散小分量拉大、主图被压扁。
    deg = dict(zip(nodes["node"], nodes["degree"]))
    membership = ig_g.connected_components().membership
    groups: dict = {}
    for v, c in enumerate(membership):
        groups.setdefault(c, []).append(names[v])
    comps = sorted(groups.values(), key=len, reverse=True)
    main_comp = comps[0] if comps else []
    mid_comps = [c for c in comps[1:] if len(c) > 1]
    iso_nodes = [nm for c in comps if len(c) == 1 for nm in c]

    sub = ig_g.subgraph([idx[nm] for nm in main_comp])
    layout = sub.layout_kamada_kawai()
    core = {nm: (float(layout[i][0]), float(layout[i][1]))
            for i, nm in enumerate(main_comp)}
    core_extent = max(
        (np.ptp([p[0] for p in core.values()]) if core else 1.0),
        (np.ptp([p[1] for p in core.values()]) if core else 1.0),
    ) or 1.0
    radius = core_extent * 1.75
    core_scale = 0.50 * radius / core_extent
    pos = {nm: (x * core_scale, y * core_scale) for nm, (x, y) in core.items()}

    # 中环：小连通分量
    mid_flat = [nm for c in mid_comps for nm in c]
    n_mid = len(mid_flat)
    for k, nm in enumerate(mid_flat):
        ang = 2 * np.pi * k / max(n_mid, 1)
        r_mid = radius * (0.62 + 0.10 * ((k % 3) / 2.0))
        pos[nm] = (r_mid * np.cos(ang), r_mid * np.sin(ang))

    # 外圈：孤立节点（按分组排布，视觉上仍能看出化学来源聚集）
    group_key = "origin_paper" if mode == "origin" else "community"
    nodes_sorted = nodes.set_index("node")
    iso_groups: dict = {}
    for nm in iso_nodes:
        iso_groups.setdefault(nodes_sorted.loc[nm, group_key], []).append(nm)
    ordered = [nm for grp in iso_groups.values() for nm in grp]
    n_iso = len(ordered)
    for k, nm in enumerate(ordered):
        ang = 2 * np.pi * k / max(n_iso, 1)
        pos[nm] = (radius * np.cos(ang), radius * np.sin(ang))

    fig, ax = plt.subplots(figsize=(13, 12))
    if mode == "origin":
        key = "origin_paper"
        vals = sorted(nodes[key].unique())
        cmap = {v: CATEGORICAL[i % len(CATEGORICAL)] for i, v in enumerate(vals)}
        draw_network(ax, nodes, edges, pos, key, cmap,
                     f"化合物共表达网络（|ρ| ≥ {thr}，Spearman）：按 8.xlsx origin_paper 着色"
                     f"（外圈为孤立节点）")
        handles = [Patch(facecolor=cmap[v], label=f"{v} ({int((nodes[key] == v).sum())})")
                   for v in vals]
    else:
        key = "community"
        sizes = nodes["community"].value_counts()
        big = list(sizes[sizes >= 2].index)
        cmap = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(big)}
        cmap.update({c: "#dfe3e8" for c in nodes["community"].unique() if c not in cmap})
        draw_network(ax, nodes, edges, pos, key, cmap,
                     f"化合物共表达网络（|ρ| ≥ {thr}，Spearman）：按 Leiden 社区着色"
                     f"（灰色为孤立节点）")
        handles = [Patch(facecolor=cmap[c], label=f"社区 {c}（{int(sizes[c])} 节点）")
                   for c in big[:12]]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=True,
              facecolor=BG, edgecolor=GRID, ncol=2)
    ax.set_xlim(-radius * 1.12, radius * 1.12)
    ax.set_ylim(-radius * 1.12, radius * 1.12)
    fig.text(0.5, 0.01, "节点大小 = 度数；边粗细 = |ρ|。AI 辅助：MiniMax Code (MiniMax-M3.1-Flash)",
             ha="center", fontsize=7, color=INK2)
    finish(fig, out)


# --------------------------------------------------------------------------- 图 3


def fig_community_heatmap(run: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    comp = pd.read_csv(run / "graph" / "community_composition.csv", encoding="utf-8")
    comp = comp[comp["n_nodes"] >= 2].sort_values("n_nodes", ascending=False).head(20)
    block_cols = [c for c in comp.columns if c.startswith("block::")]
    origin_cols = [c for c in comp.columns if c.startswith("origin::")]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    for ax, cols, title in ((axes[0], block_cols, "社区 × 测量块（block）"),
                            (axes[1], origin_cols, "社区 × 化学来源（origin_paper）")):
        mat = comp[cols].to_numpy(dtype=float)
        labels = [c.split("::", 1)[1] for c in cols]
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(comp)), [f"C{int(c)}" for c in comp["community"]],
                      fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > mat.max() * 0.55 else INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("成员数", fontsize=8)
        fig.colorbar(im, ax=ax, label="节点数", fraction=0.03)
    fig.suptitle("共表达社区的化学成分（|ρ| ≥ 0.6，非平凡社区前 20）——化学合理性主证据",
                 fontsize=12, color=INK)
    finish(fig, out)


# --------------------------------------------------------------------------- 图 4


def fig_bootstrap_bars(run: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    top = pd.read_csv(run / "top3_by_sensory.csv", keep_default_na=False, encoding="utf-8")
    dims = list(dict.fromkeys(top["sensory_dimension"]))
    x = np.arange(len(dims))
    w = 0.27
    norm = Normalize(0, 1)
    cmap = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(18, 6.6))
    for k in range(3):
        sub = top[top["rank_in_top3"] == k + 1].set_index("sensory_dimension")
        vals = [float(sub.loc[d, "bootstrap_top3_freq"]) if d in sub.index else np.nan
                for d in dims]
        ax.bar(x + (k - 1) * w, vals, width=w, color=cmap(norm(vals)),
               edgecolor="white", linewidth=0.3,
               label=f"Top-{k+1}")
    ax.axhline(qc.SELECT_FREQ_MIN, color="#2f6f9f", ls="--", lw=1.4,
               label=f"入选阈值 {qc.SELECT_FREQ_MIN}")
    ax.set_xticks(x, dims, rotation=90, fontsize=6.5)
    ax.set_ylabel("bootstrap 入选频率（B=200）", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("各感官维度 Top-3 关键化合物的 bootstrap 入选频率", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finish(fig, out)


# --------------------------------------------------------------------------- 图 5


def fig_consistency(run: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    jac = pd.read_csv(run / "jaccard_summary.csv", encoding="utf-8")
    methods = ["E1", "E2", "E3"]
    pairs = [("E1", "E2"), ("E1", "E3"), ("E2", "E3")]
    mat = np.zeros((3, 3))
    colmap = {("E1", "E2"): "jaccard_E1_E2",
              ("E1", "E3"): "jaccard_E1_E3",
              ("E2", "E3"): "jaccard_E2_E3"}
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                mat[i, j] = 1.0
            else:
                col = colmap[(a, b)] if (a, b) in colmap else colmap[(b, a)]
                mat[i, j] = float(jac[col].mean())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(3), methods)
    ax.set_yticks(range(3), methods)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] > 0.55 else INK)
    ax.set_title("跨方法 Top-3 平均 Jaccard（50 维度均值）", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axes[1]
    vals = jac["jaccard_mean"].to_numpy(dtype=float)
    order = np.argsort(vals)[::-1]
    dims = jac["sensory_dimension"].to_numpy()[order]
    colors = ["#2f6f9f" if v >= qc.SELECT_JACCARD_MIN else "#cbd2d9" for v in vals[order]]
    ax.barh(range(len(vals)), vals[order], color=colors, height=0.72)
    ax.axvline(qc.SELECT_JACCARD_MIN, color="#c15130", ls="--", lw=1.3,
               label=f"入选阈值 {qc.SELECT_JACCARD_MIN}")
    ax.set_yticks(range(len(vals)), dims, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("E1/E2/E3 三路 Top-3 的平均两两 Jaccard", fontsize=9)
    ax.set_title("逐维度方法间一致性", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finish(fig, out)


# --------------------------------------------------------------------------- 图 6


def fig_threshold(run: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    gs = json.loads((run / "graph" / "graph_summary.json").read_text(encoding="utf-8"))
    sens = pd.DataFrame(gs["sensitivity"])
    ari = pd.DataFrame(gs["threshold_community_ari"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axes[0]
    x = sens["abs_rho_threshold"].to_numpy(dtype=float)
    ax.plot(x, sens["modularity_Q"], "o-", color="#2f6f9f", lw=1.8, label="模块度 Q")
    ax.plot(x, sens["chemical_purity_weighted"], "s-", color="#3f8f6b", lw=1.8,
            label="化学纯度（来源众数占比）")
    ax.plot(x, sens["isolated_ratio"], "^-", color="#c15130", lw=1.8, label="孤立节点占比")
    ax.plot(x, sens["n_edges"] / sens["n_edges"].max(), "d--", color="#8a5fa8", lw=1.2,
            label="边数（归一化）")
    for xi, yi in zip(x, sens["modularity_Q"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), fontsize=7, color="#2f6f9f",
                    xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_xlabel("边阈值 |ρ|", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("阈值择优：模块度 / 化学纯度 / 孤立占比", fontsize=10, color=INK)

    ax = axes[1]
    ths = sorted(sens["abs_rho_threshold"].tolist())
    amat = np.eye(len(ths))
    for _, r in ari.iterrows():
        i, j = ths.index(r["threshold_a"]), ths.index(r["threshold_b"])
        amat[i, j] = amat[j, i] = float(r["ARI"])
    im = ax.imshow(amat, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(ths)), [str(t) for t in ths])
    ax.set_yticks(range(len(ths)), [str(t) for t in ths])
    for i in range(len(ths)):
        for j in range(len(ths)):
            ax.text(j, i, f"{amat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if amat[i, j] > 0.55 else INK)
    ax.set_xlabel("阈值 A", fontsize=9)
    ax.set_ylabel("阈值 B", fontsize=9)
    ax.set_title("跨阈值社区一致性（ARI）", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045)
    finish(fig, out)


# --------------------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二出图")
    ap.add_argument("--repo-root", default=str(qc.REPO_ROOT))
    args = ap.parse_args()
    font = setup_style()
    print(f"[q2_figures] 中文字体：{font}")

    run = qc.RUN_DIR
    out = qc.FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    fig_network(run, out / "fig1_coexpression_network_by_origin.png", "origin")
    fig_network(run, out / "fig2_coexpression_network_by_community.png", "community")
    fig_community_heatmap(run, out / "fig3_community_composition_heatmap.png")
    fig_bootstrap_bars(run, out / "fig4_top3_bootstrap_frequency.png")
    fig_consistency(run, out / "fig5_method_consistency_jaccard.png")
    fig_threshold(run, out / "fig6_threshold_selection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
