# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：
#   1) 中文字体统一经 scripts/plot_style.py 的 apply_style() 写入（Microsoft YaHei，
#      axes.unicode_minus=False），出图后逐张打开检查无「豆腐块」与负号乱码；
#   2) 图中所有数值直接读自 data/processed/q2_runs 下的产物，不做二次加工；
#   3) 网络图不在密集主连通分量内叠加节点文字；度数最高的 Top-6 名称按序列于图外，
#      避免标签遮挡节点与边；
#   4) 已逐张开图核对风格一致性：6 张图在 plot_style 统一风格下重出并逐一目检
#      （中文字形、负号、图内标题 ≤12 字、无 suptitle、PNG/PDF 成对落盘）。
"""问题二出图：共表达图（来源/社区着色）、社区成分热图、bootstrap 频率图、
一致性（Jaccard）热图、阈值择优对比图。

用法：
    python scripts/q2_figures.py
输出：
    data/processed/q2_figures/*.png 与同名 *.pdf（矢量），以及 figures_manifest.json

风格约定（唯一权威来源 = scripts/plot_style.py）：
    * 字体 / 字号 / 底色 / 边框 / 网格 / 配色全部由 ``plot_style.apply_style()`` 写入，
      本脚本不再自行设置 rcParams、也不再自定义配色；
    * 图内只保留 ≤12 字的极简标题，方法、单位、样本量、阈值一律移入
      ``figures_manifest.json`` 的 caption；
    * 相关性用发散色标、频次/强度用顺序色标，均取自 plot_style 契约。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_style as ps  # noqa: E402

import q2_common as qc  # noqa: E402

#: 契约色标（顺序=频次/强度，发散=相关性/有符号一致性）
SEQ_CMAP = LinearSegmentedColormap.from_list("ps_sequential", ps.SEQUENTIAL)
DIV_CMAP = LinearSegmentedColormap.from_list("ps_diverging", ps.DIVERGING)

#: 热图列名的短标签（长文字截断，完整内容见支撑材料表）
BLOCK_SHORT = {
    "GC-MS 校准化合物": "GC-MS 校准",
    "标准理化指标": "标准理化",
    "GC-FID/FPD 目标定量": "GC-FID 定量",
    "派生变量": "派生变量",
    "GC-MS 相对丰度": "GC-MS 丰度",
    "文件8无校准标记": "无校准标记",
}
ORIGIN_SHORT = {
    "Internal standard solution": "内标溶液",
    "Unknown": "未知来源",
    "Wild flora": "野生植物",
    "Spices": "香辛料",
    "Others": "其他",
    "Yeast": "酵母",
    "Malt": "麦芽",
    "Hops": "酒花",
}


def _short(text: str, limit: int = 8) -> str:
    """长标签截断（完整内容见支撑材料表）。"""
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --------------------------------------------------------------------------- 通用


def _emit(fig, out_dir: Path, stem: str, title: str, caption: str,
          width_in: float, records: list) -> None:
    """保存 PNG(300dpi) + PDF(矢量)，并登记一条 manifest 记录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ps.save(fig, out_dir / f"{stem}.png")
    plt.close(fig)
    outputs = []
    for p in paths:
        try:
            outputs.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
        except ValueError:
            outputs.append(str(p))
    records.append({
        "file": stem,
        "title": title,
        "caption": caption,
        "width_in": round(float(width_in), 2),
        "outputs": outputs,
    })
    print(f"[q2_figures] → {paths[0].name} + {paths[1].name}")


def _write_manifest(out_dir: Path, records: list) -> None:
    """写出 figures_manifest.json（图题 + 题注 caption）。"""
    payload = {
        "style_contract": ps.STYLE_CONTRACT_VERSION,
        "numbering_note": "caption 中的「图 N」为目录内占位序号，"
                          "最终图号由 scripts/build_paper.py 统一分配。",
        "figures": records,
    }
    path = out_dir / "figures_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[q2_figures] → {path.name}")


# --------------------------------------------------------------------------- 图 1/2


def _draw_network(ax, nodes: pd.DataFrame, edges: list, pos: dict, color_key: str,
                  color_map: dict, label_top: int = 6) -> list[str]:
    """画共表达网络（边按 |rho| 定粗细，节点大小=度数，缺陷节点用强调色描边）。"""
    deg = dict(zip(nodes["node"], nodes["degree"]))
    max_deg = max(max(deg.values()), 1)
    sizes = [40 + 320 * (deg.get(n, 0) / max_deg) for n in nodes["node"]]
    node_index = nodes.set_index("node")
    cols = [color_map.get(node_index.loc[n, color_key], ps.BASELINE) for n in nodes["node"]]
    defect = {n: bool(node_index.loc[n, "is_defect"]) if "is_defect" in node_index.columns
              else False for n in nodes["node"]}

    for e in edges:
        a, b = e["source"], e["target"]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=ps.BASELINE, lw=0.3 + 1.4 * e["abs_rho"], alpha=0.55, zorder=1)
    ax.scatter([pos[n][0] for n in nodes["node"]], [pos[n][1] for n in nodes["node"]],
               s=sizes, c=cols, edgecolors=ps.INK2, linewidths=0.25, zorder=2)
    # 缺陷化合物用强调色描边突出
    defect_nodes = [n for n in nodes["node"] if defect.get(n)]
    if defect_nodes:
        ax.scatter([pos[n][0] for n in defect_nodes], [pos[n][1] for n in defect_nodes],
                   s=[sizes[list(nodes["node"]).index(n)] for n in defect_nodes],
                   facecolors="none", edgecolors=ps.DEFECT, linewidths=0.9, zorder=3)

    top_nodes = nodes.sort_values(["degree", "node"], ascending=[False, True]).head(label_top)["node"].tolist()
    ax.set_axis_off()
    return top_nodes


def fig_network(run: Path, out: Path, mode: str, records: list) -> None:
    import igraph as ig

    payload = json.loads((run / "graph" / "coexpression_graph.json").read_text(encoding="utf-8"))
    nodes = pd.DataFrame(payload["nodes"])
    edges = payload["edges"]
    thr = payload["meta"]["abs_rho_threshold"]

    names = nodes["node"].tolist()
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges([(idx[e["source"]], idx[e["target"]]) for e in edges])

    # 布局策略：只对「有边相连」的节点做布局。
    #   |ρ|≥0.6 下 231 个节点中有 101 个是孤立节点（不含任何边），它们排成外圈会把
    #   Kamada-Kawai 的全局范围撑大、把真正承载共表达结构的主图压成中央一小团，
    #   反而掩盖了要表达的信息。孤立节点不含任何关联边，故不绘出，
    #   其来源构成由「社区成分热图」承担，并在题注中写明数量。
    membership = ig_g.connected_components().membership
    groups: dict = {}
    for v, c in enumerate(membership):
        groups.setdefault(c, []).append(names[v])
    comps = sorted(groups.values(), key=len, reverse=True)
    iso_nodes = [nm for c in comps if len(c) == 1 for nm in c]
    iso_set = set(iso_nodes)
    keep = [nm for nm in names if nm not in iso_set]
    keep_set = set(keep)
    edges = [e for e in edges if e["source"] in keep_set and e["target"] in keep_set]
    nodes = nodes[nodes["node"].isin(keep_set)].reset_index(drop=True)
    idx = {nm: i for i, nm in enumerate(keep)}
    ig_g = ig.Graph(n=len(keep))
    ig_g.add_edges([(idx[e["source"]], idx[e["target"]]) for e in edges])

    main_comp = comps[0] if comps else []
    mid_comps = [c for c in comps[1:] if len(c) > 1]

    sub = ig_g.subgraph([idx[nm] for nm in main_comp])
    layout = sub.layout_kamada_kawai()
    core = {nm: (float(layout[i][0]), float(layout[i][1]))
            for i, nm in enumerate(main_comp)}
    core_extent = max(
        (np.ptp([p[0] for p in core.values()]) if core else 1.0),
        (np.ptp([p[1] for p in core.values()]) if core else 1.0),
    ) or 1.0
    radius = core_extent * 1.45
    core_scale = 0.86 * radius / core_extent
    pos = {nm: (x * core_scale, y * core_scale) for nm, (x, y) in core.items()}

    # 中环：小连通分量
    mid_flat = [nm for c in mid_comps for nm in c]
    n_mid = len(mid_flat)
    for k, nm in enumerate(mid_flat):
        ang = 2 * np.pi * k / max(n_mid, 1)
        r_mid = radius * (0.66 + 0.08 * ((k % 3) / 2.0))
        pos[nm] = (r_mid * np.cos(ang), r_mid * np.sin(ang))

    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, ps.FIG_WIDTH_IN))
    if mode == "origin":
        key = "origin_paper"
        vals = sorted(nodes[key].unique())
        cmap = {v: ps.OKABE_ITO[i % len(ps.OKABE_ITO)] for i, v in enumerate(vals)}
        top_nodes = _draw_network(ax, nodes, edges, pos, key, cmap)
        handles = [Patch(facecolor=cmap[v], label=f"{v}（{int((nodes[key] == v).sum())}）")
                   for v in vals]
        title = "共表达网络（按来源）"
    else:
        key = "community"
        sizes = nodes["community"].value_counts()
        big = list(sizes[sizes >= 2].index)
        cmap = {}
        for i, c in enumerate(big[:8]):
            cmap[c] = ps.OKABE_ITO[i]
        for i, c in enumerate(big[8:12]):
            cmap[c] = SEQ_CMAP(0.90 - 0.15 * i)
        for c in nodes["community"].unique():
            cmap.setdefault(c, ps.BASELINE)
        top_nodes = _draw_network(ax, nodes, edges, pos, key, cmap)
        handles = [Patch(facecolor=cmap[c], label=f"社区 {c}（{int(sizes[c])}）")
                   for c in big[:12]]
        n_rest = int(nodes["node"].size - sum(int(sizes[c]) for c in big[:12]))
        handles.append(Patch(facecolor=ps.BASELINE, label=f"其余社区（{n_rest}）"))
        title = "共表达网络（按社区）"

    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                           markeredgecolor=ps.DEFECT, markeredgewidth=0.9, markersize=5,
                           label="缺陷化合物"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              fontsize=5.2, frameon=False, ncol=4, handlelength=1.1,
              labelspacing=0.25, borderpad=0.1, columnspacing=0.9)
    key_text = "度数最高节点\n" + "\n".join(
        f"{i}  {_short(node, 24)}" for i, node in enumerate(top_nodes, start=1)
    )
    fig.subplots_adjust(left=0.04, right=0.70, top=0.91, bottom=0.18)
    fig.text(0.72, 0.86, key_text, ha="left", va="top", fontsize=5.0,
             color=ps.INK, linespacing=1.35)
    ax.set_xlim(-radius * 1.12, radius * 1.12)
    ax.set_ylim(-radius * 1.12, radius * 1.12)
    ps.simple_title(ax, title, size=9.5)

    caption = (
        f"图 {1 if mode == 'origin' else 2}  共表达网络（按{'来源' if mode == 'origin' else '社区'}）。"
        f"基于 2.csv 的斯皮尔曼相关，取 |ρ| ≥ {thr} 的边构成网络："
        f"全图 {payload['meta']['n_nodes']} 节点 / {payload['meta']['n_edges']} 边、"
        f"{payload['meta']['n_connected_components']} 个连通分量，"
        f"其中 {len(iso_nodes)} 个节点在该阈值下不含任何关联边（孤立节点）而未绘出，"
        f"其来源构成见支撑材料社区成员表；本图绘出 {len(nodes)} 个有边节点。"
        "节点大小 = 度数，边粗细 = |ρ|，橙红描边为缺陷化合物；"
        + ("节点按化学来源着色；" if mode == "origin" else "节点按所属社区着色；")
        + "右侧按度数由高到低列出 6 个节点；完整节点清单见支撑材料表。")
    _emit(fig, out.parent, out.stem, title, caption, ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 3


def fig_community_heatmap(run: Path, out: Path, records: list) -> None:
    comp = pd.read_csv(run / "graph" / "community_composition.csv", encoding="utf-8")
    comp = comp[comp["n_nodes"] >= 2].sort_values("n_nodes", ascending=False).head(20)
    block_cols = [c for c in comp.columns if c.startswith("block::")]
    origin_cols = [c for c in comp.columns if c.startswith("origin::")]

    fig, axes = plt.subplots(1, 2, figsize=(ps.FIG_WIDTH_IN, 4.4), constrained_layout=True)
    im = None
    for ax, cols, title in ((axes[0], block_cols, "社区×测量块"),
                            (axes[1], origin_cols, "社区×化学来源")):
        mat = comp[cols].to_numpy(dtype=float)
        mat = np.nan_to_num(mat, nan=0.0)          # 稀疏透视表：缺失即 0 成员
        if title.startswith("社区×测量块"):
            labels = [_short(BLOCK_SHORT.get(c.split("::", 1)[1], c.split("::", 1)[1]), 10)
                      for c in cols]
        else:
            labels = [_short(ORIGIN_SHORT.get(c.split("::", 1)[1], c.split("::", 1)[1]), 10)
                      for c in cols]
        im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto", vmin=0)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5.0)
        ax.set_yticks(range(len(comp)), [f"C{int(c)}" for c in comp["community"]],
                      fontsize=5.5)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=4.6,
                            color="white" if mat[i, j] > mat.max() * 0.55 else ps.INK)
        ps.simple_title(ax, title, size=8.5)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    cbar = fig.colorbar(im, ax=list(axes), location="right", shrink=0.9, pad=0.02)
    cbar.set_label("成员数", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    _emit(fig, out.parent, out.stem, "社区化学成分",
          f"图 3  社区化学成分。非平凡社区（n ≥ 2）前 {len(comp)} 个，"
          "左为测量块（block）构成、右为化学来源（origin_paper）构成，"
          "单元格为该社区在该类别下的成员数（透视表缺失记 0）；完整社区成员清单见支撑材料表。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 4


def fig_bootstrap_bars(run: Path, out: Path, records: list) -> None:
    top = pd.read_csv(run / "top3_by_sensory.csv", keep_default_na=False, encoding="utf-8")
    dims = list(dict.fromkeys(top["sensory_dimension"]))
    half = int(np.ceil(len(dims) / 2))
    chunks = [dims[:half], dims[half:]]

    rank_colors = [SEQ_CMAP(0.92), SEQ_CMAP(0.68), SEQ_CMAP(0.44)]
    fig, axes = plt.subplots(2, 1, figsize=(ps.FIG_WIDTH_IN, 6.0),
                             constrained_layout=True, sharey=True)
    for k, (ax, chunk) in enumerate(zip(axes, chunks)):
        ps.tidy(ax, grid_axis="y")
        x = np.arange(len(chunk))
        w = 0.27
        for j in range(3):
            sub = top[top["rank_in_top3"] == j + 1].set_index("sensory_dimension")
            vals = [float(sub.loc[d, "bootstrap_top3_freq"]) if d in sub.index else np.nan
                    for d in chunk]
            ax.bar(x + (j - 1) * w, vals, width=w, color=rank_colors[j],
                   edgecolor="white", linewidth=0.2, label=f"Top-{j + 1}")
        ax.axhline(qc.SELECT_FREQ_MIN, color=ps.ACCENT, ls="--", lw=0.9,
                   label=f"入选阈值 {qc.SELECT_FREQ_MIN}")
        ax.set_xticks(x, [ps.dim_cn(d) for d in chunk], rotation=90, fontsize=5.5)
        ax.set_ylim(0, 1.05)
        ps.simple_title(ax, f"维度 {k * half + 1}–{k * half + len(chunk)}", size=8.5)
    axes[0].set_ylabel("bootstrap 入选频率", fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=4, frameon=False, fontsize=6.5, handlelength=1.3)

    _emit(fig, out.parent, out.stem, "Top-3 入选频率",
          "图 4  Top-3 入选频率。各感官维度 Top-1/2/3 关键化合物的 bootstrap（B=200）入选频率，"
          "横轴为 50 个感官维度（分上下两栏），蓝色虚线为入选阈值 0.60。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 5


def fig_consistency(run: Path, out: Path, records: list) -> None:
    jac = pd.read_csv(run / "jaccard_summary.csv", encoding="utf-8")
    methods = ["E1", "E2", "E3"]
    colmap = {("E1", "E2"): "jaccard_E1_E2",
              ("E1", "E3"): "jaccard_E1_E3",
              ("E2", "E3"): "jaccard_E2_E3"}
    mat = np.zeros((3, 3))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                mat[i, j] = 1.0
            else:
                col = colmap[(a, b)] if (a, b) in colmap else colmap[(b, a)]
                mat[i, j] = float(jac[col].mean())

    fig, axes = plt.subplots(2, 1, figsize=(ps.FIG_WIDTH_IN, 6.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.4, 5.0]})

    ax = axes[0]
    ps.tidy(ax, grid_axis="none")
    ax.grid(False)
    im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), methods, fontsize=6.5)
    ax.set_yticks(range(3), methods, fontsize=6.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.55 else ps.INK)
    ps.simple_title(ax, "跨方法平均 Jaccard", size=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Jaccard", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax = axes[1]
    ps.tidy(ax, grid_axis="x")
    vals = jac["jaccard_mean"].to_numpy(dtype=float)
    order = np.argsort(vals)[::-1]
    dims = jac["sensory_dimension"].to_numpy()[order]
    colors = [ps.ACCENT if v >= qc.SELECT_JACCARD_MIN else ps.BASELINE for v in vals[order]]
    ax.barh(range(len(vals)), vals[order], color=colors, height=0.72)
    ax.axvline(qc.SELECT_JACCARD_MIN, color=ps.NEGATIVE, ls="--", lw=0.9,
               label=f"入选阈值 {qc.SELECT_JACCARD_MIN}")
    ax.set_yticks(range(len(vals)), [ps.dim_cn(d) for d in dims], fontsize=5.5)
    ax.set_ylim(len(vals) - 0.2, -0.8)
    ax.set_xlabel("三路 Top-3 平均两两 Jaccard", fontsize=7.5)
    ps.simple_title(ax, "逐维度一致性", size=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", handlelength=1.3)

    _emit(fig, out.parent, out.stem, "方法一致性",
          "图 5  方法一致性。上为 E1/E2/E3 三路 Top-3 的平均两两 Jaccard 矩阵，"
          "下为 50 个感官维度的逐维度平均两两 Jaccard（红色虚线为入选阈值 0.30）。"
          "E1 仅覆盖附件 6 可映射的 44 个化合物，E2/E3 覆盖统一矩阵的 227 个建模特征，"
          "因此跨方法 Jaccard 同时受候选空间差异影响。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 6


def fig_threshold(run: Path, out: Path, records: list) -> None:
    gs = json.loads((run / "graph" / "graph_summary.json").read_text(encoding="utf-8"))
    sens = pd.DataFrame(gs["sensitivity"])
    ari = pd.DataFrame(gs["threshold_community_ari"])

    fig, axes = plt.subplots(1, 2, figsize=(ps.FIG_WIDTH_IN, 3.2), constrained_layout=True)

    ax = axes[0]
    ps.tidy(ax, grid_axis="both")
    x = sens["abs_rho_threshold"].to_numpy(dtype=float)
    series = [("modularity_Q", "模块度 Q", ps.OKABE_ITO[0], "o"),
              ("chemical_purity_weighted", "化学纯度", ps.OKABE_ITO[2], "s"),
              ("isolated_ratio", "孤立节点占比", ps.OKABE_ITO[4], "^")]
    for col, label, color, marker in series:
        ax.plot(x, sens[col], marker + "-", color=color, lw=1.0, label=label)
    ax.plot(x, sens["n_edges"] / sens["n_edges"].max(), "d--", color=ps.OKABE_ITO[3],
            lw=0.9, label="边数（归一化）")
    for xi, yi in zip(x, sens["modularity_Q"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), fontsize=5.5, color=ps.OKABE_ITO[0],
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xlabel("边阈值 |ρ|", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xlim(float(x.min()) - 0.035, float(x.max()) + 0.035)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.0, frameon=False, loc="lower left", handlelength=1.3)
    ps.simple_title(ax, "阈值敏感性", size=8.5)

    ax = axes[1]
    ps.tidy(ax, grid_axis="none")
    ax.grid(False)
    ths = sorted(sens["abs_rho_threshold"].tolist())
    amat = np.eye(len(ths))
    for _, r in ari.iterrows():
        i, j = ths.index(r["threshold_a"]), ths.index(r["threshold_b"])
        amat[i, j] = amat[j, i] = float(r["ARI"])
    vmin = 0.0
    vmax = 1.0
    im = ax.imshow(amat, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ths)), [str(t) for t in ths], fontsize=6.5)
    ax.set_yticks(range(len(ths)), [str(t) for t in ths], fontsize=6.5)
    for i in range(len(ths)):
        for j in range(len(ths)):
            ax.text(j, i, f"{amat[i, j]:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if amat[i, j] > vmax * 0.6 else ps.INK)
    ax.set_xlabel("阈值 A", fontsize=7.5)
    ax.set_ylabel("阈值 B", fontsize=7.5)
    ps.simple_title(ax, "跨阈值 ARI", size=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("ARI", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    _emit(fig, out.parent, out.stem, "阈值择优",
          "图 6  阈值择优。左为边阈值 |ρ| ∈ {0.5, 0.6, 0.7, 0.8} 扫描下的模块度 Q、"
          "化学纯度（来源众数占比）、孤立节点占比与归一化边数，右为各阈值两两组合所得社区的 "
          "ARI 矩阵（非负顺序色标，0 为浅色、1 为深色）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二出图")
    ap.add_argument("--repo-root", default=str(qc.REPO_ROOT))
    args = ap.parse_args()
    ps.apply_style()

    run = qc.RUN_DIR
    out = qc.FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    records: list = []

    fig_network(run, out / "fig1_coexpression_network_by_origin.png", "origin", records)
    fig_network(run, out / "fig2_coexpression_network_by_community.png", "community", records)
    fig_community_heatmap(run, out / "fig3_community_composition_heatmap.png", records)
    fig_bootstrap_bars(run, out / "fig4_top3_bootstrap_frequency.png", records)
    fig_consistency(run, out / "fig5_method_consistency_jaccard.png", records)
    fig_threshold(run, out / "fig6_threshold_selection.png", records)
    _write_manifest(out, records)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
