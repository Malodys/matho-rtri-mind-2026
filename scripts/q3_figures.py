# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：
#   1) 中文字体统一经 scripts/plot_style.py 的 apply_style() 写入（Microsoft YaHei，
#      axes.unicode_minus=False），出图后逐张打开检查无「豆腐块」与负号乱码；
#   2) 所有图只读 data/processed/q3_runs/ 下的产物与统一矩阵，不重复跑生成/迁移逻辑，
#      保证图与 CSV 产物严格一致；
#   3) 缺陷类图同时给出「真实 / 无约束生成 / 有约束生成 / 迁移后」四组，
#      以便直观展示约束带来的边际压缩（Tripel DMS/Diacetyl 被系统性压低）；
#   4) 已逐张开图核对风格一致性：7 张图在 plot_style 统一风格下重出并逐一目检
#      （中文字形、负号、图内标题 ≤12 字、无 suptitle、PNG/PDF 成对落盘）。
"""问题 3 出图：风格热图、边际分布对比、平行坐标、迁移雷达图、缺陷箱线图、KS 分布。

产出目录：``data/processed/q3_figures/``（每张图同时输出 PNG(300dpi) 与 PDF(矢量)，
并写 ``figures_manifest.json``，caption 内含方法/单位/样本量说明）。

风格约定（唯一权威来源 = scripts/plot_style.py）：
    * 字体 / 字号 / 底色 / 边框 / 网格 / 配色全部由 ``plot_style.apply_style()`` 写入，
      本脚本不再自行设置 rcParams、也不再自定义配色；
    * 图内只保留 ≤12 字的极简标题，方法、单位、样本量、阈值一律移入题注。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from matplotlib.projections.polar import PolarAxes

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402
import plot_style as ps  # noqa: E402
import q3_copula as q3  # noqa: E402

RUN_DIR = REPO_ROOT / "data" / "processed" / "q3_runs"
GEN_DIR = RUN_DIR / "generation"
FIG_DIR = REPO_ROOT / "data" / "processed" / "q3_figures"

#: 顺序色标（浓度/强度）：直接取 plot_style 的契约配色，不自定义颜色
SEQ_CMAP = LinearSegmentedColormap.from_list("ps_sequential", ps.SEQUENTIAL)

KEY_COMPOUNDS = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal",
                 "ethanol..v.v.", "bitterness", "esters_sum", "sulfur_sum"]
#: 热图用的固定化合物清单（化学含义明确、跨风格可比）
HEATMAP_COMPOUNDS = [
    "ethanol..v.v.", "bitterness", "DMS", "X2.3.butanedione", "X4VG",
    "X.E..2.nonenal", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum",
    "ethyl_acetate", "isoamyl_alcohol", "acetaldehyde", "benzaldehyde",
    "beta.humulene", "linalool", "geraniol", "E.caryophyllene",
]
PRETTY = {"X2.3.butanedione": "Diacetyl", "X.E..2.nonenal": "t2n",
          "ethanol..v.v.": "ethanol", "X4VG": "4-VG", "DMS": "DMS"}
TARGET_STYLES = ["Tripel", "Kriek", "Lambic"]

#: 三种来源（真实 / 无约束生成 / 有约束生成）的统一配色
SRC_COLORS = [ps.OKABE_ITO[0], ps.OKABE_ITO[1], ps.OKABE_ITO[2]]
SRC_LABELS = ["真实样本", "无约束生成", "有约束生成"]


def slug(style: str) -> str:
    return (style.replace(" ", "_").replace("/", "-")).strip("_")


def style_label(name: str) -> str:
    return {"Tripel": "三倍酒 Tripel", "Kriek": "樱桃啤酒 Kriek",
            "Lambic": "兰比克 Lambic", "Blond": "金黄 Blond",
            "Strong ale": "烈性艾尔 Strong ale", "Hoppy": "啤酒花型 Hoppy",
            "Dubbel": "Dubbel", "Saison": "赛松 Saison"}.get(name, name)


def short_name(col: str, limit: int = 16) -> str:
    """化合物显示名：优先用 PRETTY，其余长名截断（完整名见支撑材料表）。"""
    name = PRETTY.get(col, col)
    return name if len(name) <= limit else name[:limit - 1] + "…"


def _tidy(ax, grid_axis: str = "y") -> None:
    """笛卡尔坐标轴用契约的 tidy()；极坐标轴只做网格与外框处理。"""
    if isinstance(ax, PolarAxes):
        ax.grid(True, color=ps.GRID, linewidth=0.6)
        ax.spines["polar"].set_color(ps.INK2)
        ax.tick_params(which="minor", length=0)
        return
    ps.tidy(ax, grid_axis=grid_axis)


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
    print(f"[图] {paths[0].name} + {paths[1].name}")


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
    print(f"[落盘] {path}")


def load_style_frames() -> dict:
    """载入真实矩阵与各风格的生成谱（约束版 / 无约束版）。"""
    raw = dc.load_raw()
    matrix = dc.to_unified_scale(raw)
    styles = dc.load_styles(raw).reindex(matrix.index)
    out = {"matrix": matrix, "styles": styles, "gen": {}, "uncon": {}}
    for style in sorted(set(styles.astype(str))):
        p_c = GEN_DIR / f"{slug(style)}_synthetic.csv"
        p_u = GEN_DIR / f"{slug(style)}_unconstrained.csv"
        if p_c.is_file():
            out["gen"][style] = pd.read_csv(p_c, encoding="utf-8-sig")
        if p_u.is_file():
            out["uncon"][style] = pd.read_csv(p_u, encoding="utf-8-sig")
    return out


def compound_cols(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in ("beer", "tasting_category_fine",
                                                  "variant")]


# --------------------------------------------------------------------------- 图 1


def fig_style_heatmaps(data: dict, out_path: Path, records: list, top_n: int = 18) -> None:
    """风格热图：真实 / 无约束生成 / 有约束生成 的中位 log10(mg/L) 对照。"""
    styles = [s for s in TARGET_STYLES if s in data["gen"]]
    if not styles:
        print("[跳过] 无生成谱，跳过热图")
        return
    matrix, style_ser = data["matrix"], data["styles"]
    cols = [c for c in HEATMAP_COMPOUNDS if c in matrix.columns][:top_n]

    fig, axes = plt.subplots(len(styles), 1, figsize=(ps.FIG_WIDTH_IN, 3.0 * len(styles)),
                             constrained_layout=True, squeeze=False)
    im = None
    for i, style in enumerate(styles):
        real = matrix.loc[(style_ser == style).to_numpy()]
        uncon = data["uncon"].get(style)
        con = data["gen"][style]
        panels = [("真实样本", real[cols]),
                  ("无约束生成", uncon[cols] if uncon is not None else None),
                  ("有约束生成", con[cols])]
        ax = axes[i, 0]
        blocks = []
        for _, frame in panels:
            if frame is None:
                blocks.append(np.full(len(cols), np.nan))
            else:
                blocks.append(frame.median().to_numpy(dtype=float))
        mat = np.vstack(blocks).T          # 行=化合物，列=三种来源
        im = ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, vmin=-3.0, vmax=1.5)
        ax.set_xticks(range(3), ["真实", "无约束", "有约束"], fontsize=6.5)
        ax.set_yticks(range(len(cols)), [short_name(c) for c in cols], fontsize=5.0)
        ps.simple_title(ax, style_label(style), size=8.5)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                if not np.isnan(mat[r, c]):
                    ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center",
                            fontsize=4.6, color=ps.INK)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes.ravel()), location="right", shrink=0.92,
                            pad=0.02)
        cbar.set_label("中位 log10(mg/L)", fontsize=6.5)
        cbar.ax.tick_params(labelsize=6)

    _emit(fig, out_path.parent, out_path.stem, "风格化学画像",
          "图 1  风格化学画像。真实样本 / 无约束生成 / 有约束生成 的逐化合物中位 log10(mg/L) 对照"
          "（每个风格一行，三列依次为三种来源，色标为中位浓度强度）；"
          "完整逐化合物数值见支撑材料表。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 2


def fig_marginal_distributions(data: dict, out_path: Path, records: list) -> None:
    """边际分布对比：真实 / 无约束生成 / 有约束生成（箱线）。"""
    styles = [s for s in TARGET_STYLES if s in data["gen"]]
    cols = [c for c in KEY_COMPOUNDS if c in data["matrix"].columns]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ps.FIG_WIDTH_IN, 3.0 * nrows),
                             constrained_layout=True, squeeze=False)
    matrix, style_ser = data["matrix"], data["styles"]
    for j, col in enumerate(cols):
        ax = axes[j // ncols, j % ncols]
        _tidy(ax, grid_axis="y")
        series, positions, colors, groups = [], [], [], []
        pos = 0.0
        for i, style in enumerate(styles):
            real = matrix.loc[(style_ser == style).to_numpy()][col].to_numpy()
            uncon = data["uncon"].get(style)
            con = data["gen"][style]
            src_items = [(real, 0)]
            if uncon is not None:
                src_items.append((uncon[col].to_numpy(), 1))
            src_items.append((con[col].to_numpy(), 2))
            for values, ci in src_items:
                series.append(values)
                positions.append(pos + ci * 0.9)
                colors.append(SRC_COLORS[ci])
            groups.append((pos + (len(src_items) - 1) * 0.45, len(src_items)))
            pos += len(src_items) * 0.9 + 0.45
        group_centers = [g[0] for g in groups]
        bp = ax.boxplot(series, positions=positions, patch_artist=True, widths=0.62,
                        medianprops=dict(color=ps.INK, linewidth=0.8),
                        flierprops=dict(marker=".", markersize=2, alpha=0.4),
                        whiskerprops=dict(color=ps.INK2, linewidth=0.6),
                        capprops=dict(color=ps.INK2, linewidth=0.6))
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[k])
            patch.set_edgecolor("none")
            patch.set_alpha(0.85)
        thr = data["defect_thr"].get(col)
        if thr is not None:
            ax.axhline(thr, color=ps.DEFECT, ls="--", lw=0.9)
        group_centers = [g[0] for g in groups]
        ax.set_xticks(group_centers, styles, rotation=45, ha="right", fontsize=5.0)
        ps.simple_title(ax, short_name(col), size=8.0)
        ax.tick_params(labelsize=5.5)
    for k in range(len(cols), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c,
                      markeredgecolor="none", markersize=6, label=l)
               for l, c in zip(SRC_LABELS, SRC_COLORS)]
    handles.append(Line2D([0], [0], color=ps.DEFECT, ls="--", lw=0.9, label="感官阈值"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=6.5)

    _emit(fig, out_path.parent, out_path.stem, "边际分布对比",
          "图 2  边际分布对比。三种风格（Tripel / Kriek / Lambic）× 三种来源"
          "（真实样本 / 无约束生成 / 有约束生成）的逐化合物 log10(mg/L) 分布箱线图；"
          "红色虚线为感官阈值（仅缺陷化合物有阈值，t2n 无阈值）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 3


def fig_parallel_coordinates(data: dict, out_path: Path, records: list,
                              style: str = "Tripel", n_show: int = 60) -> None:
    """平行坐标：真实样本 vs 无约束生成 vs 有约束生成。"""
    if style not in data["gen"]:
        print("[跳过] 无该风格生成谱")
        return
    matrix, style_ser = data["matrix"], data["styles"]
    real = matrix.loc[(style_ser == style).to_numpy()]
    uncon = data["uncon"].get(style)
    con = data["gen"][style]
    cols = [c for c in KEY_COMPOUNDS if c in matrix.columns]
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, 3.2), constrained_layout=True)
    _tidy(ax, grid_axis="y")
    x = np.arange(len(cols))
    groups = [("真实样本", real[cols], SRC_COLORS[0]),
              ("无约束生成", uncon[cols] if uncon is not None else None, SRC_COLORS[1]),
              ("有约束生成", con[cols], SRC_COLORS[2])]
    for label, frame, color in groups:
        if frame is None:
            continue
        sub = frame.head(n_show)
        for _, row in sub.iterrows():
            ax.plot(x, row.to_numpy(dtype=float), color=color, alpha=0.18, lw=0.7)
        ax.plot([], [], color=color, lw=1.6, label=f"{label}（前 {min(n_show, len(sub))} 条）")
    ax.set_xticks(x, [short_name(c, 12) for c in cols], rotation=35, ha="right",
                  fontsize=6.5)
    ax.set_ylabel("log10(mg/L)", fontsize=7.5)
    ps.simple_title(ax, "平行坐标", size=9.0)
    ax.legend(loc="upper left", fontsize=6.0, frameon=False, handlelength=1.3)

    _emit(fig, out_path.parent, out_path.stem, "平行坐标",
          f"图 3  平行坐标。{style_label(style)} 风格的逐化合物 log10(mg/L) 平行坐标，"
          "真实样本 / 无约束生成 / 有约束生成 各取前 60 条样本。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 4


def fig_transfer_radar(out_path: Path, records: list) -> None:
    """迁移前后雷达图：四项缺陷化合物的相对水平。"""
    path = RUN_DIR / "transfer_pils_to_tripel_defects.csv"
    if not path.is_file():
        print("[跳过] 无迁移缺陷表")
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    keys = ["dms", "diacetyl", "4vg", "t2n"]
    labels = ["DMS", "Diacetyl", "4-VG", "trans-2-nonenal"]
    b_vec, a_vec = [], []
    for k in keys:
        d = df[df.defect_key == k]
        b_vec.append(float(d["value_before_log10_mgL"].mean()))
        a_vec.append(float(d["value_after_log10_mgL"].mean()))
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(ps.FIG_WIDTH_IN, 4.4), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    _tidy(ax)
    for vec, color, name in ((b_vec, ps.OKABE_ITO[0], "迁移前（Pils/Lager）"),
                             (a_vec, ps.OKABE_ITO[1], "迁移后（Tripel 边际 + 约束）")):
        v = vec + vec[:1]
        ax.plot(ang, v, color=color, lw=1.2, label=name)
        ax.fill(ang, v, color=color, alpha=0.12)
    ax.set_xticks(ang[:-1])
    # 数值直接写进轴标签，避免与多边形/网格重叠
    ax.set_xticklabels([f"{lab}\n{b:.2f} → {a:.2f}"
                         for lab, b, a in zip(labels, b_vec, a_vec)], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=5.5, colors=ps.MUTED)
    ax.set_rlabel_position(112.5)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    ps.simple_title(ax, "迁移前后对比", size=9.0, pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.5,
              frameon=False, handlelength=1.3)

    _emit(fig, out_path.parent, out_path.stem, "迁移前后对比",
          "图 4  迁移前后对比。Pils/Lager → Tripel 迁移前后四项缺陷化合物的均值 log10(mg/L) "
          "雷达图（半径为 log10 值，越靠内越低），各轴标签给出「迁移前 → 迁移后」的具体数值。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 5


def fig_defect_boxplots(data: dict, out_path: Path, records: list) -> None:
    """缺陷化合物箱线图：真实全体 / 各风格有约束 / 无约束 / 迁移后。"""
    specs = [("dms", "DMS"), ("diacetyl", "Diacetyl"),
             ("4vg", "4-VG"), ("t2n", "trans-2-nonenal")]
    matrix = data["matrix"]
    fig, axes = plt.subplots(2, 2, figsize=(ps.FIG_WIDTH_IN, 6.0), constrained_layout=True)
    transfer_path = RUN_DIR / "transfer_pils_to_tripel_defects.csv"
    for ax, (key, label) in zip(axes.ravel(), specs):
        _tidy(ax, grid_axis="y")
        col = next(s["column"] for s in q3.defect_specs(dc.defect_table())
                   if s["key"] == key)
        series, names, colors = [], [], []
        series.append(matrix[col].to_numpy())
        names.append("真实")
        colors.append(ps.OKABE_ITO[0])
        for style in ["Tripel", "Kriek", "Lambic"]:
            frame = data["gen"].get(style)
            if frame is not None:
                series.append(frame[col].to_numpy())
                names.append(f"{style[:2]}\n有约束")
                colors.append(ps.OKABE_ITO[2])
            frame_u = data["uncon"].get(style)
            if frame_u is not None:
                series.append(frame_u[col].to_numpy())
                names.append(f"{style[:2]}\n无约束")
                colors.append(ps.OKABE_ITO[1])
        if transfer_path.is_file():
            tdf = pd.read_csv(transfer_path, encoding="utf-8-sig")
            t = tdf[tdf.defect_key == key]
            if len(t):
                series.append(t["value_after_log10_mgL"].to_numpy())
                names.append("迁移后")
                colors.append(ps.OKABE_ITO[3])
        bp = ax.boxplot(series, patch_artist=True, widths=0.62, showfliers=False,
                        medianprops=dict(color=ps.INK, linewidth=0.8),
                        whiskerprops=dict(color=ps.INK2, linewidth=0.6),
                        capprops=dict(color=ps.INK2, linewidth=0.6))
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[k])
            patch.set_edgecolor("none")
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right",
                      fontsize=5.0)
        thr = data["defect_thr"].get(col)
        if thr is not None:
            ax.axhline(thr, color=ps.DEFECT, ls="--", lw=0.9)
        ps.simple_title(ax, label, size=8.5)
        ax.tick_params(labelsize=5.5)

    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c,
                      markeredgecolor="none", markersize=6, label=l)
               for l, c in [("真实全体", ps.OKABE_ITO[0]), ("有约束生成", ps.OKABE_ITO[2]),
                            ("无约束生成", ps.OKABE_ITO[1]), ("迁移后", ps.OKABE_ITO[3])]]
    handles.append(Line2D([0], [0], color=ps.DEFECT, ls="--", lw=0.9, label="感官阈值"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               fontsize=6.5)

    _emit(fig, out_path.parent, out_path.stem, "缺陷化合物分布",
          "图 5  缺陷化合物分布。DMS、Diacetyl、4-VG、trans-2-nonenal 在真实全体、"
          "三种风格的有约束/无约束生成以及迁移后（Pils→Tripel）的 log10(mg/L) 分布；"
          "红色虚线为感官阈值（t2n 无阈值）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 6


def fig_ks_distribution(out_path: Path, records: list) -> None:
    """KS 距离分布：有约束 vs 无约束，附两样本 KS 原假设中位数参考线。"""
    path = RUN_DIR / "ks_per_compound.csv"
    summary_path = RUN_DIR / "generation_summary.csv"
    if not path.is_file():
        print("[跳过] 无 ks_per_compound.csv")
        return
    ks = pd.read_csv(path, encoding="utf-8-sig")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    styles = [s for s in TARGET_STYLES if s in set(ks["style"])]
    fig, axes = plt.subplots(1, len(styles), figsize=(ps.FIG_WIDTH_IN, 3.2),
                             constrained_layout=True, squeeze=False)
    handles = []
    for ax, style in zip(axes[0], styles):
        _tidy(ax, grid_axis="y")
        d = ks[ks["style"] == style]
        free = d[~d["is_defect_with_threshold"]]
        uncon = d[~d["is_defect_with_threshold"]]
        ax.hist(free["ks_generated_vs_style"], bins=22, alpha=0.80,
                color=ps.OKABE_ITO[0], label="有约束生成")
        if "ks_generated_unconstrained_vs_style" in uncon.columns:
            ax.hist(uncon["ks_generated_unconstrained_vs_style"], bins=22, alpha=0.55,
                    color=ps.OKABE_ITO[1], label="无约束生成")
        row = summary[summary["style"] == style]
        if len(row):
            ax.axvline(float(row.iloc[0]["ks_median"]), color=ps.INK2, ls="-", lw=0.9,
                       label="KS 中位数")
            ax.axvline(float(row.iloc[0]["ks_null_median"]), color=ps.DEFECT, ls="--",
                       lw=0.9, label="KS 原假设中位数")
        ps.simple_title(ax, style_label(style), size=8.0)
        ax.set_xlabel("逐化合物 KS 距离", fontsize=7.0)
        ax.tick_params(labelsize=5.5)
        if not handles:
            handles = ax.get_legend_handles_labels()[0]
    axes[0, 0].set_ylabel("化合物数", fontsize=7.5)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=6.5)

    _emit(fig, out_path.parent, out_path.stem, "KS 距离分布",
          "图 6  KS 距离分布。三种风格下逐化合物 KS 距离直方图（参照系为该风格真实样本，"
          "已剔除 3 个受约束列）；深灰实线为有约束生成的 KS 中位数，红色虚线为两样本 KS "
          "原假设中位数。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 7


def fig_sensory_radar(out_path: Path, records: list,
                      styles: Optional[List[str]] = None) -> None:
    """感官画像雷达图：真实 vs 有约束生成 vs 无约束生成（标准化尺度）。

    依赖问题 1 的 q1_predict 产物；缺失时跳过。
    """
    styles = styles or TARGET_STYLES
    profiles = {}
    for style in styles:
        p_c = RUN_DIR / f"sensory_profile_{slug(style)}.csv"
        p_u = RUN_DIR / f"sensory_profile_{slug(style)}_unconstrained.csv"
        if not p_c.is_file():
            continue
        prof = pd.read_csv(p_c, encoding="utf-8-sig")
        dims = [c for c in prof.columns
                if c not in ("beer", "tasting_category_fine", "variant")]
        profiles[style] = {
            "gen": prof[dims].mean().to_numpy(dtype=float),
            "dims": dims,
            "uncon": (pd.read_csv(p_u, encoding="utf-8-sig")[dims].mean().to_numpy(dtype=float)
                      if p_u.is_file() else None),
        }
    if not profiles:
        print("[跳过] 无感官画像产物（Q1 未就绪）")
        return
    raw = dc.load_raw()
    real = dc.load_targets(raw)
    styles_ser = dc.load_styles(raw)
    fig, axes = plt.subplots(len(profiles), 1, figsize=(ps.FIG_WIDTH_IN, 3.0 * len(profiles)),
                             constrained_layout=True, subplot_kw=dict(polar=True),
                             squeeze=False)
    for ax, (style, prof) in zip(axes.ravel(), profiles.items()):
        dims = prof["dims"]
        style_real = real.loc[(styles_ser == style).to_numpy(), dims].mean().to_numpy(dtype=float)
        # 取该风格判别性最强的 10 个维度（相对全局均值偏差最大）
        glob = real[dims].mean().to_numpy(dtype=float)
        order = np.argsort(-np.abs(style_real - glob))[:10]
        labels = [ps.dim_cn(dims[i]) for i in order]
        ang = np.linspace(0, 2 * np.pi, len(order), endpoint=False).tolist()
        ang += ang[:1]
        series = [(style_real[order], ps.OKABE_ITO[0], "真实样本"),
                  (prof["gen"][order], ps.OKABE_ITO[2], "有约束生成"),
                  (prof["uncon"][order] if prof["uncon"] is not None else None,
                   ps.OKABE_ITO[1], "无约束生成")]
        for vec, color, name in series:
            if vec is None:
                continue
            v = list(vec) + [vec[0]]
            ax.plot(ang, v, color=color, lw=1.0, label=name)
            ax.fill(ang, v, color=color, alpha=0.10)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.tick_params(axis="y", labelsize=5.0, colors=ps.MUTED)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        _tidy(ax)
        ps.simple_title(ax, style_label(style), size=8.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, frameon=False, fontsize=6.5)

    _emit(fig, out_path.parent, out_path.stem, "感官画像",
          "图 7  感官画像。三种风格下真实样本与有约束/无约束生成的 50 维标准化感官画像雷达图，"
          "每图取该风格判别性最强的 10 个维度（A_ 前缀=香气维度，F_ 前缀=味道维度）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 入口


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="问题3 出图（统一 log10 mg/L 口径）")
    parser.add_argument("--styles", type=str, default=",".join(TARGET_STYLES))
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ps.apply_style()
    data = load_style_frames()
    data["defect_thr"] = {s["column"]: s["log10_threshold"]
                          for s in q3.defect_specs(dc.defect_table())
                          if s["has_threshold"]}
    records: list = []

    fig_style_heatmaps(data, FIG_DIR / "fig1_style_heatmaps.png", records)
    fig_marginal_distributions(data, FIG_DIR / "fig2_marginal_distributions.png", records)
    fig_parallel_coordinates(data, FIG_DIR / "fig3_parallel_coordinates_tripel.png", records)
    fig_transfer_radar(FIG_DIR / "fig4_transfer_radar.png", records)
    fig_defect_boxplots(data, FIG_DIR / "fig5_defect_boxplots.png", records)
    fig_ks_distribution(FIG_DIR / "fig6_ks_distribution.png", records)
    fig_sensory_radar(FIG_DIR / "fig7_sensory_profile_radar.png", records)
    _write_manifest(FIG_DIR, records)
    print(f"[落盘] {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
