# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 中文字体使用 Microsoft YaHei（已确认系统可用），axes.unicode_minus=False 防负号乱码；
#      出图后逐张目视核对标签未出现豆腐块。
#   2) 所有图只读 data/processed/q3_runs/ 下的产物与统一矩阵，不重复跑生成/迁移逻辑，
#      保证图与 CSV 产物严格一致。
#   3) 缺陷类图同时给出「真实 / 无约束生成 / 有约束生成 / 迁移后」四组，
#      以便直观展示约束带来的边际压缩（Tripel DMS/Diacetyl 被系统性压低）。
"""问题 3 出图：风格热图、边际分布对比、平行坐标、迁移雷达图、缺陷箱线图、KS 分布。

产出目录：``data/processed/q3_figures/``
"""
from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402
import q3_copula as q3  # noqa: E402

RUN_DIR = REPO_ROOT / "data" / "processed" / "q3_runs"
GEN_DIR = RUN_DIR / "generation"
FIG_DIR = REPO_ROOT / "data" / "processed" / "q3_figures"

CJK_FONT = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [CJK_FONT, "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

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


def slug(style: str) -> str:
    return (style.replace(" ", "_").replace("/", "-")).strip("_")


def style_label(name: str) -> str:
    return {"Tripel": "三倍酒 Tripel", "Kriek": "樱桃啤酒 Kriek",
            "Lambic": "兰比克 Lambic", "Blond": "金黄 Blond",
            "Strong ale": "烈性艾尔 Strong ale", "Hoppy": "啤酒花型 Hoppy",
            "Dubbel": "Dubbel", "Saison": "赛松 Saison"}.get(name, name)


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


def fig_style_heatmaps(data: dict, out_path: Path, top_n: int = 18) -> None:
    """风格热图：真实 / 无约束生成 / 有约束生成 的中位 log10(mg/L) 横向对照。"""
    styles = [s for s in TARGET_STYLES if s in data["gen"]]
    if not styles:
        print("[跳过] 无生成谱，跳过热图")
        return
    matrix, style_ser = data["matrix"], data["styles"]
    cols = [c for c in HEATMAP_COMPOUNDS if c in matrix.columns][:top_n]
    fig, axes = plt.subplots(len(styles), 3, figsize=(15.5, 3.6 * len(styles)),
                             squeeze=False)
    im = None
    for i, style in enumerate(styles):
        real = matrix.loc[(style_ser == style).to_numpy()]
        uncon = data["uncon"].get(style)
        con = data["gen"][style]
        panels = [("真实样本", real[cols]),
                  ("无约束生成", uncon[cols] if uncon is not None else None),
                  ("有约束生成（零超标）", con[cols])]
        for j, (title, frame) in enumerate(panels):
            ax = axes[i, j]
            if frame is None:
                ax.axis("off")
                continue
            mat = frame.median().to_numpy(dtype=float).reshape(1, -1)
            im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=-3.0, vmax=1.5)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([PRETTY.get(c, c) for c in cols], rotation=55,
                               ha="right", fontsize=7)
            ax.set_yticks([0])
            ax.set_yticklabels([style_label(style)] if j == 0 else [""], fontsize=9)
            ax.set_title(f"{title}", fontsize=10.5)
            for k, v in enumerate(mat[0]):
                ax.text(k, 0, f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                        color="black")
    fig.subplots_adjust(left=0.075, right=0.90, top=0.90, bottom=0.06, hspace=0.85,
                        wspace=0.12)
    if im is not None:
        cax = fig.add_axes([0.92, 0.08, 0.012, 0.84])
        fig.colorbar(im, cax=cax, label="中位 log10(mg/L)")
    fig.suptitle("图1  风格化学画像热图：真实 vs 无约束生成 vs 有约束生成", fontsize=13)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


# --------------------------------------------------------------------------- 图 2


def fig_marginal_distributions(data: dict, out_path: Path) -> None:
    """边际分布对比：真实 / 无约束生成 / 有约束生成（箱线 + 散点）。"""
    styles = [s for s in TARGET_STYLES if s in data["gen"]]
    cols = [c for c in KEY_COMPOUNDS if c in data["matrix"].columns]
    fig, axes = plt.subplots(len(styles), len(cols), figsize=(3.0 * len(cols),
                                                            3.0 * len(styles)),
                             squeeze=False)
    matrix, style_ser = data["matrix"], data["styles"]
    for i, style in enumerate(styles):
        real = matrix.loc[(style_ser == style).to_numpy()]
        uncon = data["uncon"].get(style)
        con = data["gen"][style]
        for j, col in enumerate(cols):
            ax = axes[i, j]
            series = [("真实", real[col].to_numpy()), ("无约束生成", uncon[col].to_numpy()),
                      ("有约束生成", con[col].to_numpy())]
            data_series = [s[1] for s in series if s[1] is not None]
            bp = ax.boxplot(data_series, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black"),
                            flierprops=dict(marker=".", markersize=2, alpha=0.4))
            colors = ["#9ecae1", "#fdae6b", "#74c476"]
            for k, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[k % len(colors)])
            ax.set_xticks(range(1, len(data_series) + 1))
            ax.set_xticklabels(["真实", "无约束", "有约束"], fontsize=8)
            # 缺陷阈值线
            thr = data["defect_thr"].get(col)
            if thr is not None:
                ax.axhline(thr, color="crimson", ls="--", lw=1)
            if i == 0:
                ax.set_title(PRETTY.get(col, col), fontsize=9)
            if j == 0:
                ax.set_ylabel(style_label(style), fontsize=9)
            ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color="w", marker="s", markersize=9,
                      markerfacecolor=c, label=l) for l, c in
               zip(["真实样本", "无约束生成", "有约束生成"],
                   ["#9ecae1", "#fdae6b", "#74c476"])]
    handles.append(Line2D([0], [0], color="crimson", ls="--", lw=1, label="感官阈值"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("图2  边际分布对比（统一 log10(mg/L) 口径；红色虚线=感官阈值）", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


# --------------------------------------------------------------------------- 图 3


def fig_parallel_coordinates(data: dict, out_path: Path, style: str = "Tripel",
                              n_show: int = 60) -> None:
    """平行坐标：真实样本 vs 无约束生成 vs 有约束生成。"""
    if style not in data["gen"]:
        print("[跳过] 无该风格生成谱")
        return
    matrix, style_ser = data["matrix"], data["styles"]
    real = matrix.loc[(style_ser == style).to_numpy()]
    uncon = data["uncon"].get(style)
    con = data["gen"][style]
    cols = [c for c in KEY_COMPOUNDS if c in matrix.columns]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cols))
    groups = [("真实样本", real[cols], "#1f77b4", 0.9),
              ("无约束生成", uncon[cols] if uncon is not None else None, "#ff7f0e", 0.9),
              ("有约束生成", con[cols], "#2ca02c", 0.9)]
    for label, frame, color, alpha in groups:
        if frame is None:
            continue
        sub = frame.head(n_show)
        for _, row in sub.iterrows():
            ax.plot(x, row.to_numpy(dtype=float), color=color, alpha=0.18, lw=0.8)
        ax.plot([], [], color=color, lw=2, label=f"{label}（前 {min(n_show, len(sub))} 条）")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(c, c) for c in cols],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("log10(mg/L)")
    ax.set_title(f"图3  {style_label(style)} 平行坐标：真实 vs 生成", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


# --------------------------------------------------------------------------- 图 4


def fig_transfer_radar(out_path: Path) -> None:
    """迁移前后雷达图：四项缺陷化合物的相对水平。"""
    path = RUN_DIR / "transfer_pils_to_tripel_defects.csv"
    if not path.is_file():
        print("[跳过] 无迁移缺陷表")
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    keys = ["dms", "diacetyl", "4vg", "t2n"]
    labels = ["DMS", "Diacetyl", "4-VG", "trans-2-nonenal"]
    before = df[df.defect_key == keys[0]]["value_before_log10_mgL"].to_numpy()
    after = df[df.defect_key == keys[0]]["value_after_log10_mgL"].to_numpy()
    b_vec, a_vec = [], []
    for k in keys:
        d = df[df.defect_key == k]
        b_vec.append(float(d["value_before_log10_mgL"].mean()))
        a_vec.append(float(d["value_after_log10_mgL"].mean()))
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, polar=True)
    for vec, color, name in ((b_vec, "#1f77b4", "迁移前（真实 Pils/Lager）"),
                             (a_vec, "#d62728", "迁移后（Tripel 边际 + 约束）")):
        v = vec + vec[:1]
        ax.plot(ang, v, color=color, lw=2, label=name)
        ax.fill(ang, v, color=color, alpha=0.15)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    # 数值标注（前→后），避免读者依赖半径判读（log10 值为负，越靠内越低）
    for i, k in enumerate(keys):
        txt = f"{b_vec[i]:.2f} → {a_vec[i]:.2f}"
        ax.text(ang[i], min(b_vec[i], a_vec[i]) - 0.12, txt, ha="center",
                va="top", fontsize=8.5, color="#333333")
    ax.set_title("图4  Pils/Lager → Tripel 迁移：四项缺陷均值对比\n"
                 "(log10 mg/L；数值越低越好，越靠内越低)", fontsize=12, pad=24)
    ax.legend(loc="lower right", bbox_to_anchor=(1.34, -0.06), fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


# --------------------------------------------------------------------------- 图 5


def fig_defect_boxplots(data: dict, out_path: Path) -> None:
    """缺陷化合物箱线图：真实全体 / 各风格有约束 / 无约束 / 迁移后。"""
    specs = [("dms", "DMS", "#1f77b4"), ("diacetyl", "Diacetyl", "#ff7f0e"),
             ("4vg", "4-VG", "#2ca02c"), ("t2n", "trans-2-nonenal", "#9467bd")]
    matrix = data["matrix"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    transfer_path = RUN_DIR / "transfer_pils_to_tripel_defects.csv"
    for ax, (key, label, color) in zip(axes, specs):
        col = next(s["column"] for s in q3.defect_specs(dc.defect_table())
                   if s["key"] == key)
        series, names = [], []
        series.append(matrix[col].to_numpy())
        names.append("真实全体(250)")
        for style in ["Tripel", "Kriek", "Lambic"]:
            frame = data["gen"].get(style)
            if frame is not None:
                series.append(frame[col].to_numpy())
                names.append(f"{style}\n有约束")
            frame_u = data["uncon"].get(style)
            if frame_u is not None:
                series.append(frame_u[col].to_numpy())
                names.append(f"{style}\n无约束")
        if transfer_path.is_file():
            tdf = pd.read_csv(transfer_path, encoding="utf-8-sig")
            t = tdf[tdf.defect_key == key]
            if len(t):
                series.append(t["value_after_log10_mgL"].to_numpy())
                names.append("迁移后\nPils→Tripel")
        bp = ax.boxplot(series, patch_artist=True, widths=0.65, showfliers=False,
                        medianprops=dict(color="black"))
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35 + 0.1 * (k % 4))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(label, fontsize=11)
        thr = data["defect_thr"].get(col)
        if thr is not None:
            ax.axhline(thr, color="crimson", ls="--", lw=1.2)
        ax.tick_params(labelsize=7)
    fig.suptitle("图5  缺陷化合物分布：真实 / 有约束生成 / 无约束生成 / 迁移后"
                 "（红色虚线=感官阈值；t2n 无阈值）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


# --------------------------------------------------------------------------- 图 6


def fig_ks_distribution(out_path: Path) -> None:
    """KS 距离分布：有约束 vs 无约束，附两样本 KS 原假设中位数参考线。"""
    path = RUN_DIR / "ks_per_compound.csv"
    summary_path = RUN_DIR / "generation_summary.csv"
    if not path.is_file():
        print("[跳过] 无 ks_per_compound.csv")
        return
    ks = pd.read_csv(path, encoding="utf-8-sig")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    styles = [s for s in TARGET_STYLES if s in set(ks["style"])]
    fig, axes = plt.subplots(1, len(styles), figsize=(5.2 * len(styles), 4.2),
                             squeeze=False)
    for ax, style in zip(axes[0], styles):
        d = ks[ks["style"] == style]
        free = d[~d["is_defect_with_threshold"]]
        uncon = d[~d["is_defect_with_threshold"]]
        ax.hist(free["ks_generated_vs_style"], bins=28, alpha=0.72, color="#1f77b4",
                label="有约束生成（剔除3个受约束列）")
        if "ks_generated_unconstrained_vs_style" in uncon.columns:
            ax.hist(uncon["ks_generated_unconstrained_vs_style"], bins=28, alpha=0.55,
                    color="#fdae6b", label="无约束生成（同口径）")
        row = summary[summary["style"] == style]
        if len(row):
            ax.axvline(float(row.iloc[0]["ks_median"]), color="navy", ls="-", lw=1.6,
                       label=f"KS 中位数={float(row.iloc[0]['ks_median']):.3f}")
            ax.axvline(float(row.iloc[0]["ks_null_median"]), color="crimson", ls="--",
                       lw=1.4,
                       label=f"KS 原假设中位数={float(row.iloc[0]['ks_null_median']):.3f}")
        ax.set_title(style_label(style), fontsize=11)
        ax.set_xlabel("逐化合物 KS 距离")
        ax.set_ylabel("化合物数")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("图6  边际 KS 距离分布：生成器还原度 vs 约束压缩\n"
                 "(参照系为该风格真实样本，样本量小故原假设中位数偏高)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


def fig_sensory_radar(out_path: Path, styles: Optional[List[str]] = None) -> None:
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
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.2 * len(profiles), 5.4),
                             subplot_kw=dict(polar=True), squeeze=False)
    for ax, (style, prof) in zip(axes[0], profiles.items()):
        dims = prof["dims"]
        style_real = real.loc[(styles_ser == style).to_numpy(), dims].mean().to_numpy(dtype=float)
        # 取该风格判别性最强的 10 个维度（相对全局均值偏差最大）
        glob = real[dims].mean().to_numpy(dtype=float)
        order = np.argsort(-np.abs(style_real - glob))[:10]
        labels = [dims[i] for i in order]
        ang = np.linspace(0, 2 * np.pi, len(order), endpoint=False).tolist()
        ang += ang[:1]
        series = [(style_real[order], "#1f77b4", "真实样本"),
                  (prof["gen"][order], "#2ca02c", "有约束生成"),
                  (prof["uncon"][order] if prof["uncon"] is not None else None,
                   "#ff7f0e", "无约束生成")]
        for vec, color, name in series:
            if vec is None:
                continue
            v = list(vec) + [vec[0]]
            ax.plot(ang, v, color=color, lw=1.8, label=name)
            ax.fill(ang, v, color=color, alpha=0.10)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(style_label(style), fontsize=11, pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), fontsize=8)
    fig.suptitle("图7  感官画像（50 维标准化尺度，示出风格判别性最强的 10 维；"
                 "A_ 前缀=香气维度，F_ 前缀=口感维度）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.5)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[图] {out_path.name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="问题3 出图（统一 log10 mg/L 口径）")
    parser.add_argument("--styles", type=str, default=",".join(TARGET_STYLES))
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_style_frames()
    data["defect_thr"] = {s["column"]: s["log10_threshold"]
                          for s in q3.defect_specs(dc.defect_table())
                          if s["has_threshold"]}

    fig_style_heatmaps(data, FIG_DIR / "fig1_style_heatmaps.png")
    fig_marginal_distributions(data, FIG_DIR / "fig2_marginal_distributions.png")
    fig_parallel_coordinates(data, FIG_DIR / "fig3_parallel_coordinates_tripel.png")
    fig_transfer_radar(FIG_DIR / "fig4_transfer_radar.png")
    fig_defect_boxplots(data, FIG_DIR / "fig5_defect_boxplots.png")
    fig_ks_distribution(FIG_DIR / "fig6_ks_distribution.png")
    fig_sensory_radar(FIG_DIR / "fig7_sensory_profile_radar.png")
    print(f"[落盘] {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
