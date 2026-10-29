# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：
#   1) 中文字体使用 Microsoft YaHei（已在 data/processed/cjk_probe.png 上验证可用），
#      统一经 scripts/plot_style.py 的 apply_style() 写入（含 axes.unicode_minus=False），
#      出图后逐张打开检查无「豆腐块」与负号乱码；
#   2) 图 1 的置信区间直接取 bootstrap_macro_rmse.csv 的 2.5%/97.5% 分位，
#      并画出 global_mean 基线位置，图注中写明「区间重叠即不可宣称显著更优」；
#   3) 图 3 的混淆矩阵按「观测等级 × 预测等级」计数，诊断线为对角线；
#   4) 图 4 的 0 分锚点画在感知阈值处、6 分锚点画在训练集 P95 浓度处，
#      散点为 250 个样本的载体浓度与锚定分数，可直接核对两锚点的位置关系；
#   5) 已逐张开图核对风格一致性：4 张图在 plot_style 统一风格下重出并逐一目检
#      （中文字形、负号、图内标题 ≤12 字、坐标标签无重叠、PNG/PDF 成对落盘）。
"""问题一 · 图表生成（中文出图）。

用法：
    python scripts/q1_figures.py --final-dir data/processed/q1_final \
        --runs-dir data/processed/q1_runs --out-dir data/processed/q1_figures

风格约定（唯一权威来源 = scripts/plot_style.py）：
    * 字体 / 字号 / 底色 / 边框 / 网格 / 配色全部由 ``plot_style.apply_style()`` 写入，
      本脚本不再自行设置 rcParams、也不再自定义配色；
    * 图内只保留 ≤12 字的极简标题，方法、单位、样本量、阈值等一律移入
      ``figures_manifest.json`` 的 caption（供论文题注使用）；
    * 每张图同时输出 PNG(300dpi) 与 PDF(矢量)，文件名（basename）保持不变。
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_style as ps  # noqa: E402

#: 顺序色标（频次/强度）：直接取 plot_style 的契约配色，不自定义颜色
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ_CMAP = LinearSegmentedColormap.from_list("ps_sequential", ps.SEQUENTIAL)

FAMILY_ORDER = ["aroma", "flavour", "mouthfeel", "fault_special", "overall"]
FAMILY_CN = {"aroma": "香气", "flavour": "滋味", "mouthfeel": "口感",
             "fault_special": "缺陷·特殊", "overall": "总体"}


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
    print(f"  → {paths[0].name} + {paths[1].name}")


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
    print(f"  → {path.name}")


# --------------------------------------------------------------------------- 图 1


def fig_model_comparison(summary: pd.DataFrame, bootstrap: pd.DataFrame,
                         out_dir: Path, records: list) -> None:
    """图 1：模型 Macro-RMSE 点图 + bootstrap 95% 置信区间。"""
    merged = summary.copy()
    for column in ("bootstrap_ci_lo", "bootstrap_ci_hi"):
        if column not in merged.columns:
            merged = merged.merge(bootstrap[["model", column]], on="model", how="left")
    merged = merged.sort_values("Macro_RMSE").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, 3.4), constrained_layout=True)
    ps.tidy(ax, grid_axis="x")
    y = np.arange(len(merged))[::-1]
    best = merged.iloc[0]
    for i, row in merged.iterrows():
        color = ps.OKABE_ITO[0] if row["model"] == best["model"] else ps.OKABE_ITO[1]
        if row["model"] == "global_mean":
            color = ps.MUTED
        ax.hlines(y[i], row["bootstrap_ci_lo"], row["bootstrap_ci_hi"],
                  color=color, linewidth=2.6, alpha=0.45, zorder=2)
        ax.scatter([row["Macro_RMSE"]], [y[i]], s=34, color=color, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.text(row["bootstrap_ci_hi"] + 0.003, y[i], f"{row['Macro_RMSE']:.4f}",
                va="center", ha="left", fontsize=6.5, color=ps.INK2)
    baseline_value = float(
        merged.loc[merged["model"] == "global_mean", "Macro_RMSE"].iloc[0])
    ax.axvline(baseline_value, color=ps.INK2, linestyle="--", linewidth=0.8)
    ax.text(baseline_value, len(merged) - 0.35, "全局均值基线", fontsize=6.5,
            color=ps.INK2, ha="center", va="bottom")

    ax.set_yticks(y, list(merged["model"]), fontsize=7)
    ax.set_ylim(-0.7, len(merged) - 0.1)
    lo = float(merged["bootstrap_ci_lo"].min())
    hi = float(merged["bootstrap_ci_hi"].max())
    ax.set_xlim(lo - 0.012, hi + 0.030)
    ax.set_xlabel("留出集 Macro-RMSE（越低越好）", fontsize=8)
    ps.simple_title(ax, "模型对比", size=9.5)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.OKABE_ITO[0],
               markeredgecolor="none", markersize=5, label="Macro-RMSE（最佳）"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.OKABE_ITO[1],
               markeredgecolor="none", markersize=5, label="Macro-RMSE（其他）"),
        Line2D([0], [0], color=ps.OKABE_ITO[1], linewidth=2.6, alpha=0.45,
               label="95% bootstrap 置信区间"),
        Line2D([0], [0], color=ps.INK2, linestyle="--", linewidth=0.8,
               label="全局均值基线"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False, fontsize=6.5, handlelength=1.6,
              columnspacing=1.0)
    _emit(fig, out_dir, "01_model_comparison_bootstrap_ci", "模型对比",
          "图 1  模型对比。留出集 Macro-RMSE 与 2000 次配对 bootstrap 的 95% 置信区间"
          "（区间重叠即不可宣称某模型显著更优），竖虚线为全局均值基线。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 2


def fig_dimension_facets(continuous: pd.DataFrame, model: str,
                         out_dir: Path, records: list) -> None:
    """图 2：逐维可预测性分面图（按五个报告族分面，画 Spearman 与 R²）。"""
    block = continuous.loc[continuous["model"] == model].copy()
    families = [f for f in FAMILY_ORDER if (block["family"] == f).any()]
    ncols = 2
    nrows = int(np.ceil((len(families) + 1) / ncols))  # +1 格放共享图例
    fig, axes = plt.subplots(nrows, ncols, figsize=(ps.FIG_WIDTH_IN, 3.0 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = list(axes.ravel())

    for ax, family in zip(flat, families):
        sub = block.loc[block["family"] == family].sort_values("Spearman")
        ps.tidy(ax, grid_axis="x")
        y = np.arange(len(sub))
        ax.barh(y, sub["Spearman"], color=ps.OKABE_ITO[2], height=0.62, alpha=0.9)
        ax.scatter(sub["R2"], y, s=11, color=ps.OKABE_ITO[1], zorder=3)
        ax.set_yticks(y, [ps.dim_cn(v) for v in sub["label"]], fontsize=6)
        ax.axvline(0, color=ps.BASELINE, linewidth=0.8)
        # 维度很少的面板（总体评价）若各自缩放，柱子会撑满整格 → 给 x 轴一个最小跨度
        lo_v = float(min(sub["Spearman"].min(), sub["R2"].min(), 0.0))
        hi_v = float(max(sub["Spearman"].max(), sub["R2"].max()))
        if hi_v - lo_v < 0.5:
            hi_v = lo_v + 0.5
        ax.set_xlim(lo_v - 0.06 * (hi_v - lo_v), hi_v + 0.06 * (hi_v - lo_v))
        ps.simple_title(ax, FAMILY_CN.get(family, family), size=8.5)

    legend_ax = flat[len(families)]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[Patch(facecolor=ps.OKABE_ITO[2], label="Spearman"),
                 Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.OKABE_ITO[1],
                        markeredgecolor="none", markersize=4, label="R²")],
        loc="center", frameon=False, fontsize=7.5, handlelength=1.4)

    for idx, ax in enumerate(flat):
        if idx == len(families):
            continue
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Spearman（柱）/ R²（点）", fontsize=7.5)

    _emit(fig, out_dir, "02_dimension_predictability_facets", "逐维可预测性",
          "图 2  逐维可预测性。留出集 75 个样本，按香气 / 滋味 / 口感 / 缺陷·特殊 / 总体"
          "五个报告族分面；柱为测试集 Spearman，点为 R²，竖直零线为无相关基准。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 3


def fig_two_track_grades(predictions: pd.DataFrame, model: str, final_dir: Path,
                         runs_dir: Path, out_dir: Path, records: list) -> None:
    """图 3：两轨分级对照（左：代理分级混淆矩阵；右：两轨逐维误差对照）。"""
    block = predictions.loc[predictions["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(ps.FIG_WIDTH_IN, 3.4),
                             constrained_layout=True)

    # ---- 左：混淆矩阵 ----
    ax = axes[0]
    ps.tidy(ax, grid_axis="x")
    ax.grid(False)
    matrix = pd.crosstab(block["observed_proxy_grade"], block["predicted_proxy_grade"])
    matrix = matrix.reindex(index=range(7), columns=range(7), fill_value=0).to_numpy()
    im = ax.imshow(matrix, cmap=SEQ_CMAP, origin="lower", aspect="auto", vmin=0)
    for i in range(7):
        for j in range(7):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else ps.INK)
    ax.set_xticks(range(7), [str(i) for i in range(7)], fontsize=6.5)
    ax.set_yticks(range(7), [str(i) for i in range(7)], fontsize=6.5)
    ax.set_xlabel("预测代理等级", fontsize=7.5)
    ax.set_ylabel("观测代理等级", fontsize=7.5)
    ps.simple_title(ax, "等级混淆矩阵", size=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("样本数", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    # ---- 右：两轨逐维误差 ----
    ax = axes[1]
    ps.tidy(ax, grid_axis="x")
    anchored = pd.read_csv(final_dir / "absolute_anchor_params.csv")
    valid_dims = anchored.loc[anchored["valid"].astype(bool), "dimension"].tolist()
    proxy_mae = {}
    detail_path = Path(runs_dir) / model / "proxy_grade_metrics_by_label.csv"
    if detail_path.is_file():
        detail = pd.read_csv(detail_path)
        detail = detail.loc[detail["model"] == model]
        proxy_mae = dict(zip(detail["label"], detail["grade_MAE"]))
    abs_mae = {}
    if valid_dims:
        preds = pd.read_csv(final_dir / "test_predictions.csv.gz")
        preds = preds.loc[preds["model"] == model]
        for dim in valid_dims:
            row = anchored.loc[anchored["dimension"] == dim].iloc[0]
            a, b = float(row["a"]), float(row["b"])
            true_abs = np.clip(
                a + b * preds.loc[preds["label"] == dim, "observed_continuous"].to_numpy(), 0, 6)
            pred_abs = np.clip(
                a + b * preds.loc[preds["label"] == dim, "predicted_continuous"].to_numpy(), 0, 6)
            abs_mae[dim] = float(np.mean(np.abs(true_abs - pred_abs)))
    dims = [d for d in proxy_mae if d in abs_mae]
    if dims:
        y = np.arange(len(dims))
        ax.barh(y - 0.2, [proxy_mae[d] for d in dims], height=0.36,
                color=ps.OKABE_ITO[0], label="主轨 等级 MAE")
        ax.barh(y + 0.2, [abs_mae[d] for d in dims], height=0.36,
                color=ps.OKABE_ITO[1], label="副轨 绝对分 MAE")
        ax.set_yticks(y, [ps.dim_cn(d) for d in dims], fontsize=7)
        ax.set_xlabel("留出集误差", fontsize=7.5)
        ax.legend(fontsize=6.5, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.3)
    else:
        ax.text(0.5, 0.5, "无通过有效性护栏的维度", ha="center", va="center",
                fontsize=8, color=ps.MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
    ps.simple_title(ax, "两轨分级误差", size=8.5)

    _emit(fig, out_dir, "03_two_track_grade_comparison", "两轨分级对照",
          "图 3  两轨分级对照。左为代理分级 0–6 级的混淆矩阵（行=观测等级、列=预测等级），"
          "右为主轨等级 MAE 与副轨绝对分 MAE 的逐维对照（仅统计通过有效性护栏的维度）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 图 4


def fig_anchor_calibration(anchor_params: pd.DataFrame, matrix: pd.DataFrame,
                           scores: pd.DataFrame, out_dir: Path, records: list) -> None:
    """图 4：绝对锚定标定图（载体浓度 → 锚定分数，两锚点标注）。"""
    anchored = anchor_params.loc[anchor_params["valid"].astype(bool)].copy()
    if anchored.empty:
        return
    dims = anchored["dimension"].tolist()
    cols = min(3, len(dims))
    rows = int(np.ceil(len(dims) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(ps.FIG_WIDTH_IN, 3.0 * rows),
                             constrained_layout=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, dim in zip(axes, dims):
        ps.tidy(ax, grid_axis="both")
        row = anchored.loc[anchored["dimension"] == dim].iloc[0]
        carrier = row["carrier"]
        conc = matrix[carrier].to_numpy(dtype=float)
        abs_col = f"{dim}_abs"
        values = scores[abs_col].to_numpy(dtype=float) if abs_col in scores.columns else None
        if values is None:
            ax.set_visible(False)
            continue
        is_threshold = row["anchor_type"] == "threshold_anchored"
        color = ps.OKABE_ITO[0] if is_threshold else ps.OKABE_ITO[3]
        ax.scatter(conc, values, s=6, alpha=0.40, color=color, linewidths=0)
        x0 = float(row["anchor0_log10_conc"])
        x6 = float(row["anchor6_log10_conc"])
        ax.axvline(x0, color=ps.OKABE_ITO[1] if is_threshold else ps.MUTED,
                   linestyle="--", linewidth=0.9)
        ax.axvline(x6, color=ps.INK2, linestyle="--", linewidth=0.9)
        ax.set_ylim(-0.3, 6.4)
        ps.simple_title(ax, ps.dim_cn(dim), size=8.5)
        ax.set_xlabel(f"载体 {carrier} 的 log10 浓度", fontsize=6.5)
        ax.set_ylabel("锚定绝对分", fontsize=7)
    for ax in axes[len(dims):]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.OKABE_ITO[0],
               markeredgecolor="none", markersize=4.5, label="阈值锚定维度"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.OKABE_ITO[3],
               markeredgecolor="none", markersize=4.5, label="相对约束替代"),
        Line2D([0], [0], color=ps.OKABE_ITO[1], linestyle="--", linewidth=0.9,
               label="0 分锚点（阈值）"),
        Line2D([0], [0], color=ps.MUTED, linestyle="--", linewidth=0.9,
               label="0 分锚点（P05）"),
        Line2D([0], [0], color=ps.INK2, linestyle="--", linewidth=0.9,
               label="6 分锚点（P95）"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=6.5, handlelength=1.4)
    _emit(fig, out_dir, "04_absolute_anchor_calibration", "绝对锚定标定",
          "图 4  绝对锚定标定。仅显示通过有效性护栏的维度；横轴为载体化合物的 log10 浓度，"
          "纵轴为锚定绝对分；橙色虚线为 0 分锚点（感官阈值），灰色虚线为 0 分锚点（训练集 P05），"
          "深灰虚线为 6 分锚点（训练集 P95）。",
          ps.FIG_WIDTH_IN, records)


# --------------------------------------------------------------------------- 入口


def make_all(final_dir: Path, runs_dir: Path, out_dir: Path) -> None:
    """读取交付物与模型运行产物，生成四张图（并写 manifest）。"""
    final_dir = Path(final_dir)
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.apply_style()

    summary = pd.read_csv(final_dir / "model_summary.csv")
    bootstrap = pd.read_csv(final_dir / "bootstrap_macro_rmse.csv")
    scores = pd.read_csv(final_dir / "sensory_scores_0_6.csv")
    anchor_params = pd.read_csv(final_dir / "absolute_anchor_params.csv")
    best = summary.loc[summary["selected_as_final_model"].astype(bool), "model"]
    best = str(best.iloc[0]) if len(best) else str(summary.sort_values("Macro_RMSE").iloc[0]["model"])

    records: list = []
    fig_model_comparison(summary, bootstrap, out_dir, records)

    continuous = pd.read_csv(runs_dir / best / "continuous_metrics_by_label.csv")
    fig_dimension_facets(continuous, best, out_dir, records)

    predictions = pd.read_csv(final_dir / "test_predictions.csv.gz")
    fig_two_track_grades(predictions, best, final_dir, runs_dir, out_dir, records)

    matrix = pd.read_csv(REPO_ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    matrix.index = matrix.index.astype(int)
    fig_anchor_calibration(anchor_params, matrix, scores, out_dir, records)

    _write_manifest(out_dir, records)


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 · 图表生成")
    ap.add_argument("--final-dir", default="data/processed/q1_final")
    ap.add_argument("--runs-dir", default="data/processed/q1_runs")
    ap.add_argument("--out-dir", default="data/processed/q1_figures")
    args = ap.parse_args()
    final_dir = (REPO_ROOT / args.final_dir).resolve()
    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    make_all(final_dir, runs_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
