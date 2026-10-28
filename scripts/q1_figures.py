# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 中文字体已在 data/processed/cjk_probe.png 上验证可用，本脚本统一使用
#      Microsoft YaHei + axes.unicode_minus=False，出图后逐张打开检查无「豆腐块」与负号乱码；
#   2) 图 1 的置信区间直接取 bootstrap_macro_rmse.csv 的 2.5%/97.5% 分位，
#      并画出 global_mean 基线位置，图注中写明「区间重叠即不可宣称显著更优」；
#   3) 图 3 的混淆矩阵按「观测等级 × 预测等级」计数，诊断线为对角线；
#   4) 图 4 的 0 分锚点画在感知阈值处、6 分锚点画在训练集 P95 浓度处，
#      散点为 250 个样本的载体浓度与锚定分数，可直接核对两锚点的位置关系。
"""问题一 · 图表生成（中文出图）。

用法：
    python scripts/q1_figures.py --final-dir data/processed/q1_final \
        --runs-dir data/processed/q1_runs --out-dir data/processed/q1_figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                              "#898781", "#e1e0d9", "#c3c2b7")
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_ORDER = ["aroma", "flavour", "mouthfeel", "fault_special", "overall"]
FAMILY_CN = {"aroma": "香气 Aroma", "flavour": "滋味 Flavour", "mouthfeel": "口感 Mouthfeel",
             "fault_special": "缺陷/特殊", "overall": "总体 Overall"}


def _style(ax):
    """统一图表风格：浅底、去顶右框、淡刻度。"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def _figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"  → {out_dir / name}")


# --------------------------------------------------------------------------- 图


def fig_model_comparison(summary: pd.DataFrame, bootstrap: pd.DataFrame, out_dir: Path) -> None:
    """图 1：模型 Macro-RMSE 点图 + bootstrap 95% 置信区间。"""
    merged = summary.copy()
    for column in ("bootstrap_ci_lo", "bootstrap_ci_hi"):
        if column not in merged.columns:
            merged = merged.merge(bootstrap[["model", column]], on="model", how="left")
    merged = merged.sort_values("Macro_RMSE").reset_index(drop=True)
    fig, ax = _figure(7.6, 4.6)
    y = np.arange(len(merged))[::-1]
    best = merged.iloc[0]
    for i, row in merged.iterrows():
        color = CATEGORICAL[0] if row["model"] == best["model"] else CATEGORICAL[1]
        if row["model"] == "global_mean":
            color = MUTED
        ax.plot([row["bootstrap_ci_lo"], row["bootstrap_ci_hi"]], [y[i], y[i]],
                color=color, linewidth=3, alpha=0.35, solid_capstyle="round")
        ax.scatter([row["Macro_RMSE"]], [y[i]], s=70, color=color, zorder=3)
        ax.text(row["bootstrap_ci_hi"] + 0.004, y[i], f"{row['Macro_RMSE']:.4f}",
                va="center", fontsize=8, color=INK_2)
    baseline_value = float(merged.loc[merged["model"] == "global_mean", "Macro_RMSE"].iloc[0])
    ax.axvline(baseline_value, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(baseline_value - 0.006, -0.75, "全局均值基线", fontsize=8, color=MUTED,
            ha="right", va="center")
    ax.set_yticks(y, merged["model"], fontsize=9)
    ax.set_xlabel("留出集 Macro-RMSE（越低越好）", fontsize=9)
    ax.set_title("模型对比：Macro-RMSE 与 bootstrap 95% 置信区间\n"
                 "区间重叠即不可宣称某模型显著更优（2000 次配对重采样）",
                 fontsize=10.5, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "01_model_comparison_bootstrap_ci.png")


def fig_dimension_facets(continuous: pd.DataFrame, model: str, out_dir: Path) -> None:
    """图 2：逐维可预测性分面图（按五个报告族分面，画 Spearman 与 R²）。"""
    block = continuous.loc[continuous["model"] == model].copy()
    families = [f for f in FAMILY_ORDER if (block["family"] == f).any()]
    fig, axes = plt.subplots(1, len(families), figsize=(2.5 * len(families) + 1.6, 5.2),
                             constrained_layout=True, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = block.loc[block["family"] == family].sort_values("Spearman")
        _style(ax)
        y = np.arange(len(sub))
        ax.barh(y, sub["Spearman"], color=CATEGORICAL[2], height=0.6, alpha=0.85)
        ax.scatter(sub["R2"], y, s=16, color=CATEGORICAL[1], zorder=3, label="R²")
        ax.set_yticks(y, sub["label"], fontsize=6.5)
        ax.axvline(0, color=BASELINE, linewidth=0.8)
        ax.set_title(f"{FAMILY_CN.get(family, family)}（n={len(sub)}）", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_xlabel("测试集 Spearman（柱）/ R²（点）", fontsize=9)
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"逐维可预测性分面图 · {model}（留出集 75 样本）", fontsize=11, color=INK)
    _save(fig, out_dir, "02_dimension_predictability_facets.png")


def fig_two_track_grades(predictions: pd.DataFrame, model: str, final_dir: Path,
                         runs_dir: Path, out_dir: Path) -> None:
    """图 3：两轨分级对照（左：代理分级混淆矩阵；右：两轨逐维误差对照）。"""
    block = predictions.loc[predictions["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    _style(ax)
    matrix = pd.crosstab(block["observed_proxy_grade"], block["predicted_proxy_grade"])
    matrix = matrix.reindex(index=range(7), columns=range(7), fill_value=0).to_numpy()
    im = ax.imshow(matrix, cmap="Blues", origin="lower", aspect="equal")
    for i in range(7):
        for j in range(7):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else INK)
    ax.set_xticks(range(7), [str(i) for i in range(7)])
    ax.set_yticks(range(7), [str(i) for i in range(7)])
    ax.set_xlabel("预测代理等级", fontsize=9)
    ax.set_ylabel("观测代理等级", fontsize=9)
    ax.set_title("主轨·相对分级：等级混淆矩阵", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axes[1]
    _style(ax)
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
            true_abs = np.clip(a + b * preds.loc[preds["label"] == dim, "observed_continuous"].to_numpy(), 0, 6)
            pred_abs = np.clip(a + b * preds.loc[preds["label"] == dim, "predicted_continuous"].to_numpy(), 0, 6)
            abs_mae[dim] = float(np.mean(np.abs(true_abs - pred_abs)))
    dims = [d for d in proxy_mae if d in abs_mae]
    if dims:
        y = np.arange(len(dims))
        ax.barh(y - 0.2, [proxy_mae[d] for d in dims], height=0.38, color=CATEGORICAL[0], label="主轨 等级 MAE")
        ax.barh(y + 0.2, [abs_mae[d] for d in dims], height=0.38, color=CATEGORICAL[1], label="副轨 绝对分 MAE")
        ax.set_yticks(y, dims, fontsize=7)
        ax.set_xlabel("留出集误差", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "无通过有效性护栏的维度", ha="center", va="center", fontsize=9, color=MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"两轨分级误差对照（仅通过护栏的 {len(dims)} 个维度）", fontsize=10, color=INK)
    fig.suptitle(f"双轨分级对照 · {model}", fontsize=11, color=INK)
    _save(fig, out_dir, "03_two_track_grade_comparison.png")


def fig_anchor_calibration(anchor_params: pd.DataFrame, matrix: pd.DataFrame,
                           scores: pd.DataFrame, out_dir: Path) -> None:
    """图 4：绝对锚定标定图（载体浓度 → 锚定分数，两锚点标注）。"""
    anchored = anchor_params.loc[anchor_params["valid"].astype(bool)].copy()
    if anchored.empty:
        return
    dims = anchored["dimension"].tolist()
    cols = 4
    rows = int(np.ceil(len(dims) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.7 * rows), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, dim in zip(axes, dims):
        _style(ax)
        row = anchored.loc[anchored["dimension"] == dim].iloc[0]
        carrier = row["carrier"]
        conc = matrix[carrier].to_numpy(dtype=float)
        abs_col = f"{dim}_abs"
        values = scores[abs_col].to_numpy(dtype=float) if abs_col in scores.columns else None
        if values is None:
            ax.set_visible(False)
            continue
        color = CATEGORICAL[0] if row["anchor_type"] == "threshold_anchored" else CATEGORICAL[3]
        ax.scatter(conc, values, s=8, alpha=0.45, color=color, linewidths=0)
        x0 = float(row["anchor0_log10_conc"])
        x6 = float(row["anchor6_log10_conc"])
        ax.axvline(x0, color=CATEGORICAL[3], linestyle="--", linewidth=1.1)
        ax.axvline(x6, color=CATEGORICAL[7], linestyle="--", linewidth=1.1)
        ax.text(x0, 6.25, "0 分锚点" + ("（阈值）" if row["anchor_type"] == "threshold_anchored" else "（P05）"),
                fontsize=6.5, color=CATEGORICAL[3], ha="center", va="bottom")
        ax.text(x6, 6.25, "6 分锚点（P95）", fontsize=6.5, color=CATEGORICAL[7], ha="center", va="bottom")
        ax.set_ylim(-0.3, 6.9)
        ax.set_title(f"{dim} · {carrier}", fontsize=8.5, color=INK)
        ax.set_xlabel("载体 log10 浓度", fontsize=7.5)
        ax.set_ylabel("锚定绝对分", fontsize=7.5)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(dims):]:
        ax.set_visible(False)
    fig.suptitle("绝对锚定标定图：化合物浓度 → 绝对 0–6 分\n"
                 "（基于感官阈值的一阶近似，本文提出；仅显示通过有效性护栏的维度，"
                 "蓝=阈值锚定，橙=相对约束替代规则）",
                 fontsize=11, color=INK)
    _save(fig, out_dir, "04_absolute_anchor_calibration.png")


# --------------------------------------------------------------------------- 入口


def make_all(final_dir: Path, runs_dir: Path, out_dir: Path) -> None:
    """读取交付物与模型运行产物，生成四张图。"""
    final_dir = Path(final_dir)
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(final_dir / "model_summary.csv")
    bootstrap = pd.read_csv(final_dir / "bootstrap_macro_rmse.csv")
    scores = pd.read_csv(final_dir / "sensory_scores_0_6.csv")
    anchor_params = pd.read_csv(final_dir / "absolute_anchor_params.csv")
    best = summary.loc[summary["selected_as_final_model"].astype(bool), "model"]
    best = str(best.iloc[0]) if len(best) else str(summary.sort_values("Macro_RMSE").iloc[0]["model"])

    fig_model_comparison(summary, bootstrap, out_dir)

    continuous = pd.read_csv(runs_dir / best / "continuous_metrics_by_label.csv")
    fig_dimension_facets(continuous, best, out_dir)

    predictions = pd.read_csv(final_dir / "test_predictions.csv.gz")
    fig_two_track_grades(predictions, best, final_dir, runs_dir, out_dir)

    matrix = pd.read_csv(REPO_ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    matrix.index = matrix.index.astype(int)
    fig_anchor_calibration(anchor_params, matrix, scores, out_dir)


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
