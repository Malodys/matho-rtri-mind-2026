# -*- coding: utf-8 -*-
# AI 辅助：OpenAI Codex (GPT-5), OpenAI, 2026-10-30
# 人工核验点：图中均值/标准差均从 holdout_seed_metrics.csv 按风格现场聚合；
# 三态顺序固定为真实训练参照、无约束、受约束；横轴从 0 起。
"""绘制 Q3 留出验证三态 energy distance 对照图。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_style import (  # noqa: E402
    ACCENT, BASELINE, DEFECT, FIG_WIDTH_IN, INK2, apply_style, save,
    simple_title, tidy,
)

OUT_DIR = Path(__file__).resolve().parent
SOURCE = OUT_DIR / "holdout_seed_metrics.csv"


def main() -> int:
    data = pd.read_csv(SOURCE, encoding="utf-8-sig")
    specs = [
        ("真实训练参照", "energy_real_reference", BASELINE),
        ("无约束", "energy_unconstrained", ACCENT),
        ("受约束", "energy_constrained", DEFECT),
    ]
    means = data.groupby("style")[[x[1] for x in specs]].mean()
    stds = data.groupby("style")[[x[1] for x in specs]].std()
    order = means["energy_unconstrained"].sort_values().index.tolist()

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4.15))
    y = np.arange(len(order), dtype=float)
    height = 0.22
    offsets = (-height, 0.0, height)
    for offset, (label, col, color) in zip(offsets, specs):
        vals = means.loc[order, col].to_numpy()
        errs = stds.loc[order, col].to_numpy()
        ax.barh(y + offset, vals, height=height * 0.88, color=color, alpha=0.88,
                label=label, xerr=errs, error_kw={"ecolor": INK2, "elinewidth": 0.6,
                                                  "capsize": 1.5})
    ax.set_yticks(y, order)
    ax.set_xlim(left=0)
    ax.set_xlabel("Energy distance（稳健标准化）")
    simple_title(ax, "留出分布距离")
    tidy(ax, grid_axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    files = save(fig, OUT_DIR / "holdout_energy_comparison")
    plt.close(fig)

    manifest = [{
        "file": "holdout_energy_comparison",
        "title": "留出分布距离",
        "caption": (
            "8 个风格在 3 个分层留出随机种子、2 种 Copula 相关结构下的 energy distance "
            "均值与标准差；真实训练参照、无约束生成和受约束生成使用相同稳健标准化。"
            "误差条表示划分、抽样与模型选择的合并波动，不是置信区间。AI 辅助绘图，数值由审计脚本现场聚合。"
        ),
        "source": "holdout_seed_metrics.csv",
    }]
    (OUT_DIR / "figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(str(x) for x in files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
