"""Audit agreement between expert sensory scores and RateBeer text descriptors.

The forum columns are published enrichment scores, not consumer liking ratings.
The locked Q1 holdout predictions are used without fitting on forum data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/modeling/forum_bridge_v1"
PAIRS = [
    ("麦芽香气", "A_malt_all", "aroma_malt"),
    ("麦芽滋味", "F_malt_all", "taste_malt"),
    ("酒花香气", "A_hops_all", "aroma_hoppy"),
    ("酒花滋味", "F_hops_all", "taste_hoppy"),
    ("酯香", "A_esters_all", "aroma_estery"),
    ("酯味", "F_esters_all", "taste_estery"),
    ("酸味", "acidity", "taste_sour"),
    ("苦味", "bitternes", "taste_bitter"),
    ("甜味", "sweetness", "taste_sweet"),
    ("酒精感", "alcohol", "taste_alcohol"),
    ("气泡感", "co2", "palate_carbonation"),
    ("双乙酰", "diacetyl", "aroma_diacetyl"),
    ("芫荽", "coriander", "aroma_coriander"),
    ("丁香", "clove", "aroma_clove"),
    ("橙香", "orange", "aroma_orange"),
]


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def style_residual(d, col):
    return (d[col] - d.groupby("tasting_category_fine")[col].transform("mean")).to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expert = pd.read_csv(ROOT / "data/raw/4.csv")
    forum = pd.read_csv(ROOT / "data/raw/7.csv")
    pred = pd.read_csv(ROOT / "reports/modeling/q1_mainline_retrain_v1/test_predictions.csv.gz")
    pred = pred.loc[pred.model.eq("grouped_extra_trees")]
    joined = expert.merge(forum.drop(columns=["beer_id", "tasting_category_fine"]), on="beer", validate="one_to_one")
    assert len(joined) == 250
    rows = []
    for name, panel_col, forum_col in PAIRS:
        d = joined[["beer", "tasting_category_fine", panel_col, forum_col]].copy()
        p = pred.loc[pred.label.eq(panel_col), ["beer", "predicted_continuous"]]
        h = d.merge(p, on="beer", validate="one_to_one")
        rows.append({
            "pair": name, "expert": panel_col, "forum": forum_col,
            "all_rho": rho(d[panel_col].to_numpy(), d[forum_col].to_numpy()),
            "all_within_style_rho": rho(style_residual(d, panel_col), style_residual(d, forum_col)),
            "holdout_n": len(h),
            "holdout_observed_rho": rho(h[panel_col].to_numpy(), h[forum_col].to_numpy()),
            "holdout_predicted_rho": rho(h.predicted_continuous.to_numpy(), h[forum_col].to_numpy()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "paired_descriptor_correlations.csv", index=False)

    plt.rcParams.update({"font.sans-serif": ["Arial Unicode MS", "Heiti SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9, 5.3))
    y = np.arange(len(result))
    ax.barh(y + .19, result.all_rho, height=.36, label="全部250款：专家 vs 评论", color="#31688e")
    ax.barh(y - .19, result.all_within_style_rho, height=.36, label="风格内去均值后", color="#67a9cf")
    ax.axvline(0, color="#555", linewidth=.8)
    ax.set_yticks(y, result.pair)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman 相关系数")
    ax.set_xlim(-.25, .85)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("专家品评与消费者评论：哪些感官词一致？")
    fig.tight_layout()
    fig.savefig(OUT / "expert_forum_bridge.png", dpi=200)
    plt.close(fig)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
