"""Data relationship views for the full modeling paper."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
STYLES = ["Blond", "Tripel", "Strong ale", "Hoppy", "Stout/Porter", "Wheat", "Amber", "Brown"]
COLORS = plt.get_cmap("tab10").colors


def standardized_pca(data):
    z = StandardScaler().fit_transform(data)
    model = PCA(n_components=2)
    return model.fit_transform(z), model.explained_variance_ratio_


def beer_spaces():
    x = pd.read_csv(ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0).drop(columns=["esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"])
    expert = pd.read_csv(ROOT / "data/raw/4.csv").set_index("beer").reindex(x.index)
    forum = pd.read_csv(ROOT / "data/raw/7.csv").set_index("beer").reindex(x.index)
    forum = forum.filter(regex="^(aroma|palate|taste)_").dropna(axis=1, how="all")
    matrices = [("化学指纹（227维）", x), ("专家感官（50维）", expert.loc[:, "A_malt_all":"overall"]),
                ("论坛描述词（183维）", forum)]
    styles = expert.tasting_category_fine.to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.8))
    for ax, (name, matrix) in zip(axs, matrices):
        points, ev = standardized_pca(matrix)
        ax.scatter(points[:, 0], points[:, 1], s=12, c="#d9dfe2", alpha=.55, label="其他风格", rasterized=True)
        for j, style in enumerate(STYLES):
            mask = styles == style
            ax.scatter(points[mask, 0], points[mask, 1], s=20, color=COLORS[j], alpha=.78,
                       label=f"{style} ({mask.sum()})", rasterized=True)
        ax.set_title(name, fontsize=11, weight="bold")
        ax.set_xlabel(f"PC1 · {ev[0]:.0%}方差")
        ax.set_ylabel(f"PC2 · {ev[1]:.0%}方差")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#eef1f2", lw=.5)
        ax.set_axisbelow(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[1:], labels[1:], loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("同一批250款啤酒，在三类测量中的样品结构", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, .17, 1, .9], w_pad=1.0)
    fig.savefig(OUT / "beer_spaces.png", dpi=220)
    plt.close(fig)


def module_sensory_map():
    x = pd.read_csv(ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    y = pd.read_csv(ROOT / "data/raw/4.csv").set_index("beer").reindex(x.index)
    nodes = pd.read_csv(ROOT / "reports/modeling/q2_mainline_retrain_v1/nodes.csv")
    modules = nodes.module.value_counts().head(6).index.tolist()
    labels = ["A_malt_all", "A_hops_all", "A_esters_all", "F_malt_all", "F_hops_all", "F_esters_all",
              "acidity", "bitternes", "sweetness", "body", "co2", "overall", "diacetyl", "lactic", "clove"]
    names = ["麦芽香", "酒花香", "酯香", "麦芽味", "酒花味", "酯味", "酸度", "苦味", "甜味", "酒体", "气泡", "总体", "双乙酰", "乳酸", "丁香"]
    z = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)
    matrix = []
    row_labels = []
    descriptions = []
    for module in modules:
        members = nodes.loc[nodes.module.eq(module)].sort_values("degree", ascending=False)
        score = z[members.feature].mean(axis=1)
        matrix.append([spearmanr(score, y[label]).statistic for label in labels])
        row_labels.append(f"社区{module+1} · {len(members)}物质")
        descriptions.append(f"社区{module+1}: {', '.join(members.feature.head(3))}")
    mat = np.asarray(matrix)
    pd.DataFrame(mat, index=row_labels, columns=labels).to_csv(ROOT / "reports/modeling/q2_mainline_retrain_v1/module_sensory_map.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("化合物社区与感官维度的关系图", fontsize=13, weight="bold", pad=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) >= .3: ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                                         color="white" if abs(mat[i,j]) > .43 else "#152836")
    cb = fig.colorbar(im, ax=ax, fraction=.022, pad=.015)
    cb.set_label("Spearman 相关")
    ax.spines[:].set_visible(False)
    fig.text(.06, .055, "社区得分＝成员化合物标准化丰度的平均值；社区来自211节点共表达图。", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, .09, 1, .97])
    fig.savefig(OUT / "module_sensory_map.png", dpi=220)
    plt.close(fig)
    (ROOT / "reports/modeling/q2_mainline_retrain_v1/module_sensory_map_key.txt").write_text("\n".join(descriptions), encoding="utf-8")


def candidate_space():
    x = pd.read_csv(ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    profiles = pd.read_csv(ROOT / "reports/modeling/q3_source_guided_v2/selected_profiles_logscale.csv")
    selected = profiles.loc[profiles.arm.eq("guided") & profiles["rank"].eq(1)]
    features = list(selected.columns[5:])
    x = x[features]
    styles = pd.read_csv(ROOT / "data/raw/4.csv").set_index("beer").reindex(x.index).tasting_category_fine
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=2).fit(scaler.transform(x))
    real_xy = pca.transform(scaler.transform(x))
    cand_xy = pca.transform(scaler.transform(selected[features]))
    targets = ["Tripel", "Kriek", "Lambic"]
    colors = ["#335e87", "#c47745", "#2b9884"]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.8), sharex=True, sharey=True)
    for ax, target, color in zip(axs, targets, colors):
        real_mask = styles.to_numpy() == target
        cand_mask = selected.target_style.to_numpy() == target
        ax.scatter(real_xy[:, 0], real_xy[:, 1], color="#d8dfe3", s=12, alpha=.35, label="其他真实啤酒", rasterized=True)
        ax.scatter(real_xy[real_mask, 0], real_xy[real_mask, 1], color=color, s=38, edgecolor="white", linewidth=.5,
                   label=f"真实{target} ({real_mask.sum()})")
        ax.scatter(cand_xy[cand_mask, 0], cand_xy[cand_mask, 1], color=color, marker="x", s=30, alpha=.7,
                   label=f"引导候选 ({cand_mask.sum()})", rasterized=True)
        if target == "Tripel":
            idx = list(x.index).index(48)
            ax.scatter(real_xy[idx, 0], real_xy[idx, 1], marker="*", s=130, color="#1c2733", label="源啤酒48")
            case = selected.loc[selected.candidate_id.eq("Tripel-0460") & selected.source_beer.eq(48)]
            if len(case):
                loc = selected.index.get_loc(case.index[0])
                ax.scatter(cand_xy[loc, 0], cand_xy[loc, 1], marker="D", s=70, color="#1c2733", label="候选0460")
        ax.set_title(target, weight="bold")
        ax.set_xlabel(f"化学PC1 · {pca.explained_variance_ratio_[0]:.0%}方差")
        ax.grid(color="#eef1f2", lw=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7.7, loc="best")
    axs[0].set_ylabel(f"化学PC2 · {pca.explained_variance_ratio_[1]:.0%}方差")
    fig.suptitle("候选谱在真实啤酒化学空间中的位置", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .92], w_pad=1.2)
    fig.savefig(OUT / "candidate_chemical_space.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.sans-serif": ["Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)
    beer_spaces()
    module_sensory_map()
    candidate_space()
