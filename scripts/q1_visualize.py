#!/usr/bin/env python3
"""Interpretability figures for Q1: what the models learned, beyond test metrics.

Reads the locked feature matrix and the artifacts under reports/modeling/q1_runs/,
refits the final models on the 175 training beers only, and produces seven
figures plus CSV tables and a notes file under reports/modeling/q1_interpret_v1/.

No test-sample information is used anywhere in the refits or the importances;
test predictions are only re-read for the baseline-skill comparison.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/q1_matplotlib")
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from q1_evaluation import load_inputs, official_family
from q1_models import semantic_group

# Validated chart palette (light surface). Categorical slots in fixed order.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                              "#898781", "#e1e0d9", "#c3c2b7")
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_GRAY = "#898781"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue",
                                             ["#cde2fb", "#6da7ec", "#3987e5",
                                              "#256abf", "#0d366b"])
DIV_BLUE_RED = LinearSegmentedColormap.from_list("div_blue_red",
                                                 ["#2a78d6", "#f0efec", "#e34948"])
FAMILY_ORDER = ["aroma", "flavour", "mouthfeel", "fault_special", "overall"]
RANDOM_STATE = 20261026
OUT_DIR = Path("reports/modeling/q1_interpret_v1")
RUNS_DIR = Path("reports/modeling/q1_runs")


def new_axes(width, height):
    """Create one styled axes on the light chart surface."""
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    return fig, ax


def finish(fig, name):
    """Save one figure at report resolution."""
    fig.savefig(OUT_DIR / name, dpi=220, facecolor=SURFACE)
    plt.close(fig)


def make_evaluation_figures(continuous, prediction_table, summary, out):
    """Create cross-model figures; skip the comparison bar chart for one model."""
    global OUT_DIR
    previous = OUT_DIR
    OUT_DIR = Path(out) / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = list(summary.index)
    if len(order) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, metric in zip(axes, ["Macro_RMSE", "Macro_MAE"]):
            values = summary.loc[order, metric]
            ax.barh(order[::-1], values.iloc[::-1], color=CATEGORICAL[0])
            ax.set_title(f"Test {metric.replace('_', '-')}")
            ax.grid(axis="x", color=GRID, linewidth=.6)
        finish(fig, "01_model_overall_performance.png")
    family = continuous.groupby(["model", "family"])["RMSE"].mean().unstack()
    family = family.reindex(index=order, columns=[c for c in FAMILY_ORDER if c in family])
    fig, ax = new_axes(9, 5.5)
    im = ax.imshow(family, aspect="auto", cmap=SEQ_BLUE)
    ax.set_xticks(range(len(family.columns)), family.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(family.index)), family.index)
    for i in range(family.shape[0]):
        for j in range(family.shape[1]):
            ax.text(j, i, f"{family.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Test RMSE by sensory family")
    fig.colorbar(im, ax=ax, label="Mean per-target RMSE")
    finish(fig, "02_family_rmse_heatmap.png")
    r2 = continuous.pivot(index="label", columns="model", values="R2").reindex(columns=order)
    labels = (continuous[["label", "family"]].drop_duplicates()
              .assign(family=lambda d: pd.Categorical(d.family, FAMILY_ORDER, ordered=True))
              .sort_values(["family", "label"]).label)
    r2 = r2.reindex(labels)
    fig, ax = new_axes(11, 14)
    limit = max(1.0, float(np.nanmax(np.abs(r2.to_numpy()))))
    im = ax.imshow(r2, aspect="auto", cmap=DIV_BLUE_RED, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(r2.columns)), r2.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(r2.index)), r2.index, fontsize=7)
    ax.set_title("Per-target test R²")
    fig.colorbar(im, ax=ax, label="R²")
    finish(fig, "03_target_r2_heatmap.png")
    best = next((m for m in order if m != "global_mean"), order[0])
    frame = prediction_table[prediction_table.model.eq(best)].copy()
    frame["residual"] = frame.observed_continuous - frame.predicted_continuous
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].scatter(frame.observed_continuous, frame.predicted_continuous, s=9, alpha=.35)
    lo = min(frame.observed_continuous.min(), frame.predicted_continuous.min())
    hi = max(frame.observed_continuous.max(), frame.predicted_continuous.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color=INK, linewidth=1)
    axes[0].set(xlabel="Observed", ylabel="Predicted", title=f"Observed vs predicted: {best}")
    axes[1].scatter(frame.predicted_continuous, frame.residual, s=9, alpha=.35)
    axes[1].axhline(0, linestyle="--", color=INK, linewidth=1)
    axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residual diagnostic")
    finish(fig, "04_best_model_diagnostics.png")
    OUT_DIR = previous
    return family


def make_linear_explanation_figures(fitted_models, feature_names, labels, out):
    """Plot top standardized coefficients for fitted linear mainline models."""
    global OUT_DIR
    previous = OUT_DIR
    OUT_DIR = Path(out) / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_families = np.array([official_family(label) for label in labels])
    for name in ["ridge", "multitask_elastic_net", "pls2"]:
        if name not in fitted_models:
            continue
        model = fitted_models[name].best_estimator_.named_steps["model"]
        coef = np.asarray(model.coef_)
        if coef.shape == (len(feature_names), len(labels)):
            coef = coef.T
        if coef.shape != (len(labels), len(feature_names)):
            continue
        chosen = np.argsort(np.linalg.norm(coef, axis=0))[-20:][::-1]
        grouped = np.column_stack([
            coef[target_families == family][:, chosen].mean(axis=0) for family in FAMILY_ORDER])
        bound = max(float(np.max(np.abs(grouped))), 1e-12)
        fig, ax = new_axes(8.5, 8)
        im = ax.imshow(grouped, aspect="auto", cmap=DIV_BLUE_RED, vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(FAMILY_ORDER)), FAMILY_ORDER, rotation=25, ha="right")
        ax.set_yticks(range(len(chosen)), np.asarray(feature_names)[chosen], fontsize=8)
        ax.set_title(f"{name}: top standardized coefficients by family")
        fig.colorbar(im, ax=ax, label="Mean signed coefficient")
        finish(fig, f"05_{name}_coefficient_heatmap.png")
    OUT_DIR = previous


def load_split(root):
    """Return train/test frames for X and Y plus styles, from the fixed manifest."""
    x, y, styles, dictionary = load_inputs(root)
    plan = pd.read_csv(root / "data/splits/seed_0/split_manifest.csv",
                       encoding="utf-8-sig")
    train_ids = plan.loc[plan.split.eq("train"), "beer"]
    test_ids = plan.loc[plan.split.eq("test"), "beer"]
    return (x.loc[train_ids], y.loc[train_ids], x.loc[test_ids], y.loc[test_ids],
            styles.loc[train_ids], dictionary)


# ---------------------------------------------------------------------------
# Figure 1: skill over the global-mean baseline, with bootstrap intervals.
# ---------------------------------------------------------------------------

def fig_skill_vs_baseline():
    """Bootstrap the RMSE improvement of each model over the mean baseline."""
    frames = {}
    for model_dir in sorted(RUNS_DIR.iterdir()):
        path = model_dir / "test_predictions.csv.gz"
        if path.exists():
            frames[model_dir.name] = pd.read_csv(path)
    wide = {}
    for name, frame in frames.items():
        obs = frame.pivot(index="beer", columns="label", values="observed_continuous")
        pred = frame.pivot(index="beer", columns="label", values="predicted_continuous")
        pred = pred.reindex(index=obs.index, columns=obs.columns)
        wide[name] = (obs.to_numpy(), pred.to_numpy())
    base = wide["global_mean"][0]                       # observed values are identical across models
    rng = np.random.default_rng(RANDOM_STATE)
    n_beers = base.shape[0]
    rows = []
    for _ in range(2000):
        idx = rng.integers(0, n_beers, n_beers)
        base_rmse = np.sqrt(((base[idx] - base[idx].mean(axis=0)) ** 2).mean()).mean()
        for name, (obs, pred) in wide.items():
            rmse = np.sqrt(((obs[idx] - pred[idx]) ** 2).mean()).mean()
            rows.append({"model": name,
                         "improvement": 100 * (base_rmse - rmse) / base_rmse})
    boot = pd.DataFrame(rows)
    summary = boot.groupby("model")["improvement"].agg(
        improvement="mean", lo=lambda s: s.quantile(0.025), hi=lambda s: s.quantile(0.975))
    summary = summary.drop(index="global_mean").sort_values("improvement")
    summary.to_csv(OUT_DIR / "skill_vs_baseline.csv")

    fig, ax = new_axes(7.5, 4.2)
    y_pos = np.arange(len(summary))
    for i, (name, row) in enumerate(summary.iterrows()):
        ax.errorbar(row.improvement, i,
                    xerr=[[row.improvement - row.lo], [row.hi - row.improvement]],
                    fmt="o", color=CATEGORICAL[0], markersize=6, capsize=3,
                    elinewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.text(row.hi + 0.35, i, f"{row.improvement:.1f}%", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(y_pos, summary.index)
    ax.set_xlabel("RMSE improvement over global-mean baseline (%)")
    ax.set_title("Skill over baseline, 95% bootstrap CI (2000 resamples)", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    finish(fig, "01_skill_vs_baseline.png")


# ---------------------------------------------------------------------------
# Figure 2: which sensory targets are predictable at all.
# ---------------------------------------------------------------------------

def fig_target_predictability():
    """Per-target test Spearman of grouped Extra Trees, faceted by family."""
    metrics = pd.read_csv(RUNS_DIR / "grouped_extra_trees/continuous_metrics_by_label.csv")
    fig, axes = plt.subplots(1, 5, figsize=(12, 5.2), sharey=True,
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, family in zip(axes, FAMILY_ORDER):
        block = (metrics[metrics.family.eq(family)]
                 .sort_values("Spearman", ascending=True))
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=7)
        y_pos = np.arange(len(block))
        ax.hlines(y_pos, 0, block.Spearman, color=GRID, linewidth=1)
        ax.scatter(block.Spearman, y_pos, s=14, color=CATEGORICAL[0], zorder=3)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_title(f"{family} (n={len(block)})", fontsize=9, color=INK)
        ax.set_yticks(y_pos, block.label, fontsize=6.5)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_xlabel("Test Spearman", fontsize=8)
    fig.suptitle("Per-target ranking skill, grouped Extra Trees (test, n=75)",
                 color=INK, fontsize=11)
    finish(fig, "02_target_predictability.png")


# ---------------------------------------------------------------------------
# Figure 3: PLS2 scores, loadings, and VIP.
# ---------------------------------------------------------------------------

def pls_vip(pls, feature_names):
    """Variable Importance in Projection, from fitted PLS weights and scores."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, n_comp = w.shape
    assert np.allclose((w ** 2).sum(axis=0), 1)
    ssy_per_comp = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)
    vip = np.sqrt(p * (w ** 2).dot(ssy_per_comp) / ssy_per_comp.sum())
    return pd.Series(vip, index=feature_names)


def fig_pls(x_train, y_train, styles_train):
    """Scores colored by dominant styles, plus top loadings and VIP features."""
    scaler = StandardScaler().fit(x_train)
    pls = PLSRegression(n_components=10, scale=False, max_iter=2000)
    pls.fit(scaler.transform(x_train), y_train)
    vip = pls_vip(pls, list(x_train.columns)).sort_values(ascending=False)
    vip.head(25).to_csv(OUT_DIR / "pls_vip_top25.csv")

    top_styles = styles_train.value_counts().head(8).index
    fig = plt.figure(figsize=(13.5, 5.5), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    scores = pls.x_scores_

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for i, style in enumerate(top_styles):
        mask = styles_train.eq(style).to_numpy()
        ax.scatter(scores[mask, 0], scores[mask, 1], s=16, alpha=.8,
                   color=CATEGORICAL[i], label=style)
    mask_other = ~styles_train.isin(top_styles).to_numpy()
    ax.scatter(scores[mask_other, 0], scores[mask_other, 1], s=16, alpha=.5,
               color=OTHER_GRAY, label=f"Other ({mask_other.sum()})")
    ax.set_xlabel("PLS component 1", fontsize=8)
    ax.set_ylabel("PLS component 2", fontsize=8)
    ax.set_title("Training scores by beer style", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="best", labelcolor=INK_2)

    for panel, comp in [(1, 0), (2, 1)]:
        ax = fig.add_subplot(gs[panel])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        loading = pd.Series(pls.x_loadings_[:, comp], index=x_train.columns)
        loading = loading.reindex(loading.abs().sort_values(ascending=False).index)
        top = loading.head(12).iloc[::-1]
        colors = [CATEGORICAL[0] if v > 0 else CATEGORICAL[7] for v in top]
        ax.barh(range(len(top)), top, color=colors, height=.62)
        ax.set_yticks(range(len(top)), top.index, fontsize=6.5)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_title(f"Component {comp + 1}: top loadings", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.suptitle("PLS2: does the chemical space recover sensory structure?",
                 color=INK, fontsize=11)
    finish(fig, "03_pls_scores_loadings_vip.png")
    return pls, vip


# ---------------------------------------------------------------------------
# Figure 4: grouped Extra Trees permutation importance.
# ---------------------------------------------------------------------------

def fig_grouped_importance(x_train, y_train, inner):
    """Permutation importance per semantic group, averaged over inner folds."""
    params = pd.read_csv(RUNS_DIR / "grouped_extra_trees/grouped_model_best_params.csv")
    groups = y_train.columns.map(semantic_group)
    rows = []
    for _, row in params.iterrows():
        group = row.model_group
        members = [c for c, g in zip(y_train.columns, groups) if g == group]
        target = y_train[members]
        depth = None if pd.isna(row.max_depth) else int(row.max_depth)
        for fold_i, (tri, vai) in enumerate(inner.split(x_train)):
            model = ExtraTreesRegressor(n_estimators=500, max_features=float(row.max_features),
                                        min_samples_leaf=int(row.min_samples_leaf),
                                        max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(x_train.iloc[tri], target.iloc[tri])
            imp = permutation_importance(model, x_train.iloc[vai], target.iloc[vai],
                                         scoring="neg_mean_squared_error",
                                         n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
            for name, value in zip(x_train.columns, imp.importances_mean):
                rows.append({"group": group, "fold": fold_i,
                             "feature": name, "importance": value})
    long = pd.DataFrame(rows)
    long.to_csv(OUT_DIR / "grouped_extra_trees_permutation_importance.csv", index=False)
    table = long.groupby(["group", "feature"])["importance"].mean().reset_index()
    groups_order = table.group.drop_duplicates().tolist()

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, group in zip(axes.flat, groups_order):
        block = (table[table.group.eq(group)]
                 .sort_values("importance", ascending=False).head(8).iloc[::-1])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.barh(range(len(block)), block.importance, color=CATEGORICAL[0], height=.62)
        ax.set_yticks(range(len(block)), block.feature, fontsize=6.5)
        ax.set_title(group, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.suptitle("Grouped Extra Trees: permutation importance "
                 "(mean over 5 inner folds, test-set MSE units)", color=INK, fontsize=11)
    finish(fig, "04_grouped_extra_trees_importance.png")
    return table


# ---------------------------------------------------------------------------
# Figure 5: GBR partial dependence for the best-predicted targets.
# ---------------------------------------------------------------------------

def fig_gbr_pdp(x_train, y_train, inner):
    """PDP curves for the top features of the most predictable targets."""
    metrics = pd.read_csv(RUNS_DIR / "independent_gbr/continuous_metrics_by_label.csv")
    best_params = pd.read_csv(RUNS_DIR / "independent_gbr/gbr_best_params_by_target.csv"
                              ).set_index("target")
    top_targets = metrics.sort_values("Spearman", ascending=False).label.head(4)
    panels = []
    for target in top_targets:
        p = best_params.loc[target]
        model = GradientBoostingRegressor(
            n_estimators=int(p.n_estimators), learning_rate=float(p.learning_rate),
            max_depth=int(p.max_depth), min_samples_leaf=int(p.min_samples_leaf),
            max_features=p.max_features, subsample=float(p.subsample),
            random_state=RANDOM_STATE)
        tri, _ = next(iter(inner.split(x_train)))
        model.fit(x_train.iloc[tri], y_train.iloc[tri][target])
        imp = permutation_importance(model, x_train.iloc[tri], y_train.iloc[tri][target],
                                     n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        ranked = pd.Series(imp.importances_mean, index=x_train.columns)
        ranked = ranked.reindex(ranked.abs().sort_values(ascending=False).index)
        for feature in ranked.head(2).index:
            panels.append((target, feature, model))

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (target, feature, model) in zip(axes.flat, panels):
        result = partial_dependence(model, x_train, [feature], grid_resolution=60)
        grid = result["grid_values"][0]
        mean_curve = result["average"][0]
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.plot(grid, mean_curve, color=CATEGORICAL[0], linewidth=2)
        values = x_train[feature].to_numpy()
        ax.scatter(values, np.full_like(values, ax.get_ylim()[0]), marker="|",
                   s=8, color=MUTED, alpha=.5)
        ax.set_title(f"{target}\nvs {feature}", fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.suptitle("GBR partial dependence: response shape of top relationships "
                 "(rug = training values)", color=INK, fontsize=11)
    finish(fig, "05_gbr_partial_dependence.png")


# ---------------------------------------------------------------------------
# Figure 6: Elastic Net sparsity structure.
# ---------------------------------------------------------------------------

def fig_elastic_net(x_train, y_train):
    """Heatmap of nonzero coefficients, features by strength, targets by family."""
    scaler = StandardScaler().fit(x_train)
    model = MultiTaskElasticNet(alpha=0.248163, l1_ratio=0.95, max_iter=30000,
                                tol=1e-4, selection="random",
                                random_state=RANDOM_STATE)
    model.fit(scaler.transform(x_train), y_train)
    coefs = pd.DataFrame(model.coef_, index=y_train.columns, columns=x_train.columns).T
    retained = coefs.loc[coefs.abs().max(axis=1) > 0]
    retained = retained.reindex(retained.abs().max(axis=1).sort_values(ascending=False).index)
    order = [c for fam in FAMILY_ORDER for c in y_train.columns if official_family(c) == fam]
    retained = retained[order]
    retained.to_csv(OUT_DIR / "elastic_net_retained_coefficients.csv")

    fig, ax = new_axes(11, 1.2 + 0.24 * len(retained))
    limit = float(np.abs(retained.to_numpy()).max())
    ax.imshow(retained.to_numpy(), aspect="auto", cmap=DIV_BLUE_RED,
              vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(range(len(order)), [c[:14] for c in order], rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(retained)), retained.index, fontsize=5.5)
    ax.set_title(f"Elastic Net nonzero coefficients "
                 f"({len(retained)}/{x_train.shape[1]} features retained)", color=INK)
    finish(fig, "06_elastic_net_sparsity.png")
    return coefs


# ---------------------------------------------------------------------------
# Figure 7: cross-model agreement on important features.
# ---------------------------------------------------------------------------

def fig_cross_model_agreement(x_train, y_train, pls_vip_series, grouped_importance,
                              elastic_coefs):
    """Count how many of four methods rank a feature in their top 20."""
    scaler = StandardScaler().fit(x_train)
    ridge = Ridge(alpha=366.524).fit(scaler.transform(x_train), y_train)
    ridge_rank = (pd.DataFrame(ridge.coef_.T, index=x_train.columns)
                  .abs().mean(axis=1).sort_values(ascending=False))
    en_rank = elastic_coefs.abs().mean(axis=1).sort_values(ascending=False)
    en_rank = en_rank[en_rank > 0]
    pls_rank = pls_vip_series.sort_values(ascending=False)
    tree_rank = (grouped_importance.groupby("feature")["importance"].mean()
                 .sort_values(ascending=False))
    top_sets = {"ridge": set(ridge_rank.head(20).index),
                "elastic_net": set(en_rank.head(20).index),
                "pls_vip": set(pls_rank.head(20).index),
                "extra_trees": set(tree_rank.head(20).index)}
    counts = {}
    for method, top in top_sets.items():
        for feature in top:
            counts[feature] = counts.get(feature, 0) + 1
    table = (pd.DataFrame([{"feature": f, "methods": n} for f, n in counts.items()])
             .sort_values(["methods", "feature"], ascending=[False, True]))
    table.to_csv(OUT_DIR / "cross_model_feature_agreement.csv", index=False)
    detail = pd.DataFrame({m: pd.Series(1, index=sorted(s)) for m, s in top_sets.items()})
    detail.to_csv(OUT_DIR / "cross_model_feature_membership.csv")

    show = table.head(25).iloc[::-1]
    fig, ax = new_axes(7.5, 7)
    colors = {1: "#cde2fb", 2: "#9ec5f4", 3: "#5598e7", 4: "#1c5cab"}
    ax.barh(range(len(show)), show.methods,
            color=[colors[n] for n in show.methods], height=.62)
    ax.set_yticks(range(len(show)), show.feature, fontsize=7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Number of methods ranking the feature in their top 20")
    ax.set_title("Cross-model agreement on key chemicals "
                 "(ridge / elastic net / PLS VIP / extra trees)", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    finish(fig, "07_cross_model_agreement.png")
    return table


def write_notes(pls, vip, grouped_importance, elastic_coefs, agreement):
    """Summarize the interpretable numbers that the figures carry."""
    retained = elastic_coefs.loc[elastic_coefs.abs().max(axis=1) > 0]
    agreed = agreement[agreement.methods >= 3].feature.tolist()
    lines = [
        "# Q1 interpretability notes",
        "",
        f"- Elastic Net (alpha=0.248, l1_ratio=0.95) retains "
        f"**{len(retained)} of {elastic_coefs.shape[0]}** chemical features.",
        f"- Strongest retained features: "
        f"{', '.join(retained.abs().max(axis=1).sort_values(ascending=False).head(10).index)}.",
        f"- PLS VIP top 10: {', '.join(vip.head(10).index)}.",
        f"- Extra Trees: strongest features per group are in "
        f"`grouped_extra_trees_permutation_importance.csv`.",
        f"- Endorsed by >=3 of 4 methods: "
        f"{', '.join(agreed) if agreed else '(none)'}." ,
    ]
    (OUT_DIR / "interpretation_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib_cache"))
    OUT_DIR.joinpath(".matplotlib_cache").mkdir(exist_ok=True)

    x_train, y_train, _, _, styles_train, _ = load_split(args.root)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fig_skill_vs_baseline()
    fig_target_predictability()
    pls, vip = fig_pls(x_train, y_train, styles_train)
    grouped_importance = fig_grouped_importance(x_train, y_train, kfold)
    fig_gbr_pdp(x_train, y_train, kfold)
    elastic_coefs = fig_elastic_net(x_train, y_train)
    agreement = fig_cross_model_agreement(x_train, y_train, vip, grouped_importance,
                                          elastic_coefs)
    write_notes(pls, vip, grouped_importance, elastic_coefs, agreement)
    print(f"figures and tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()
