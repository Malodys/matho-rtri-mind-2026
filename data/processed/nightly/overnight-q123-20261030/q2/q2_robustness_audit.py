# -*- coding: utf-8 -*-
# AI 辅助：OpenAI Codex (GPT-5), OpenAI, 2026-10-30
# 人工核验点：本脚本只读主 Q2 产物与 data_contract 数据入口；所有新结果仅写入本 nightly/q2 目录。
"""Q2 夜间稳健关联复核：有符号 Spearman、训练选择/留出置换 FDR、相关组冗余。

该脚本不修改 ``data/processed/q2_runs``、``data/processed/q2_figures`` 或 ``data/raw``。
留出置换检验使用训练集重新选择的 Top-3，避免主 E2 全样本相关对留出集的信息泄漏。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROBUST_LABEL = "稳健关键因素"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_unit(frame: pd.DataFrame) -> np.ndarray:
    """逐列平均秩、中心化并归一到单位二范数；列间点积即 Spearman rho。"""
    ranks = frame.rank(axis=0, method="average").to_numpy(dtype=float).copy()
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt((ranks * ranks).sum(axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    return ranks / norms


def spearman_matrix(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    values = rank_unit(x).T @ rank_unit(y)
    return pd.DataFrame(values, index=x.columns, columns=y.columns)


def rank_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    ranks = frame.rank(axis=0, method="average")
    return (ranks - 1.0) / max(frame.shape[0] - 1, 1)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg 调整，保持输入顺序。"""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def permutation_pvalues(
    x_unit: np.ndarray,
    y_unit: np.ndarray,
    selected_feature_indices: list[int],
    observed: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> np.ndarray:
    """对一个目标的三个预先选择特征做双侧留出置换检验。"""
    x_selected = x_unit[:, selected_feature_indices]
    counts = np.zeros(len(selected_feature_indices), dtype=np.int64)
    n = y_unit.shape[0]
    done = 0
    while done < permutations:
        size = min(batch_size, permutations - done)
        # 每行是一组独立置换；argsort 随机连续数产生均匀随机排列。
        perm_indices = np.argsort(rng.random((size, n)), axis=1)
        permuted_y = y_unit[perm_indices]
        permuted_rho = permuted_y @ x_selected
        counts += (np.abs(permuted_rho) >= np.abs(observed)[None, :] - 1e-15).sum(axis=0)
        done += size
    return (counts + 1.0) / (permutations + 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Q2 夜间稳健关联复核")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import data_contract as dc  # 唯一允许读取 data/raw 的仓库入口

    run_dir = root / "data" / "processed" / "q2_runs"
    graph_dir = run_dir / "graph"
    evidence_dir = run_dir / "evidence"
    input_paths = {
        "summary": run_dir / "summary.json",
        "top3": run_dir / "top3_by_sensory.csv",
        "stability": run_dir / "stability.csv",
        "bootstrap": run_dir / "bootstrap_top3_freq.csv",
        "e2": evidence_dir / "e2_spearman_full.csv",
        "e3": evidence_dir / "e3_shap_importance.csv",
        "communities": graph_dir / "communities.csv",
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(input_paths["summary"].read_text(encoding="utf-8"))
    top = pd.read_csv(input_paths["top3"], keep_default_na=False, encoding="utf-8")
    e3 = pd.read_csv(input_paths["e3"], index_col=0, encoding="utf-8")
    communities = pd.read_csv(input_paths["communities"], encoding="utf-8")

    raw = dc.load_raw(root / "data" / "raw")
    x = dc.load_modeling_matrix(raw)
    y = dc.load_targets(raw)
    split = dc.load_split(root / "data" / "splits" / "seed_0")
    train_beers = set(split.loc[split["split"] == "train", "beer"].astype(int))
    test_beers = set(split.loc[split["split"] == "test", "beer"].astype(int))
    x_train = x.loc[x.index.isin(train_beers)]
    y_train = y.loc[y.index.isin(train_beers)]
    x_test = x.loc[x.index.isin(test_beers)]
    y_test = y.loc[y.index.isin(test_beers)]
    if x_train.shape != (175, 227) or y_train.shape != (175, 50):
        raise ValueError(f"训练形状异常: X={x_train.shape}, Y={y_train.shape}")
    if x_test.shape != (75, 227) or y_test.shape != (75, 50):
        raise ValueError(f"留出形状异常: X={x_test.shape}, Y={y_test.shape}")

    # 1) 当前主结果：显式保留有符号 rho，并锁定三个要求的不变量。
    signed = top.copy()
    signed["rho_sign"] = np.where(
        pd.to_numeric(signed["e2_rho"]) > 0,
        "positive",
        np.where(pd.to_numeric(signed["e2_rho"]) < 0, "negative", "zero"),
    )
    signed["is_robust"] = signed["selection_status"].eq(ROBUST_LABEL)
    signed["is_robust_and_chem_plausible"] = signed["is_robust"] & signed[
        "chem_plausible"
    ].astype(str).str.lower().eq("true")
    signed.to_csv(out_dir / "signed_spearman_main_top3.csv", index=False, encoding="utf-8")

    robust = signed[signed["is_robust"]]
    robust_dims = sorted(robust["sensory_dimension"].unique())
    all_dims = list(y.columns)
    nonrobust_dims = sorted(set(all_dims) - set(robust_dims))

    # 2) 训练集重新选 Top-3：E2(train) + 既有 E3(train SHAP)，再在留出集做置换检验。
    rho_train = spearman_matrix(x_train, y_train)
    rho_test = spearman_matrix(x_test, y_test)
    if list(e3.index) != list(x.columns) or list(e3.columns) != list(y.columns):
        e3 = e3.reindex(index=x.columns, columns=y.columns)
    fused_train = 0.5 * rank_normalize(rho_train.abs()) + 0.5 * rank_normalize(e3)

    x_test_unit = rank_unit(x_test)
    y_test_unit = rank_unit(y_test)
    feat_index = {name: idx for idx, name in enumerate(x_test.columns)}
    rng = np.random.default_rng(args.seed)
    validation_rows: list[dict] = []
    main_sets = {
        dim: set(group["compound"].astype(str))
        for dim, group in top.groupby("sensory_dimension", sort=False)
    }
    stability_rows: list[dict] = []

    for target_idx, target in enumerate(y.columns):
        frame = pd.DataFrame(
            {"compound": fused_train.index.astype(str), "score": fused_train[target].to_numpy()}
        ).sort_values(["score", "compound"], ascending=[False, True], kind="mergesort")
        selected = frame.head(3)["compound"].tolist()
        indices = [feat_index[name] for name in selected]
        observed = np.array([rho_test.loc[name, target] for name in selected], dtype=float)
        pvalues = permutation_pvalues(
            x_test_unit,
            y_test_unit[:, target_idx],
            indices,
            observed,
            args.permutations,
            rng,
        )
        for rank, (compound, pvalue) in enumerate(zip(selected, pvalues), start=1):
            train_rho = float(rho_train.loc[compound, target])
            holdout_rho = float(rho_test.loc[compound, target])
            validation_rows.append(
                {
                    "sensory_dimension": target,
                    "train_selected_rank": rank,
                    "compound": compound,
                    "train_rho": train_rho,
                    "holdout_rho": holdout_rho,
                    "train_sign": int(np.sign(train_rho)),
                    "holdout_sign": int(np.sign(holdout_rho)),
                    "sign_consistent": bool(np.sign(train_rho) == np.sign(holdout_rho)),
                    "holdout_permutation_p_two_sided": float(pvalue),
                    "in_published_main_top3": bool(compound in main_sets[target]),
                }
            )
        overlap = len(set(selected) & main_sets[target])
        stability_rows.append(
            {
                "sensory_dimension": target,
                "published_main_top3": "; ".join(sorted(main_sets[target])),
                "train_only_top3": "; ".join(selected),
                "overlap_count": overlap,
                "jaccard": overlap / (6 - overlap),
            }
        )

    validation = pd.DataFrame(validation_rows)
    validation["holdout_bh_q_150"] = bh_adjust(
        validation["holdout_permutation_p_two_sided"].to_numpy(dtype=float)
    )
    validation["holdout_fdr_0p05"] = validation["holdout_bh_q_150"] <= 0.05
    validation.to_csv(
        out_dir / "train_selected_holdout_permutation_fdr.csv", index=False, encoding="utf-8"
    )
    published_robust_validation = robust.merge(
        validation,
        on=["sensory_dimension", "compound"],
        how="left",
        suffixes=("_published", "_validation"),
    )
    published_robust_validation.to_csv(
        out_dir / "published_robust_holdout_review.csv", index=False, encoding="utf-8"
    )
    selection_stability = pd.DataFrame(stability_rows)
    selection_stability.to_csv(
        out_dir / "selection_stability_by_dimension.csv", index=False, encoding="utf-8"
    )

    # 3) 相关特征组审查：沿用 |rho|>=0.6 Leiden 社区，不重新发明分组。
    community_map = dict(
        zip(communities["file1_column"].astype(str), communities["community"].astype(str))
    )
    community_sizes = communities["community"].astype(str).value_counts().to_dict()
    group_rows: list[dict] = []
    for dim, group in top.groupby("sensory_dimension", sort=False):
        groups = [community_map.get(str(c), "unmapped") for c in group["compound"]]
        duplicates = sorted({g for g in groups if groups.count(g) > 1})
        group_rows.append(
            {
                "sensory_dimension": dim,
                "top3_compounds": "; ".join(group["compound"].astype(str)),
                "rho_0p6_communities": "; ".join(groups),
                "distinct_communities": len(set(groups)),
                "has_correlated_group_redundancy": bool(duplicates),
                "duplicated_communities": "; ".join(duplicates),
                "largest_selected_community_size": max(
                    [int(community_sizes.get(g, 0)) for g in groups] or [0]
                ),
            }
        )
    group_review = pd.DataFrame(group_rows)
    group_review.to_csv(out_dir / "correlated_group_review.csv", index=False, encoding="utf-8")

    method_review = {
        "recommended_primary_followup": "grouped_permutation_importance",
        "group_definition": {
            "primary": "abs_rho>=0.6 Leiden communities from q2_runs/graph/communities.csv",
            "sensitivity": "repeat at abs_rho>=0.5 and report community-level agreement",
            "singleton_policy": "isolated nodes remain singleton groups",
        },
        "evaluation_unit": "permute all columns in one correlated group jointly within a fixed validation fold",
        "metrics": ["delta holdout Spearman", "delta holdout RMSE"],
        "why_preferred": [
            "227 chemical features contain strongly correlated communities",
            "single-feature permutation can split credit arbitrarily among correlated proxies",
            "the existing graph supplies auditable groups without fitting a new conditional density model",
        ],
        "conditional_shap_disposition": "not_run",
        "conditional_shap_reason": (
            "n=175 training samples is small relative to 227 features; no validated conditional sampler or "
            "cross-fitted conditional SHAP implementation is present. Running it now would add modeling risk "
            "and could be mistaken for confirmatory evidence."
        ),
        "causal_scope": "Both methods assess predictive attribution under correlation; neither identifies causality.",
    }
    (out_dir / "method_review.json").write_text(
        json.dumps(method_review, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fdr_hits = validation[validation["holdout_fdr_0p05"]]
    payload = {
        "run_id": "overnight-q123-20261030",
        "main_result_preserved": {
            "graph_abs_rho_threshold": summary["graph"]["threshold"],
            "graph_nodes": summary["graph"]["n_nodes"],
            "graph_edges": summary["graph"]["n_edges"],
            "graph_components": summary["graph"]["n_connected_components"],
        },
        "required_invariants": {
            "robust_rows": int(len(robust)),
            "robust_dimensions": int(len(robust_dims)),
            "dimensions_without_robust_item": int(len(nonrobust_dims)),
            "robust_and_chem_plausible_rows": int(signed["is_robust_and_chem_plausible"].sum()),
            "robust_dimension_names": robust_dims,
            "dimensions_without_robust_item_names": nonrobust_dims,
            "checks": {
                "robust_dimensions_is_22": len(robust_dims) == 22,
                "dimensions_without_robust_item_is_28": len(nonrobust_dims) == 28,
                "robust_and_chem_plausible_rows_is_9": int(
                    signed["is_robust_and_chem_plausible"].sum()
                )
                == 9,
            },
        },
        "signed_spearman": {
            "published_top3_positive": int((signed["rho_sign"] == "positive").sum()),
            "published_top3_negative": int((signed["rho_sign"] == "negative").sum()),
            "robust_positive": int((robust["rho_sign"] == "positive").sum()),
            "robust_negative": int((robust["rho_sign"] == "negative").sum()),
        },
        "train_selection_holdout_permutation": {
            "permutations": args.permutations,
            "seed": args.seed,
            "family_size": int(len(validation)),
            "fdr_0p05_rows": int(len(fdr_hits)),
            "fdr_0p05_dimensions": int(fdr_hits["sensory_dimension"].nunique()),
            "sign_consistent_rows": int(validation["sign_consistent"].sum()),
            "sign_consistent_fraction": float(validation["sign_consistent"].mean()),
            "mean_published_vs_train_top3_jaccard": float(selection_stability["jaccard"].mean()),
            "dimensions_with_full_top3_overlap": int(
                (selection_stability["overlap_count"] == 3).sum()
            ),
            "scope_note": (
                "Top-3 is reselected using training-only E2 plus the existing training-fitted E3, then tested "
                "on the fixed 75-sample holdout. BH controls the 150 preselected holdout association tests; "
                "it does not validate chemistry or causality."
            ),
        },
        "published_main_robust_holdout_review": {
            "published_robust_rows": int(len(published_robust_validation)),
            "retained_by_train_only_top3": int(
                published_robust_validation["train_selected_rank"].notna().sum()
            ),
            "sign_consistent_rows": int(
                published_robust_validation["sign_consistent"].fillna(False).sum()
            ),
            "holdout_fdr_0p05_rows": int(
                published_robust_validation["holdout_fdr_0p05"].fillna(False).sum()
            ),
            "robust_and_chem_plausible_holdout_fdr_0p05_rows": int(
                (
                    published_robust_validation["is_robust_and_chem_plausible"].astype(bool)
                    & published_robust_validation["holdout_fdr_0p05"].fillna(False)
                ).sum()
            ),
            "rows_not_passing_holdout_fdr": published_robust_validation.loc[
                ~published_robust_validation["holdout_fdr_0p05"].fillna(False),
                [
                    "sensory_dimension",
                    "compound",
                    "holdout_rho",
                    "holdout_bh_q_150",
                    "chem_plausibility",
                ],
            ].to_dict(orient="records"),
            "scope_note": (
                "This is an association robustness check. A row can reproduce statistically while remaining "
                "chemically weak, and no row is promoted to a causal claim."
            ),
        },
        "correlated_group_review": {
            "dimensions_with_top3_group_redundancy": int(
                group_review["has_correlated_group_redundancy"].sum()
            ),
            "dimensions_total": int(len(group_review)),
        },
        "historical_discrepancy": {
            "review_round2_claimed_robust_rows": 33,
            "current_recomputed_robust_rows": int(len(robust)),
            "disposition": "report_only_do_not_overwrite_main_result",
        },
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
    }
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
