#!/usr/bin/env python3
"""Build explicit, versioned input contracts for Q1/Q2/Q3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SUMS = ["esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "interim"
    paths = {
        "matrix": out / "chem_log10_matrix_v1.csv",
        "provenance": out / "chem_feature_provenance_v1.csv",
        "status": out / "chem_observation_status_v1.csv",
        "status_dictionary": out / "chem_observation_status_v1_dictionary.json",
        "nondetect": out / "chem_nondetect_mask_v1.csv",
        "label_dictionary": out / "label_dictionary_v1.csv",
        "raw_labels": root / "raw_data" / "4.csv",
        "correlation": root / "raw_data" / "2.csv",
        "metadata": root / "raw_data" / "8.xlsx",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    x = pd.read_csv(paths["matrix"], index_col=0)
    x.index = x.index.astype(int)
    provenance = pd.read_csv(paths["provenance"], index_col=0).reindex(x.columns)
    status = pd.read_csv(paths["status"], index_col=0).reindex(index=x.index, columns=x.columns)
    raw_y = pd.read_csv(paths["raw_labels"], encoding="utf-8-sig").set_index("beer")
    raw_y.index = raw_y.index.astype(int)
    raw_y = raw_y.reindex(x.index)
    label_dictionary = pd.read_csv(paths["label_dictionary"], index_col=0)
    assert x.shape == status.shape == (250, 231)
    assert not x.isna().any().any() and not status.isna().any().any()
    assert set(SUMS) == {c for c in x if c.endswith("_sum")}

    registry = provenance.copy()
    registry.index.name = "feature"
    registry["feature_set"] = "chemical"
    registry.loc[registry.block.eq("标准理化指标"), "feature_set"] = "physicochemical"
    registry.loc[registry.index.isin(SUMS), "feature_set"] = "derived_sum"
    registry["q1_primary"] = registry.feature_set.isin(["chemical", "physicochemical"])
    registry["q1_sum_sensitivity"] = True
    registry["q2_primary_network"] = registry.feature_set.eq("chemical")
    registry["q2_top3_candidate"] = registry.feature_set.eq("chemical")
    registry["q2_context_layer"] = registry.feature_set.isin(["physicochemical", "derived_sum"])
    registry["q3_primary_generation"] = registry.feature_set.eq("chemical")
    registry["q3_optional_context"] = registry.feature_set.eq("physicochemical")
    registry["q3_generate_directly"] = registry.feature_set.eq("chemical")
    registry["is_derived"] = registry.feature_set.eq("derived_sum")
    registry["formula_verified"] = ~registry.is_derived
    registry["has_left_censoring"] = status.eq(1).any().to_numpy()
    registry["has_model_imputation"] = status.eq(2).any().to_numpy()
    registry["rule"] = registry.feature_set.map({
        "chemical": "retain registered scale; carry observation status",
        "physicochemical": "retain measurement-aware scale; no blanket log",
        "derived_sum": "exclude from primary Q2/Q3 until formula verified",
    })
    registry_path = out / "chem_feature_sets_v1.csv"
    registry.to_csv(registry_path, encoding="utf-8-sig")

    y = raw_y[list(label_dictionary.index)].apply(pd.to_numeric, errors="coerce")
    assert y.shape == (250, 50) and not y.isna().any().any()
    y.index.name = "beer"
    y_path = out / "sensory_targets_v1.csv"
    y.to_csv(y_path)

    styles = raw_y[["tasting_category_fine"]].rename(columns={"tasting_category_fine": "style"})
    styles["style_n"] = styles["style"].map(styles["style"].value_counts())
    styles["rare_style_lt5"] = styles.style_n < 5
    styles.index.name = "beer"
    styles_path = out / "beer_style_v1.csv"
    styles.to_csv(styles_path, encoding="utf-8-sig")

    rules = {
        "version": "v1",
        "frozen_primary_matrix": "data/interim/chem_log10_matrix_v1.csv",
        "principles": [
            "No blanket log transform; use each feature's registered scale_action.",
            "Flag outliers without deleting samples or changing the frozen matrix.",
            "Carry observation status; censored and model-imputed cells are not ordinary observations.",
            "Fit data-dependent imputation, scaling, selection and dimension reduction inside training folds.",
            "Manage the four derived sums separately until their formulas, scales and weights are verified.",
        ],
        "feature_sets": {"chemical": 211, "physicochemical": 16, "derived_sum": 4},
        "derived_sum_policy": {
            "features": SUMS, "q2_primary": False, "q3_direct_generation": False,
            "recompute": "only after formulas, component scales and weights are verified",
            "aroma_hops_sum": "NNLS fill preserves matrix completeness but does not verify its formula",
        },
        "status_codes": json.loads(paths["status_dictionary"].read_text(encoding="utf-8")),
    }
    rules_path = out / "data_processing_rules_v1.json"
    dump(rules_path, rules)

    common = {
        "contract_version": "v1", "join_key": "beer", "row_count": 250,
        "chemistry_matrix": reference(paths["matrix"], root),
        "feature_registry": reference(registry_path, root),
        "observation_status": reference(paths["status"], root),
        "nondetect_mask": reference(paths["nondetect"], root),
        "processing_rules": reference(rules_path, root),
    }
    manifests = {
        "q1_input_manifest_v1.json": {**common, "task": "Q1 prediction",
            "feature_selector": {"column": "q1_primary", "value": True, "expected_count": 227},
            "sum_sensitivity_selector": {"column": "q1_sum_sensitivity", "value": True, "expected_count": 231},
            "targets": reference(y_path, root), "target_dictionary": reference(paths["label_dictionary"], root),
            "styles": reference(styles_path, root)},
        "q2_input_manifest_v1.json": {**common, "task": "Q2 network and sensory Top-3",
            "primary_network_selector": {"column": "q2_primary_network", "value": True, "expected_count": 211},
            "top3_selector": {"column": "q2_top3_candidate", "value": True, "expected_count": 211},
            "context_selector": {"column": "q2_context_layer", "value": True, "expected_count": 20},
            "published_descriptive_correlation": reference(paths["correlation"], root),
            "bootstrap_source": "chemistry_matrix", "targets": reference(y_path, root),
            "required_edge_audits": ["shared censoring/detection pattern", "bootstrap sign/selection stability", "community stability"]},
        "q3_input_manifest_v1.json": {**common, "task": "Q3 conditional generation and style transfer",
            "generation_selector": {"column": "q3_primary_generation", "value": True, "expected_count": 211},
            "optional_context_selector": {"column": "q3_optional_context", "value": True, "expected_count": 16},
            "styles": reference(styles_path, root), "chemical_metadata_and_thresholds": reference(paths["metadata"], root),
            "derived_sum_policy": "exclude from direct generation; recompute only after formula verification",
            "sensory_integration": "use final Q1 predictor for sample-level profiles"},
    }
    manifest_paths = []
    for name, manifest in manifests.items():
        p = out / name
        dump(p, manifest)
        manifest_paths.append(p)

    shared = {
        "contract_version": "v1", "produced_by": "src/build_data_contracts.py",
        "upstream_notebooks": ["notebooks/02_chem_process.ipynb", "notebooks/03_label_prediction.ipynb"],
        "canonical_artifacts": {"chemistry_matrix": reference(paths["matrix"], root),
            "feature_registry": reference(registry_path, root), "observation_status": reference(paths["status"], root),
            "sensory_targets": reference(y_path, root), "beer_styles": reference(styles_path, root),
            "processing_rules": reference(rules_path, root)},
        "task_manifests": [reference(p, root) for p in manifest_paths],
        "forbidden_legacy_entry_points": ["data/interim/chem_log10_matrix.csv"],
    }
    dump(out / "shared_data_manifest_v1.json", shared)
    print("contracts built: chemical=211, physicochemical=16, derived_sum=4, targets=50, styles=22")


if __name__ == "__main__":
    main()
