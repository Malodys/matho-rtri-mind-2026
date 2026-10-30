# -*- coding: utf-8 -*-
"""Regression tests for release-only contracts added during hardening."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import acceptance_check  # noqa: E402


def test_independent_model_artifact_contract():
    artifact = ROOT / "data" / "processed" / "q1_final" / "model_independent_gbr.joblib"
    metadata_path = ROOT / "data" / "processed" / "q1_final" / "model_independent_gbr_metadata.json"
    assert artifact.is_file() and artifact.stat().st_size > 1_000_000
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["feature_columns"]) == 227
    assert len(metadata["target_columns"]) == 50
    assert metadata["holdout_macro_rmse"] < metadata["holdout_baseline_macro_rmse"]


def test_remote_branch_classification_is_explicit():
    heads = "abc123\trefs/heads/release/toolchain-hardening-20261029\n"
    assert acceptance_check._classify_remote_branch("reachable", heads, "release/toolchain-hardening-20261029") == "branch_present"
    assert acceptance_check._classify_remote_branch("reachable", heads, "release/other") == "branch_missing"
    assert acceptance_check._classify_remote_branch("unreachable", "", "release/other") == "unreachable"
    assert acceptance_check._classify_remote_branch("local_environment_broken", "", "release/other") == "local_environment_broken"


def test_release_manifest_records_model_and_external_contract():
    manifest_path = ROOT / "data" / "processed" / "release_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["model_contract"]
    assert contract["n_features"] == 227
    assert contract["n_targets"] == 50
    assert contract["artifact"].endswith("model_independent_gbr.joblib")
    assert manifest["external_data_policy"] == "validation_and_metadata_only; no external rows in main training set"
    assert isinstance(manifest["external_sources"], list)
