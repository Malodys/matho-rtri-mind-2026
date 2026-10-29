# -*- coding: utf-8 -*-
"""Regression tests for the Q1 absolute-track aggregate contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "data" / "processed" / "q1_final"


@pytest.fixture(scope="module")
def absolute_outputs():
    summary_path = FINAL_DIR / "model_summary.csv"
    detail_path = FINAL_DIR / "absolute_grade_metrics_by_label.csv"
    anchor_path = FINAL_DIR / "absolute_anchor_params.csv"
    if not (summary_path.is_file() and detail_path.is_file() and anchor_path.is_file()):
        pytest.skip("Q1 absolute-track outputs are not available")
    return (
        pd.read_csv(summary_path),
        pd.read_csv(detail_path),
        pd.read_csv(anchor_path),
    )


def test_absolute_summary_has_separate_populations_and_valid_alias(absolute_outputs):
    summary, detail, anchor = absolute_outputs
    assert {"abs_threshold_MAE", "abs_fallback_MAE", "abs_valid_MAE", "abs_MAE"} <= set(summary.columns)

    valid = anchor.loc[anchor["valid"].astype(bool), "dimension"].tolist()
    threshold = anchor.loc[
        anchor["valid"].astype(bool) & anchor["anchor_type"].eq("threshold_anchored"),
        "dimension",
    ].tolist()
    fallback = anchor.loc[
        anchor["valid"].astype(bool) & anchor["anchor_type"].eq("relative_fallback"),
        "dimension",
    ].tolist()

    assert set(detail["dimension"]) == set(valid)
    assert threshold and fallback and len(valid) == len(threshold) + len(fallback)

    metric_map = {
        "MAE": "MAE",
        "exact_accuracy": "exact_accuracy",
        "within_one_accuracy": "within_one_accuracy",
        "QWK": "QWK",
    }
    best = summary.iloc[0]
    for summary_metric, detail_metric in metric_map.items():
        valid_mean = detail[detail["dimension"].isin(valid)][detail_metric].mean()
        threshold_mean = detail[detail["dimension"].isin(threshold)][detail_metric].mean()
        fallback_mean = detail[detail["dimension"].isin(fallback)][detail_metric].mean()
        assert np.isclose(best[f"abs_valid_{summary_metric}"], valid_mean)
        assert np.isclose(best[f"abs_threshold_{summary_metric}"], threshold_mean)
        assert np.isclose(best[f"abs_fallback_{summary_metric}"], fallback_mean)
        assert np.isclose(best[f"abs_{summary_metric}"], best[f"abs_valid_{summary_metric}"])
        assert not np.isclose(best[f"abs_{summary_metric}"], best[f"abs_fallback_{summary_metric}"])
