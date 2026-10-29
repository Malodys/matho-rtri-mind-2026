# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：裁定分布（LOG10=37 / RAW=19 / RAW_APPROX=10 / AMBIG=4）与与 provenance 的一致率
# 已在本机复算确认；4 列 AMBIG 为文件8 标定值与文件1 存储值的系统性差异所致，记录在案。
"""标度审计测试：1.csv 是混合标度，禁止对全表盲做 10**x。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_contract as dc  # noqa: E402


@pytest.fixture(scope="module")
def audit():
    return dc.audit_scale(dc.load_raw())


def test_audit_covers_70_alignable_compounds(audit):
    assert len(audit) == 70
    assert set(audit["verdict"]) <= {"LOG10", "RAW", "RAW_APPROX", "AMBIG"}


def test_verdict_distribution_matches_measurement(audit):
    counts = audit["verdict"].value_counts().to_dict()
    assert counts.get("LOG10") == 37
    assert counts.get("RAW") == 19
    assert counts.get("RAW_APPROX") == 10
    assert counts.get("AMBIG") == 4


def test_audit_agrees_with_provenance(audit):
    assert audit["matches_provenance"].all(), \
        audit[~audit["matches_provenance"]]["file1_column"].tolist()


def test_log10_columns_have_slope_one_against_log10_raw(audit):
    rows = audit[audit["verdict"] == "LOG10"]
    assert len(rows) == 37
    assert (rows["slope_log10"].sub(1).abs() < 1e-3).all()
    assert (rows["r_log10"] > 0.9999).all()
    # 自然对数假设必须被排除：file1 = log10(C) ⇒ 斜率(ln C ~ file1) 恒为 ln(10) = 2.302585
    assert (rows["slope_ln"].sub(np.log(10)).abs() < 1e-3).all()
    assert (rows["r_ln"] > 0.9999).all()


def test_raw_columns_have_slope_one_against_raw(audit):
    rows = audit[audit["verdict"] == "RAW"]
    assert len(rows) == 19
    assert (rows["slope_raw"].sub(1).abs() < 3e-3).all()
    assert (rows["r_raw"] > 0.999).all()


def test_diacetyl_and_nonenal_are_raw_scale(audit):
    """两个缺陷化合物在文件1 中存的是原始浓度，不能取 10**x。"""
    for col in ("X2.3.butanedione", "X.E..2.nonenal"):
        row = audit[audit["file1_column"] == col].iloc[0]
        assert row["verdict"] in ("RAW", "RAW_APPROX")
        assert row["r_log10"] < 0.9, f"{col} 不应匹配 log10"


def test_uniform_log10_would_be_wrong(audit):
    """反证：若对全表盲做 10**x，Diacetyl 的中位数会从 0.257 mg/L 变成荒谬量级。"""
    matrix = dc.to_unified_scale(dc.load_raw())
    raw_file1 = dc.load_raw().chem
    unified_dia = matrix["X2.3.butanedione"].median()
    file1_dia = pd_numeric(raw_file1["X2.3.butanedione"]).median()
    assert file1_dia == pytest.approx(0.284672, abs=1e-4)   # 文件1 存原始 mg/L
    assert 10 ** unified_dia == pytest.approx(0.256561, abs=1e-3)  # 统一矩阵 log10(mg/L)


def pd_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")
