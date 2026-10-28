# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：断言阈值来自本轮实测（化学桥 275/275、感官桥 100/100、矩阵 250×227、
# 目标 250×50、切分 175/75），逐条核对后写定。
"""共享数据层契约测试：锁住本轮经实测裁定的数据事实。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_contract as dc  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return dc.load_raw()


@pytest.fixture(scope="module")
def bridges(raw):
    return dc.build_bridges(raw)


def test_raw_shapes_match_attachment_description(raw):
    assert raw.chem.shape == (250, 234)
    assert raw.corr_chem.shape == (231, 231)
    assert raw.sensory.shape == (250, 53)
    assert raw.corr_sensory.shape == (50, 50)
    assert raw.cross.shape == (44, 50)
    assert raw.forum.shape[0] == 250


def test_chemical_bridge_resolves_every_node(bridges, raw):
    chem_bridge, _ = bridges
    m2 = chem_bridge[chem_bridge["matrix"] == "2.csv"]
    m6 = chem_bridge[chem_bridge["matrix"] == "6.csv"]
    assert len(m2) == 231
    assert (m2["file1_column"] != "").all(), m2[m2["file1_column"] == ""]["matrix_name"].tolist()
    assert len(m6) == 44
    assert (m6["file1_column"] != "").all()
    # 桥接后的 1.csv 列必须真实存在
    file1_columns = set(raw.chem.columns)
    assert set(m2["file1_column"]) <= file1_columns
    assert set(m6["file1_column"]) <= file1_columns


def test_sensory_bridge_resolves_every_column(bridges):
    _, sens_bridge = bridges
    m5 = sens_bridge[sens_bridge["matrix"] == "5.csv"]
    m6 = sens_bridge[sens_bridge["matrix"] == "6.csv"]
    assert len(m5) == 50 and len(m6) == 50
    assert (m5["file4_column"] != "").all()
    assert (m6["file4_column"] != "").all()
    targets = set(dc.load_targets().columns)
    assert set(m5["file4_column"]) <= targets
    assert set(m6["file4_column"]) <= targets


def test_unified_matrix_shape_and_no_missing():
    matrix = dc.load_modeling_matrix()
    assert matrix.shape == (250, 227)
    assert not matrix.isna().any().any()
    assert all(str(c).endswith("_sum") is False for c in matrix.columns)
    assert not [c for c in matrix.columns if str(c).endswith("_sum")]


def test_split_is_locked_175_75():
    split = dc.load_split()
    assert int((split["split"] == "train").sum()) == 175
    assert int((split["split"] == "test").sum()) == 75
    assert split["beer"].is_unique
    assert len(set(split["beer"])) == 250


def test_targets_shape_and_style_alignment():
    targets = dc.load_targets()
    styles = dc.load_styles()
    assert targets.shape == (250, 50)
    assert list(targets.index) == list(styles.index)
    assert styles.nunique() == 22


def test_defect_thresholds_use_unified_log10_scale():
    table = dc.defect_table()
    assert len(table) == 4
    dms = table[table["key"] == "dms"].iloc[0]
    # 17.25 µg/L → 0.01725 mg/L → log10
    assert dms["log10_threshold"] == pytest.approx(np.log10(0.01725), abs=1e-6)
    dia = table[table["key"] == "diacetyl"].iloc[0]
    assert dia["log10_threshold"] == pytest.approx(np.log10(0.29925), abs=1e-6)
    vg = table[table["key"] == "4vg"].iloc[0]
    assert vg["log10_threshold"] == pytest.approx(np.log10(0.206333), abs=1e-6)
    t2n = table[table["key"] == "t2n"].iloc[0]
    assert not t2n["has_threshold"]
    assert np.isnan(t2n["log10_threshold"])


def test_defect_exceedance_matches_measured_rates():
    matrix = dc.to_unified_scale(dc.load_raw())
    exc = dc.defect_exceedance(matrix).set_index("key")
    assert exc.loc["dms", "n_exceed"] == 168
    assert exc.loc["4vg", "n_exceed"] == 16
    assert exc.loc["dms", "exceed_pct"] == pytest.approx(67.2, abs=0.5)


def test_manifest_records_raw_hashes(raw):
    for name in ("1.csv", "2.csv", "3.xlsx", "4.csv", "5.csv", "6.csv", "7.csv", "8.xlsx"):
        assert name in raw.manifest
        assert len(raw.manifest[name]["sha256"]) == 64
