# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：Spearman 判定基于「重算 vs 给定矩阵」的逐元素比对，max|Δ|=0 已复算确认。
"""相关类型回归测试：锁定 2.csv / 5.csv / 6.csv 为 Spearman（防止后续被误改成 Pearson）。

背景：官方赛题 PDF 称「表格 2 单元格为皮尔逊相关系数」，而《数据说明.txt》称 Spearman。
本轮以「从 1.csv / 4.csv 重算并逐元素比对」裁定为 Spearman，本测试把该结论固化为回归保护。
"""
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


def _upper(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M.to_numpy(dtype=float)[iu]


def test_5csv_equals_spearman_of_4csv(raw):
    """5.csv = 4.csv 目标列两两 Spearman（max|Δ| 必须为 0）。"""
    bridge = dc.build_sensory_bridge(raw)
    cols = [n for n in raw.corr_sensory.index
            if bridge[(bridge["matrix"] == "5.csv") & (bridge["matrix_name"] == n)]["file4_column"].iloc[0]]
    assert len(cols) == 50
    targets = dc.load_targets(raw)
    names = [bridge[(bridge["matrix"] == "5.csv") & (bridge["matrix_name"] == n)]["file4_column"].iloc[0]
             for n in cols]
    M = targets[names].copy()
    M.columns = list(cols)
    given = raw.corr_sensory.loc[cols, cols]
    spearman = M.corr(method="spearman").to_numpy()
    pearson = M.corr(method="pearson").to_numpy()
    iu = np.triu_indices(len(cols), 1)
    g = _upper(given)
    d_s = np.abs(g - spearman[iu])
    d_p = np.abs(g - pearson[iu])
    assert np.nanmax(d_s) < 1e-8
    assert np.nanmedian(d_p) > 1e-3, "Pearson 不应匹配，若匹配说明数据或桥表已变"


def test_6csv_equals_spearman_cross_matrix(raw):
    """6.csv = 1.csv 化合物 × 4.csv 感官 的 Spearman（max|Δ| 必须为 0）。"""
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    rows = [n for n in raw.cross.index
            if chem_bridge[(chem_bridge["matrix"] == "6.csv") & (chem_bridge["matrix_name"] == n)]["file1_column"].iloc[0]]
    cols = [n for n in raw.cross.columns
            if sens_bridge[(sens_bridge["matrix"] == "6.csv") & (sens_bridge["matrix_name"] == n)]["file4_column"].iloc[0]]
    assert len(rows) == 44 and len(cols) == 50
    X = raw.chem.set_index("beer")
    Xc = X[[chem_bridge[(chem_bridge["matrix"] == "6.csv") & (chem_bridge["matrix_name"] == n)]["file1_column"].iloc[0]
            for n in rows]].apply(pd.to_numeric, errors="coerce")
    Xc.columns = list(rows)
    Y = dc.load_targets(raw)
    Yc = Y[[sens_bridge[(sens_bridge["matrix"] == "6.csv") & (sens_bridge["matrix_name"] == n)]["file4_column"].iloc[0]
            for n in cols]].copy()
    Yc.columns = list(cols)
    both = pd.concat([Xc, Yc], axis=1)
    # 化合物名与感官名可能重名（如 Diacetyl / Lactic acid 同时出现在行与列），必须加前缀去歧义
    both.columns = ["X_" + c for c in Xc.columns] + ["Y_" + c for c in Yc.columns]
    spearman = both.corr(method="spearman").loc[["X_" + n for n in rows],
                                                ["Y_" + n for n in cols]].to_numpy()
    pearson = both.corr(method="pearson").loc[["X_" + n for n in rows],
                                              ["Y_" + n for n in cols]].to_numpy()
    g = raw.cross.loc[rows, cols].to_numpy(dtype=float)
    assert np.nanmax(np.abs(g - spearman)) < 1e-8
    assert np.nanmedian(np.abs(g - pearson)) > 1e-3


def test_2csv_equals_spearman_of_file1_stored_scale(raw):
    """2.csv = 1.csv 原始存储标度上的 Spearman（max|Δ| 必须为 0）。"""
    chem_bridge, _ = dc.build_bridges(raw)
    nodes = [n for n in raw.corr_chem.index
             if chem_bridge[(chem_bridge["matrix"] == "2.csv") & (chem_bridge["matrix_name"] == n)]["file1_column"].iloc[0]]
    assert len(nodes) == 231
    X = raw.chem.set_index("beer")
    cols = [chem_bridge[(chem_bridge["matrix"] == "2.csv") & (chem_bridge["matrix_name"] == n)]["file1_column"].iloc[0]
            for n in nodes]
    M = X[cols].apply(pd.to_numeric, errors="coerce")
    M.columns = list(nodes)
    g = raw.corr_chem.loc[nodes, nodes].to_numpy(dtype=float)
    iu = np.triu_indices(len(nodes), 1)
    d_s = np.abs(g[iu] - M.corr(method="spearman").to_numpy()[iu])
    d_p = np.abs(g[iu] - M.corr(method="pearson").to_numpy()[iu])
    assert np.nanmax(d_s) < 1e-8
    assert np.nanmedian(d_p) > 1e-3


def test_correlation_matrices_are_symmetric_with_unit_diagonal(raw):
    for mat in (raw.corr_chem, raw.corr_sensory):
        v = mat.to_numpy(dtype=float)
        assert np.allclose(v, v.T, atol=1e-10)
        assert np.allclose(np.diag(v), 1.0, atol=1e-10)
