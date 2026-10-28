# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物；所有结构假设须在探查输出中逐项确认。
"""问题二结构探查：打印桥表 / 溯源 / MDI / 阈值 / 标签字典的实际结构。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import data_contract as dc


def main() -> int:
    raw = dc.load_raw()
    print("=== raw shapes ===")
    print("chem", raw.chem.shape, "corr_chem", raw.corr_chem.shape)
    print("sensory", raw.sensory.shape, "corr_sensory", raw.corr_sensory.shape)
    print("cross", raw.cross.shape, "forum", raw.forum.shape)
    print("thresholds", raw.thresholds.shape, "sensory_names", raw.sensory_names.shape)
    print("gbm_mdi", raw.gbm_mdi.shape, "gbm_shap", raw.gbm_shap.shape)

    print("\n=== 2.csv index head ===")
    print(list(raw.corr_chem.index[:12]))
    print("=== 6.csv index head ===")
    print(list(raw.cross.index[:12]))
    print("=== 6.csv columns head ===")
    print(list(raw.cross.columns[:12]))

    chem_bridge, sens_bridge = dc.build_bridges(raw)
    print("\n=== chem bridge ===")
    print(chem_bridge.groupby("matrix").size())
    print(chem_bridge["resolve_method"].value_counts())
    print("resolved 2.csv:", (chem_bridge.query("matrix=='2.csv'")["file1_column"] != "").sum())
    print("resolved 6.csv:", (chem_bridge.query("matrix=='6.csv'")["file1_column"] != "").sum())
    print(chem_bridge.head(8).to_string())
    print("\n=== sens bridge ===")
    print(sens_bridge.groupby("matrix").size())
    print("resolved 5.csv:", (sens_bridge.query("matrix=='5.csv'")["file4_column"] != "").sum())
    print("resolved 6.csv:", (sens_bridge.query("matrix=='6.csv'")["file4_column"] != "").sum())
    print(sens_bridge.head(8).to_string())
    dup = sens_bridge[(sens_bridge["matrix"] == "6.csv") & (sens_bridge["file4_column"] != "")]
    print("6.csv sensory unique targets:", dup["file4_column"].nunique(), "of", len(dup))

    prov = dc.load_provenance()
    print("\n=== provenance ===")
    print(prov.shape, list(prov.columns))
    print(prov["block"].value_counts())
    print(prov.head(5).to_string())
    print("origin_paper in thresholds?", "origin_paper" in raw.thresholds.columns)
    print(raw.thresholds["origin_paper"].value_counts())
    thr_map = raw.thresholds.set_index("field_name_R")
    print("thresholds field_name_R sample:", list(thr_map.index[:8]))
    print("thresholds cols:", list(raw.thresholds.columns))
    print(raw.thresholds[["chemical_name", "field_name_R", "origin_paper", "unit", "aging",
                           "off_flavors", "threshold_mean", "threshold_median"]].head(8).to_string())

    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    print("\n=== modeling matrix ===", X.shape, "targets", Y.shape)
    print("X cols head:", list(X.columns[:10]))
    print("Y cols:", list(Y.columns))

    print("\n=== gbm_mdi ===")
    print(list(raw.gbm_mdi.columns))
    print(raw.gbm_mdi.head(6).to_string())
    print("\n=== gbm_shap ===")
    print(list(raw.gbm_shap.columns))
    print(raw.gbm_shap.head(4).to_string())

    print("\n=== thresholds ===")
    print(list(raw.thresholds.columns))
    print(raw.thresholds.head(5).to_string())
    print("threshold field_name_R n unique:", raw.thresholds["field_name_R"].nunique())

    print("\n=== sensory_names ===")
    print(list(raw.sensory_names.columns))
    print(raw.sensory_names.head(5).to_string())

    labels = dc.load_label_dictionary()
    print("\n=== label dictionary ===", labels.shape, list(labels.columns))
    print(labels.to_string())

    # 覆盖度检查：2.csv 节点 -> 1.csv 列是否落在建模矩阵中
    b2 = chem_bridge.query("matrix=='2.csv'")
    inX = b2["file1_column"].isin(X.columns).sum()
    print("\n2.csv -> 1.csv col in X:", inX, "/", len(b2))
    b6 = chem_bridge.query("matrix=='6.csv'")
    print("6.csv -> 1.csv col in X:", b6["file1_column"].isin(X.columns).sum(), "/", len(b6))

    # 6.csv 列 -> 4.csv 目标 覆盖
    s6 = sens_bridge.query("matrix=='6.csv'")
    print("6.csv -> target in Y:", s6["file4_column"].isin(Y.columns).sum(), "/", len(s6))

    # 缺陷表
    print("\n=== defect table ===")
    print(dc.defect_table()[["key", "column", "threshold_mg_per_L", "log10_threshold"]].to_string())
    print(dc.defect_exceedance(X, raw).to_string())

    # 文件1 原始列 vs 建模矩阵列 差异
    chem_cols = [c for c in raw.chem.columns if c not in dc.KEY_COLUMNS]
    print("\nfile1 chem cols:", len(chem_cols), "X cols:", X.shape[1])
    print("in chem not in X:", sorted(set(chem_cols) - set(X.columns))[:20])
    print("in X not in chem:", sorted(set(X.columns) - set(chem_cols))[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
