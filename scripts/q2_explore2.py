# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物。
"""问题二探查第二批：MDI/SHAP 细节、拆分文件、缺失节点、off_flavors、时序基准。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import data_contract as dc


def main() -> int:
    raw = dc.load_raw()
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)

    # 1) 2.csv 节点里哪些不在 X
    b2 = chem_bridge.query("matrix=='2.csv'")
    miss = b2[~b2["file1_column"].isin(X.columns)]
    print("=== 2.csv nodes not in X ===")
    print(miss.to_string())

    # 2) MDI 细节
    mdi = raw.gbm_mdi.copy()
    mdi.columns = ["importance", "compound"]
    mdi = mdi.sort_values("importance", ascending=False)
    print("\n=== GBM importance MDI (top 20) ===")
    print(mdi.head(20).to_string())
    print("MDI sum:", mdi["importance"].sum(), "n:", len(mdi))
    print("MDI cols cover X?", mdi["compound"].isin(X.columns).sum())

    # 3) SHAP 表：定位其对应目标
    sh = raw.gbm_shap.copy()
    print("\n=== GBM SHAP Values ===", sh.shape)
    print("first cols:", list(sh.columns[:6]))
    print("beer dup count:", sh["beer"].nunique(), "rows:", len(sh))
    print(sh[["data", "beer", "beer_id", "tasting_category_fine", "base_prediction"]].head(8).to_string())
    print("data unique:", sh["data"].unique()[:10])
    # 用 SHAP 还原预测，看与哪个目标相关
    feat_cols = [c for c in sh.columns if c not in ("data", "beer", "beer_id", "tasting_category_fine", "base_prediction")]
    print("n feature cols in shap:", len(feat_cols))
    S = sh[feat_cols].to_numpy(dtype=float)
    base = sh["base_prediction"].to_numpy(dtype=float)
    pred = S.sum(axis=1) + base
    print("pred head:", np.round(pred[:5], 4))
    sh2 = sh.drop_duplicates("beer").set_index("beer")
    best = []
    for t in Y.columns:
        y = Y[t].to_numpy(dtype=float)
        idx = [i for i, b in enumerate(sh2.index) if b in set(Y.index)]
        yv = Y.loc[[b for b in sh2.index if b in set(Y.index)], t].to_numpy(dtype=float)
        pv = pred[:len(sh2)] if len(pred) == len(sh2) else None
        if pv is None:
            continue
        r = float(np.corrcoef(pv, yv)[0, 1])
        best.append((abs(r), r, t))
    best.sort(reverse=True)
    print("best target match by |corr(pred, y)|:", [(t, round(r, 4)) for _, r, t in best[:6]])

    # 4) 拆分文件
    sp = dc.load_split()
    print("\n=== split manifest ===", sp.shape, list(sp.columns))
    print(sp.head(3).to_string())
    print(sp["split"].value_counts() if "split" in sp.columns else "")
    tr1 = pd.read_csv(dc.SPLIT_DIR / "1_train.csv")
    tr4 = pd.read_csv(dc.SPLIT_DIR / "4_train.csv")
    te1 = pd.read_csv(dc.SPLIT_DIR / "1_test.csv")
    print("1_train", tr1.shape, "4_train", tr4.shape, "1_test", te1.shape)
    print("1_train cols == file1 chem cols?", [c for c in tr1.columns if c not in dc.KEY_COLUMNS] == list(raw.chem.columns))
    print("train beers:", tr1["beer"].nunique())

    # 5) off_flavors / aging 非空比例
    thr = raw.thresholds
    print("\n=== thresholds off_flavors ===")
    print("non-null off_flavors:", thr["off_flavors"].notna().sum(), "/", len(thr))
    print(thr[thr["off_flavors"].notna()][["chemical_name", "field_name_R", "origin_paper", "off_flavors"]].head(30).to_string())

    # 6) 2.csv 对称性 / 值域
    C = raw.corr_chem.to_numpy(dtype=float)
    print("\n=== 2.csv ===")
    print("symmetric max|diff|:", float(np.nanmax(np.abs(C - C.T))))
    print("min/max:", float(np.nanmin(C)), float(np.nanmax(C)))
    print("diag unique:", np.unique(np.round(np.diag(C), 6))[:5])
    C5 = raw.corr_sensory.to_numpy(dtype=float)
    print("5.csv symmetric max|diff|:", float(np.nanmax(np.abs(C5 - C5.T))))

    # 7) GBDT+SHAP 时序基准（LightGBM）
    import lightgbm as lgb
    import shap as shp
    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          verbose=-1, random_state=0, n_jobs=1)
    m.fit(X, Y["A_malt_all"])
    t1 = time.time()
    ex = shp.TreeExplainer(m)
    sv = ex.shap_values(X.iloc[:175])
    t2 = time.time()
    print(f"\nLightGBM fit {t1-t0:.3f}s, shap {t2-t1:.3f}s")
    ma = np.abs(np.asarray(sv)).mean(axis=0)
    print("shap top5:", [X.columns[i] for i in np.argsort(ma)[::-1][:5]])
    print("lgb feature_importances top5:", [X.columns[i] for i in np.argsort(m.feature_importances_)[::-1][:5]])

    # 8) 单次 Spearman 全量重算耗时
    t0 = time.time()
    R = pd.concat([X.corrwith(Y, method="spearman")], axis=1)
    print(f"\nspearman X~Y 227x50 time {time.time()-t0:.3f}s")
    print("R shape", R.shape)
    print("bitternes top5:", R["bitternes"].abs().sort_values(ascending=False).head(5).to_dict())
    print("alcohol top5:", R["alcohol"].abs().sort_values(ascending=False).head(5).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
