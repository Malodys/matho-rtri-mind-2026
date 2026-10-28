# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物。
"""问题二探查第三批：拆分对齐、SHAP 结构与目标定位、bootstrap 计时。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import data_contract as dc


def spearman_matrix(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """按列取秩后算 Pearson，等价 Spearman（向量化，避免 scipy 循环）。"""
    from scipy.stats import rankdata
    Rx = rankdata(X.to_numpy(dtype=float), axis=0)
    Ry = rankdata(Y.to_numpy(dtype=float), axis=0)
    Rx = (Rx - Rx.mean(0)) / Rx.std(0)
    Ry = (Ry - Ry.mean(0)) / Ry.std(0)
    return pd.DataFrame(Rx.T @ Ry / len(Rx), index=X.columns, columns=Y.columns)


def main() -> int:
    import lightgbm as lgb
    import os

    raw = dc.load_raw()
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    print("cpu count:", os.cpu_count())

    # 1) 拆分对齐
    tr1 = pd.read_csv(dc.SPLIT_DIR / "1_train.csv")
    print("1_train shape", tr1.shape)
    set_equal = set(tr1.columns) == set(raw.chem.columns)
    print("1_train cols set == file1 cols:", set_equal)
    print("diff1:", sorted(set(tr1.columns) - set(raw.chem.columns)))
    print("diff2:", sorted(set(raw.chem.columns) - set(tr1.columns)))
    print("1_train head cols:", list(tr1.columns[:8]))

    sp = dc.load_split()
    train_beers = sp.loc[sp["split"] == "train", "beer"].to_numpy()
    print("train beers n:", len(train_beers), "dtype", train_beers.dtype)
    print("X index dtype:", X.index.dtype, "Y index dtype:", Y.index.dtype)
    print("X index == beer set?", set(X.index) == set(raw.chem["beer"]))
    Xtr = X.loc[X.index.isin(train_beers)]
    Ytr = Y.loc[Y.index.isin(train_beers)]
    print("Xtr", Xtr.shape, "Ytr", Ytr.shape, "aligned:", list(Xtr.index) == list(Ytr.index))

    # 2) gbm_shap 结构
    sh = raw.gbm_shap.copy()
    print("\n=== shap blocks ===")
    print(sh.groupby("data").size())
    shS = sh[sh["data"] == "SHAP"].copy()
    shC = sh[sh["data"] == "concentration"].copy()
    feat_cols = [c for c in sh.columns if c not in
                 ("data", "beer", "beer_id", "tasting_category_fine", "base_prediction")]
    print("shap rows:", len(shS), "conc rows:", len(shC))
    print("base_prediction unique in SHAP block:", shS["base_prediction"].nunique())
    print("conc block head:")
    print(shC[["data", "beer"] + feat_cols[:6]].head(4).to_string())
    # 还原预测并找目标
    S = shS[feat_cols].to_numpy(dtype=float)
    pred = S.sum(1) + shS["base_prediction"].to_numpy(dtype=float)
    shS = shS.reset_index(drop=True)
    shS["pred"] = pred
    print("pred head:", np.round(pred[:5], 4))
    best = []
    for t in Y.columns:
        y = Y[t].reindex(shS["beer"].to_numpy()).to_numpy(dtype=float)
        m = ~np.isnan(y)
        r = float(np.corrcoef(pred[m], y[m])[0, 1])
        best.append((abs(r), r, t))
    best.sort(reverse=True)
    print("target match |corr(pred,y)|:", [(t, round(r, 4)) for _, r, t in best[:8]])
    # 浓度块 vs 文件1 是否同标度
    conc_beers = shC["beer"].to_numpy()
    c1 = raw.chem.set_index("beer")
    for col in feat_cols[:6]:
        if col in c1.columns:
            a = shC.set_index("beer")[col].reindex(conc_beers).to_numpy(dtype=float)
            b = c1[col].reindex(conc_beers).to_numpy(dtype=float)
            print(f"conc vs file1 {col}: max|diff|={np.nanmax(np.abs(a-b)):.6g}")

    # 3) 全量 Spearman 复算（227x50）+ 与 6.csv 的一致性
    R = spearman_matrix(X, Y)
    print("\n=== E2 全量 Spearman ===", R.shape)
    for t in ["bitternes", "alcohol", "A_hops_citrus", "A_malt_bread", "A_esters_fruity", "diacetyl"]:
        s = R[t].abs().sort_values(ascending=False).head(4)
        print(t, "->", [(k, round(R[t][k], 4)) for k in s.index])

    # 4) 计时：精简 LightGBM + pred_contrib
    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.06, num_leaves=15,
                          min_child_samples=20, subsample=0.9, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=1)
    m.fit(Xtr, Ytr["A_malt_all"])
    t1 = time.time()
    sv = m.predict(Xtr, pred_contrib=True)
    t2 = time.time()
    ma = np.abs(sv[:, :-1]).mean(0)
    print(f"\nlean lgb fit {t1-t0:.3f}s pred_contrib {t2-t1:.3f}s")
    print("contrib top5:", [X.columns[i] for i in np.argsort(ma)[::-1][:5]])
    print("contrib shape", sv.shape)

    # 5) 6.csv 与 E2 的一致性校验（在 44 个共有化合物上）
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    b6 = chem_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file1_column"]
    s6 = sens_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file4_column"]
    cross = raw.cross
    diffs, n = [], 0
    for rname in cross.index:
        c = b6.get(rname)
        if not c:
            continue
        for cname in cross.columns:
            t = s6.get(cname)
            if not t or c not in R.index or t not in R.columns:
                continue
            diffs.append(abs(cross.loc[rname, cname] - R.loc[c, t]))
            n += 1
    diffs = np.array(diffs)
    print(f"\n6.csv vs E2(Spearman on X) n={n} max|Δ|={diffs.max():.6g} frac<1e-8={np.mean(diffs<1e-8):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
