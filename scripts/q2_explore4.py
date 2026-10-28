# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物。
"""问题二探查第四批：定位 6.csv 与 E2 的差异来源（标度 / 插补）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import data_contract as dc
from scipy.stats import rankdata


def spearman_matrix(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    Rx = rankdata(X.to_numpy(dtype=float), axis=0)
    Ry = rankdata(Y.to_numpy(dtype=float), axis=0)
    Rx = (Rx - Rx.mean(0)) / Rx.std(0)
    Ry = (Ry - Ry.mean(0)) / Ry.std(0)
    return pd.DataFrame(Rx.T @ Ry / len(Rx), index=X.columns, columns=Y.columns)


def main() -> int:
    raw = dc.load_raw()
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    chem_bridge, sens_bridge = dc.build_bridges(raw)
    b6 = chem_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file1_column"]
    s6 = sens_bridge.query("matrix=='6.csv'").set_index("matrix_name")["file4_column"]

    # (a) 在 1.csv 原始存储标度上重算 Spearman
    chem = raw.chem.set_index("beer")
    chem_cols = [c for c in chem.columns if c not in dc.KEY_COLUMNS]
    Xraw = chem[chem_cols]
    Rraw = spearman_matrix(Xraw, Y)
    Runi = spearman_matrix(X, Y)
    print("Xraw", Xraw.shape, "NaN per col total:", int(Xraw.isna().sum().sum()))

    cross = raw.cross
    rows = []
    for rname in cross.index:
        c = b6.get(rname)
        if not c:
            continue
        for cname in cross.columns:
            t = s6.get(cname)
            if not t:
                continue
            v6 = float(cross.loc[rname, cname])
            rr = Rraw.loc[c, t] if c in Rraw.index else np.nan
            ru = Runi.loc[c, t] if (c in Runi.index and t in Runi.columns) else np.nan
            rows.append({"comp": c, "target": t, "v6": v6, "rho_raw": rr, "rho_unified": ru,
                         "d_raw": abs(v6 - rr), "d_uni": abs(v6 - ru),
                         "nan_in_comp": int(Xraw[c].isna().sum()) if c in Xraw.columns else -1})
    df = pd.DataFrame(rows)
    print("\nvs E2raw (原始存储标度): max|Δ|=%.3g frac<1e-8=%.4f"
          % (df["d_raw"].max(), (df["d_raw"] < 1e-8).mean()))
    print("vs E2uni (统一 log10 矩阵): max|Δ|=%.3g frac<1e-8=%.4f"
          % (df["d_uni"].max(), (df["d_uni"] < 1e-8).mean()))
    bad = df[df["d_uni"] > 1e-8]
    print("\n差异单元格数:", len(bad), "涉及化合物:", bad["comp"].nunique())
    print(bad.groupby("comp").agg(n=("d_uni", "size"), maxd=("d_uni", "max"),
                                  nan_in_comp=("nan_in_comp", "first"),
                                  d_raw_max=("d_raw", "max")).sort_values("maxd", ascending=False).to_string())

    # (b) 差异化合物的 NaN 情况
    if len(bad):
        cols = sorted(bad["comp"].unique())
        print("\n这些化合物在 1.csv 中的 NaN 数:")
        for c in cols:
            print(f"  {c}: NaN={int(Xraw[c].isna().sum())}, "
                  f"file1 中是否含负值={bool((Xraw[c].dropna() < 0).any())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
