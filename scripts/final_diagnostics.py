# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：5.csv/6.csv 的相关类型判定、样本量分析与 8.xlsx 明细结论已人工复核。
"""Final diagnostics: 5.csv correlation type (positional alignment),
6.csv reproducibility, sample-size analysis, 8.xlsx detail."""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
KEY = ["beer", "beer_id", "tasting_category_fine"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


d1 = pd.read_csv(os.path.join(RAW, "1.csv"))
d2 = pd.read_csv(os.path.join(RAW, "2.csv"), index_col=0)
d4 = pd.read_csv(os.path.join(RAW, "4.csv"))
d5 = pd.read_csv(os.path.join(RAW, "5.csv"), index_col=0)
d6 = pd.read_csv(os.path.join(RAW, "6.csv"), index_col=0)
x8 = pd.read_excel(os.path.join(RAW, "8.xlsx"), sheet_name=None)
th, sn = x8["Sensory thresholds"], x8["Sensory names"]

chem1 = [c for c in d1.columns if c not in KEY]
sens4 = [c for c in d4.columns if c not in KEY]

print("=" * 80)
print("5.csv : NAME + CORRELATION TYPE")
print("=" * 80)
print("5.csv index (first 6 / last 3):", list(d5.index)[:6], "...", list(d5.index)[-3:])
print("4.csv sensory cols (first 6 / last 3):", sens4[:6], "...", sens4[-3:])
print(f"counts: 5.csv index={len(d5.index)}  5.csv cols={len(d5.columns)}  4.csv sensory={len(sens4)}")

sn_map = {}
for _, r in sn.iterrows():
    sn_map.setdefault(r.get("CleanName_PaperLong"), []).append(r.get("R_name"))
pos_ok = 0
name_ok = 0
for i, (nm, col) in enumerate(zip(d5.index, sens4)):
    cand = sn_map.get(nm, [])
    if col in cand:
        name_ok += 1
print(f"positional alignment consistent with 'Sensory names' dictionary: {name_ok}/{len(sens4)}")

if len(d5.index) == len(sens4) == len(d5.columns):
    M5 = d4[sens4].apply(pd.to_numeric, errors="coerce")
    i5 = np.triu_indices(len(sens4), 1)
    g5 = d5.to_numpy(dtype=float)[i5]
    p5 = M5.corr(method="pearson").to_numpy()[i5]
    s5 = M5.corr(method="spearman").to_numpy()[i5]
    dp, ds = np.abs(g5 - p5), np.abs(g5 - s5)
    print(f"\n  |given - pearson |  max={np.nanmax(dp):.8f} median={np.nanmedian(dp):.8f} frac<1e-8={np.mean(dp<1e-8):.3f}")
    print(f"  |given - spearman|  max={np.nanmax(ds):.8f} median={np.nanmedian(ds):.8f} frac<1e-8={np.mean(ds<1e-8):.3f}")
    print("  >>> VERDICT 5.csv = " + ("PEARSON" if np.nanmedian(dp) < np.nanmedian(ds) else "SPEARMAN"))

print("\n" + "=" * 80)
print("6.csv : IS IT REPRODUCIBLE FROM 1.csv x 4.csv ?")
print("=" * 80)
R2PY = dict(zip(th["chemical_name"], th["field_name_R"]))
n1 = {norm(c): c for c in chem1}
for c in chem1:
    n1.setdefault(norm(c), c)


def to_chem(name):
    if name in R2PY and R2PY[name] in chem1:
        return R2PY[name]
    return n1.get(norm(name))


r6, c6 = list(d6.index), list(d6.columns)
m6 = {r: to_chem(r) for r in r6}
ok6 = {k: v for k, v in m6.items() if v in chem1}
cc = [c for c in c6 if c in sens4]
print(f"rows mapped to 1.csv: {len(ok6)}/{len(r6)}   sensory cols matched to 4.csv: {len(cc)}/{len(c6)}")
print(f"unmapped compound rows: {[r for r in r6 if r not in ok6][:10]}")
if len(ok6) >= 8 and len(cc) >= 8:
    rr = list(ok6.keys())
    Xc = d1[[ok6[r] for r in rr]].apply(pd.to_numeric, errors="coerce")
    Ys = d4[cc].apply(pd.to_numeric, errors="coerce")
    gc = d6.loc[rr, cc].to_numpy(dtype=float)
    for nm, meth in [("pearson", "pearson"), ("spearman", "spearman")]:
        dd = np.abs(gc - Xc.corr(method=meth).to_numpy())
        print(f"  {nm:9s} max={np.nanmax(dd):8.5f} median={np.nanmedian(dd):8.5f} frac<1e-6={np.mean(dd<1e-6):.3f}")
    stg = np.nanmax(np.abs(gc), axis=0)
    print("\n  strongest sensory attributes in 6.csv:")
    for j in np.argsort(-stg)[:10]:
        print(f"    {cc[j]:24s} max|r|={stg[j]:.3f}")
    print("\n  the 44 compounds in 6.csv:")
    print("   ", [r for r in r6])

print("\n" + "=" * 80)
print("SAMPLE SIZE")
print("=" * 80)
vc = d1["tasting_category_fine"].value_counts()
print(f"n={len(d1)}  p_chem={len(chem1)}  p_sensory={len(sens4)}  p/n={len(chem1)/len(d1):.2f}")
print(f"styles={len(vc)}   n>=20: {(vc>=20).sum()}   10<=n<20: {((vc>=10)&(vc<20)).sum()}   "
      f"5<=n<10: {((vc>=5)&(vc<10)).sum()}   n<5: {(vc<5).sum()}")
print(f"styles with n<10: {vc[vc<10].to_dict()}")
print(f"samples in styles n<10: {vc[vc<10].sum()}/{len(d1)} ({100*vc[vc<10].sum()/len(d1):.1f}%)")
print(f"smallest cell n={vc.min()} -> style-stratified 5-fold per-fold train size ~ {vc.min()*4/5:.1f}")
# effective n if we include style dummies
print(f"if style is one-hot encoded: design matrix rank ~ {len(chem1)} predictors vs n={len(d1)}")

print("\n" + "=" * 80)
print("8.xlsx DETAIL")
print("=" * 80)
for k, v in x8.items():
    print(f"\n--- {k}  shape={v.shape}  NA={int(v.isna().sum().sum())} ---")
    print(f"    cols: {list(v.columns)}")

print("\n[Sensory thresholds] key flags:")
for c in ["aging", "off_flavors", "calibrated", "origin_paper", "unit"]:
    if c in th.columns:
        print(f"  {c}: {th[c].value_counts(dropna=False).head(8).to_dict()}")
print(f"  threshold_mean non-null: {int(th['threshold_mean'].notna().sum())}/{len(th)}")
for c in ["threshold_mean", "threshold_median"]:
    if c in th.columns and th[c].notna().any():
        print(f"  {c} range: {th[c].min():.6g} .. {th[c].max():.6g}")

mi = x8["GBM importance MDI"]
print(f"\n[GBM importance MDI] shape={mi.shape} cols={list(mi.columns)}")
print(mi.head(15).to_string(index=False))

sh = x8["GBM SHAP Values"]
print(f"\n[GBM SHAP Values] shape={sh.shape}")
print(f"  first 6 cols: {list(sh.columns)[:6]}   last 3: {list(sh.columns)[-3:]}")
print(sh.iloc[:3, :6].to_string(index=False))

cal = x8["Calibrated chemical dataset"]
print(f"\n[Calibrated chemical dataset] shape={cal.shape}")
print(f"  cols: {list(cal.columns)}")

bp = x8["Beer Prices"]
print(f"\n[Beer Prices] shape={bp.shape} cols={list(bp.columns)}")
print(bp.head(5).to_string(index=False))
