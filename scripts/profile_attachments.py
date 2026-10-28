# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：附件 1–8 的字段清单与 8.xlsx 提供的两套名称桥已人工核对。
"""Full inventory of attachments 1-8 + decisive test of correlation type.

Name bridges come from 8.xlsx:
  'Sensory thresholds' : field_name_R / field_name_python  <-> chemical_name
  'Sensory names'      : R_name (4.csv col)                 <-> CleanName_PanelFeedback
"""
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
d7 = pd.read_csv(os.path.join(RAW, "7.csv"))
x3 = pd.read_excel(os.path.join(RAW, "3.xlsx"), sheet_name=None)
x8 = pd.read_excel(os.path.join(RAW, "8.xlsx"), sheet_name=None)

chem1 = [c for c in d1.columns if c not in KEY]
sens4 = [c for c in d4.columns if c not in KEY]
th = x8["Sensory thresholds"]
sn = x8["Sensory names"]

print("#" * 80)
print("# A. FILE INVENTORY")
print("#" * 80)
for nm, obj in [("1.csv", d1), ("2.csv", d2), ("4.csv", d4), ("5.csv", d5),
                ("6.csv", d6), ("7.csv", d7)]:
    na = int(obj.isna().sum().sum())
    print(f"\n{nm}: shape={obj.shape}  NA={na} ({100*na/obj.size:.2f}%)")
    if nm == "1.csv":
        na_c = obj[chem1].isna().sum()
        print(f"    features={len(chem1)}   features containing NA={int((na_c>0).sum())}  cells NA={int(na_c.sum())}")
        print(na_c[na_c > 0].sort_values(ascending=False).to_string())
    if nm in ("2.csv", "5.csv", "6.csv"):
        v = obj.to_numpy(dtype=float)
        sq = v.shape[0] == v.shape[1]
        sym = np.allclose(v, v.T, atol=1e-8) if sq else "n/a"
        print(f"    {'square' if sq else 'NON-square'}  range [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]  symmetric={sym}")

print("\n" + "#" * 80)
print("# B. NAME BRIDGES FROM 8.xlsx")
print("#" * 80)
print(f"thresholds sheet: {th.shape}  cols={list(th.columns)}")
print(f"  field_name_R in 1.csv cols      : {int(th['field_name_R'].isin(chem1).sum())} / {len(th)}")
print(f"  field_name_python in 1.csv cols : {int(th['field_name_python'].isin(chem1).sum())} / {len(th)}")
print(f"  chemical_name in 2.csv index    : {int(th['chemical_name'].isin(d2.index).sum())} / {len(th)}")

R2PY = dict(zip(th["chemical_name"], th["field_name_R"]))
R2PY2 = dict(zip(th["chemical_name"], th["field_name_python"]))
n1 = {norm(c): c for c in chem1}
for c in chem1:
    n1.setdefault(norm(c), c)


def to_chem(name):
    if name in R2PY and R2PY[name] in chem1:
        return R2PY[name]
    if name in R2PY2 and R2PY2[name] in chem1:
        return R2PY2[name]
    k = norm(name)
    return n1.get(k)


S4N = dict(zip(sn["R_name"], sn["CleanName_PanelFeedback"]))
print(f"\nSensory names: R_name covering 4.csv cols = {int(sn['R_name'].isin(sens4).sum())} / {len(sens4)}")
print(f"  CleanName_PanelFeedback in 5.csv index  : "
      f"{int(sn['CleanName_PanelFeedback'].isin(d5.index).sum())} / {len(d5)}")

print("\n" + "#" * 80)
print("# C. IS 2.csv PEARSON OR SPEARMAN?  (recomputed from 1.csv)")
print("#" * 80)
m2 = {r: to_chem(r) for r in d2.index}
ok2 = {k: v for k, v in m2.items() if v in chem1}
print(f"2.csv rows mapped to 1.csv columns: {len(ok2)} / {len(d2.index)}")
cols_map = {c: to_chem(c) for c in d2.columns}
okc = {k: v for k, v in cols_map.items() if v in chem1}
print(f"2.csv cols mapped to 1.csv columns: {len(okc)} / {len(d2.columns)}")

shared = sorted(set(v for v in ok2.values() if v in set(okc.values())))
print(f"fully mapped chemicals usable for the test: {len(shared)}")
sub = shared[:150]
M = d1[sub].apply(pd.to_numeric, errors="coerce")
pear = M.corr(method="pearson").to_numpy()
spear = M.corr(method="spearman").to_numpy()
row_of = {v: k for k, v in ok2.items()}
col_of = {v: k for k, v in okc.items()}
gi, gj = [], []
for i, ci in enumerate(sub):
    for j, cj in enumerate(sub):
        if i < j:
            gi.append(row_of[ci]); gj.append(col_of[cj])
g = d2.loc[gi, gj].to_numpy(dtype=float)
dp = np.abs(g - pear[np.triu_indices(len(sub), 1)])
ds = np.abs(g - spear[np.triu_indices(len(sub), 1)])
print(f"\noff-diagonal pairs tested: {len(g)}")
print(f"  |given - pearson |  max={np.nanmax(dp):.8f}  median={np.nanmedian(dp):.8f}  frac<1e-8={np.mean(dp<1e-8):.3f}")
print(f"  |given - spearman |  max={np.nanmax(ds):.8f}  median={np.nanmedian(ds):.8f}  frac<1e-8={np.mean(ds<1e-8):.3f}")
print("\n  >>> VERDICT 2.csv = " + ("PEARSON" if np.nanmedian(dp) < np.nanmedian(ds) else "SPEARMAN"))

print("\n" + "#" * 80)
print("# D. IS 5.csv PEARSON OR SPEARMAN?  (recomputed from 4.csv)")
print("#" * 80)
s4n = {c: S4N.get(c) for c in d5.index}
ok5 = {k: v for k, v in s4n.items() if v in sens4}
print(f"5.csv rows mapped to 4.csv columns: {len(ok5)} / {len(d5.index)}")
sub5 = list(ok5.values())
M5 = d4[sub5].apply(pd.to_numeric, errors="coerce")
p5 = M5.corr(method="pearson").to_numpy()
s5 = M5.corr(method="spearman").to_numpy()
i5 = np.triu_indices(len(sub5), 1)
r5 = {v: k for k, v in ok5.items()}
gi5 = [r5[c] for c in sub5]
g5 = d5.loc[gi5, gi5].to_numpy(dtype=float)
d5p = np.abs(g5 - p5[i5])
d5s = np.abs(g5 - s5[i5])
print(f"\noff-diagonal pairs tested: {len(g5)}")
print(f"  |given - pearson |  max={np.nanmax(d5p):.8f}  median={np.nanmedian(d5p):.8f}  frac<1e-8={np.mean(d5p<1e-8):.3f}")
print(f"  |given - spearman |  max={np.nanmax(d5s):.8f}  median={np.nanmedian(d5s):.8f}  frac<1e-8={np.mean(d5s<1e-8):.3f}")
print("\n  >>> VERDICT 5.csv = " + ("PEARSON" if np.nanmedian(d5p) < np.nanmedian(d5s) else "SPEARMAN"))

print("\n" + "#" * 80)
print("# E. IS 6.csv REPRODUCIBLE?  (44 compounds x 50 sensory)")
print("#" * 80)
r6 = list(d6.index)
c6 = list(d6.columns)
print(f"6.csv shape {d6.shape}; sensory cols matched to 4.csv: {sum(c in sens4 for c in c6)} / {len(c6)}")
m6 = {r: to_chem(r) for r in r6}
ok6 = {k: v for k, v in m6.items() if v in chem1}
print(f"compound rows mapped to 1.csv: {len(ok6)} / {len(r6)}")
print(f"unmapped rows: {[r for r in r6 if r not in ok6][:10]}")
cc = [c for c in c6 if c in sens4]
rr = list(ok6.keys())
if len(rr) >= 10 and len(cc) >= 10:
    Xc = d1[[ok6[r] for r in rr]].apply(pd.to_numeric, errors="coerce")
    Ys = d4[cc].apply(pd.to_numeric, errors="coerce")
    pc = Xc.corr(method="pearson").to_numpy()
    sc = Xc.corr(method="spearman").to_numpy()
    gc = d6.loc[rr, cc].to_numpy(dtype=float)
    dpc = np.abs(gc - pc)
    dsc = np.abs(gc - sc)
    print(f"\n  |given - pearson |  max={np.nanmax(dpc):.8f}  median={np.nanmedian(dpc):.8f}")
    print(f"  |given - spearman |  max={np.nanmax(dsc):.8f}  median={np.nanmedian(dsc):.8f}")
    print("  >>> VERDICT 6.csv = " + ("PEARSON" if np.nanmedian(dpc) < np.nanmedian(dsc) else "SPEARMAN"))
    stg = np.nanmax(np.abs(gc), axis=0)
    print("\n  strongest sensory attributes in 6.csv (max |r| over 44 compounds):")
    for j in np.argsort(-stg)[:12]:
        print(f"    {cc[j]:22s} max|r|={stg[j]:.3f}")

print("\n" + "#" * 80)
print("# F. DOES 2.csv COVER ALL 1.csv FEATURES?")
print("#" * 80)
allmap = set(v for v in list(ok2.values()) + list(okc.values()) if v in chem1)
print(f"2.csv covers {len(allmap)} / {len(chem1)} chemistry features")
print(f"  missing from 2.csv: {sorted(set(chem1)-allmap)[:12]}")
derived = [c for c in chem1 if c in ("esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum")]
print(f"  derived summary cols {derived} -> in 2.csv: {[d for d in derived if d in allmap]}")

print("\n" + "#" * 80)
print("# G. SAMPLE-SIZE REALITY CHECK")
print("#" * 80)
vc = d1["tasting_category_fine"].value_counts()
print(f"n = {len(d1)} samples, p = {len(chem1)} features, p/n = {len(chem1)/len(d1):.2f}")
print(f"styles: {len(vc)}")
print(f"  n>=20 : {(vc>=20).sum()}   10<=n<20 : {((vc>=10)&(vc<20)).sum()}   "
      f"5<=n<10 : {((vc>=5)&(vc<10)).sum()}   n<5 : {(vc<5).sum()}")
print(f"  n<10 styles -> {vc[vc<10].to_dict()}")
print(f"  samples in styles with n<10 : {vc[vc<10].sum()} / {len(d1)} ({100*vc[vc<10].sum()/len(d1):.1f}%)")
print(f"  smallest cell n={vc.min()} -> style-stratified 5-fold leaves {vc.min()/4:.2f} per fold")

print("\n" + "#" * 80)
print("# H. 8.xlsx DETAIL")
print("#" * 80)
for k, v in x8.items():
    print(f"\n--- {k}  shape={v.shape}  NA={int(v.isna().sum().sum())} ---")
    print(f"    cols: {list(v.columns)}")

st = x8["Sensory thresholds"]
print("\n[Sensory thresholds] off_flavors / aging flags:")
for c in ["aging", "off_flavors", "calibrated", "origin_paper"]:
    if c in st.columns:
        print(f"  {c}: {st[c].value_counts(dropna=False).to_dict()}")
if "threshold_mean" in st.columns:
    print(f"  threshold_mean non-null: {int(st['threshold_mean'].notna().sum())}/{len(st)}")
    print(st[st["threshold_mean"].notna()].head(8).to_string(index=False))

mi = x8["GBM importance MDI"]
print(f"\n[GBM importance MDI] shape={mi.shape} cols={list(mi.columns)}")
print(mi.head(12).to_string(index=False))
print(f"  top-10 by importance: {mi.iloc[:10, 0].tolist()}")

sh = x8["GBM SHAP Values"]
print(f"\n[GBM SHAP Values] shape={sh.shape}")
print(f"  first cols: {list(sh.columns)[:6]} ... last: {list(sh.columns)[-3:]}")

cal = x8["Calibrated chemical dataset"]
print(f"\n[Calibrated chemical dataset] shape={cal.shape}")
print(f"  cols: {list(cal.columns)[:12]}")
print(cal.iloc[:4, :8].to_string(index=False))

bp = x8["Beer Prices"]
print(f"\n[Beer Prices] shape={bp.shape} cols={list(bp.columns)}")
print(bp.head(4).to_string(index=False))
