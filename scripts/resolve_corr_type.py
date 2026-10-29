# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：2.csv / 5.csv 的相关系数类型与对数底数判定已人工复核。
"""Resolve two open questions:

  Q1  What scale / correlation type is 2.csv actually computed on?
      Candidates: Pearson|Spearman  x  log10(1.csv) | raw concentrations
      x  pairwise-complete | listwise-complete deletion
  Q2  Is 5.csv Pearson or Spearman?  (5.csv index uses CleanName_PaperLong)
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
x8 = pd.read_excel(os.path.join(RAW, "8.xlsx"), sheet_name=None)
th, sn, cal = x8["Sensory thresholds"], x8["Sensory names"], x8["Calibrated chemical dataset"]

chem1 = [c for c in d1.columns if c not in KEY]
sens4 = [c for c in d4.columns if c not in KEY]

print("=" * 80)
print("Q2: 5.csv CORRELATION TYPE")
print("=" * 80)
print(f"Sensory names cols: {list(sn.columns)}")
cand = {}
for cn in ["CleanName_PaperLong", "CleanName_PanelFeedback", "CleanName_Paper"]:
    if cn in sn.columns:
        hit = int(sn[cn].isin(d5.index).sum())
        print(f"  {cn:24s} matches 5.csv index: {hit}/{len(d5)}")
        if hit >= len(d5) * 0.8:
            cand[cn] = dict(zip(sn[cn], sn["R_name"]))
bridge = cand.get("CleanName_PaperLong") or cand.get("CleanName_PanelFeedback")
ok5 = {k: v for k, v in bridge.items() if v in sens4}
print(f"\n5.csv rows mapped to 4.csv columns: {len(ok5)} / {len(d5.index)}")
if len(ok5) >= 40:
    sub5 = list(ok5.values())
    M5 = d4[sub5].apply(pd.to_numeric, errors="coerce")
    i5 = np.triu_indices(len(sub5), 1)
    p5 = M5.corr(method="pearson").to_numpy()[i5]
    s5 = M5.corr(method="spearman").to_numpy()[i5]
    r5 = {v: k for k, v in ok5.items()}
    gi5 = [r5[c] for c in sub5]
    g5 = d5.loc[gi5, gi5].to_numpy(dtype=float)[i5]
    dp, ds = np.abs(g5 - p5), np.abs(g5 - s5)
    print(f"  |given-pearson |  max={np.nanmax(dp):.8f} median={np.nanmedian(dp):.8f} frac<1e-8={np.mean(dp<1e-8):.3f}")
    print(f"  |given-spearman|  max={np.nanmax(ds):.8f} median={np.nanmedian(ds):.8f} frac<1e-8={np.mean(ds<1e-8):.3f}")
    print("  >>> VERDICT 5.csv = " + ("PEARSON" if np.nanmedian(dp) < np.nanmedian(ds) else "SPEARMAN"))

print("\n" + "=" * 80)
print("Q1: WHAT SCALE IS 2.csv COMPUTED ON?")
print("=" * 80)
print(f"Calibrated chemical dataset shape={cal.shape}")
print(f"  cols (first 14): {list(cal.columns)[:14]}")
print(f"  any identifier cols? {[c for c in cal.columns if c.lower() in ('beer','beer_id')]}")

R2PY = dict(zip(th["chemical_name"], th["field_name_R"]))
n1 = {norm(c): c for c in chem1}
for c in chem1:
    n1.setdefault(norm(c), c)


def to_chem(name):
    if name in R2PY and R2PY[name] in chem1:
        return R2PY[name]
    k = norm(name)
    return n1.get(k)


ok2 = {r: to_chem(r) for r in d2.index}
ok2 = {k: v for k, v in ok2.items() if v in chem1}
shared = sorted(set(ok2.values()))
print(f"\n2.csv rows mapped: {len(ok2)}/{len(d2.index)}")

derived = {"esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"}
sub = [c for c in shared if c not in derived][:150]
M = d1[sub].apply(pd.to_numeric, errors="coerce")
row_of = {v: k for k, v in ok2.items()}
g = np.array([[d2.loc[row_of[a], row_of[b]] for b in sub] for a in sub], dtype=float)
iu = np.triu_indices(len(sub), 1)
gv = g[iu]

variants = {}
variants["pearson_pairwise"] = M.corr(method="pearson").to_numpy()[iu]
variants["spearman_pairwise"] = M.corr(method="spearman").to_numpy()[iu]
Mlw = M.dropna()
variants["pearson_listwise"] = Mlw.corr(method="pearson").to_numpy()[iu]
variants["spearman_listwise"] = Mlw.corr(method="spearman").to_numpy()[iu]

print(f"\n--- vs 2.csv (excluding derived sum columns, {len(gv)} pairs) ---")
for k, v in variants.items():
    d = np.abs(gv - v)
    print(f"  {k:20s} max={np.nanmax(d):8.5f}  median={np.nanmedian(d):8.5f}  "
          f"mean={np.nanmean(d):8.5f}  frac<1e-6={np.mean(d<1e-6):.3f}")

print("\n--- calibration check: is 2.csv on RAW (pre-log) concentrations? ---")
print(f"Calibrated dataset cols sample: {list(cal.columns)[:8]}")
keycol = [c for c in cal.columns if c.lower() in ("beer", "beer_id", "sample")]
print(f"identifier candidates: {keycol}")
cal_cols = [c for c in cal.columns if c not in keycol]
# map calibrated display names -> 1.csv column
cal_map = {}
for c in cal_cols:
    if c in chem1:
        cal_map[c] = c
    elif c in R2PY and R2PY[c] in chem1:
        cal_map[c] = R2PY[c]
    else:
        k = norm(c)
        if k in n1:
            cal_map[c] = n1[k]
print(f"calibrated columns mapped to 1.csv: {len(cal_map)} / {len(cal_cols)}")
print(f"  example mappings: {list(cal_map.items())[:6]}")

if len(cal_map) >= 8:
    shared_cal = [c for c in cal_map if cal_map[c] in set(shared)]
    print(f"  usable for pairwise test: {len(shared_cal)}")
    if len(shared_cal) >= 8:
        raw = cal[list(shared_cal)].apply(pd.to_numeric, errors="coerce")
        lg = d1[[cal_map[c] for c in shared_cal]].apply(pd.to_numeric, errors="coerce")
        rp = raw.corr(method="pearson").to_numpy()
        rs = raw.corr(method="spearman").to_numpy()
        lp = lg.corr(method="pearson").to_numpy()
        ls = lg.corr(method="spearman").to_numpy()
        iu2 = np.triu_indices(len(shared_cal), 1)
        gv2 = np.array([[d2.loc[row_of[cal_map[a]], row_of[cal_map[b]]]
                         for b in shared_cal] for a in shared_cal], dtype=float)[iu2]
        for nm, v in [("pearson_RAW", rp[iu2]), ("spearman_RAW", rs[iu2]),
                      ("pearson_LOG10", lp[iu2]), ("spearman_LOG10", ls[iu2])]:
            dd = np.abs(gv2 - v)
            print(f"  {nm:16s} max={np.nanmax(dd):8.5f} median={np.nanmedian(dd):8.5f} "
                  f"frac<1e-6={np.mean(dd<1e-6):.3f}")

print("\n--- spot check: 6 individual pairs, given vs all candidates ---")
pairs = [(sub[i], sub[i + 1]) for i in range(0, 6)]
for a, b in pairs:
    ga = d2.loc[row_of[a], row_of[b]]
    pa = M[a].corr(M[b])
    sa = M[a].corr(M[b], method="spearman")
    print(f"  {a:24s} x {b:24s} given={ga: .6f}  pearson={pa: .6f}  spearman={sa: .6f}")

print("\n" + "=" * 80)
print("Q1b: DOES 6.csv MATCH 1.csv x 4.csv?")
print("=" * 80)
r6, c6 = list(d6.index), list(d6.columns)
m6 = {r: to_chem(r) for r in r6}
ok6 = {k: v for k, v in m6.items() if v in chem1}
cc = [c for c in c6 if c in sens4]
print(f"rows mapped {len(ok6)}/{len(r6)},  sensory cols matched {len(cc)}/{len(c6)}")
if len(ok6) >= 8 and len(cc) >= 8:
    rr = list(ok6.keys())
    Xc = d1[[ok6[r] for r in rr]].apply(pd.to_numeric, errors="coerce")
    Ys = d4[cc].apply(pd.to_numeric, errors="coerce")
    for nm, meth in [("pearson", "pearson"), ("spearman", "spearman")]:
        cmtx = Xc.corr(method=meth).to_numpy()
        gc = d6.loc[rr, cc].to_numpy(dtype=float)
        dd = np.abs(gc - cmtx)
        print(f"  {nm:9s} max={np.nanmax(dd):8.5f} median={np.nanmedian(dd):8.5f} "
              f"frac<1e-6={np.mean(dd<1e-6):.3f}")
