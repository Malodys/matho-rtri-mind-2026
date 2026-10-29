# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：5.csv / 6.csv 的名称归一化映射结果与 GBM 重要性方向已人工逐项核对。
"""Final: name-normalised 5.csv / 6.csv verdicts + correct GBM importance orientation."""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
KEY = ["beer", "beer_id", "tasting_category_fine"]


def clean(s):
    s = str(s).replace("_x000D_", "\r").replace("_x000A_", "\n")
    return re.sub(r"\s+", " ", s.replace("\r", " ").replace("\n", " ")).strip()


d1 = pd.read_csv(os.path.join(RAW, "1.csv"))
d4 = pd.read_csv(os.path.join(RAW, "4.csv"))
d5 = pd.read_csv(os.path.join(RAW, "5.csv"), index_col=0)
d6 = pd.read_csv(os.path.join(RAW, "6.csv"), index_col=0)
x8 = pd.read_excel(os.path.join(RAW, "8.xlsx"), sheet_name=None)
sn, th = x8["Sensory names"], x8["Sensory thresholds"]
sens4 = [c for c in d4.columns if c not in KEY]
chem1 = [c for c in d1.columns if c not in KEY]

bridge = {}
for _, r in sn.iterrows():
    if r.get("R_name") not in sens4:
        continue
    for cn in ["CleanName_PaperLong", "CleanName_PanelFeedback", "CleanName_Paper"]:
        nm = clean(r.get(cn)) if isinstance(r.get(cn), str) else None
        if nm:
            bridge.setdefault(nm, set()).add(r["R_name"])


def resolve(disp):
    cand = sorted(bridge.get(clean(disp), []))
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]
    low = clean(disp).lower()
    pref = "A_" if "aroma" in low else ("F_" if "taste" in low else None)
    if pref:
        hit = [c for c in cand if c.startswith(pref)]
        if hit:
            return hit[0]
    return cand[0]


print("=" * 80)
print("5.csv")
print("=" * 80)
res5 = [resolve(n) for n in d5.index]
print(f"resolved {sum(r is not None for r in res5)}/{len(res5)}   unique={len(set(res5))}")
if all(res5) and len(set(res5)) == len(res5):
    M5 = d4[res5].apply(pd.to_numeric, errors="coerce")
    i5 = np.triu_indices(len(res5), 1)
    g5 = d5.to_numpy(dtype=float)[i5]
    dp = np.abs(g5 - M5.corr(method="pearson").to_numpy()[i5])
    ds = np.abs(g5 - M5.corr(method="spearman").to_numpy()[i5])
    print(f"  |given - pearson |  max={np.nanmax(dp):.8f} median={np.nanmedian(dp):.8f} frac<1e-8={np.mean(dp<1e-8):.3f}")
    print(f"  |given - spearman|  max={np.nanmax(ds):.8f} median={np.nanmedian(ds):.8f} frac<1e-8={np.mean(ds<1e-8):.3f}")
    print("  >>> VERDICT 5.csv = " + ("PEARSON" if np.nanmedian(dp) < np.nanmedian(ds) else "SPEARMAN"))

print("\n" + "=" * 80)
print("6.csv")
print("=" * 80)
R2PY = dict(zip(th["chemical_name"], th["field_name_R"]))
n1 = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in chem1}
ok6 = {}
for r in d6.index:
    ok6[r] = R2PY[r] if (r in R2PY and R2PY[r] in chem1) else n1.get(re.sub(r"[^a-z0-9]+", "", str(r).lower()))
ok6 = {k: v for k, v in ok6.items() if v in chem1}
cols6 = [resolve(c) for c in d6.columns]
print(f"compounds mapped {len(ok6)}/{len(d6.index)}  sensory cols resolved {sum(c is not None for c in cols6)}/{len(d6.columns)}")
rr = list(ok6.keys())
d6_cols = list(d6.columns)
cc = [resolve(c) for c in d6_cols]
print(f"  resolved sensory order: {cc[:5]} ...")
if len(ok6) > 5 and all(cc):
    Xc = d1[[ok6[r] for r in rr]].apply(pd.to_numeric, errors="coerce")
    Ys = d4[cc].apply(pd.to_numeric, errors="coerce")
    gc = d6.loc[rr, d6_cols].to_numpy(dtype=float)
    for nm, meth in [("pearson", "pearson"), ("spearman", "spearman")]:
        both = pd.concat([Xc, Ys], axis=1)
        cm = both.corr(method=meth).loc[Xc.columns, Ys.columns].to_numpy()
        dd = np.abs(gc - cm)
        print(f"  {nm:9s} max={np.nanmax(dd):8.5f} median={np.nanmedian(dd):8.5f} frac<1e-6={np.mean(dd<1e-6):.3f}")
    stg = np.nanmax(np.abs(gc), axis=0)
    print("\n  strongest sensory attributes in 6.csv:")
    for j in np.argsort(-stg)[:12]:
        print(f"    {cc[j]:24s} max|r|={stg[j]:.3f}")
    print("\n  44 compounds in 6.csv:")
    for i in range(0, len(rr), 3):
        print("    " + " | ".join(f"{r}" for r in rr[i:i + 3]))

print("\n" + "=" * 80)
print("GBM importance MDI (correct orientation) + SHAP halves")
print("=" * 80)
mi = x8["GBM importance MDI"]
imp = pd.to_numeric(mi.iloc[:, 0], errors="coerce")
cmp_ = mi.iloc[:, 1].astype(str).str.strip()
mdf = pd.DataFrame({"compound": cmp_, "importance": imp}).dropna()
print(f"rows={len(mdf)}  sum={mdf['importance'].sum():.4f}")
print("\n  TOP-20 by prior-model importance:")
print(mdf.sort_values("importance", ascending=False).head(20).to_string(index=False))

sh = x8["GBM SHAP Values"]
shap_part = sh[sh["data"] == "SHAP"]
conc_part = sh[sh["data"] == "concentration"]
print(f"\nSHAP half: {shap_part.shape}   concentration half: {conc_part.shape}")
comp_cols = [c for c in sh.columns if c not in ("data", "beer", "beer_id", "tasting_category_fine", "base_prediction")]
d1i = d1.set_index("beer")
conc_idx = conc_part.set_index("beer")
common = [b for b in conc_idx.index if b in d1i.index]
cmp_common = [c for c in comp_cols if c in d1i.columns]
print(f"  concentration rows matched to 1.csv by 'beer': {len(common)}/{len(conc_idx)}")
print(f"  compound cols shared with 1.csv: {len(cmp_common)}/{len(comp_cols)}")
if common and cmp_common:
    A = conc_idx.loc[common, cmp_common].to_numpy(dtype=float)
    B = d1i.loc[common, cmp_common].to_numpy(dtype=float)
    print(f"  max |conc_half - 1.csv| = {np.nanmax(np.abs(A-B)):.6f}   "
          f"frac identical = {np.mean(np.isclose(A, B, atol=1e-9, equal_nan=True)):.4f}")
    # does the SHAP half reconstruct a prediction from base_prediction?
    sv = shap_part.set_index("beer")
    common2 = [b for b in sv.index if b in d1i.index]
    S = sv.loc[common2, cmp_common].to_numpy(dtype=float)
    base = sv.loc[common2, "base_prediction"].to_numpy(dtype=float)
    pred = base + S.sum(axis=1)
    print(f"  base+sum(SHAP) -> prediction range [{pred.min():.4f}, {pred.max():.4f}]")
    print("  (if this is a z-scored sensory target, the prior model predicts one target only)")
