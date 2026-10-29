# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：7.csv 的表结构、量纲与缺失机制（MCAR/结构化缺失）判定已人工复核。
"""Probe the true structure of 7.csv (sensory descriptor scores).

Questions answered here:
  1. What is actually inside 7.csv (shape, prefixes, scaling, NA)?
  2. Is its NA pattern random (MCAR) or structured (left-censored / subset)?
  3. Is 7.csv a re-standardised *copy* of 4.csv (redundant) or an independent
     representation (adds information)?
  4. Does 7.csv tie back to the chemistry in 1.csv?
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
KEY = ["beer", "beer_id", "tasting_category_fine"]

d1 = pd.read_csv(os.path.join(RAW, "1.csv"))
d4 = pd.read_csv(os.path.join(RAW, "4.csv"))
d7 = pd.read_csv(os.path.join(RAW, "7.csv"))

desc7 = [c for c in d7.columns if c not in KEY]
sens4 = [c for c in d4.columns if c not in KEY]
chem1 = [c for c in d1.columns if c not in KEY]

print("=" * 78)
print("1. SHAPES & COMPOSITION")
print("=" * 78)
print(f"1.csv (chem)      {d1.shape}   features={len(chem1)}")
print(f"4.csv (sensory)   {d4.shape}   features={len(sens4)}")
print(f"7.csv (descr)     {d7.shape}   features={len(desc7)}")

pref = Counter(c.split("_")[0] for c in desc7)
print(f"\n7.csv prefixes: {dict(pref)}")
pref4 = Counter(c.split("_")[0] for c in sens4)
print(f"4.csv prefixes: {dict(pref4)}")

names7 = sorted({c.split("_", 1)[1] for c in desc7 if "_" in c})
print(f"\ndistinct descriptors in 7.csv: {len(names7)}")
print(names7)

print("\n" + "=" * 78)
print("2. ROW ALIGNMENT ACROSS TABLES")
print("=" * 78)
print("1 vs 4 : beer sequence identical :", bool((d1["beer"].values == d4["beer"].values).all()))
print("1 vs 7 : beer sequence identical :", bool((d1["beer"].values == d7["beer"].values).all()))
print("4 vs 7 : beer sequence identical :", bool((d4["beer"].values == d7["beer"].values).all()))
print("7.csv category == 4.csv category  :",
      bool((d7["tasting_category_fine"].values == d4["tasting_category_fine"].values).all()))

print("\n" + "=" * 78)
print("3. NA STRUCTURE IN 7.csv")
print("=" * 78)
m7 = d7[desc7]
na_cells = int(m7.isna().sum().sum())
total = m7.size
print(f"NA cells      : {na_cells} / {total} ({100 * na_cells / total:.2f}%)")
print(f"cols w/ NA    : {int((m7.isna().sum() > 0).sum())} / {len(desc7)}")
print(f"rows w/ NA    : {int(m7.isna().any(axis=1).sum())} / {len(d7)}")

rowna = m7.isna().sum(axis=1)
print(f"\nNA-per-row distribution: {dict(sorted(Counter(rowna.tolist()).items()))}")

colna = m7.isna().sum().sort_values(ascending=False)
print("\ncolumns with NA (top 25):")
print(colna[colna > 0].head(25).to_string())

# signature grouping: which rows share the exact same missing-pattern
sig = m7.isna().astype(int).astype(str).agg("".join, axis=1)
print(f"\ndistinct NA signatures: {sig.nunique()}")
for s, cnt in Counter(sig).most_common(6):
    idx = np.where(sig.values == s)[0]
    missing = [desc7[j] for j in range(len(desc7)) if s[j] == "1"][:8]
    cats = d7["tasting_category_fine"].iloc[idx].value_counts().to_dict()
    print(f"  n={cnt:3d}  missing={missing}  categories={cats}")

# is NA related to style / to chemistry?
if na_cells:
    df_na = pd.DataFrame({"beer": d7["beer"], "n_na": rowna,
                          "cat": d7["tasting_category_fine"]})
    print("\nNA per style (styles with any NA):")
    g = df_na.groupby("cat")["n_na"].agg(["sum", "mean", "size"])
    print(g[g["sum"] > 0].sort_values("sum", ascending=False).to_string())

print("\n" + "=" * 78)
print("4. IS 7.csv A RESCALED COPY OF 4.csv ?  (max |r| per 7-column)")
print("=" * 78)


def corr_matrix(X, Y, min_n=30):
    out = np.full((X.shape[1], Y.shape[1]), np.nan)
    for i in range(X.shape[1]):
        xi = X[:, i].astype(float)
        for j in range(Y.shape[1]):
            yj = Y[:, j].astype(float)
            msk = ~np.isnan(xi) & ~np.isnan(yj)
            k = int(msk.sum())
            if k >= min_n:
                a = xi[msk] - xi[msk].mean()
                b = yj[msk] - yj[msk].mean()
                da = np.sqrt((a * a).sum())
                db = np.sqrt((b * b).sum())
                if da > 0 and db > 0:
                    out[i, j] = (a * b).sum() / (da * db)
    return out


X7 = m7.to_numpy(dtype=float)
Y4 = d4[sens4].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
R74 = corr_matrix(X7, Y4)

best = []
for i, c in enumerate(desc7):
    row = R74[i]
    if np.isnan(row).all():
        continue
    j = int(np.nanargmax(np.abs(row)))
    best.append((c, sens4[j], row[j], int((~np.isnan(row)).sum())))
best.sort(key=lambda x: -abs(x[2]))

print(f"{'7.csv column':26s} {'best 4.csv match':24s} {'r':>8s}")
for c, t, r, _ in best[:30]:
    print(f"{c:26s} {t:24s} {r: .4f}")

arrmax = np.array([abs(x[2]) for x in best])
print(f"\nn_7cols={len(best)}")
for thr in (0.999, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2):
    print(f"  max|r| >= {thr:<6}: {int((arrmax >= thr).sum()):3d}  ({100 * (arrmax >= thr).mean():.1f}%)")
print(f"  max|r| <  0.2     : {int((arrmax < 0.2).sum()):3d}")

# test exact affine relationship for the r~1 cases
print("\n-- affine check for the strongest pairs (is 7.csv an affine rescale of 4.csv?) --")
for c, t, r, _ in best[:8]:
    if abs(r) > 0.9:
        x = d7[c].astype(float)
        y = d4[t].astype(float)
        msk = x.notna() & y.notna()
        b1, b0 = np.polyfit(x[msk], y[msk], 1)
        pred = b1 * x[msk] + b0
        resid = float(np.sqrt(np.mean((pred - y[msk]) ** 2)))
        print(f"  {c:24s} ~ {t:22s} r={r: .5f}  slope={b1: .4f} intercept={b0: .4f} RMSE={resid:.6f}")

print("\n" + "=" * 78)
print("5. DOES 7.csv TIE BACK TO CHEMISTRY (1.csv)?")
print("=" * 78)
aroma_cols = [c for c in desc7 if c.startswith("aroma_")]
Xa = d7[aroma_cols].to_numpy(dtype=float)
Y1 = d1[chem1].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
R71 = corr_matrix(Xa, Y1)

print(f"{'7.csv aroma column':26s} {'best compound':26s} {'r':>8s}")
rows71 = []
for i, c in enumerate(aroma_cols):
    row = R71[i]
    if np.isnan(row).all():
        continue
    j = int(np.nanargmax(np.abs(row)))
    rows71.append((c, chem1[j], row[j]))
rows71.sort(key=lambda x: -abs(x[2]))
for c, t, r in rows71[:20]:
    print(f"{c:26s} {t:26s} {r: .4f}")

print("\n-- named spot checks (descriptor vs its own compound) --")
spot = [("aroma_acetaldehyde", "acetaldehyde"), ("aroma_acetate", "ethyl_acetate"),
        ("aroma_alcohol", "ethanol"), ("aroma_citrus", "limonene"),
        ("aroma_hoppy", "humulene"), ("aroma_diacetyl", "diacetyl"),
        ("aroma_sulfur", "DMS"), ("aroma_cabbage", "dimethyl_sulfide"),
        ("aroma_acetic", "acetic_acid"), ("aroma_lactic", "lactic_acid")]
for c, t in spot:
    if c in d7.columns and t in d1.columns:
        x = d7[c].astype(float)
        y = d1[t].astype(float)
        msk = x.notna() & y.notna()
        r = float(np.corrcoef(x[msk], y[msk])[0, 1]) if msk.sum() > 30 else float("nan")
        print(f"  {c:22s} vs {t:20s} r={r: .4f}  (n={int(msk.sum())})")

print("\n" + "=" * 78)
print("6. SCALING OF 7.csv COLUMNS")
print("=" * 78)
cm = m7.mean()
cs = m7.std()
print(f"mean of column means : {cm.mean(): .4f}   (|mean|>0.05 in {int((cm.abs() > 0.05).sum())} cols)")
print(f"mean of column sds   : {cs.mean(): .4f}   (sd outside [0.8,1.2] in {int(((cs < 0.8) | (cs > 1.2)).sum())} cols)")
print(f"global min/max       : {np.nanmin(m7.to_numpy(dtype=float)): .4f} / {np.nanmax(m7.to_numpy(dtype=float)): .4f}")
cm4 = d4[sens4].apply(pd.to_numeric, errors="coerce").mean()
cs4 = d4[sens4].apply(pd.to_numeric, errors="coerce").std()
print(f"[4.csv] mean of col means {cm4.mean(): .4f} | mean of col sds {cs4.mean(): .4f}")

print("\n" + "=" * 78)
print("7. INTERNAL STRUCTURE: aroma vs palate vs taste blocks")
print("=" * 78)
blocks = {}
for pfx in ("aroma", "palate", "taste"):
    cols = [c for c in desc7 if c.startswith(pfx + "_")]
    blocks[pfx] = cols
    print(f"{pfx:8s} n={len(cols)}")
pairs = [("aroma", "palate"), ("aroma", "taste"), ("palate", "taste")]
for a, b in pairs:
    sub = corr_matrix(d7[blocks[a]].to_numpy(dtype=float), d7[blocks[b]].to_numpy(dtype=float))
    v = sub[~np.isnan(sub)]
    print(f"  r({a:6s},{b:6s}) mean={v.mean(): .3f} median={np.median(v): .3f} "
          f"p90={np.percentile(v, 90): .3f} max={v.max(): .3f}")

# same descriptor across the three contexts
print("\n  same descriptor, different context (mean r):")
rows = []
for nm in names7:
    vals = {}
    for pfx in ("aroma", "palate", "taste"):
        c = f"{pfx}_{nm}"
        if c in d7.columns:
            vals[pfx] = d7[c].astype(float)
    if len(vals) == 3:
        sub = corr_matrix(np.column_stack(list(vals.values())), np.column_stack(list(vals.values())))
        rows.append((nm, sub[0, 1], sub[0, 2], sub[1, 2]))
    elif len(vals) == 2:
        sub = corr_matrix(np.column_stack(list(vals.values())), np.column_stack(list(vals.values())))
        rows.append((nm, sub[0, 1], np.nan, np.nan))
rr = [r for r in rows if not np.isnan(r[1])]
if rr:
    print(f"    aroma-palate mean r = {np.mean([r[1] for r in rr]): .3f}")
    print(f"    aroma-taste  mean r = {np.nanmean([r[2] for r in rr]): .3f}")
    print(f"    palate-taste mean r = {np.nanmean([r[3] for r in rr]): .3f}")
    print("    highest cross-context agreement (same descriptor):")
    for r in sorted(rows, key=lambda x: -(x[1] if not np.isnan(x[1]) else -9))[:10]:
        print(f"      {r[0]:16s} aroma-palate={r[1]: .3f}  aroma-taste={r[2]: .3f}")

print("\n" + "=" * 78)
print("8. HOW MANY 7.csv COLUMNS ARE ESSENTIALLY CONSTANT / LOW-VARIANCE?")
print("=" * 78)
print(f"columns with sd < 0.1  : {int((cs < 0.1).sum())}")
print(f"columns with sd < 0.5  : {int((cs < 0.5).sum())}")
print("lowest-variance columns:")
print(cs.sort_values().head(12).to_string())
