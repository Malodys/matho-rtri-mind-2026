# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：计数型地板质量分析与"7.csv 是否为 4.csv 冗余"的判别结论已人工复核。
"""Follow-up probe v2: what is 7.csv actually good for?

  A. Effective dimensionality of 7.csv vs 4.csv (dropping the 3 all-NA columns).
  B. Do aroma/palate/taste replicate one latent signal per descriptor?
  C. COUNT-DATA SIGNATURE: do columns show a point mass at their floor
     (log10(count+1) of forum mentions -> z-scored)?  Does "floor mass"
     predict how weakly a descriptor correlates with chemistry?
  D. Practical test: does 7.csv add predictive value for the 4.csv targets (Q1)?
  E. Decode descriptors using 8.xlsx 'Sensory names'.
"""
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
KEY = ["beer", "beer_id", "tasting_category_fine"]

d1 = pd.read_csv(os.path.join(RAW, "1.csv"))
d4 = pd.read_csv(os.path.join(RAW, "4.csv"))
d7 = pd.read_csv(os.path.join(RAW, "7.csv"))
x8 = pd.read_excel(os.path.join(RAW, "8.xlsx"), sheet_name=None)

desc7 = [c for c in d7.columns if c not in KEY]
sens4 = [c for c in d4.columns if c not in KEY]
chem1 = [c for c in d1.columns if c not in KEY]

m7raw = d7[desc7].apply(pd.to_numeric, errors="coerce")
allna = [c for c in desc7 if m7raw[c].isna().all()]
desc7_ok = [c for c in desc7 if c not in allna]
print(f"dropped all-NA columns: {allna}")
m7 = m7raw[desc7_ok]
m4 = d4[sens4].apply(pd.to_numeric, errors="coerce")

print("\n" + "=" * 78)
print("A. EFFECTIVE DIMENSIONALITY")
print("=" * 78)
p7 = PCA(n_components=10).fit(m7.fillna(m7.mean()))
p4 = PCA(n_components=10).fit(m4.fillna(m4.mean()))
print(f"7.csv ({len(desc7_ok)} cols) : " + "  ".join(f"PC{i+1}={v*100:4.1f}%" for i, v in enumerate(p7.explained_variance_ratio_[:8])))
print(f"   PC1={p7.explained_variance_ratio_[0]*100:.1f}%  PC10 cum={p7.explained_variance_ratio_[:10].sum()*100:.1f}%")
print(f"4.csv ({len(sens4)} cols)  : " + "  ".join(f"PC{i+1}={v*100:4.1f}%" for i, v in enumerate(p4.explained_variance_ratio_[:8])))
print(f"   PC1={p4.explained_variance_ratio_[0]*100:.1f}%  PC10 cum={p4.explained_variance_ratio_[:10].sum()*100:.1f}%")

print("\n" + "=" * 78)
print("B. CROSS-CONTEXT REPLICATION PER DESCRIPTOR")
print("=" * 78)
names7 = sorted({c.split("_", 1)[1] for c in desc7_ok if "_" in c})
rows = []
for nm in names7:
    cols = [f"{p}_{nm}" for p in ("aroma", "palate", "taste") if f"{p}_{nm}" in d7.columns]
    if len(cols) < 2:
        continue
    vals = [d7[c].astype(float) for c in cols]
    R = np.corrcoef(np.column_stack(vals), rowvar=False)
    rows.append((nm, R[0, 1] if len(cols) > 1 else np.nan,
                 R[0, 2] if len(cols) > 2 else np.nan,
                 R[1, 2] if len(cols) > 2 else np.nan))
df_ctx = pd.DataFrame(rows, columns=["descriptor", "r_AP", "r_AT", "r_PT"])
print(df_ctx.sort_values("r_AT", ascending=False).head(10).to_string(index=False))
print("\n  r(A,T) distribution:")
for thr in (0.95, 0.8, 0.6, 0.4, 0.2):
    print(f"    r_AT >= {thr:<5}: {int((df_ctx['r_AT'] >= thr).sum()):3d} / {len(df_ctx)}")
print("\n  weakest cross-context agreement (r_AT < 0.2):")
print(df_ctx[df_ctx["r_AT"] < 0.2].to_string(index=False))

print("\n" + "=" * 78)
print("C. COUNT-DATA SIGNATURE (is 7.csv forum word counts?)")
print("=" * 78)
sk = m7.skew()
floor_frac = m7.apply(lambda s: float((s == s.min()).mean()))
colmin = m7.min()
print(f"skewness: median={sk.median():.2f}  mean={sk.mean():.2f}  "
      f"cols with skew>2: {int((sk>2).sum())} / {len(desc7_ok)}")
print(f"fraction of rows sitting exactly at each column's minimum: "
      f"median={floor_frac.median():.3f}  max={floor_frac.max():.3f}")
print(f"column-minimum values: median={colmin.median():.3f}  "
      f"cols with min in [-5.5,-4.0]: {int(((colmin>=-5.5)&(colmin<=-4.0)).sum())}")
print(f"cells with z > 3 : {int((m7 > 3).sum().sum())}   z > 4 : {int((m7 > 4).sum().sum())}"
      f"   z > 6 : {int((m7 > 6).sum().sum())}")
print(f"cells with z < -3: {int((m7 < -3).sum().sum())}")


def corr_matrix(X, Y, min_n=30):
    out = np.full((X.shape[1], Y.shape[1]), np.nan)
    for i in range(X.shape[1]):
        xi = X[:, i].astype(float)
        for j in range(Y.shape[1]):
            yj = Y[:, j].astype(float)
            msk = ~np.isnan(xi) & ~np.isnan(yj)
            if int(msk.sum()) >= min_n:
                a = xi[msk] - xi[msk].mean()
                b = yj[msk] - yj[msk].mean()
                da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
                if da > 0 and db > 0:
                    out[i, j] = (a * b).sum() / (da * db)
    return out


aroma_cols = [c for c in desc7_ok if c.startswith("aroma_")]
Y1 = d1[chem1].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
R71 = corr_matrix(m7[aroma_cols].to_numpy(dtype=float), Y1)

recs = []
for i, c in enumerate(aroma_cols):
    row = R71[i]
    j = int(np.nanargmax(np.abs(row)))
    recs.append({"col": c, "best_chem": chem1[j], "best_r": row[j],
                 "floor_frac": float(floor_frac[c]), "skew": float(sk[c]),
                 "min_val": float(colmin[c])})
df_rec = pd.DataFrame(recs).sort_values("best_r", key=lambda s: -s.abs())
print("\n  aroma descriptor vs best-matching compound:")
print(df_rec.head(18).to_string(index=False))

pr_floor = df_rec["floor_frac"].abs().corr(df_rec["best_r"].abs(), method="spearman")
pr_skew = df_rec["skew"].abs().corr(df_rec["best_r"].abs(), method="spearman")
print(f"\n  Spearman( floor_frac , max|r| to chemistry ) = {pr_floor: .3f}")
print(f"  Spearman( |skew|   , max|r| to chemistry ) = {pr_skew: .3f}")
print("  (strong negative => rare/heavily-floored descriptors are the ones that")
print("   fail to track chemistry, exactly what a mention-count table predicts)")

print("\n  highest floor-fraction descriptors (rarest 'mentions'):")
print(df_rec.sort_values("floor_frac", ascending=False).head(12)[
    ["col", "floor_frac", "min_val", "skew", "best_chem", "best_r"]].to_string(index=False))

print("\n" + "=" * 78)
print("D. DOES 7.csv ADD PREDICTIVE VALUE FOR 4.csv TARGETS?  (Q1 relevance)")
print("=" * 78)
Xc = d1[chem1].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
for c in Xc.columns:
    if Xc[c].nunique() == 1:
        Xc[c] = 0.0
Xc = Xc.fillna(Xc.mean())
X7 = m7.fillna(m7.mean())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for t in ["A_esters_fruity", "A_malt_cara", "F_hops_citrus", "A_esters_all", "overall"]:
    if t not in d4.columns:
        continue
    y = pd.to_numeric(d4[t], errors="coerce")
    if y.isna().all():
        continue
    out = {}
    for tag, X in [("chem", Xc), ("7only", X7), ("chem+7", pd.concat([Xc, X7], axis=1))]:
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        s = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
        out[tag] = float(np.sqrt(s.mean()))
    b = out["chem"]
    print(f"  {t:16s} RMSE  chem={b:.4f}  7only={out['7only']:.4f}  chem+7={out['chem+7']:.4f}"
          f"  delta={100*(out['chem+7']-b)/b:+.1f}%")

print("\n" + "=" * 78)
print("E. DECODING DESCRIPTORS VIA 8.xlsx 'Sensory names'")
print("=" * 78)
print("8.xlsx sheets:", list(x8.keys()))
sn = x8.get("Sensory names")
if sn is not None:
    print(f"'Sensory names' shape={sn.shape} cols={list(sn.columns)}")
    print(sn.head(20).to_string(index=False))
st = x8.get("Sensory thresholds")
if st is not None:
    print(f"\n'Sensory thresholds' shape={st.shape} cols={list(st.columns)}")
    print(st.head(10).to_string(index=False))
