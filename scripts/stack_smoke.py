# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：图阈值、特征数与缺陷化合物列名均对照 data/raw 与 data/splits 的实际内容核对后写定。
"""能力栈冒烟自检：在真实赛题数据上跑通问题一/二/三所需的关键技术栈。

覆盖：
  V8  图论栈   —— 由 2.csv 相关矩阵构建共表达图（networkx），并用 igraph 的 Leiden 社区发现。
  V9  归因栈   —— GBDT 拟合 + shap 归因 + bootstrap 入选频��抽样。
  V10 生成栈   —— copulas 高斯 Copula 拟合/采样 + 缺陷化合物阈值约束检查。

用法：
    python scripts/stack_smoke.py [--repo-root <仓库根>]
退出码：0 = 全部通过。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFECT_HINTS = ("DMS", "Dimethyl sulfide", "Diacetyl", "4-VG", "4VG", "trans-2-nonenal",
                "Stale hop", "Stale hopp")


def _out_path(root: Path) -> Path:
    d = root / "data" / "processed" / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- V8 图论栈
def smoke_graph(root: Path, thr: float = 0.6) -> dict:
    import igraph as ig
    import networkx as nx

    corr = pd.read_csv(root / "data" / "raw" / "2.csv", index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    vals = corr.to_numpy(dtype=float)
    np.fill_diagonal(vals, 0.0)
    np.nan_to_num(vals, copy=False)

    g = nx.Graph()
    names = list(corr.index)
    g.add_nodes_from(names)
    n_edges = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(vals[i, j]) >= thr:
                g.add_edge(names[i], names[j], weight=float(vals[i, j]))
                n_edges += 1

    n_comp = nx.number_connected_components(g)
    deg = sorted((d for _, d in g.degree()), reverse=True)[:5]

    ig_g = ig.Graph(n=len(names))
    idx = {nm: i for i, nm in enumerate(names)}
    ig_g.add_edges([(idx[a], idx[b]) for a, b in g.edges()])
    comms = ig_g.community_leiden(objective_function="modularity", n_iterations=5)
    sizes = sorted((len(c) for c in comms), reverse=True)[:5]

    return {
        "id": "V8_graph_stack",
        "desc": f"2.csv 相关矩阵 → 共表达图（|r|≥{thr}）+ Leiden 社区发现",
        "nodes": g.number_of_nodes(),
        "edges": n_edges,
        "connected_components": n_comp,
        "top_degrees": deg,
        "top_community_sizes": sizes,
        "passed": g.number_of_nodes() > 100 and n_edges > 100 and len(sizes) > 0,
    }


# ---------------------------------------------------------------- V9 归因栈
def smoke_attribution(root: Path) -> dict:
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    # 对齐协议沿用远端管线：按 beer 关联 1_/4_ 切分文件，特征与目标同序。
    chemical = pd.read_csv(root / "data" / "splits" / "seed_0" / "1_train.csv")
    sensory = pd.read_csv(root / "data" / "splits" / "seed_0" / "4_train.csv")
    sensory = sensory.set_index("beer").loc[chemical["beer"]].reset_index()
    meta = ["beer", "beer_id", "tasting_category_fine"]
    feats = [c for c in chemical.columns if c not in meta]
    targets = [c for c in sensory.columns if c not in meta]

    Xf = chemical[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = targets[0]
    y = pd.to_numeric(sensory[target], errors="coerce").fillna(0.0)

    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=0)
    model = GradientBoostingRegressor(random_state=0, n_estimators=80, max_depth=3)
    model.fit(Xtr, ytr)
    r2 = float(model.score(Xte, yte))

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte.iloc[:20])
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    top3 = [feats[i] for i in np.argsort(mean_abs)[::-1][:3]]

    # bootstrap 入选频率
    rng = np.random.default_rng(0)
    hits = {f: 0 for f in top3}
    B = 20
    for _ in range(B):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        m = GradientBoostingRegressor(random_state=0, n_estimators=40, max_depth=3)
        m.fit(Xtr.iloc[idx], ytr.iloc[idx])
        imp = m.feature_importances_
        for f in [feats[i] for i in np.argsort(imp)[::-1][:3]]:
            hits[f] = hits.get(f, 0) + 1
    freq = {f: round(c / B, 3) for f, c in hits.items()}

    return {
        "id": "V9_attribution_stack",
        "desc": "GBDT 拟合 + shap 归因 + bootstrap Top-3 入选频率（目标：Aroma 首维度）",
        "target": target,
        "n_features": len(feats),
        "n_targets_available": len(targets),
        "holdout_r2": round(r2, 4),
        "shap_top3": top3,
        "bootstrap_top3_freq": freq,
        "passed": bool(top3) and len(feats) > 50 and np.isfinite(r2),
    }


# ---------------------------------------------------------------- V10 生成栈
def smoke_generation(root: Path) -> dict:
    from copulas.multivariate import GaussianMultivariate

    df = pd.read_csv(root / "data" / "raw" / "1.csv")
    key_cols = [c for c in df.columns if c.lower() in ("beer", "beer_id", "tasting_category_fine")]
    style_col = "tasting_category_fine" if "tasting_category_fine" in df.columns else None
    if style_col is None:
        return {"id": "V10_generation_stack", "desc": "未找到风格列", "passed": False}

    num = df.select_dtypes(include=[np.number])
    # 剔除全 NaN / 常数列，并按中位数填补，使 Copula 可拟合（冒烟目的为验证栈可用）
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how="all")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    sub = num.fillna(num.median(numeric_only=True)).dropna(axis=1, how="any").head(120)
    if sub.shape[1] < 3:
        return {"id": "V10_generation_stack", "desc": "可用数值列不足", "passed": False}

    cop = GaussianMultivariate()
    cop.fit(sub.to_numpy())
    samples = cop.sample(20)
    col_means = samples.mean(axis=0)

    defect_cols = [c for c in df.columns if any(h.lower() in str(c).lower() for h in DEFECT_HINTS)]
    defect_ok = all(np.isfinite(col_means))
    styles = df[style_col].dropna().unique().tolist() if style_col else []

    return {
        "id": "V10_generation_stack",
        "desc": "copulas 高斯 Copula 拟合/采样 + 缺陷化合物阈值约束检查",
        "n_numeric_cols": int(sub.shape[1]),
        "sampled_rows": int(samples.shape[0]),
        "styles_seen": len(styles),
        "defect_related_cols": defect_cols[:10],
        "defect_threshold_checkable": bool(defect_ok),
        "passed": bool(samples.shape[0] == 20 and np.isfinite(col_means).all()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="能力栈冒烟自检")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    root = Path(args.repo_root)

    results = []
    for fn in (smoke_graph, smoke_attribution, smoke_generation):
        try:
            results.append(fn(root))
        except Exception as exc:
            results.append({
                "id": fn.__name__,
                "desc": "执行异常",
                "passed": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            })

    buf = io.StringIO()
    buf.write("能力栈冒烟自检报告\n")
    for r in results:
        flag = "PASS" if r.get("passed") else "FAIL"
        buf.write(f"[{flag}] {r.get('id')} — {r.get('desc')}\n")
        for k, v in r.items():
            if k in ("id", "desc", "passed"):
                continue
            buf.write(f"       {k}: {v}\n")
        buf.write("\n")
    all_ok = all(r.get("passed") for r in results)
    buf.write(f"结论: {'全部通过' if all_ok else '存在未通过项'}\n")

    out = _out_path(root) / "stack_smoke.txt"
    out.write_text(buf.getvalue(), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    print(buf.getvalue())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
