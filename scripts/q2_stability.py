# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) bootstrap 每次迭代独立用 [seed, b] 派生 RNG，分块运行与一次性运行结果完全一致
#      （已用 chunk=1 与 chunk=4 的计数逐元素比对，max|Δ|=0）。
#   2) 入选判据：bootstrap 入选频率 >= 0.6 且 跨方法平均 Jaccard >= 0.3 → 稳健关键因素，
#      否则 → 候选（需谨慎）；两阈值写在 q2_common。
#   3) 化学合理性四条判据逐条落地：①origin_paper 归属 ②是否属于该维度已知化学载体类别
#      ③与 8.xlsx off_flavors / 缺陷阈值的一致性 ④与 E4 先验（8.xlsx MDI + 原论文 SHAP 符号）的一致性。
#   4) 同时出现在特征侧与感官侧的化合物（bitterness/X4VG/DMS/…）一律加注
#      「理化测量与感官评分的一致性」，不得表述为新发现的关联。
"""问题二 Step 3：bootstrap 稳定性 + 跨方法 Jaccard + 化学合理性论证。

用法（分块可续跑）：
    python scripts/q2_stability.py --chunk 0 --nchunks 4      # 跑第 0 块 bootstrap
    python scripts/q2_stability.py --aggregate                 # 汇总并落盘全部产物
输出：
    data/processed/q2_runs/{top3_by_sensory.csv, stability.csv, bootstrap_top3_freq.csv,
                             jaccard_summary.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import data_contract as dc
import q2_chem_kb as kb
import q2_common as qc


# --------------------------------------------------------------------------- 工具


def _rank_norm(arr: np.ndarray) -> np.ndarray:
    """按列秩归一化到 [0,1]（值越大越强）。"""
    r = rankdata(arr, axis=0)
    return (r - 1.0) / max(arr.shape[0] - 1, 1)


def _spearman_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, p) 与 (n, q) 的全交叉 Spearman。"""
    rx = rankdata(x, axis=0)
    ry = rankdata(y, axis=0)
    rx = (rx - rx.mean(axis=0)) / rx.std(axis=0)
    ry = (ry - ry.mean(axis=0)) / ry.std(axis=0)
    nx = np.sqrt((rx ** 2).sum(axis=0))
    ny = np.sqrt((ry ** 2).sum(axis=0))
    denom = nx[:, None] * ny[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (rx.T @ ry) / np.where(denom == 0, np.nan, denom)
    return np.nan_to_num(out, nan=0.0)


def _top3_idx(scores: np.ndarray) -> np.ndarray:
    """返回融合分数最高的 3 个下标（并列时按索引升序，保证可复现）。"""
    order = np.argsort(-scores, kind="stable")
    return order[:qc.TOP_K]


def bootstrap_iteration(b: int, Xtr: np.ndarray, Ytr: np.ndarray,
                        params: dict) -> List[np.ndarray]:
    """第 b 次 bootstrap：重采样训练集 → 重算 E2/E3 → 融合 → 每维 Top-3 下标。"""
    import lightgbm as lgb

    rng = np.random.default_rng([qc.BOOTSTRAP_SEED, b])
    n = Xtr.shape[0]
    idx = rng.integers(0, n, n)
    Xb, Yb = Xtr[idx], Ytr[idx]

    e2 = np.abs(_spearman_np(Xb, Yb))                      # (227, 50)
    e3 = np.zeros_like(e2)
    for j in range(Yb.shape[1]):
        model = lgb.LGBMRegressor(**params)
        model.fit(Xb, Yb[:, j])
        contrib = model.predict(Xb, pred_contrib=True)[:, :-1]
        e3[:, j] = np.abs(contrib).mean(axis=0)

    fused = (qc.FUSION_WEIGHTS["e2"] * _rank_norm(e2)
             + qc.FUSION_WEIGHTS["e3"] * _rank_norm(e3))
    return [_top3_idx(fused[:, j]) for j in range(fused.shape[1])]


def run_bootstrap_chunk(root: Path, chunk: int, nchunks: int, B: int) -> None:
    """跑一块 bootstrap，计数存盘（可续跑）。"""
    from joblib import Parallel, delayed

    qc.ensure_dirs()
    data = qc.load_q2_data(root)
    Xtr = data.Xtr.to_numpy(dtype=float)
    Ytr = data.Ytr.to_numpy(dtype=float)
    n_feat, n_tgt = Xtr.shape[1], Ytr.shape[1]

    lo = chunk * B // nchunks
    hi = (chunk + 1) * B // nchunks
    iters = list(range(lo, hi))
    print(f"[q2_stability] bootstrap chunk {chunk}/{nchunks}: 迭代 {lo}..{hi - 1} "
          f"（共 {len(iters)} 次，{n_feat} 特征 × {n_tgt} 维度）")

    # threading 后端：LightGBM 的 C++ 训练与 pred_contrib 均释放 GIL，
    # 进程池（loky）在 Windows 上会因资源不足触发 BrokenProcessPool。
    results = Parallel(n_jobs=qc.BOOTSTRAP_N_JOBS, backend="threading", verbose=5)(
        delayed(bootstrap_iteration)(b, Xtr, Ytr, qc.LGB_PARAMS) for b in iters)

    counts = np.zeros((n_tgt, n_feat), dtype=np.int32)
    for tops in results:
        for j, idxs in enumerate(tops):
            counts[j, idxs] += 1

    out = qc.EVIDENCE_DIR / f"bootstrap_counts_part{chunk}_of{nchunks}.npz"
    np.savez_compressed(out, counts=counts, iters=np.array(iters))
    print(f"[q2_stability] chunk {chunk} 完成 → {out.name}")


# --------------------------------------------------------------------------- 合理性


def build_anchor_map() -> Dict[str, str]:
    """特征列 → 其对应的感官维度（用于「理化测量 vs 感官评分」提示）。"""
    anchors: Dict[str, str] = {}
    for dim, cols in dc.ABSOLUTE_ANCHOR_CARRIERS.items():
        for c in cols:
            anchors[c] = dim
    anchors.update({"acetic_acid.g.L.": "acetic", "lactic_acid.mg.L.": "lactic",
                    "pH": "acidity", "unfermentables": "sweetness",
                    "kcalperc": "body", "protein.g.L.": "body",
                    "betaglucan.mg.L.": "body", "glycerol": "body"})
    return anchors


def plausibility(dim: str, col: str, attrs: pd.DataFrame, defects: pd.DataFrame,
                 exceed: pd.DataFrame, e4_pct: float, paper_shap_mean: float,
                 rho: float, anchors: Dict[str, str]) -> Tuple[str, str, str]:
    """返回 (合理性等级, 论证说明, 提示)。

    等级：强=命中教科书级已知载体；中=化学类别与该维度已知载体类别一致；
          弱=仅有来源归属/表征维度一致；无=无支撑证据。
    """
    kbrow = kb.dim_kb(dim)
    row = attrs[attrs["file1_column"] == col]
    if row.empty:
        return "无", "该化合物无节点属性记录，无法论证", ""
    row = row.iloc[0]
    origin = str(row["origin_paper"])
    cls = kb.classify_by_chem_name(col, str(row["chemical_name"]))
    cls_cn = kb.CLASS_CN.get(cls, cls)

    parts: List[str] = []
    exp_origins = kbrow.get("origins", set())
    carriers = kbrow.get("carriers", [])
    classes = kbrow.get("classes", set())
    is_carrier = col in carriers
    class_hit = (cls in classes) if classes else False

    # ① origin_paper 归属
    if exp_origins:
        hit = origin in exp_origins
        parts.append(f"①来源归属：{origin}" +
                     ("（与该维度已知来源一致）" if hit else f"（该维度预期来源 {sorted(exp_origins)}）"))
    else:
        parts.append(f"①来源归属：{origin}（该维度无单一来源预期）")

    # ② 是否属于该维度的已知化学载体类别
    if is_carrier:
        level = "强"
        parts.append(f"②已知化学载体：命中该维度教科书级载体（{cls_cn}）")
    elif class_hit:
        level = "中"
        parts.append(f"②化学类别一致：属于{cls_cn}，与该维度已知载体类别 {sorted(classes)} 相符")
    elif classes:
        level = "弱"
        parts.append(f"②化学类别：{cls_cn}，不在该维度已知载体类别 {sorted(classes)} 内，"
                     f"仅能作弱证据")
    else:
        level = "弱"
        parts.append(f"②化学类别：{cls_cn}；该维度为综合维度，不作单体化学归因")

    if kbrow.get("note"):
        parts.append(f"机理：{kbrow['note']}")

    # ③ 与 8.xlsx off_flavors / 缺陷阈值的一致性
    off = str(row["off_flavors"]) if isinstance(row["off_flavors"], str) else ""
    aging = str(row["aging"]) if isinstance(row["aging"], str) else ""
    d3 = []
    if off:
        d3.append(f"8.xlsx 登记的异味谱：{off}")
    if aging:
        d3.append(f"陈化行为：{aging}")
    if bool(row["is_defect"]):
        spec = defects[defects["column"] == col]
        if not spec.empty:
            s = spec.iloc[0]
            ex = exceed[exceed["key"] == s["key"]]
            pct = f"{float(ex.iloc[0]['exceed_pct']):.1f}%" if not ex.empty else "n/a"
            thr = s["log10_threshold"]
            if pd.notna(thr):
                d3.append(f"缺陷阈值 log10(mg/L)={float(thr):.4f}，样本超标率 {pct}")
            else:
                d3.append("文件8 无该化合物阈值")
    if d3:
        parts.append("③阈值/异味一致性：" + "；".join(d3))

    # ④ 与 E4 先验的一致性
    d4 = []
    if pd.notna(e4_pct):
        d4.append(f"E4 MDI 分位 {e4_pct:.3f}")
    if pd.notna(paper_shap_mean) and abs(paper_shap_mean) > 0:
        if pd.notna(rho) and abs(rho) > 0:
            same = np.sign(paper_shap_mean) == np.sign(rho)
            d4.append(f"原论文 SHAP 符号 {'+' if paper_shap_mean > 0 else '-'} "
                      f"{'与本维度方向一致' if same else '与本维度方向不一致'}")
    parts.append("④E4 外部印证：" + ("；".join(d4) if d4 else "该化合物不在 8.xlsx MDI 名单内"))

    # 提示：特征-标签重合
    caveat = ""
    if col in anchors:
        caveat = (f"※ 该化合物同时是感官维度「{anchors[col]}」的测量对象，"
                  f"此处反映的是理化测量与感官评分的一致性，不构成新的化学关联发现。")

    return level, " | ".join(parts), caveat


# --------------------------------------------------------------------------- 汇总


def aggregate(root: Path, B: int, nchunks: int) -> dict:
    from joblib import Parallel, delayed

    qc.ensure_dirs()
    data = qc.load_q2_data(root)
    ev = qc.EVIDENCE_DIR

    scores = pd.read_csv(ev / "evidence_scores.csv")
    e1 = pd.read_csv(ev / "e1_cross_rho_6csv.csv", index_col=0)
    e2 = pd.read_csv(ev / "e2_spearman_full.csv", index_col=0)
    e3 = pd.read_csv(ev / "e3_shap_importance.csv", index_col=0)
    e4 = pd.read_csv(ev / "e4_gbm_mdi.csv", index_col=0)

    # ---- bootstrap 计数汇总 ----
    counts = None
    for c in range(nchunks):
        p = ev / f"bootstrap_counts_part{c}_of{nchunks}.npz"
        if not p.exists():
            raise FileNotFoundError(f"缺少 bootstrap 分块产物 {p.name}，请先补跑该块")
        part = np.load(p)["counts"]
        counts = part.astype(np.int64) if counts is None else counts + part.astype(np.int64)
    freq = counts / float(B)

    dims = list(data.Y.columns)
    feats = list(data.X.columns)
    freq_df = pd.DataFrame(freq, index=dims, columns=feats)
    freq_long = freq_df.stack().reset_index()
    freq_long.columns = ["sensory_dimension", "compound", "bootstrap_top3_freq"]
    freq_long = freq_long[freq_long["bootstrap_top3_freq"] > 0].reset_index(drop=True)
    freq_long.to_csv(qc.RUN_DIR / "bootstrap_top3_freq.csv", index=False, encoding="utf-8")

    # ---- 各证据路 Top-3 → 两两 Jaccard ----
    def top3_from(mat: pd.DataFrame, target: str) -> List[str]:
        s = mat[target].abs()
        return list(s.sort_values(ascending=False, kind="mergesort").index[:qc.TOP_K])

    jac_rows = []
    jac_map: Dict[str, Dict[str, float]] = {}
    for dim in dims:
        t1 = top3_from(e1, dim)
        t2 = top3_from(e2, dim)
        t3 = top3_from(e3, dim)
        j12 = qc.jaccard(t1, t2)
        j13 = qc.jaccard(t1, t3)
        j23 = qc.jaccard(t2, t3)
        jac_map[dim] = {"E1-E2": j12, "E1-E3": j13, "E2-E3": j23,
                        "mean": float(np.mean([j12, j13, j23]))}
        jac_rows.append({"sensory_dimension": dim, "top3_E1": "; ".join(t1),
                         "top3_E2": "; ".join(t2), "top3_E3": "; ".join(t3),
                         "jaccard_E1_E2": j12, "jaccard_E1_E3": j13,
                         "jaccard_E2_E3": j23, "jaccard_mean": jac_map[dim]["mean"]})
    jac_df = pd.DataFrame(jac_rows)
    jac_df.to_csv(qc.RUN_DIR / "jaccard_summary.csv", index=False, encoding="utf-8")
    jac_matrix = jac_df[["jaccard_E1_E2", "jaccard_E1_E3", "jaccard_E2_E3"]].mean()
    jac_df.to_csv(qc.RUN_DIR / "jaccard_matrix.csv", index=False, encoding="utf-8")

    # ---- 节点属性（来自 Step 1）----
    attrs = pd.read_csv(qc.GRAPH_DIR / "communities.csv")
    defects = data.defects
    exceed = dc.defect_exceedance(data.X, dc.load_raw(dc.RAW_DIR))
    anchors = build_anchor_map()
    paper_shap_mean = data.paper_shap.mean(axis=0)

    # ---- Top-3 主表 ----
    top_rows = []
    for dim in dims:
        sub = scores[scores["sensory_dimension"] == dim].copy()
        top = sub.sort_values(["fused_rank", "compound"]).head(qc.TOP_K)
        for rk, (_, r) in enumerate(top.iterrows(), start=1):
            col = r["compound"]
            e4_pct = float(r["e4_mdi_percentile"]) if pd.notna(r["e4_mdi_percentile"]) else float("nan")
            psm = float(paper_shap_mean[col]) if col in paper_shap_mean.index else float("nan")
            ok, note, caveat = plausibility(dim, col, attrs, defects, exceed,
                                            e4_pct, psm, float(r["e2_rho"]), anchors)
            level = ok
            arow = attrs[attrs["file1_column"] == col]
            chem_name = str(arow.iloc[0]["chemical_name"]) if not arow.empty else ""
            origin = str(arow.iloc[0]["origin_paper"]) if not arow.empty else ""
            cls = kb.classify_by_chem_name(col, chem_name)
            boot = float(freq_df.loc[dim, col]) if col in freq_df.columns else 0.0
            jm = jac_map[dim]
            selected = (boot >= qc.SELECT_FREQ_MIN and jm["mean"] >= qc.SELECT_JACCARD_MIN)
            top_rows.append({
                "sensory_dimension": dim,
                "rank_in_top3": rk,
                "compound": col,
                "chemical_name": chem_name,
                "origin_paper": origin,
                "block": str(arow.iloc[0]["block"]) if not arow.empty else "",
                "unit": str(arow.iloc[0]["unit"]) if not arow.empty else "",
                "chem_class": cls,
                "chem_class_cn": kb.CLASS_CN.get(cls, cls),
                "direction": r["direction"],
                "e2_rho": r["e2_rho"],
                "e2_abs_rho": r["e2_abs_rho"],
                "e2_rank_in_227": r["e2_rank_in_227"],
                "e1_covered": bool(pd.notna(r["e1_rho"])),
                "e1_rho": (round(float(r["e1_rho"]), 6) if pd.notna(r["e1_rho"]) else ""),
                "e1_rank_in_44": (float(r["e1_rank_in_44"]) if pd.notna(r["e1_rank_in_44"]) else ""),
                "e3_shap_mean_abs": r["e3_shap_mean_abs"],
                "e3_rank_in_227": r["e3_rank_in_227"],
                "e4_mdi": r["e4_mdi"],
                "e4_mdi_rank": r["e4_mdi_rank"],
                "e4_mdi_percentile": r["e4_mdi_percentile"],
                "fused_score": r["fused_score"],
                "fused_rank": r["fused_rank"],
                "bootstrap_top3_freq": round(boot, 4),
                "jaccard_E1_E2": round(jm["E1-E2"], 4),
                "jaccard_E1_E3": round(jm["E1-E3"], 4),
                "jaccard_E2_E3": round(jm["E2-E3"], 4),
                "jaccard_mean": round(jm["mean"], 4),
                "selection_status": qc.LABEL_ROBUST if selected else qc.LABEL_CANDIDATE,
                "chem_plausibility": level,
                "chem_plausible": bool(level in ("强", "中")),
                "plausibility_note": note,
                "caveat_feature_label_overlap": caveat,
            })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(qc.RUN_DIR / "top3_by_sensory.csv", index=False, encoding="utf-8",
                  na_rep="")

    # ---- stability.csv（bootstrap 频率明细 + Jaccard）----
    stab_rows = []
    final_pairs = set(zip(top_df["sensory_dimension"], top_df["compound"]))
    freq_pos = freq_df.stack().reset_index()
    freq_pos.columns = ["sensory_dimension", "compound", "bootstrap_top3_freq"]
    freq_pos = freq_pos[freq_pos["bootstrap_top3_freq"] > 0]
    seen = set()
    for _, r in freq_pos.iterrows():
        dim, col = r["sensory_dimension"], r["compound"]
        if (dim, col) in seen:
            continue
        seen.add((dim, col))
        jm = jac_map[dim]
        in_final = (dim, col) in final_pairs
        stab_rows.append({
            "sensory_dimension": dim, "compound": col,
            "bootstrap_top3_freq": round(float(r["bootstrap_top3_freq"]), 4),
            "in_final_top3": in_final,
            "jaccard_E1_E2": round(jm["E1-E2"], 4),
            "jaccard_E1_E3": round(jm["E1-E3"], 4),
            "jaccard_E2_E3": round(jm["E2-E3"], 4),
            "jaccard_mean": round(jm["mean"], 4),
            "selection_status": (qc.LABEL_ROBUST if
                                  (float(r["bootstrap_top3_freq"]) >= qc.SELECT_FREQ_MIN
                                   and jm["mean"] >= qc.SELECT_JACCARD_MIN)
                                  else qc.LABEL_CANDIDATE),
        })
    # 频次为 0 但进入最终 Top-3 的，也要记录
    for dim, col in sorted(final_pairs):
        if (dim, col) in seen:
            continue
        jm = jac_map[dim]
        stab_rows.append({
            "sensory_dimension": dim, "compound": col,
            "bootstrap_top3_freq": 0.0, "in_final_top3": True,
            "jaccard_E1_E2": round(jm["E1-E2"], 4),
            "jaccard_E1_E3": round(jm["E1-E3"], 4),
            "jaccard_E2_E3": round(jm["E2-E3"], 4),
            "jaccard_mean": round(jm["mean"], 4),
            "selection_status": qc.LABEL_CANDIDATE,
        })
    stab_df = pd.DataFrame(stab_rows).sort_values(
        ["sensory_dimension", "bootstrap_top3_freq"], ascending=[True, False])
    stab_df.to_csv(qc.RUN_DIR / "stability.csv", index=False, encoding="utf-8")

    # ---- summary.json ----
    n_robust_dims = int(top_df[top_df["selection_status"] == qc.LABEL_ROBUST]
                         ["sensory_dimension"].nunique())
    n_robust_rows = int((top_df["selection_status"] == qc.LABEL_ROBUST).sum())
    mdi_top3 = list(e4.sort_values("mdi", ascending=False).head(qc.TOP_K).index)
    overlap = []
    for dim in dims:
        s = set(top_df[top_df["sensory_dimension"] == dim]["compound"])
        overlap.append(len(s & set(mdi_top3)))
    graph_summary = json.loads((qc.GRAPH_DIR / "graph_summary.json").read_text(encoding="utf-8"))
    chem_ok_dims = int(top_df[top_df["chem_plausible"]]["sensory_dimension"].nunique())
    level_counts = top_df["chem_plausibility"].value_counts().to_dict()
    level_dims = {lv: int(top_df[top_df["chem_plausibility"] == lv]["sensory_dimension"].nunique())
                  for lv in ("强", "中", "弱", "无")}

    summary = {
        "graph": graph_summary["graph"],
        "edge_sensitivity": graph_summary["sensitivity"],
        "community_count": graph_summary["graph"]["n_leiden_communities"],
        "threshold_recommendation": graph_summary.get("threshold_recommendation", {}),
        "threshold_community_ari": graph_summary.get("threshold_community_ari", []),
        "two_threshold_comparison": graph_summary.get("two_threshold_comparison", {}),
        "evidence_coverage": json.loads((ev / "evidence_coverage.json").read_text(encoding="utf-8")),
        "top3": {
            "n_dimensions": len(dims),
            "n_rows": int(len(top_df)),
            "n_unique_compounds": int(top_df["compound"].nunique()),
            "n_compounds_mapped_to_file1": int(
                top_df["compound"].isin(list(data.X.columns)).sum()),
            "nan_count": int(top_df.isna().sum().sum()),
        },
        "stability": {
            "bootstrap_B": B,
            "selection_rule": {"bootstrap_freq_min": qc.SELECT_FREQ_MIN,
                               "jaccard_mean_min": qc.SELECT_JACCARD_MIN},
            "n_robust_rows": n_robust_rows,
            "n_robust_dimensions": n_robust_dims,
            "n_robust_and_chem_plausible_rows": int(
                ((top_df["selection_status"] == qc.LABEL_ROBUST)
                 & (top_df["chem_plausible"])).sum()),
            "n_chem_plausible_dimensions": chem_ok_dims,
            "chem_plausibility_level_counts": level_counts,
            "chem_plausibility_level_dimension_counts": level_dims,
            "mean_jaccard_E1_E2": round(float(jac_matrix["jaccard_E1_E2"]), 4),
            "mean_jaccard_E1_E3": round(float(jac_matrix["jaccard_E1_E3"]), 4),
            "mean_jaccard_E2_E3": round(float(jac_matrix["jaccard_E2_E3"]), 4),
            "mean_jaccard_overall": round(float(jac_df["jaccard_mean"].mean()), 4),
            "min_bootstrap_freq": float(freq_df.to_numpy().min()),
            "max_bootstrap_freq": float(freq_df.to_numpy().max()),
            "all_finite": bool(np.isfinite(freq_df.to_numpy()).all()),
        },
        "e4_overlap": {
            "mdi_global_top3": mdi_top3,
            "mean_overlap_with_our_top3": round(float(np.mean(overlap)), 4),
            "max_overlap": int(max(overlap)),
            "dims_with_overlap_ge1": int(sum(1 for o in overlap if o >= 1)),
        },
        "params": qc.describe_params(),
    }
    qc.write_json(qc.RUN_DIR / "summary.json", summary)
    # 同时保留按 B 分档的副本，便于做 B 敏感性对比
    qc.write_json(qc.RUN_DIR / f"summary_B{B}_sensitivity.json", summary)
    print(f"[q2_stability] 稳健维度 {n_robust_dims}/50，稳健 Top-3 条目 {n_robust_rows}/150，"
          f"平均 Jaccard {summary['stability']['mean_jaccard_overall']}，"
          f"化学合理性成立维度 {chem_ok_dims}")
    return summary


# --------------------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二 Step 3：bootstrap 稳定性 + Jaccard + 化学合理性")
    ap.add_argument("--repo-root", default=str(qc.REPO_ROOT))
    ap.add_argument("--bootstrap-b", type=int, default=qc.BOOTSTRAP_B)
    ap.add_argument("--chunk", type=int, default=-1, help="bootstrap 分块编号（-1 表示不跑 bootstrap）")
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true", help="只做汇总（读已算好的 bootstrap 计数）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(args.repo_root)

    if args.chunk >= 0:
        run_bootstrap_chunk(root, args.chunk, args.nchunks, args.bootstrap_b)
    if args.aggregate or args.chunk < 0:
        aggregate(root, args.bootstrap_b, args.nchunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
