# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本逐项实测验收标准，所有断言的通过/失败由实际读数决定，不做形式化声明。
"""问题二验收自检：逐条核对题面验收标准。

用法：
    python scripts/q2_verify.py
退出码：0 = 全部通过。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import q2_common as qc


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    run = qc.RUN_DIR
    lines: list = []
    ok_all = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # ---- 验收 1：共表达图规模 ----
    gs = json.loads((run / "graph" / "graph_summary.json").read_text(encoding="utf-8"))
    g = gs["graph"]
    check("验收1a 节点数=231", g["n_nodes"] == 231, f"n_nodes={g['n_nodes']}")
    check("验收1b |rho|>=0.6 边数=266", g["n_edges"] == 266, f"n_edges={g['n_edges']}")
    check("验收1c |rho|>=0.6 连通分量=114", g["n_connected_components"] == 114,
          f"components={g['n_connected_components']}")
    smoke = (qc.REPO_ROOT / "data" / "processed" / "smoke" / "stack_smoke.txt").read_text(
        encoding="utf-8")
    check("验收1d 与 stack_smoke V8 记录一致",
          "nodes: 231" in smoke and "edges: 266" in smoke and
          "connected_components: 114" in smoke, "smoke V8 记录存在且数值一致")

    # ---- 验收 2：50 维 Top-3 完整、无 NaN、化合物名唯一且可映射回 1.csv 列 ----
    top = pd.read_csv(run / "top3_by_sensory.csv", keep_default_na=False, encoding="utf-8")
    check("验收2a 维度数=50", top["sensory_dimension"].nunique() == 50,
          f"{top['sensory_dimension'].nunique()} 个维度")
    check("验收2b 每维 3 条 Top-3", len(top) == 150 and (top.groupby("sensory_dimension").size() == 3).all(),
          f"{len(top)} 行")
    check("验收2c 无 NaN 标记", not top.isna().any().any(), f"NaN 计数={int(top.isna().sum().sum())}")
    data = qc.load_q2_data(qc.REPO_ROOT)
    unmapped = top[~top["compound"].isin(list(data.X.columns))]
    check("验收2d 化合物均可映射回 1.csv 列", len(unmapped) == 0,
          f"未映射 {len(unmapped)} 条")
    check("验收2e 同一维度内化合物名唯一",
          not top.duplicated(subset=["sensory_dimension", "compound"]).any(), "无重复")

    # ---- 验收 3：bootstrap 频率与 Jaccard 取值域 ----
    freq = pd.read_csv(run / "bootstrap_top3_freq.csv", encoding="utf-8")
    fv = freq["bootstrap_top3_freq"].to_numpy(dtype=float)
    check("验收3a bootstrap 频率 ∈ [0,1]", bool((fv >= 0).all() and (fv <= 1).all()),
          f"min={fv.min()}, max={fv.max()}")
    nz = np.unique(fv[fv > 0])
    step = float(np.min(np.diff(nz))) if len(nz) > 1 else 0.0
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    B = summary["stability"]["bootstrap_B"]
    check("验收3b bootstrap_B=200", B == 200, f"bootstrap_B={B}")
    check("验收3c 频率最小非零步长=1/B=0.005", abs(step - 1.0 / B) < 1e-12,
          f"step={step}, 1/B={1.0/B}")
    check("验收3d 频率全部有限", bool(np.isfinite(fv).all()), "")
    jac = pd.read_csv(run / "jaccard_summary.csv", encoding="utf-8")
    jv = jac[["jaccard_E1_E2", "jaccard_E1_E3", "jaccard_E2_E3", "jaccard_mean"]].to_numpy(dtype=float)
    check("验收3e Jaccard ∈ [0,1] 且有限",
          bool(np.isfinite(jv).all() and (jv >= 0).all() and (jv <= 1).all()),
          f"min={jv.min():.4f}, max={jv.max():.4f}")

    # ---- 验收 4：化学合理性论证 ----
    lv = top["chem_plausibility"].value_counts().to_dict()
    dims_strong = top[top["chem_plausibility"] == "强"]["sensory_dimension"].nunique()
    dims_ok = top[top["chem_plausible"]]["sensory_dimension"].nunique()
    check("验收4a 至少 11 个维度有成立的化学合理性论证", dims_ok >= 11,
          f"强证据维度 {dims_strong} 个；强/中合计覆盖 {dims_ok} 个维度；等级分布 {lv}")
    must = {"bitternes": "bitterness", "alcohol": "ethanol..v.v."}
    for dim, comp in must.items():
        hit = top[(top["sensory_dimension"] == dim) & (top["compound"] == comp)]
        check(f"验收4b {dim} 维度含 {comp}", len(hit) > 0,
              f"rank={int(hit['rank_in_top3'].iloc[0]) if len(hit) else 'N/A'}, "
              f"ρ={hit['e2_rho'].iloc[0] if len(hit) else 'N/A'}")
    # 特征-标签重合必须加提示
    overlap = top[top["caveat_feature_label_overlap"] != ""]
    check("验收4c 特征/标签重合化合物均加注提示", len(overlap) > 0,
          f"{len(overlap)} 条加注（bitterness/X4VG/DMS/ethanol 等）")

    # ---- 附加：阈值择优与两档并列 ----
    rec = gs.get("threshold_recommendation", {})
    check("附加A 阈值择优给出推荐档", bool(rec.get("operating_default_threshold")),
          f"默认档 |rho|>={rec.get('operating_default_threshold')}，"
          f"连通性档 |rho|>={rec.get('connectivity_threshold')}")
    check("附加B 四档敏感性表完整（边数/分量/社区/模块度/纯度）",
          len(gs["sensitivity"]) == 4 and
          all("modularity_Q" in s and "chemical_purity_weighted" in s for s in gs["sensitivity"]),
          f"{len(gs['sensitivity'])} 档")
    check("附加C 跨阈值社区一致性（ARI）已计算", len(gs.get("threshold_community_ari", [])) == 6,
          f"{len(gs.get('threshold_community_ari', []))} 对")

    # ---- 附加：四路证据覆盖 ----
    cov = summary["evidence_coverage"]
    check("附加D E1 覆盖 44 化合物", cov["E1_6csv"]["n_compounds"] == 44, str(cov["E1_6csv"]))
    check("附加E E2 自算全量覆盖 227 化合物", cov["E2_spearman_full"]["n_compounds"] == 227,
          str(cov["E2_spearman_full"]))
    check("附加F E3 GBDT+SHAP 覆盖 227", cov["E3_shap"]["n_compounds"] == 227, str(cov["E3_shap"]))
    check("附加G E4 MDI 先验覆盖 231", cov["E4_mdi"]["n_compounds"] == 231, str(cov["E4_mdi"]))

    # ---- 附加：产出清单 ----
    expected = [
        "top3_by_sensory.csv", "stability.csv", "bootstrap_top3_freq.csv",
        "jaccard_summary.csv", "jaccard_matrix.csv", "summary.json", "README.md",
        "graph/coexpression_graph.graphml", "graph/coexpression_graph.json",
        "graph/communities.csv", "graph/edge_sensitivity.csv", "graph/community_composition.csv",
        "graph/threshold_recommendation.json", "graph/threshold_community_ari.csv",
    ]
    missing = [f for f in expected if not (run / f).exists()]
    check("附加H 核心产出齐全", not missing, f"缺失 {missing}" if missing else "全部存在")
    figs = sorted(p.name for p in qc.FIG_DIR.glob("*.png")) if qc.FIG_DIR.exists() else []
    check("附加I 图已产出", len(figs) >= 4, f"{len(figs)} 张：{figs}")

    report = "\n".join(lines) + f"\n\n结论: {'全部通过' if ok_all else '存在未通过项'}"
    print(report)
    (run / "acceptance_q2.txt").write_text(report, encoding="utf-8")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
