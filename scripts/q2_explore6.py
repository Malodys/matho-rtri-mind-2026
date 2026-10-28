# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物。
"""Leiden 参数探查：比较 n_iterations / 加权 / resolution 对社区划分的影响。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx

import data_contract as dc


def main() -> int:
    raw = dc.load_raw(dc.RAW_DIR)
    corr = raw.corr_chem
    names = list(corr.index)
    vals = corr.to_numpy(dtype=float)
    np.fill_diagonal(vals, 0.0)
    np.nan_to_num(vals, copy=False)

    g = nx.Graph()
    g.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(vals[i, j]) >= 0.6:
                g.add_edge(names[i], names[j], weight=float(vals[i, j]))
    idx = {nm: i for i, nm in enumerate(names)}
    edges = [(idx[a], idx[b]) for a, b in g.edges()]

    for weighted in (False, True):
        ig_g = ig.Graph(n=len(names))
        ig_g.add_edges(edges)
        if weighted:
            ig_g.es["weight"] = [abs(g[a][b]["weight"]) for a, b in g.edges()]
        for nit in (5, -1):
            for res in (1.0,):
                c = ig_g.community_leiden(objective_function="modularity",
                                          n_iterations=nit, resolution=res,
                                          weights="weight" if weighted else None)
                sizes = sorted(c.sizes(), reverse=True)
                print(f"weighted={weighted} n_iterations={nit} resolution={res}: "
                      f"n_comm={len(c)} top5={sizes[:5]} "
                      f"modularity={ig_g.modularity(c):.4f} singletons={sum(1 for s in sizes if s==1)}")

    # 参考：Louvain / walktrap / fastgreedy 作为对照
    ig_g = ig.Graph(n=len(names))
    ig_g.add_edges(edges)
    for algo in ("louvain", "walktrap", "fastgreedy", "label_propagation"):
        try:
            if algo == "louvain":
                c = ig_g.community_multilevel()
            elif algo == "walktrap":
                c = ig_g.community_walktrap()
            elif algo == "fastgreedy":
                c = ig_g.community_fastgreedy()
            else:
                c = ig_g.community_label_propagation()
            sizes = sorted(c.sizes(), reverse=True)
            print(f"{algo}: n_comm={len(c)} top5={sizes[:5]} modularity={ig_g.modularity(c):.4f}")
        except Exception as e:
            print(f"{algo}: 失败 {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
