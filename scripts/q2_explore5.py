# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读探查并打印，不产出交付物。
"""问题二探查第五批：231 个化合物的 origin_paper / block / unit 清单。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import data_contract as dc


def main() -> int:
    raw = dc.load_raw()
    chem_bridge, _ = dc.build_bridges(raw)
    prov = dc.load_provenance()
    thr = raw.thresholds.set_index("field_name_R")
    b2 = chem_bridge.query("matrix=='2.csv'")

    rows = []
    for _, r in b2.iterrows():
        c = r["file1_column"]
        rows.append({
            "node": r["matrix_name"], "col": c,
            "origin_paper": thr["origin_paper"].get(c, "") if c in thr.index else "",
            "unit": thr["unit"].get(c, "") if c in thr.index else "",
            "block": prov["block"].get(c, "") if c in prov.index else "",
            "chem_name": thr["chemical_name"].get(c, "") if c in thr.index else "",
            "aging": thr["aging"].get(c, "") if c in thr.index else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "q2_runs" / "_probe_inventory.csv",
              index=False, encoding="utf-8") if False else None
    for op in df["origin_paper"].value_counts().items():
        print(f"\n### origin_paper = {op[0]}  (n={op[1]})")
        sub = df[df["origin_paper"] == op[0]]
        print(", ".join(f"{a}[{b}]" for a, b in zip(sub["col"], sub["chem_name"])))
    print("\n=== unit value counts ===")
    print(df["unit"].value_counts(dropna=False))
    print("\n=== missing origin_paper rows ===")
    print(df[df["origin_paper"] == ""].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
