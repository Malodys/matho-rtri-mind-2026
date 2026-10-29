# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本脚本仅做只读打印，用于摘录 README 所需的代表性数值。
"""问题二：代表性维度的 Top-3 明细摘录。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="问题二：代表性维度 Top-3 明细摘录（只读）")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    t = pd.read_csv(Path(args.repo_root) / "data" / "processed" / "q2_runs" / "top3_by_sensory.csv",
                    keep_default_na=False, encoding="utf-8")
    dims = ["bitternes", "alcohol", "A_hops_citrus", "X4vg", "dms",
            "A_malt_bread", "A_esters_isoaa", "stale_hops", "overall"]
    for d in dims:
        s = t[t["sensory_dimension"] == d]
        parts = []
        for _, r in s.iterrows():
            e1 = r["e1_rho"] if r["e1_rho"] != "" else "NA"
            parts.append("{} ({}, rho={:+.3f}, E1={}, freq={}, J={}, cls={})".format(
                r["compound"], r["direction"], r["e2_rho"], e1,
                r["bootstrap_top3_freq"], r["jaccard_mean"], r["chem_plausibility"]))
        print(d, "|", " ; ".join(parts))
    print()
    print("caveats:")
    for _, r in t[t["caveat_feature_label_overlap"] != ""].head(4).iterrows():
        print(" ", r["sensory_dimension"], r["compound"], "|", r["caveat_feature_label_overlap"])
    print()
    print("robust dims:", t[t["selection_status"] == "稳健关键因素"]["sensory_dimension"].nunique())
    print("robust+strong rows:",
          len(t[(t["selection_status"] == "稳健关键因素") & (t["chem_plausibility"] == "强")]))
    print(t[(t["selection_status"] == "稳健关键因素") & (t["chem_plausibility"] == "强")]
          [["sensory_dimension", "compound", "e2_rho", "bootstrap_top3_freq"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
