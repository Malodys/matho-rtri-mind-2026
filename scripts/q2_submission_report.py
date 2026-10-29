#!/usr/bin/env python3
"""Build the chemical-audit table for the canonical Q2 submission baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIRECT = {
    "X4vg": "X4VG",
    "diacetyl": "X2.3.butanedione",
    "dms": "DMS",
    "t2n": "X.E..2.nonenal",
}


def chemistry_assessment(label: str, feature: str, chemical_name: str, origin: str) -> tuple[str, str]:
    text = f"{feature} {chemical_name}".lower()
    if DIRECT.get(label) == feature:
        return "direct_marker", "感官词与同名或直接对应化合物一致。"
    ester = any(term in text for term in (
        "acetate", "butyrate", "hexanoate", "octanoate", "decanoate",
        "propanoate", "isovalerate", "ester",
    ))
    hop = any(term in text for term in (
        "linalool", "geraniol", "citronellol", "humulene", "caryophyll",
        "farnes", "terpine", "ocimene",
    ))
    malt = any(term in text for term in (
        "pyrazine", "furfural", "acetylpyrrole", "benzaldehyde", "isomaltol",
    ))
    if "esters" in label and ester:
        return "pathway_consistent", "酯类或相关发酵产物，与果香、花香或酯香表达一致。"
    if "hops" in label and hop:
        return "pathway_consistent", "酒花萏烯或含氧萏类成分，与酒花类感官表达一致。"
    if "malt" in label and malt:
        return "pathway_consistent", "麦芽烘焙或美拉德反应相关产物，与麦芽类感官表达一致。"
    if label in {
        "alcohol", "body", "aftertaste", "overall", "bitternes", "co2",
        "sweetness", "acidity", "lactic", "acetic",
    }:
        return "process_proxy", "可反映发酵、酒花投料或风格共变，但不是该感官维度的专一指标。"
    if origin and origin != "nan":
        return "plausible_proxy", f"文献来源分类为{origin}，可作风格或工艺关联标志物。"
    return "plausible_proxy", "缺少直接感官通路证据，作为化学关联候选保留。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    result_dir = args.result_dir or root / "reports/modeling/q2_final_v1"

    top = pd.read_csv(result_dir / "sensory_top3.csv")
    dictionary = pd.read_csv(root / "data/interim/chemical_dictionary.csv").rename(
        columns={"column": "feature"}
    )
    columns = ["feature", "chemical_name", "origin_paper", "off_flavors"]
    output = top.merge(dictionary[columns], on="feature", how="left")
    assessments = [
        chemistry_assessment(row.label, row.feature, str(row.chemical_name), str(row.origin_paper))
        for row in output.itertuples()
    ]
    output["chemical_plausibility"] = [value[0] for value in assessments]
    output["chemical_rationale"] = [value[1] for value in assessments]
    output.to_csv(result_dir / "sensory_top3_enriched.csv", index=False)
    print(result_dir / "sensory_top3_enriched.csv")


if __name__ == "__main__":
    main()
