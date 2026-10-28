# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：判定规则（log10 列 slope≈1 且 r>0.9999；raw 列 slope≈1 且 r>0.999）与
# 本轮人工回归结论（37 log10 / 19 raw / 14 ambiguous）逐条核对。
"""逐列审计 1.csv 的标度，并与队友 provenance 表对账。

用法：
    python scripts/audit_scale.py [--repo-root <仓库根>]

输出：
    data/interim/chem_scale_audit_v1.csv
    控制台摘要（各裁定类别的化合物清单 + 与 provenance 的不一致项）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_contract as dc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="1.csv 逐列标度审计")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args()
    if args.repo_root:
        dc.REPO_ROOT = args.repo_root.resolve()
        dc.RAW_DIR = dc.REPO_ROOT / "data" / "raw"
        dc.INTERIM_DIR = dc.REPO_ROOT / "data" / "interim"

    raw = dc.load_raw()
    audit = dc.audit_scale(raw)

    print("=" * 78)
    print("标度裁定（1.csv vs 8.xlsx Calibrated chemical dataset）")
    print("=" * 78)
    print(f"  可对齐化合物 : {len(audit)}")
    for verdict in ("LOG10", "RAW", "AMBIG"):
        cols = sorted(audit.loc[audit["verdict"] == verdict, "file1_column"].tolist())
        print(f"  {verdict:6s} ({len(cols):2d}): {', '.join(cols)}")

    mismatched = audit[~audit["matches_provenance"]]
    print()
    print(f"  与 provenance 不一致: {len(mismatched)}")
    for _, r in mismatched.iterrows():
        print(f"    {r['file1_column']:28s} 裁定={r['verdict']:6s} "
              f"provenance={r['provenance_transform'] or '(空)'}/{r['provenance_block']}")

    out = dc.INTERIM_DIR / "chem_scale_audit_v1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(audit.to_csv(index=False), encoding="utf-8")
    dc.write_manifest(out, {"data_version": dc.DATA_VERSION,
                             "raw_manifest": raw.manifest,
                             "verdicts": audit["verdict"].value_counts().to_dict(),
                             "n_mismatched_vs_provenance": int(len(mismatched))})
    print()
    print(f"  已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
