# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：输出的解析率与裁定计数已逐项对照本轮实测（化学桥 231/231、感官桥 50/50、
# LOG10=37 / RAW=19 / AMBIG=14）核对后才写定通过条件。
"""构建共享数据层的版本化中间产物。

用法：
    python scripts/build_data_contract.py [--repo-root <仓库根>]

产物（均在 data/interim/）：
    bridges_v1.csv          化学桥 + 感官桥（含解析方法与解析率）
    chem_scale_audit_v1.csv 逐列标度审计（log10 / raw / ambiguous）
    defect_thresholds_v1.csv 缺陷化合物阈值表（含统一矩阵上的 log10 阈值）
    bridges_v1_manifest.json 上游 sha256 与关键计数

退出码：0 = 全部自检通过。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_contract as dc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="构建共享数据层中间产物")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args()
    if args.repo_root:
        dc.REPO_ROOT = args.repo_root.resolve()
        dc.RAW_DIR = dc.REPO_ROOT / "data" / "raw"
        dc.INTERIM_DIR = dc.REPO_ROOT / "data" / "interim"
        dc.SPLIT_DIR = dc.REPO_ROOT / "data" / "splits" / "seed_0"

    raw = dc.load_raw()
    print("=" * 78)
    print("原始附件清单（sha256 前 12 位）")
    print("=" * 78)
    for name, meta in raw.manifest.items():
        print(f"  {name:10s} {meta['bytes']:>10d} B  {meta['sha256'][:12]}")

    payload = dc.persist_interim(raw)
    print()
    print("=" * 78)
    print("自检")
    print("=" * 78)
    print(f"  化学桥解析        : {payload['chemical_bridge_resolved']}/{payload['chemical_bridge_total']}")
    print(f"  感官桥解析        : {payload['sensory_bridge_resolved']}/{payload['sensory_bridge_total']}")
    print(f"  标度裁定分布      : {payload['scale_audit_verdicts']}")
    print(f"  与 provenance 一致 : {payload['scale_audit_matches_provenance']}/{payload['scale_audit_total']}")

    matrix = dc.load_modeling_matrix(raw)
    print(f"  建模特征矩阵      : {matrix.shape}（已剔除 4 个 *_sum 派生列）")
    targets = dc.load_targets(raw)
    print(f"  感官目标矩阵      : {targets.shape}")
    split = dc.load_split()
    print(f"  切分              : train={int((split.split == 'train').sum())} "
          f"test={int((split.split == 'test').sum())}")

    print()
    print("=" * 78)
    print("缺陷化合物超阈值（统一矩阵 log10(mg/L) 口径）")
    print("=" * 78)
    exc = dc.defect_exceedance(dc.to_unified_scale(raw), raw)
    for _, r in exc.iterrows():
        if r["has_threshold"]:
            print(f"  {r['key']:9s} {r['column']:22s} log10阈值={r['log10_threshold']:+.4f} "
                  f"超阈值 {r['n_exceed']}/{r['n']} = {r['exceed_pct']:.1f}%")
        else:
            print(f"  {r['key']:9s} {r['column']:22s} 无阈值（以相对不恶化约束兜底）")

    ok = (payload["chemical_bridge_resolved"] == payload["chemical_bridge_total"]
          and payload["sensory_bridge_resolved"] == payload["sensory_bridge_total"]
          and matrix.shape == (250, 227) and targets.shape == (250, 50))
    print()
    print(f"结论: {'全部自检通过' if ok else '存在未通过项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
