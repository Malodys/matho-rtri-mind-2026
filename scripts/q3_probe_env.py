# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 统一矩阵形状 (250, 231)、缺陷阈值列是否真实存在于统一矩阵中 —— 逐列打印核对。
#   2) 风格样本量分布 —— 与赛题简报给出的 22 类计数逐项比对。
#   3) 常量列（全局/风格内无变异）清单 —— 决定 Copula 相关矩阵的对角处理方式。
#   4) 中文字体可用性 —— 仅探测 matplotlib 能识别的 CJK 字体名，供出图脚本使用。
"""环境与数据事实探针（只读，不写任何产物）。

用途：在正式生成前把「统一矩阵形状 / 缺陷列存在性 / 风格计数 / 常量列 / 中文字体 /
Q1 接口是否就绪」一次性打印出来，避免在建模脚本里靠假设推进。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import data_contract as dc  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("[1] 统一矩阵")
    raw = dc.load_raw()
    full = dc.to_unified_scale(raw)
    model = dc.load_modeling_matrix(raw)
    print(f"  to_unified_scale      : {full.shape}")
    print(f"  load_modeling_matrix  : {model.shape}")
    print(f"  NaN 数               : {int(full.isna().sum().sum())}")
    print(f"  inf 数               : {int(np.isinf(full.to_numpy(dtype=float)).sum())}")
    derived = [c for c in full.columns if c.endswith("_sum")]
    print(f"  派生列(*{dc.DERIVED_SUFFIX})  : {derived}")

    print("=" * 78)
    print("[2] 缺陷阈值表")
    table = dc.defect_table()
    for _, r in table.iterrows():
        col = r["column"]
        present = col in full.columns
        print(f"  {r['key']:<9} column={col:<22} log10_thr={r['log10_threshold']!s:<10} "
              f"has_thr={r['has_threshold']!s:<5} in_matrix={present}")
    print("  实测超标率（统一矩阵口径）:")
    print(dc.defect_exceedance(full).to_string(index=False))

    print("=" * 78)
    print("[3] 风格计数")
    styles = dc.load_styles()
    counts = styles.value_counts()
    print(f"  风格类别数: {counts.size}")
    for name, n in counts.items():
        print(f"    {name:<22} {n}")

    print("=" * 78)
    print("[4] 常量列")
    arr = full.to_numpy(dtype=float)
    gstd = arr.std(axis=0)
    const_cols = [c for c, s in zip(full.columns, gstd) if s < 1e-12]
    print(f"  全局常量列: {len(const_cols)} -> {const_cols}")
    style_const = {}
    for st in ["Tripel", "Kriek", "Lambic", "Blond", "Strong ale", "Hoppy", "Dubbel",
               "Saison", "Pils/Lager"]:
        sub = full.loc[styles.index[styles == st]].to_numpy(dtype=float)
        if sub.shape[0] > 1:
            bad = [c for c, s in zip(full.columns, sub.std(axis=0)) if s < 1e-12]
            if bad:
                style_const[st] = bad
    print(f"  风格内常量列: {style_const}")

    print("=" * 78)
    print("[5] 分层切分")
    split = dc.load_split()
    print(f"  split 表形状: {split.shape}, 列: {list(split.columns)}")
    for c in split.columns:
        if split[c].dtype == object or split[c].nunique() < 12:
            print(f"    {c}: {split[c].value_counts().to_dict()}")
    print(f"  前 3 行:\n{split.head(3).to_string(index=False)}")

    print("=" * 78)
    print("[6] Q1 接口 & 中文字体")
    q1 = Path(__file__).resolve().parent / "q1_predict.py"
    print(f"  scripts/q1_predict.py 存在: {q1.exists()}")
    try:
        import matplotlib
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        cjk = [n for n in ("Microsoft YaHei", "SimHei", "SimSun", "Microsoft JhengHei",
                            "KaiTi", "FangSong", "Noto Sans CJK SC", "Source Han Sans CN",
                            "DengXian") if n in names]
        print(f"  matplotlib {matplotlib.__version__} 可用 CJK 字体: {cjk}")
    except Exception as exc:  # pragma: no cover
        print(f"  matplotlib 探测失败: {exc}")

    print("=" * 78)
    print("[7] 目标风格样本量与阈值列数值域")
    for st in ["Tripel", "Kriek", "Lambic", "Pils/Lager"]:
        mask = styles == st
        sub = full.loc[styles.index[mask]]
        print(f"  --- {st} (n={int(mask.sum())}) ---")
        for _, r in table.iterrows():
            col = r["column"]
            if col in sub.columns:
                v = sub[col]
                print(f"    {r['key']:<9} min={v.min():+.4f} med={v.median():+.4f} "
                      f"max={v.max():+.4f} thr={r['log10_threshold']!s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
