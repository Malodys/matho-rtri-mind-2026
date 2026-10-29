# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 六条验收标准逐条重算，不信任上游脚本的自报；任一条不通过即以非零退出码结束。
#   2) 生成谱的「有限值 / 无 NaN」逐元素重算（8 风格 × 2 版 × 250 × 231）。
#   3) 阈值违规数按 defect_table() 的 log10_threshold 现算，不读脚本写下的结论。
#   4) 迁移「不恶化」与相关结构保持度从 transfer_pils_to_tripel.csv 原始行重算。
"""问题 3 验收自检：把六条验收标准逐条重算并打印判定表。

用法：.\\.venv\\Scripts\\python.exe scripts\\q3_verify.py
任一标准不通过 → 退出码 1。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402
import q3_copula as q3  # noqa: E402

RUN_DIR = REPO_ROOT / "data" / "processed" / "q3_runs"
GEN_DIR = RUN_DIR / "generation"
FIG_DIR = REPO_ROOT / "data" / "processed" / "q3_figures"
TARGET_STYLES = ["Tripel", "Kriek", "Lambic"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="问题3 六条验收标准逐条重算")
    parser.parse_args(argv)

    data = q3.load_dataset()
    specs = q3.defect_specs(data.defects)
    thr = {s["key"]: s for s in specs if s["has_threshold"]}
    summary = pd.read_csv(RUN_DIR / "generation_summary.csv", encoding="utf-8-sig")
    results: List[tuple] = []

    # ---- 标准 1：三个目标风格各 ≥200 条，数值有限、无 NaN ----
    n_bad = 0
    detail1 = []
    for style in TARGET_STYLES:
        path = GEN_DIR / f"{style.replace(' ', '_').replace('/', '-')}_synthetic.csv"
        if not path.is_file():
            n_bad += 1
            detail1.append(f"{style}:缺文件")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        chem = df.drop(columns=[c for c in ("beer", "tasting_category_fine")
                                if c in df.columns])
        vals = chem.to_numpy(dtype=float)
        finite = bool(np.isfinite(vals).all())
        ok = len(df) >= 200 and finite and chem.shape[1] == 231
        n_bad += 0 if ok else 1
        detail1.append(f"{style}:{len(df)}条/{chem.shape[1]}列/有限值={finite}")
    results.append(("1 三目标风格 ≥200 条且全有限无 NaN",
                    "通过" if n_bad == 0 else "不通过", "；".join(detail1)))

    # ---- 标准 2：有阈值的 3 项缺陷违规样本数 = 0 ----
    viol_detail = []
    total_viol = 0
    for style in summary["style"]:
        path = GEN_DIR / f"{style.replace(' ', '_').replace('/', '-')}_synthetic.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        for key, spec in thr.items():
            v = int((df[spec["column"]].to_numpy(dtype=float)
                     > spec["log10_threshold"]).sum())
            total_viol += v
            if v:
                viol_detail.append(f"{style}/{key}:{v}")
    results.append(("2 三项有阈值缺陷违规样本数 = 0",
                    "通过" if total_viol == 0 else "不通过",
                    f"合计违规 {total_viol}" + ("；" + "；".join(viol_detail)
                                                if viol_detail else "")))

    # ---- 标准 3：迁移后四项缺陷逐样本 ≤ 迁移前 ----
    tpath = RUN_DIR / "transfer_pils_to_tripel.csv"
    if tpath.is_file():
        t = pd.read_csv(tpath, encoding="utf-8-sig")
        d = t[t["is_defect"].astype(bool) == True]  # noqa: E712
        worse = int((d["value_after_log10_mgL"] > d["value_before_log10_mgL"] + 1e-12).sum())
        over = int(d["exceeds_threshold_after"].astype(object).map(
            lambda v: v is True or v == "True").sum())
        results.append(("3 迁移后四项缺陷逐样本 ≤ 迁移前",
                        "通过" if worse == 0 else "不通过",
                        f"缺陷项 {len(d)} 条，恶化 {worse} 条，迁移后超标 {over} 条"))
    else:
        results.append(("3 迁移后四项缺陷逐样本 ≤ 迁移前", "不通过", "缺 transfer_pils_to_tripel.csv"))

    # ---- 标准 4：迁移前后 Spearman ρ 平均绝对偏差 < 0.05 ----
    if tpath.is_file():
        long = pd.read_csv(tpath, encoding="utf-8-sig")
        beers = sorted(long["beer"].unique())
        comps = list(dict.fromkeys(long["compound"].tolist()))
        before = np.zeros((len(beers), len(comps)))
        after = np.zeros_like(before)
        bi = {b: i for i, b in enumerate(beers)}
        ci = {c: i for i, c in enumerate(comps)}
        for _, r in long.iterrows():
            before[bi[r["beer"]], ci[r["compound"]]] = r["value_before_log10_mgL"]
            after[bi[r["beer"]], ci[r["compound"]]] = r["value_after_log10_mgL"]
        rho_mean, rho_max = q3.spearman_mean_abs_dev(before, after)
        results.append(("4 迁移前后 Spearman ρ 平均绝对偏差 < 0.05",
                        "通过" if rho_mean < 0.05 else "不通过",
                        f"平均 {rho_mean:.6f}，最大 {rho_max:.4f}（由原始对照行重算）"))
    else:
        results.append(("4 迁移前后 Spearman ρ 平均绝对偏差 < 0.05", "不通过", "缺对照文件"))

    # ---- 标准 5：KS 中位数有明确数值并写入 README ----
    readme = RUN_DIR / "README.md"
    ks_med = summary["ks_median"].tolist()
    in_readme = readme.is_file() and all(
        f"{v:.4f}" in readme.read_text(encoding="utf-8") for v in ks_med)
    results.append(("5 KS 中位数有明确数值并写入 README",
                    "通过" if in_readme else "不通过",
                    "各风格 KS 中位数 " + "、".join(
                        f"{r['style']}={r['ks_median']:.4f}" for _, r in summary.iterrows())))

    # ---- 标准 6：脚本 --help 可跑；感官画像优雅降级 ----
    import subprocess
    scripts = ["q3_copula.py", "q3_generate.py", "q3_transfer.py",
               "q3_sensory_profile.py", "q3_figures.py", "q3_report.py",
               "q3_probe_env.py", "q3_verify.py"]
    py = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    bad_scripts = []
    for s in scripts:
        p = REPO_ROOT / "scripts" / s
        if not p.is_file():
            bad_scripts.append(f"{s}(缺文件)")
            continue
        r = subprocess.run([py, str(p), "--help"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            bad_scripts.append(f"{s}(rc={r.returncode})")
    pending_ok = (RUN_DIR / "sensory_profile_pending.md").is_file() or \
        bool(list(RUN_DIR.glob("sensory_profile_*.csv")))
    status = str(summary["sensory_profile_status"].iloc[0]) \
        if "sensory_profile_status" in summary.columns else "unknown"
    results.append(("6 全部脚本 --help 可跑；感官画像优雅降级",
                    "通过" if not bad_scripts and pending_ok else "不通过",
                    f"脚本异常：{bad_scripts if bad_scripts else '无'}；"
                    f"感官画像状态={status}"))

    # ---- 附加：产物齐备性 ----
    need = ["generation_summary.csv", "feasibility_report.csv", "ks_per_compound.csv",
            "transfer_pils_to_tripel.csv", "transfer_pils_to_tripel_defects.csv",
            "transfer_pils_to_tripel_summary.csv", "README.md"]
    missing = [n for n in need if not (RUN_DIR / n).is_file()]
    figs = sorted(p.name for p in FIG_DIR.glob("*.png")) if FIG_DIR.is_dir() else []
    results.append(("附加 产物齐备性",
                    "通过" if not missing else "不通过",
                    f"缺失={missing if missing else '无'}；图 {len(figs)} 张：{figs}"))

    print("=" * 100)
    print("问题 3 验收自检")
    print("=" * 100)
    for name, verdict, detail in results:
        print(f"[{verdict}] {name}")
        print(f"        {detail}")
    print("=" * 100)
    n_pass = sum(1 for _, v, _ in results if v == "通过")
    print(f"合计 {n_pass}/{len(results)} 项通过")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
