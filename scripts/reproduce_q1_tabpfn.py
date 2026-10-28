# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：折划分、TabPFN 超参与指标口径均直接复用队友分支 tabpfn_cv.py 的实现（load_split / scores /
#             KFold 参数），仅将"目标维度"这一层循环并行化；单次拟合的随机种子与模型构造参数未作任何改动。
"""问题一 · TabPFN 3.5 五折复现（并行版）。

与远端 experiments/prediction-models 分支的 scripts/tabpfn_cv.py 语义等价：
  · 数据与对齐：直接 import 远端模块的 load_split（按 beer 关联 1_/4_ 切分文件）；
  · 折划分：KFold(n_splits=5, shuffle=True, random_state=seed)；
  · 模型：TabPFNRegressor(model_path=..., device, random_state=seed, n_estimators=4)；
  · 指标：直接 import 远端模块的 scores()，基线为训练折均值。
唯一差别：把"目标维度"循环改为多进程并行（单机 CPU 上 50 目标 × 5 折约需 70 分钟，并行后约 10 分钟）。

用法：
    python scripts/reproduce_q1_tabpfn.py \
        --worktree <...\\worktrees\\prediction-models> \
        --model-path <...\\tabpfn-v3.5-20260909.safetensors> \
        --output-dir data/processed/q1_repro_tabpfn --workers 8
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _import_remote(worktree: Path):
    """把远端分支的 scripts/ 加入 sys.path，复用其 load_split / scores 实现。"""
    scripts = worktree / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(f"未找到远端脚本目录: {scripts}")
    sys.path.insert(0, str(scripts))
    from tabpfn_cv import load_split, scores  # noqa: E402
    return load_split, scores


def _fit_one(payload):
    """单个 (折, 目标) 的拟合与预测。payload 含特征矩阵与目标向量（numpy，避免跨进程 DataFrame 序列化）。"""
    fold, target, Xtr, ytr, Xva, model_path, seed, n_estimators, device, threads = payload
    import torch

    torch.set_num_threads(threads)
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        model_path=str(model_path), device=device, random_state=seed,
        n_estimators=n_estimators, show_progress_bar=False,
    )
    model.fit(Xtr, ytr)
    predicted = np.asarray(model.predict(Xva)).reshape(-1)
    if not np.isfinite(predicted).all():
        raise ValueError(f"非法预测: fold={fold}, target={target}")
    return fold, target, predicted


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 TabPFN 3.5 五折并行复现")
    ap.add_argument("--worktree", required=True, help="远端管线 worktree 路径")
    ap.add_argument("--model-path", required=True, help="TabPFN 3.5 权重文件路径")
    ap.add_argument("--data-dir", default="data/splits/seed_0", help="切分数据目录（相对 worktree）")
    ap.add_argument("--output-dir", default="data/processed/q1_repro_tabpfn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="每个 worker 的 torch 线程数")
    ap.add_argument("--limit-targets", type=int, default=0, help=">0 时只跑前 N 个目标（冒烟用）")
    args = ap.parse_args()

    worktree = Path(args.worktree).resolve()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        print(f"[FAIL] 权重不存在: {model_path}", file=sys.stderr)
        return 1

    load_split, scores = _import_remote(worktree)
    from sklearn.model_selection import KFold

    data_dir = worktree / args.data_dir
    features, targets, metadata = load_split(data_dir)
    if args.limit_targets:
        targets = targets.iloc[:, :args.limit_targets]

    splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(features))

    jobs = []
    for fold, (train, valid) in enumerate(splits, 1):
        Xtr = features.iloc[train].to_numpy(dtype=float)
        Xva = features.iloc[valid].to_numpy(dtype=float)
        for target in targets.columns:
            ytr = targets[target].iloc[train].to_numpy(dtype=float)
            jobs.append((fold, target, Xtr, ytr, Xva, model_path, args.seed,
                         args.n_estimators, args.device, args.threads))

    print(f"任务数={len(jobs)} (5 折 × {targets.shape[1]} 目标)，workers={args.workers}", flush=True)

    from joblib import Parallel, delayed

    started = time.perf_counter()
    results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
        delayed(_fit_one)(job) for job in jobs
    )
    elapsed = time.perf_counter() - started

    preds = {(f, t): p for f, t, p in results}
    fold_metrics, oof_rows = [], []
    for fold, (train, valid) in enumerate(splits, 1):
        for target in targets.columns:
            predicted = preds[(fold, target)]
            actual = targets[target].iloc[valid].to_numpy(dtype=float)
            baseline = np.full(len(valid), targets[target].iloc[train].mean())
            fold_metrics.append({
                "fold": fold, "target": target,
                "n_train": len(train), "n_valid": len(valid),
                **scores(actual, predicted, baseline),
            })
            rows = metadata.iloc[valid].copy()
            rows["fold"], rows["target"] = fold, target
            rows["actual"], rows["prediction"] = actual, predicted
            rows["baseline_prediction"] = baseline
            oof_rows.append(rows)

    fm = pd.DataFrame(fold_metrics)
    oof = pd.concat(oof_rows, ignore_index=True)
    metric_columns = ["mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2"]
    by_fold = fm.groupby("fold")[metric_columns].mean()
    summary = by_fold.copy()
    summary.index = [f"fold_{i}" for i in summary.index]
    summary.loc["cv_mean"] = by_fold.mean()
    summary.loc["cv_std"] = by_fold.std(ddof=1)
    summary.index.name = "evaluation"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fm.to_csv(out / "fold_target_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "cv_summary.csv", encoding="utf-8-sig")
    pd.Series({
        "python": sys.version.split()[0], "device": args.device, "seed": args.seed,
        "folds": 5, "n_estimators": args.n_estimators, "workers": args.workers,
        "threads_per_worker": args.threads, "samples": len(features),
        "features": features.shape[1], "targets": targets.shape[1],
        "elapsed_seconds": round(elapsed, 1),
        "model_path": str(model_path),
    }, name="value").rename_axis("setting").to_csv(out / "run.csv", encoding="utf-8-sig")

    buf = io.StringIO()
    buf.write(summary.round(5).to_string() + "\n")
    buf.write(f"Elapsed: {elapsed:.1f}s\n")
    (out / "cv_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print(f"五折平均 R2 = {by_fold['r2'].mean():.4f}（远端 RESULTS.md 参考值 0.2055）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
