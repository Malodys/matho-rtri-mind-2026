# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 折划分、模型构造与指标口径直接复用队友 worktrees/prediction-models/scripts/tabpfn_cv.py 的
#      load_split() / scores()（import 而非重写），仅把「目标维度」循环改为多进程并行；
#   2) 与参考实现逐项核对：KFold(n_splits=5, shuffle=True, random_state=0)、特征=1.csv 的 231 列
#      （含 4 个 *_sum 派生列与 55 个左删失 NaN，TabPFN 原生支持 NaN）、基线=训练折目标均值；
#   3) 落盘的 cv_summary.csv 指标列与远端 tabpfn_cv.py 完全一致（mae/rmse/r2 + baseline_* + pooled_oof），
#      并与远端已归档的 fold 1 逐目标 R² 做过对照；
#   4) --resume 以「(fold, target) 粒度的独立缓存文件」判定跳过，缓存文件损坏或内容非有限数时自动重算。
#   5) 分块（chunk）策略：每个 worker 分块只加载一次 876 MB 权重并逐目标复用同一估计器；
#      本机 13.9 GB 内存下「每拟合一次加载一遍权重 + 6 workers」会触发 WinError 1450（系统资源不足），
#      已改为按 chunk 复用权重并把 worker 数降到 4，单拟合结果与原实现逐位一致。
"""问题一 · TabPFN 3.5 五折 × 50 目标并行复现（带断点续跑）。

与远端 worktrees/prediction-models/scripts/tabpfn_cv.py 语义等价：
  · 数据：import 远端 ``load_split``（读 data/splits/seed_0 的 1_/4_ 切分文件）；
  · 折划分：KFold(n_splits=5, shuffle=True, random_state=seed)；
  · 模型：TabPFNRegressor(model_path=..., device, random_state=seed, n_estimators=...)；
  · 指标：import 远端 ``scores``，基线为训练折目标均值。
唯一差别：把 (折, 目标) 网格改为多进程并行，并支持 ``--resume`` 按 (fold, target) 粒度跳过已完成项。

用法：
    python scripts/q1_tabpfn_cv.py \
        --worktree worktrees/prediction-models \
        --model-path worktrees/pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors \
        --output-dir data/processed/q1_repro_tabpfn --workers 6 --threads 2 --resume
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_remote(worktree: Path):
    """把远端分支的 scripts/ 加入 sys.path，复用其 load_split / scores 实现。"""
    scripts = worktree / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(f"未找到远端脚本目录: {scripts}")
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from tabpfn_cv import load_split, scores  # noqa: E402

    return load_split, scores


def _fit_one(payload):
    """单个 (折, 目标) 的拟合与预测；payload 只含 numpy，避免跨进程 DataFrame 序列化。"""
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
    if len(predicted) != len(Xva) or not np.isfinite(predicted).all():
        raise ValueError(f"非法预测: fold={fold}, target={target}")
    return fold, target, predicted.tolist()


def _fit_chunk(payload):
    """一个 (折, 目标子集) 分块：只加载一次 TabPFN 权重，逐目标复用同一个估计器。

    人工核验点：分块复用权重后，876 MB 权重从「每拟合一次加载一遍」降为「每分块加载一遍」，
    既避免本机 13.9 GB 内存下的 WinError 1450（系统资源不足），也把磁盘读取量降了一个量级；
    每个目标拟合完立即写缓存文件，因此中断后 --resume 可按 (fold, target) 粒度续跑。
    """
    fold, targets, Xtr, Xva, ytr_map, model_path, seed, n_estimators, device, threads, cache_dir = payload
    cache_dir = Path(cache_dir)
    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    import torch

    torch.set_num_threads(threads)
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        model_path=str(model_path), device=device, random_state=seed,
        n_estimators=n_estimators, show_progress_bar=False,
    )
    results = []
    for target in targets:
        model.fit(Xtr, ytr_map[target])
        predicted = np.asarray(model.predict(Xva)).reshape(-1)
        if len(predicted) != len(Xva) or not np.isfinite(predicted).all():
            raise ValueError(f"非法预测: fold={fold}, target={target}")
        _cache_path(cache_dir, fold, target).write_text(
            json.dumps({"fold": fold, "target": target, "predicted": predicted.tolist()}),
            encoding="utf-8",
        )
        results.append((fold, target, predicted.tolist()))
    return results


def _cache_path(cache_dir: Path, fold: int, target: str) -> Path:
    safe = target.replace("/", "_").replace("\\", "_")
    return cache_dir / f"fold{fold}__{safe}.json"


def _load_cached(cache_dir: Path, fold: int, target: str):
    """读取缓存预测；缺失/损坏/含非有限数时返回 None（触发重算）。"""
    path = _cache_path(cache_dir, fold, target)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = np.asarray(payload["predicted"], dtype=float)
    except Exception:
        return None
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 TabPFN 3.5 五折并行复现（支持 --resume）")
    ap.add_argument("--worktree", default="worktrees/prediction-models", help="远端管线 worktree 路径")
    ap.add_argument("--model-path",
                    default="worktrees/pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors")
    ap.add_argument("--data-dir", default="data/splits/seed_0", help="切分数据目录（相对 worktree）")
    ap.add_argument("--output-dir", default="data/processed/q1_repro_tabpfn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="每个 worker 的 torch 线程数")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="每个分块的目标数（0=按 workers 自动，均分 targets/workers）")
    ap.add_argument("--limit-targets", type=int, default=0, help=">0 时只跑前 N 个目标（冒烟用）")
    ap.add_argument("--resume", action="store_true", help="跳过缓存中已完成的 (fold, target)")
    ap.add_argument("--refresh-final", action="store_true", help="强制重算汇总产物")
    ap.add_argument("--protocol", choices=("legacy", "fair_main"), default="legacy",
                    help="legacy=队友原始协议；fair_main=227 特征、固定 175/75 留出协议")
    args = ap.parse_args()

    worktree = (REPO_ROOT / args.worktree).resolve() if not Path(args.worktree).is_absolute() else Path(args.worktree)
    model_path = (REPO_ROOT / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    if not model_path.is_file():
        print(f"[FAIL] 权重不存在: {model_path}", file=sys.stderr)
        return 1

    load_split, scores = _import_remote(worktree)
    from sklearn.model_selection import KFold

    if args.protocol == "legacy":
        data_dir = worktree / args.data_dir
        features, targets, metadata = load_split(data_dir)
        protocol_label = "legacy teammate protocol: raw split features, KFold 5, seed 0"
    else:
        # Fair comparison uses exactly the same audited 227-feature matrix and
        # fixed 175/75 style-stratified holdout as q1_pipeline.py.  It is a
        # separate product because it is not comparable to the legacy KFold
        # reproduction already stored in q1_repro_tabpfn.
        from data_contract import load_modeling_matrix, load_split as load_main_split, load_styles, load_targets

        features = load_modeling_matrix()
        targets = load_targets().reindex(features.index)
        styles = load_styles().reindex(features.index)
        metadata = pd.DataFrame({
            "beer": features.index,
            "tasting_category_fine": styles.to_numpy(),
        })
        manifest = load_main_split()
        train_ids = manifest.loc[manifest["split"].eq("train"), "beer"].to_numpy(dtype=int)
        valid_ids = manifest.loc[manifest["split"].eq("test"), "beer"].to_numpy(dtype=int)
        train_pos = features.index.get_indexer(train_ids)
        valid_pos = features.index.get_indexer(valid_ids)
        if (train_pos < 0).any() or (valid_pos < 0).any():
            raise ValueError("固定主模型切分与统一矩阵的 beer 标识无法对齐")
        splits = [(train_pos, valid_pos)]
        protocol_label = "fair main protocol: audited 227 features, fixed 175/75 holdout, seed 0"
    if args.limit_targets:
        targets = targets.iloc[:, : args.limit_targets]

    if args.protocol == "legacy":
        splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(features))

    out = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 划分作业清单，并按 --resume 过滤已完成项
    chunk_size = args.chunk_size or max(1, int(np.ceil(len(targets.columns) / max(args.workers, 1))))
    jobs, cached = [], {}
    for fold, (train, valid) in enumerate(splits, 1):
        Xtr = features.iloc[train].to_numpy(dtype=float)
        Xva = features.iloc[valid].to_numpy(dtype=float)
        pending = []
        for target in targets.columns:
            hit = _load_cached(cache_dir, fold, target) if args.resume else None
            if hit is not None:
                cached[(fold, target)] = hit
                continue
            pending.append(target)
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            ytr_map = {t: targets[t].iloc[train].to_numpy(dtype=float) for t in chunk}
            jobs.append((fold, chunk, Xtr, Xva, ytr_map, model_path, args.seed,
                         args.n_estimators, args.device, args.threads, str(cache_dir)))

    print(f"任务数={len(jobs)} 个分块（缓存命中 {len(cached)} 个 fold/target），"
          f"workers={args.workers}，chunk_size={chunk_size}，"
          f"targets={targets.shape[1]}，features={features.shape[1]}", flush=True)

    started = time.perf_counter()
    if jobs:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
            delayed(_fit_chunk)(job) for job in jobs
        )
        for chunk_results in results:
            for fold, target, predicted in chunk_results:
                cached[(fold, target)] = np.asarray(predicted, dtype=float)
    elapsed = time.perf_counter() - started

    n_folds = len(splits)
    missing = [(f, t) for f in range(1, n_folds + 1)
               for t in targets.columns if (f, t) not in cached]
    if missing:
        print(f"[FAIL] 仍有 {len(missing)} 个 (fold,target) 未完成，例：{missing[:3]}", file=sys.stderr)
        return 2

    # ---- 汇总：与远端 tabpfn_cv.py 的指标口径一致
    fold_metrics, oof_rows = [], []
    for fold, (train, valid) in enumerate(splits, 1):
        for target in targets.columns:
            predicted = np.asarray(cached[(fold, target)], dtype=float)
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

    per_target = []
    for target, group in oof.groupby("target", sort=False):
        per_target.append({
            "target": target, "n_samples": len(group),
            **scores(group["actual"], group["prediction"], group["baseline_prediction"]),
        })
    target_metrics = pd.DataFrame(per_target)

    metric_columns = ["mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2"]
    by_fold = fm.groupby("fold")[metric_columns].mean()
    summary = by_fold.copy()
    summary.index = [f"fold_{i}" for i in summary.index]
    summary.loc["cv_mean"] = by_fold.mean()
    summary.loc["cv_std"] = by_fold.std(ddof=1)
    summary.loc["pooled_oof"] = target_metrics[metric_columns].mean()
    summary.index.name = "evaluation"

    fm.to_csv(out / "fold_target_metrics.csv", index=False, encoding="utf-8-sig")
    target_metrics.to_csv(out / "target_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "cv_summary.csv", encoding="utf-8-sig")

    pd.Series({
        "python": sys.version.split()[0], "device": args.device, "seed": args.seed,
        "folds": n_folds, "n_estimators": args.n_estimators, "workers": args.workers,
        "threads_per_worker": args.threads, "samples": len(features),
        "features": features.shape[1], "targets": targets.shape[1],
        "elapsed_seconds": round(elapsed, 1),
        "model_path": str(model_path),
        "protocol": protocol_label,
        "feature_policy": "audited unified 227 features" if args.protocol == "fair_main"
                          else "raw teammate split features",
        "split_policy": "fixed 175/75 style-stratified holdout" if args.protocol == "fair_main"
                        else "KFold 5, shuffle=True, random_state=seed",
    }, name="value").rename_axis("setting").to_csv(out / "run.csv", encoding="utf-8-sig")

    buf = io.StringIO()
    buf.write(summary.round(5).to_string() + "\n")
    buf.write(f"Elapsed: {elapsed:.1f}s\n")
    (out / "cv_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print(f"五折平均 R2 = {by_fold['r2'].mean():.4f}（远端 RESULTS.md 参考值 0.2055，验收容差 ±0.03）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
