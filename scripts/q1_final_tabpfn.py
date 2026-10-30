# -*- coding: utf-8 -*-
# AI 辅助：Codex (GPT-6), OpenAI, 2026-10-30
# 人工复核状态：待团队核验；独立整理论文最终TabPFN的五折与留出评估入口。
"""Evaluate the final TabPFN with a uniformly doubled training context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PARAMETERS = {
    "n_estimators": 4,
    "softmax_temperature": 1.0,
    "average_before_softmax": True,
}


def doubled_context(features, targets):
    # AI辅助实现；训练行顺序与历史最终实验一致，待人工复核。
    rows = np.tile(np.arange(len(features)), 2)
    return features.iloc[rows], targets.iloc[rows]


def run(args, predictor=None):
    # AI辅助实现；使用原实验的数据入口、折划分、预测函数和指标函数。
    project = args.experiment_root.resolve()
    sys.path.insert(0, str(project / "scripts"))
    from tabpfn_cv import load_split
    from tuning_protocol import load_protocol, export_predictions, macro_scores, data_signature
    if predictor is None:
        from tune_tabpfn import fit_predict
        predictor = fit_predict
    def relative(path):
        return path if path.is_absolute() else project / path
    checkpoint = relative(args.model_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    x, y, meta, folds = load_protocol(relative(args.data_dir), args.seed)
    xt, yt, mt = load_split(relative(args.data_dir), "test")
    xt, yt = xt.loc[:, x.columns], yt.loc[:, y.columns]
    if set(meta.beer) & set(mt.beer):
        raise ValueError("Training and holdout beers overlap.")
    output = relative(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof = np.full(y.shape, np.nan)
    baseline = np.full(y.shape, np.nan)
    assignments = meta.copy()
    assignments["fold"] = 0
    fold_metrics = []
    def predict(train_x, train_y, evaluation):
        context_x, context_y = doubled_context(train_x, train_y)
        values = np.asarray(predictor(
            PARAMETERS, context_x, context_y, evaluation,
            model_path=checkpoint, device=args.device, seed=args.seed,
        ))
        if values.shape != (len(evaluation), y.shape[1]) or not np.isfinite(values).all():
            raise ValueError("Invalid TabPFN predictions.")
        return values
    for fold, (train, valid) in enumerate(folds, 1):
        oof[valid] = predict(x.iloc[train], y.iloc[train], x.iloc[valid])
        baseline[valid] = y.iloc[train].mean().to_numpy()
        assignments.loc[valid, "fold"] = fold
        metric = macro_scores(y.iloc[valid].to_numpy(), oof[valid])
        fold_metrics.append({"fold": fold, **metric})
        print(f"Fold {fold}/5 R2={metric['r2']:.6f}", flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete out-of-fold predictions.")
    export_predictions(output, y, oof, assignments, baseline, folds)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    test_predictions = predict(x, y, xt)
    export_predictions(output / "test", yt, test_predictions, mt,
                       np.broadcast_to(y.mean().to_numpy(), yt.shape))
    summary = {
        "model": "tabpfn",
        "mean_cv_r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "n_train": len(x), "n_test": len(xt),
        **macro_scores(yt.to_numpy(), test_predictions),
    }
    pd.DataFrame([summary]).to_csv(output / "comparison.csv", index=False)
    manifest = {
        "parameters": PARAMETERS, "seed": args.seed, "folds": len(folds),
        "features": x.shape[1], "targets": y.shape[1],
        "context_multiplicity": 2, "dataset_sha256": data_signature(x, y, meta),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def main():
    # AI辅助命令行入口，待人工复核。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("prediction-models"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/seed_0"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabpfn_final_seed_0"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
