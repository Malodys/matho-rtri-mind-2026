# -*- coding: utf-8 -*-
# AI 辅助：OpenAI Codex (GPT-5), OpenAI, 2026-10-30
# 人工核验点：
#   1) 训练/留出索引逐风格互斥，所有评估指标只以真实留出样本为 target；
#   2) 无约束、有约束和真实训练参照使用同一标准化与同一 MMD 带宽；
#   3) 约束接受率由实际候选池 accepted/(accepted+rejected) 重算；
#   4) Ledoit-Wolf 仅调用 sklearn.covariance.LedoitWolf，不实现未经验证的潜变量因子 Copula；
#   5) 合成重复只评估 Monte Carlo 稳定性，不增加真实样本证据量。
"""Q3 夜间有界验证：留出、重复种子、两样本距离、覆盖率与约束接受率。

所有输出严格写入 ``data/processed/nightly/overnight-q123-20261030/q3``，
不改写正式 ``data/processed/q3_runs`` 产物。
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import q3_copula as q3  # noqa: E402

RUN_ID = "overnight-q123-20261030"
OUT_DIR = REPO_ROOT / "data" / "processed" / "nightly" / RUN_ID / "q3"
DEFAULT_SEEDS = (7, 42, 20261030)
TARGET_STYLES = tuple(q3.DEFAULT_TARGET_STYLES)


def _seed_parts(seed: int, *labels: str) -> List[int]:
    return [seed] + [zlib.crc32(x.encode("utf-8")) for x in labels]


def fit_ledoit_wolf_copula(matrix: pd.DataFrame) -> Tuple[q3.GlobalCopula, float]:
    """用 Ledoit-Wolf 收缩正态分数协方差，并转换为相关矩阵。"""
    values = matrix.to_numpy(dtype=float)
    n, _ = values.shape
    ranks = rankdata(values, axis=0)
    z = norm.ppf((ranks - 0.5) / n)
    sample_corr = np.corrcoef(z, rowvar=False)
    sample_corr = (sample_corr + sample_corr.T) / 2.0
    eig_raw = np.linalg.eigvalsh(sample_corr)

    estimator = LedoitWolf(assume_centered=False).fit(z)
    cov = np.asarray(estimator.covariance_, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eig = np.linalg.eigvalsh(corr)
    if eig.min() <= 0:
        corr += (abs(float(eig.min())) + 1e-9) * np.eye(corr.shape[0])
        d2 = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d2, d2)
        np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    fitted = q3.GlobalCopula(
        columns=list(matrix.columns),
        chol=chol,
        corr=corr,
        corr_raw=sample_corr,
        eig_raw=eig_raw,
        eig_clipped=np.linalg.eigvalsh(corr),
        n_clipped=0,
        ridge=0.0,
        global_sorted=np.sort(values, axis=0).T,
    )
    return fitted, float(estimator.shrinkage_)


def pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def robust_scale(train: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    center = np.median(train, axis=0)
    q75, q25 = np.quantile(train, [0.75, 0.25], axis=0)
    scale = q75 - q25
    sd = np.std(train, axis=0, ddof=1)
    scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
    return tuple((x - center) / scale for x in arrays)


def bandwidth_sq(reference: np.ndarray) -> float:
    d2 = pairwise_sq(reference, reference)
    vals = d2[np.triu_indices(reference.shape[0], k=1)]
    vals = vals[vals > 1e-12]
    return float(np.median(vals)) if vals.size else 1.0


def mmd2_biased(a: np.ndarray, b: np.ndarray, bw2: float) -> float:
    denom = max(2.0 * bw2, 1e-12)
    kaa = np.exp(-pairwise_sq(a, a) / denom)
    kbb = np.exp(-pairwise_sq(b, b) / denom)
    kab = np.exp(-pairwise_sq(a, b) / denom)
    return float(max(kaa.mean() + kbb.mean() - 2.0 * kab.mean(), 0.0))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    dab = np.sqrt(pairwise_sq(a, b))
    daa = np.sqrt(pairwise_sq(a, a))
    dbb = np.sqrt(pairwise_sq(b, b))
    return float(max(2.0 * dab.mean() - daa.mean() - dbb.mean(), 0.0))


def interval_coverage(source: np.ndarray, target: np.ndarray, alpha: float = 0.10) -> float:
    lo = np.quantile(source, alpha / 2.0, axis=0)
    hi = np.quantile(source, 1.0 - alpha / 2.0, axis=0)
    return float(((target >= lo) & (target <= hi)).mean())


def within_target_range(source: np.ndarray, target: np.ndarray) -> float:
    lo = target.min(axis=0)
    hi = target.max(axis=0)
    return float(((source >= lo) & (source <= hi)).mean())


def compliant_fraction(values: np.ndarray, columns: Sequence[str], specs: Sequence[dict]) -> float:
    return float((~q3.threshold_violations(values, columns, specs, 0.0)).mean())


def iqr_ratio(constrained: np.ndarray, unconstrained: np.ndarray,
              indices: Iterable[int] | None = None) -> float:
    idx = np.arange(unconstrained.shape[1]) if indices is None else np.asarray(list(indices))
    uq = np.quantile(unconstrained[:, idx], [0.25, 0.75], axis=0)
    cq = np.quantile(constrained[:, idx], [0.25, 0.75], axis=0)
    den = uq[1] - uq[0]
    num = cq[1] - cq[0]
    good = den > 1e-9
    return float(np.median(num[good] / den[good])) if good.any() else float("nan")


def evaluate_three_way(train_real: np.ndarray, test_real: np.ndarray,
                       unconstrained: np.ndarray, constrained: np.ndarray,
                       columns: Sequence[str], specs: Sequence[dict]) -> Dict[str, float]:
    tr, te, un, co = robust_scale(train_real, train_real, test_real,
                                  unconstrained, constrained)
    bw2 = bandwidth_sq(tr)
    out: Dict[str, float] = {"mmd_bandwidth_sq": bw2}
    for label, source in (("real_reference", tr), ("unconstrained", un),
                          ("constrained", co)):
        out[f"mmd2_{label}"] = mmd2_biased(source, te, bw2)
        out[f"energy_{label}"] = energy_distance(source, te)
    for label, source in (("real_reference", train_real),
                          ("unconstrained", unconstrained),
                          ("constrained", constrained)):
        out[f"interval90_coverage_{label}"] = interval_coverage(source, test_real)
        out[f"within_test_range_{label}"] = within_target_range(source, test_real)
        out[f"constraint_compliance_{label}"] = compliant_fraction(source, columns, specs)
    return out


def stratified_holdout(styles: pd.Series, seed: int,
                       target_styles: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    held: Dict[str, np.ndarray] = {}
    held_all: List[int] = []
    style_values = styles.astype(str).to_numpy()
    for style in target_styles:
        idx = np.flatnonzero(style_values == style)
        rng = np.random.default_rng(_seed_parts(seed, style, "split"))
        perm = rng.permutation(idx)
        n_test = min(max(3, int(round(0.30 * idx.size))), idx.size - 3)
        held[style] = np.sort(perm[:n_test])
        held_all.extend(held[style].tolist())
    train_mask = np.ones(len(styles), dtype=bool)
    train_mask[np.asarray(held_all, dtype=int)] = False
    return train_mask, held


def generate_pair(copula: q3.GlobalCopula, marginal: q3.ConditionalMarginal,
                  specs: Sequence[dict], seed: int, style: str, model: str,
                  n_gen: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(_seed_parts(seed, style, model, "generate"))
    constrained, info, unconstrained = q3.enforce_defect_constraints(
        copula, marginal, n_gen, specs, rng, mode="reject", pool_factor=30,
        max_rounds=6, return_unconstrained=True)
    if unconstrained is None:
        raise RuntimeError("未返回配对无约束样本")
    total = int(info["pool_size"]) + int(info["n_candidates_rejected"])
    info["candidate_acceptance_rate"] = (
        float(info["pool_size"]) / total if total else float("nan"))
    return unconstrained, constrained, info


def failure_rows(values: np.ndarray, columns: Sequence[str], specs: Sequence[dict],
                 seed: int, style: str, model: str, limit: int = 3) -> List[dict]:
    bad = q3.threshold_violations(values, columns, specs, 0.0)
    rows: List[dict] = []
    for i in np.flatnonzero(bad)[:limit]:
        row = {"protocol": "stratified_style_holdout", "seed": seed,
               "style": style, "model": model, "candidate_row": int(i)}
        for spec in specs:
            if not spec["has_threshold"]:
                continue
            j = columns.index(spec["column"])
            value = float(values[i, j])
            row[f"{spec['key']}_value_log10_mgL"] = value
            row[f"{spec['key']}_threshold_log10_mgL"] = float(spec["log10_threshold"])
            row[f"{spec['key']}_violates"] = bool(value > spec["log10_threshold"])
        rows.append(row)
    return rows


def run_stratified(data: q3.Dataset, seeds: Sequence[int], n_gen: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = data.matrix.to_numpy(dtype=float)
    specs = q3.defect_specs(data.defects)
    rows: List[dict] = []
    failures: List[dict] = []
    for seed in seeds:
        train_mask, held = stratified_holdout(data.styles, seed, TARGET_STYLES)
        train_df = data.matrix.iloc[np.flatnonzero(train_mask)]
        fits: Dict[str, Tuple[q3.GlobalCopula, float | None]] = {
            "gaussian_eig_ridge": (q3.fit_global_copula(train_df), None),
            "ledoit_wolf": fit_ledoit_wolf_copula(train_df),
        }
        for model, (copula, shrinkage) in fits.items():
            for style, test_idx in held.items():
                style_all = np.flatnonzero((data.styles.astype(str) == style).to_numpy())
                train_style_idx = np.setdiff1d(style_all, test_idx, assume_unique=True)
                train_real = values[train_style_idx]
                test_real = values[test_idx]
                marginal = q3.build_marginal(copula, train_real, style, mode="cdf")
                uncon, con, info = generate_pair(copula, marginal, specs, seed,
                                                  style, model, n_gen)
                metrics = evaluate_three_way(train_real, test_real, uncon, con,
                                             copula.columns, specs)
                defect_idx = [copula.columns.index(s["column"])
                              for s in specs if s["has_threshold"]]
                rows.append({
                    "protocol": "stratified_style_holdout",
                    "seed": seed,
                    "model": model,
                    "style": style,
                    "n_real_total": int(style_all.size),
                    "n_real_train": int(train_real.shape[0]),
                    "n_real_test": int(test_real.shape[0]),
                    "n_synthetic_per_state": n_gen,
                    "lambda_train": float(marginal.lam),
                    "lw_shrinkage": shrinkage,
                    "candidate_acceptance_rate": info["candidate_acceptance_rate"],
                    "candidate_rejected": int(info["n_candidates_rejected"]),
                    "constraint_fallback_compress": bool(info.get("fallback_compress", False)),
                    "iqr_ratio_constrained_to_unconstrained_all": iqr_ratio(con, uncon),
                    "iqr_ratio_constrained_to_unconstrained_defects": iqr_ratio(con, uncon, defect_idx),
                    **metrics,
                })
                failures.extend(failure_rows(uncon, copula.columns, specs, seed,
                                             style, model))
                print(f"[分层留出] seed={seed} model={model:<18} style={style:<11} "
                      f"energy U/C/R={metrics['energy_unconstrained']:.3f}/"
                      f"{metrics['energy_constrained']:.3f}/"
                      f"{metrics['energy_real_reference']:.3f} "
                      f"accept={info['candidate_acceptance_rate']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(failures)


def run_loso(data: q3.Dataset, seed: int, n_gen: int) -> pd.DataFrame:
    """整风格留一：无目标风格样本时，明确退化为训练集全局边际（λ=0）。"""
    values = data.matrix.to_numpy(dtype=float)
    specs = q3.defect_specs(data.defects)
    style_values = data.styles.astype(str).to_numpy()
    rows: List[dict] = []
    for style in TARGET_STYLES:
        test_idx = np.flatnonzero(style_values == style)
        train_idx = np.flatnonzero(style_values != style)
        train_df = data.matrix.iloc[train_idx]
        held_real = values[test_idx]
        rng = np.random.default_rng(_seed_parts(seed, style, "loso_reference"))
        perm = rng.permutation(held_real.shape[0])
        cut = max(3, held_real.shape[0] // 2)
        ref_a, ref_b = held_real[perm[:cut]], held_real[perm[cut:]]
        if ref_b.shape[0] < 3:
            ref_a, ref_b = held_real[:-3], held_real[-3:]
        fits: Dict[str, Tuple[q3.GlobalCopula, float | None]] = {
            "gaussian_eig_ridge": (q3.fit_global_copula(train_df), None),
            "ledoit_wolf": fit_ledoit_wolf_copula(train_df),
        }
        for model, (copula, shrinkage) in fits.items():
            marginal = q3.build_marginal(copula, train_df.to_numpy(dtype=float),
                                         style, mode="cdf", lam=0.0)
            uncon, con, info = generate_pair(copula, marginal, specs, seed,
                                              style, model + "_loso", n_gen)
            metrics = evaluate_three_way(ref_a, ref_b, uncon, con,
                                         copula.columns, specs)
            rows.append({
                "protocol": "leave_one_style_out_global_fallback",
                "seed": seed,
                "model": model,
                "style": style,
                "n_real_style": int(held_real.shape[0]),
                "n_reference_a": int(ref_a.shape[0]),
                "n_reference_b": int(ref_b.shape[0]),
                "n_synthetic_per_state": n_gen,
                "lambda_target": 0.0,
                "lw_shrinkage": shrinkage,
                "candidate_acceptance_rate": info["candidate_acceptance_rate"],
                "constraint_fallback_compress": bool(info.get("fallback_compress", False)),
                **metrics,
            })
            print(f"[整风格留一] model={model:<18} style={style:<11} "
                  f"energy U/C/R={metrics['energy_unconstrained']:.3f}/"
                  f"{metrics['energy_constrained']:.3f}/"
                  f"{metrics['energy_real_reference']:.3f}")
    return pd.DataFrame(rows)


def model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mmd2_real_reference", "mmd2_unconstrained", "mmd2_constrained",
        "energy_real_reference", "energy_unconstrained", "energy_constrained",
        "interval90_coverage_real_reference", "interval90_coverage_unconstrained",
        "interval90_coverage_constrained", "within_test_range_real_reference",
        "within_test_range_unconstrained", "within_test_range_constrained",
        "constraint_compliance_real_reference", "constraint_compliance_unconstrained",
        "constraint_compliance_constrained", "candidate_acceptance_rate",
        "iqr_ratio_constrained_to_unconstrained_all",
        "iqr_ratio_constrained_to_unconstrained_defects",
    ]
    grouped = metrics.groupby("model", sort=True)[cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q3 夜间有界验证")
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--n-gen", type=int, default=128,
                        help="每个风格/种子/模型的每种合成状态样本数；不增加真实证据量")
    parser.add_argument("--skip-loso", action="store_true")
    args = parser.parse_args(argv)
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = q3.load_dataset()
    metrics, failures = run_stratified(data, seeds, args.n_gen)
    loso = pd.DataFrame() if args.skip_loso else run_loso(data, seeds[-1], args.n_gen)
    comparison = model_summary(metrics)

    metrics.to_csv(OUT_DIR / "holdout_seed_metrics.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(OUT_DIR / "constraint_failure_samples.csv", index=False,
                    encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if not loso.empty:
        loso.to_csv(OUT_DIR / "loso_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": RUN_ID,
        "protocol": "3 repeated stratified-style holdouts plus one leave-one-style-out diagnostic",
        "seeds": list(seeds),
        "styles": list(TARGET_STYLES),
        "n_synthetic_per_state": args.n_gen,
        "real_rows_total": int(data.matrix.shape[0]),
        "features": int(data.matrix.shape[1]),
        "holdout_rows": int(metrics.shape[0]),
        "loso_rows": int(loso.shape[0]),
        "failure_sample_rows": int(failures.shape[0]),
        "fallback_compress_count": int(metrics["constraint_fallback_compress"].sum()),
        "synthetic_evidence_policy": (
            "Repeated synthetic draws estimate Monte Carlo stability only; "
            "the evidence count remains the held-out real beers."),
        "outputs": [
            "holdout_seed_metrics.csv", "loso_metrics.csv", "model_comparison.csv",
            "constraint_failure_samples.csv",
        ],
    }
    (OUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    method_review = {
        "gaussian_copula": "existing q3_copula.py eigen/ridge implementation reused unchanged",
        "shrinkage_comparator": (
            "scikit-learn LedoitWolf covariance shrinkage on normal scores; "
            "this is a covariance-shrinkage Gaussian Copula comparator"),
        "latent_factor_copula": (
            "not implemented: no reliable project implementation or validated factor-count "
            "selection protocol exists; creating one overnight would be unvalidated"),
        "leave_one_style_out": (
            "without any target-style observations the current conditional marginal is not "
            "identifiable; the diagnostic therefore uses an explicitly labelled global "
            "fallback with lambda=0, not a claimed target-style generator"),
    }
    (OUT_DIR / "method_review.json").write_text(
        json.dumps(method_review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[落盘] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
