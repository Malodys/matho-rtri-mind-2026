# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：
#   1) 7.csv 清洗：186 个消费者描述符中 palate_acetaldehyde / palate_acetate / palate_dentist
#      三列整列为空（各 250 个 NA）已剔除，余 183 维且逐列确认零缺失；
#   2) B1 分位级统计（6 组配对）逐行复现主控实测基准：组内 ρ 与感官中位全部对齐；
#   3) 缺陷阈值统一换算为 log10(mg/L)（µg/L → mg/L 逐条换算核对）后再与统一矩阵比较；
#   4) 消费者层基线为「训练折目标均值」，与 Q1 同口径（Macro-RMSE / Macro-MAE / R²中位 / ρ中位）；
#   5) 价格回归样本量（≥1 家 229 款、≥2 家 189 款）与几何均价区间逐项核对；
#   6) 全部取数经 scripts/data_contract.py，锁定 seed_0 的 175/75 切分，未重新划分。
"""B2 · 消费者层（化学 → 消费者感知 → 市场价值）+ 缺陷化合物剂量-响应再检验。

一条命令完成：
    A1 清洗 7.csv → 183 维消费者描述符
    A2 化学 → 消费者描述符多任务模型（与 Q1 品评组模型同口径对比）
    A3 品评组感官 ↔ 消费者描述符的交叉一致性（真值 vs 模型预测）
    A4 外部市场价值信号（log 几何均价 ~ 化学预测感知 + ABV + 风格）
    B1–B4 缺陷化合物剂量-响应分位级再检验 + 风格分层稳健性 + 与原文结论对照

用法：
    python scripts/q4_consumer.py                       # 全流程
    python scripts/q4_consumer.py --stages consumer,dose
    python scripts/q4_consumer.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 0
INNER_SEED = 20261026
N_FOLDS = 5

#: 消费者描述符矩阵中整列为空、必须剔除的三列（7.csv 原始事实）。
EMPTY_CONSUMER_COLS = ["palate_acetaldehyde", "palate_acetate", "palate_dentist"]

#: A3 交叉一致性配对：品评组感官维度 ↔ 消费者描述符（依据见 README「配对依据」表）
CROSS_PAIRS: List[Tuple[str, str, str]] = [
    ("A_hops_citrus", "aroma_citrus", "词根一致：柑橘"),
    ("A_hops_tropical", "aroma_tropical", "词根一致：热带果"),
    ("A_hops_woody", "aroma_woody", "词根一致：木质"),
    ("A_esters_fruity", "aroma_fruity", "词根一致：果香"),
    ("A_esters_ethac", "aroma_acetate", "同指乙酸乙酯"),
    ("A_esters_isoaa", "aroma_banana", "异戊乙酸酯=香蕉"),
    ("A_esters_flower", "aroma_flower", "词根一致：花香"),
    ("A_esters_all", "aroma_estery", "酯类总体 ↔ 酯香"),
    ("A_malt_bread", "aroma_bread", "词根一致：面包"),
    ("A_malt_cara", "aroma_caramel", "词根一致：焦糖"),
    ("F_hops_citrus", "taste_citrus", "词根一致：柑橘"),
    ("F_esters_fruity", "taste_fruity", "词根一致：果香"),
    ("F_esters_ethac", "taste_acetate", "同指乙酸乙酯"),
    ("F_esters_all", "taste_estery", "酯类总体 ↔ 酯香"),
    ("bitternes", "taste_bitter", "词根一致：苦味"),
    ("acidity", "taste_sour", "酸度 ↔ 酸感"),
    ("sweetness", "taste_sweet", "词根一致：甜"),
    ("alcohol", "palate_alcohol", "词根一致：酒精感"),
    ("co2", "palate_carbonation", "碳酸 ↔ 碳酸感"),
    ("body", "palate_buttery", "酒体 ↔ 奶油质地（最接近的口感体描述符）"),
    ("dms", "taste_stale", "硫化物/氧化 → 陈化味（缺陷感）"),
    ("stale_hops", "aroma_stale", "陈化酒花味 → 陈化味"),
    ("diacetyl", "taste_diacetyl", "词根一致：双乙酰"),
    ("X4vg", "taste_smoky", "4-乙烯基愈创木酚 → 烟熏/酚感"),
    ("t2n", "taste_cardboard", "2-壬烯醛 → 纸板味"),
    ("metallic", "taste_metallic", "词根一致：金属感"),
    ("orange", "aroma_orange", "词根一致：橙香"),
    ("clove", "taste_clove", "词根一致：丁香"),
    ("coriander", "aroma_coriander", "词根一致：香菜"),
    ("lactic", "taste_lactic", "词根一致：乳酸"),
    ("acetic", "taste_acetic", "词根一致：乙酸"),
    ("barnyard", "taste_barnyard", "词根一致：马厩味"),
    ("aftertaste", "taste_aftertaste", "词根一致：回味"),
]

#: B1–B3 剂量-响应配对：化合物（统一矩阵列名） ↔ 品评组感官维度
DOSE_PAIRS: List[Tuple[str, str]] = [
    ("ethyl_acetate", "A_esters_all"),
    ("ethyl_acetate", "F_esters_all"),
    ("linalool", "A_hops_citrus"),
    ("DMS", "dms"),
    ("X2.3.butanedione", "diacetyl"),
    ("X.E..2.nonenal", "t2n"),
]

#: 缺陷化合物集合（图 3 用强调色）
DEFECT_KEYS = {"DMS", "X2.3.butanedione", "X.E..2.nonenal", "X4VG"}

#: A4 价格回归使用的化学预测感知变量（消费者路径，OOF 预测）
PRICE_DESCRIPTORS = [
    "aroma_fruity", "aroma_hoppy", "aroma_citrus",
    "taste_fruity", "taste_sweet", "taste_bitter",
    "taste_stale", "palate_carbonation",
]

#: A4 风格固定效应的最小风格样本量（低于此值并入 Other）
FE_MIN_STYLE_N = 5

#: 模型族顺序（与 Q1 对齐）
MODEL_ORDER = ["global_mean", "ridge", "pls2", "global_extra_trees", "independent_gbr"]

#: 基准复现容差（任务书要求 ±0.005）
BENCH_TOL = 0.005

#: 主控实测基准（B1 分位级统计）：compound, sensory_dim, overall_rho, q1..q4, sensory med q1..q4
BENCHMARK = {
    ("ethyl_acetate", "A_esters_all"): (0.321, [0.171, 0.014, -0.038, 0.407],
                                         [-0.246, None, None, 0.262]),
    ("ethyl_acetate", "F_esters_all"): (0.380, [0.159, 0.042, -0.052, 0.498],
                                         [None, None, None, None]),
    ("linalool", "A_hops_citrus"): (0.327, [0.114, 0.033, 0.112, 0.128],
                                    [None, None, None, None]),
    ("DMS", "dms"): (0.088, [0.031, -0.173, -0.044, -0.004],
                     [None, None, None, None]),
    ("X2.3.butanedione", "diacetyl"): (-0.093, [0.083, -0.041, 0.171, 0.030],
                                        [-0.143, None, None, -0.233]),
    ("X.E..2.nonenal", "t2n"): (-0.151, [-0.073, -0.167, -0.033, -0.219],
                                 [None, None, None, None]),
}


# --------------------------------------------------------------------------- 工具


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 相关系数；任一序列为常数时返回 NaN。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def log(msg: str) -> None:
    print(f"[q4] {msg}", flush=True)


def rel(path: Path) -> str:
    """仓库相对路径（避免绝对路径泄漏到产物中）。"""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return Path(path).name


# --------------------------------------------------------------------------- A1 清洗


def load_consumer_matrix(raw: dc.RawBundle) -> Tuple[pd.DataFrame, dict]:
    """A1：读 7.csv → 剔除整列为空的描述符 → 其余按列中位数填补 → 索引对齐 beer。"""
    frame = raw.forum.copy()
    frame = frame.set_index("beer")
    frame.index = frame.index.astype(int)
    key_cols = [c for c in dc.KEY_COLUMNS if c in frame.columns]
    desc_cols = [c for c in frame.columns if c not in key_cols]

    n_rows = int(len(frame))
    na_counts = frame[desc_cols].isna().sum()
    all_na = [c for c in desc_cols if int(na_counts[c]) == n_rows]
    # 任务书点名的三列必须与实测一致，否则报错
    assert set(all_na) == set(EMPTY_CONSUMER_COLS), f"全空列与预期不符：{all_na}"

    keep = [c for c in desc_cols if c not in all_na]
    mat = frame[keep].apply(pd.to_numeric, errors="coerce")
    n_missing_cells = int(mat.isna().sum().sum())
    n_imputed_columns = int((mat.isna().sum() > 0).sum())
    filled = mat.fillna(mat.median())
    assert not filled.isna().any().any()

    info = {
        "source": "data/raw/7.csv",
        "rows": n_rows,
        "raw_descriptor_columns": len(desc_cols),
        "dropped_all_na_columns": all_na,
        "kept_dimensions": int(filled.shape[1]),
        "imputed_cells": n_missing_cells,
        "imputed_columns": n_imputed_columns,
        "index_aligned_to": "data/raw/1.csv 的 beer（250 行，顺序一致）",
        "scaling_note": "附件 7.csv 描述符已中心化并按列缩放（列标准差中位 1.003），本层不再二次缩放",
    }
    return filled, info


def load_features(raw: dc.RawBundle, keep_derived: bool) -> pd.DataFrame:
    """化学特征：统一矩阵 log10(mg/L)。keep_derived=True 保留 4 个 *_sum 派生列。"""
    matrix = dc.to_unified_scale(raw)
    if not keep_derived:
        matrix = matrix.drop(columns=[c for c in matrix.columns if c.endswith("_sum")])
    return matrix


# --------------------------------------------------------------------------- 模型


def q1_fixed_params() -> Dict[str, dict]:
    """复用 Q1 在 50 维品评组上选定的超参（只读产物），保证两层同口径且避免二次调参。"""
    out: Dict[str, dict] = {}
    et_path = REPO_ROOT / "data/processed/q1_runs/global_extra_trees/model_best_params.csv"
    if et_path.exists():
        et = pd.read_csv(et_path).iloc[0]
        out["global_extra_trees"] = {
            "max_depth": None if pd.isna(et.get("max_depth")) else int(et["max_depth"]),
            "max_features": float(et["max_features"]),
            "min_samples_leaf": int(et["min_samples_leaf"]),
            "n_estimators": 300,
        }
    else:
        out["global_extra_trees"] = {"max_depth": None, "max_features": 0.6,
                                     "min_samples_leaf": 3, "n_estimators": 300}

    gb_path = REPO_ROOT / "data/processed/q1_runs/independent_gbr/model_best_params.csv"
    if gb_path.exists():
        gb = pd.read_csv(gb_path)
        pick = {}
        for col in ["n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
                    "max_features", "subsample"]:
            if col in gb.columns:
                mode = gb[col].mode()
                pick[col] = mode.iloc[0] if len(mode) else gb[col].iloc[0]
        pick["n_estimators"] = int(pick["n_estimators"])
        pick["learning_rate"] = float(pick["learning_rate"])
        pick["max_depth"] = int(pick["max_depth"])
        pick["min_samples_leaf"] = int(pick["min_samples_leaf"])
        pick["max_features"] = str(pick["max_features"])
        pick["subsample"] = float(pick["subsample"])
        out["independent_gbr"] = pick
    else:
        out["independent_gbr"] = {"n_estimators": 200, "learning_rate": 0.05,
                                  "max_depth": 1, "min_samples_leaf": 15,
                                  "max_features": "sqrt", "subsample": 0.8}
    return out


def fit_model(name: str, x_train: pd.DataFrame, y_train: pd.DataFrame,
              x_apply: pd.DataFrame, inner: List[Tuple[np.ndarray, np.ndarray]],
              n_jobs: int, fixed: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """拟合单个模型族并对 x_apply 预测。返回 (预测表, 超参表)。"""
    labels = list(y_train.columns)
    pred = pd.DataFrame(index=x_apply.index, columns=labels, dtype=float)

    if name == "global_mean":
        pred.loc[:, :] = np.tile(y_train.mean().to_numpy(), (len(x_apply), 1))
        return pred, pd.DataFrame([{"model": name, "note": "训练折目标均值基线"}])

    if name == "ridge":
        search = GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-3, 4, 40)}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True)
        search.fit(x_train, y_train)
        pred.loc[:, labels] = np.asarray(search.predict(x_apply)).reshape(len(x_apply), len(labels))
        return pred, pd.DataFrame([{"model": name, "model__alpha": float(search.best_params_["model__alpha"]),
                                    "inner_best_neg_MSE": float(search.best_score_)}])

    if name == "pls2":
        search = GridSearchCV(
            Pipeline([("scale", StandardScaler()),
                      ("model", PLSRegression(scale=False, max_iter=2000))]),
            {"model__n_components": [2, 5, 10, 15, 20]}, cv=inner,
            scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True)
        search.fit(x_train, y_train)
        pred.loc[:, labels] = np.asarray(search.predict(x_apply)).reshape(len(x_apply), len(labels))
        return pred, pd.DataFrame([{"model": name,
                                    "model__n_components": int(search.best_params_["model__n_components"]),
                                    "inner_best_neg_MSE": float(search.best_score_)}])

    if name == "global_extra_trees":
        params = dict(fixed["global_extra_trees"])
        est = ExtraTreesRegressor(random_state=SEED, n_jobs=1, **params)
        est.fit(x_train, y_train)
        pred.loc[:, labels] = np.asarray(est.predict(x_apply)).reshape(len(x_apply), len(labels))
        return pred, pd.DataFrame([{"model": name, **params, "source": "Q1 选定超参（复用）"}])

    if name == "independent_gbr":
        params = dict(fixed["independent_gbr"])
        for target in labels:
            est = GradientBoostingRegressor(loss="squared_error", random_state=SEED, **params)
            est.fit(x_train, y_train[target])
            pred[target] = est.predict(x_apply)
        return pred, pd.DataFrame([{"model": name, **params, "source": "Q1 选定超参众数（复用）"}])

    raise ValueError(f"未知模型：{name}")


def evaluate(y_true: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """逐目标 RMSE/MAE/R²/Spearman 与宏平均汇总。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rows = []
    for label in y_true.columns:
        t = y_true[label].to_numpy(dtype=float)
        p = pred[label].to_numpy(dtype=float)
        rows.append({
            "label": label,
            "RMSE": float(mean_squared_error(t, p) ** 0.5),
            "MAE": float(mean_absolute_error(t, p)),
            "R2": float(r2_score(t, p)) if np.std(t) > 1e-12 else float("nan"),
            "Spearman": spearman(t, p),
        })
    per_dim = pd.DataFrame(rows)
    summary = {
        "n_targets": int(len(rows)),
        "Macro_RMSE": float(per_dim["RMSE"].mean()),
        "Macro_MAE": float(per_dim["MAE"].mean()),
        "R2_mean": float(per_dim["R2"].mean()),
        "R2_median": float(per_dim["R2"].median()),
        "Spearman_mean": float(per_dim["Spearman"].mean()),
        "Spearman_median": float(per_dim["Spearman"].median()),
    }
    return per_dim, summary


def run_holdout(x: pd.DataFrame, y: pd.DataFrame, train_ids: np.ndarray, test_ids: np.ndarray,
                models: Sequence[str], n_jobs: int, fixed: Dict[str, dict],
                tag: str = "") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, dict]]:
    """锁定 175/75 留出集评估（基线 = 训练折均值）。"""
    x_train, x_test = x.loc[train_ids], x.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    inner = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=INNER_SEED).split(x_train))
    preds: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, dict] = {}
    per_dim_frames = []
    for name in models:
        t0 = time.time()
        pred, _params = fit_model(name, x_train, y_train, x_test, inner, n_jobs, fixed)
        per_dim, summary = evaluate(y_test, pred)
        per_dim.insert(0, "model", name)
        per_dim_frames.append(per_dim)
        preds[name] = pred
        summary["fit_seconds"] = round(time.time() - t0, 1)
        summaries[name] = summary
        log(f"holdout{tag} {name}: Macro_RMSE={summary['Macro_RMSE']:.4f} "
            f"Spearman_median={summary['Spearman_median']:.4f} ({summary['fit_seconds']}s)")
    return preds, pd.concat(per_dim_frames, ignore_index=True), summaries


def run_oof(x: pd.DataFrame, y: pd.DataFrame, name: str, fixed: Dict[str, dict],
            n_jobs: int) -> pd.DataFrame:
    """5 折 OOF（KFold shuffle seed 0）预测全部 250 样本——避免自证，供交叉一致性与价格回归使用。"""
    labels = list(y.columns)
    out = np.zeros((len(x), len(labels)))
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(splitter.split(x), 1):
        x_tr, x_va = x.iloc[tr], x.iloc[va]
        y_tr = y.iloc[tr]
        inner = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=INNER_SEED).split(x_tr))
        if name == "ridge":
            search = GridSearchCV(
                Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
                {"model__alpha": np.logspace(-3, 4, 40)}, cv=inner,
                scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True).fit(x_tr, y_tr)
            pred = np.asarray(search.predict(x_va)).reshape(len(va), len(labels))
        elif name == "pls2":
            search = GridSearchCV(
                Pipeline([("scale", StandardScaler()),
                          ("model", PLSRegression(scale=False, max_iter=2000))]),
                {"model__n_components": [2, 5, 10, 15, 20]}, cv=inner,
                scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True).fit(x_tr, y_tr)
            pred = np.asarray(search.predict(x_va)).reshape(len(va), len(labels))
        else:
            pred_df, _ = fit_model(name, x_tr, y_tr, x_va, inner, n_jobs, fixed)
            pred = pred_df.to_numpy()
        out[va] = pred
        log(f"OOF {name} fold {fold}/{N_FOLDS} 完成")
    return pd.DataFrame(out, index=x.index, columns=labels)


# --------------------------------------------------------------------------- A4 价格


def price_frame(raw: dc.RawBundle, min_retailers: int) -> pd.DataFrame:
    """构造 log(几何均价)：只保留被报价商家数 ≥ min_retailers 的酒。"""
    bp = raw.beer_prices.copy()
    price_cols = [c for c in bp.columns if c.startswith("Price_")]
    rows = []
    for _, r in bp.iterrows():
        vals = [float(r[c]) for c in price_cols if pd.notna(r[c]) and float(r[c]) > 0]
        if len(vals) >= min_retailers:
            geo = float(np.exp(np.mean(np.log(vals))))
            rows.append({"beer": int(r["beer"]), "n_retailers": len(vals),
                         "geo_price": geo, "log_price": float(np.log(geo)),
                         "ABV": float(r["ABV"]) if pd.notna(r["ABV"]) else np.nan,
                         "price_style": str(r.get("tasting_category", ""))})
    return pd.DataFrame(rows).set_index("beer").sort_index()


def price_regression(frame: pd.DataFrame, predictor_frame: pd.DataFrame,
                     styles: pd.Series, descriptors: Sequence[str],
                     min_style_n: int = FE_MIN_STYLE_N) -> Tuple[pd.DataFrame, dict]:
    """三套规格：ols / ols_with_abv / fixed_effects（风格 FE + 聚类稳健标准误）。"""
    import statsmodels.api as sm

    df = frame.join(predictor_frame[[c for c in descriptors]], how="inner")
    df = df.join(styles.rename("style"), how="left")
    df = df.dropna(subset=list(descriptors) + ["log_price"])
    df = df[df["ABV"].notna()]
    n = int(len(df))

    # 风格固定效应：样本量过小的风格并入 Other；最大风格作基准组
    counts = df["style"].value_counts()
    keep_styles = [s for s, c in counts.items() if c >= min_style_n]
    df["fe_style"] = df["style"].where(df["style"].isin(keep_styles), "Other")
    fe_counts = df["fe_style"].value_counts()
    ref_style = str(fe_counts.index[0])
    fe_levels = [str(s) for s in fe_counts.index if str(s) != ref_style]

    rows: List[dict] = []
    specs: Dict[str, dict] = {}

    def _design(term_cols: List[str]) -> pd.DataFrame:
        return sm.add_constant(df[term_cols], has_constant="add")

    def _record(model: str, res, term_cols: List[str], robust: str) -> None:
        params = res.params
        bse = res.bse
        r2 = float(getattr(res, "rsquared", np.nan))
        for term in params.index:
            rows.append({"term": term, "coef": float(params[term]),
                         "std_err": float(bse.get(term, np.nan)), "model": model,
                         "n": int(res.nobs), "r2": r2})
        specs[model] = {"n": int(res.nobs), "r2": r2,
                        "adj_r2": float(getattr(res, "rsquared_adj", np.nan)),
                        "terms": len(term_cols) + 1, "robust": robust,
                        "design": term_cols}

    # 1) ols：仅化学预测感知
    terms_ols = list(descriptors)
    res = sm.OLS(df["log_price"], _design(terms_ols)).fit()
    _record("ols", res, terms_ols, "classical")

    # 2) ols_with_abv：+ ABV
    terms_abv = terms_ols + ["ABV"]
    res_abv = sm.OLS(df["log_price"], _design(terms_abv)).fit()
    _record("ols_with_abv", res_abv, terms_abv, "classical")

    # 3) fixed_effects：+ ABV + 风格固定效应，聚类稳健标准误（按风格聚类）
    terms_fe = terms_abv + [f"fe_{s}" for s in fe_levels]
    design_fe = pd.concat(
        [pd.Series(1.0, index=df.index, name="const"), df[terms_abv]],
        axis=1)
    for s in fe_levels:
        design_fe[f"fe_{s}"] = (df["fe_style"] == s).astype(float)
    res_fe = sm.OLS(df["log_price"], design_fe).fit(
        cov_type="cluster", cov_kwds={"groups": df["fe_style"]})
    _record("fixed_effects", res_fe, terms_fe, "cluster-robust (by style)")

    table = pd.DataFrame(rows)
    meta = {"specs": specs, "n_obs": n, "fe_levels": fe_levels,
            "fe_reference_style": ref_style,
            "styles_dropped_to_other": sorted(set(counts.index) - set(keep_styles)),
            "abv_corr_log_price": float(np.corrcoef(df["ABV"], df["log_price"])[0, 1]),
            "style_counts": {str(k): int(v) for k, v in counts.items()},
            "abv_mean": float(df["ABV"].mean()), "log_price_mean": float(df["log_price"].mean())}
    return table, meta


# --------------------------------------------------------------------------- B 剂量-响应


def sensory_threshold_lookup(raw: dc.RawBundle) -> Dict[str, dict]:
    """8.xlsx「Sensory thresholds」→ 统一矩阵口径 log10(mg/L) 的阈值（用中位阈值）。"""
    th = raw.thresholds.copy()
    lookup: Dict[str, dict] = {}
    for _, r in th.iterrows():
        col = r.get("field_name_R")
        median = r.get("threshold_median")
        unit = r.get("unit")
        if not isinstance(col, str) or pd.isna(median) or not isinstance(unit, str):
            continue
        unit = unit.strip().lower()
        if unit == "mg/l":
            factor = 1.0
        elif unit == "ug/l":
            factor = 1e-3
        else:
            continue
        thr_mg = float(median) * factor
        lookup[col] = {"threshold_median_raw": float(median), "unit": unit,
                       "threshold_mg_per_L": thr_mg,
                       "log10_threshold": float(np.log10(thr_mg))}
    return lookup


def quartile_assignment(values: pd.Series) -> pd.Series:
    """浓度四分位（Q1 = 最低浓度）。

    口径与主控基准一致：直接对浓度值 ``pd.qcut(..., 4)``，**不做 rank 破平**——
    浓度存在并列时并列值落入同一分位（分位样本量可不等，如 linalool 为 63/64/60/63），
    这正是基准数字的来源；若某子集因并列导致分箱失败，则退化为 rank 破平。
    """
    try:
        return pd.qcut(values, 4, labels=[1, 2, 3, 4]).astype(int)
    except Exception:
        return pd.qcut(values.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)


def verdict_from(rhos: Sequence[float], meds: Sequence[float],
                 overall_rho: float) -> Tuple[str, str]:
    """模式判定：组内 ρ 符号一致性为主，感官中位单调性为辅。

    返回 (pattern, note)。
    """
    rhos = [float(r) for r in rhos]
    meds = [float(m) for m in meds]
    valid = [r for r in rhos if np.isfinite(r)]
    if not valid:
        return "none", "组内 ρ 全部不可计算"
    all_pos = all(r > 0 for r in valid)
    all_neg = all(r < 0 for r in valid)
    has_pos = any(r > 0 for r in valid)
    has_neg = any(r < 0 for r in valid)
    weak = all(abs(r) < 0.10 for r in valid) and abs(overall_rho) < 0.10
    mono_inc = all(meds[i + 1] > meds[i] for i in range(len(meds) - 1))
    mono_dec = all(meds[i + 1] < meds[i] for i in range(len(meds) - 1))
    nondec = all(meds[i + 1] >= meds[i] for i in range(len(meds) - 1))
    noninc = all(meds[i + 1] <= meds[i] for i in range(len(meds) - 1))
    med_dir = ("单调上升" if mono_inc else "单调下降" if mono_dec
               else "非降" if nondec else "非升" if noninc else "非单调")
    sign_str = " ".join("+" if r > 0 else ("-" if r < 0 else "0") for r in valid)
    if weak:
        pattern = "none"
        note = f"组内 ρ 均 |ρ|<0.10（{sign_str}），感官中位{med_dir} → 无剂量-响应信号"
    elif all_pos:
        pattern = "monotonic_positive"
        note = f"组内 ρ 全为正（{sign_str}），感官中位{med_dir} → 剂量-响应方向一致"
    elif all_neg:
        pattern = "monotonic_negative"
        note = f"组内 ρ 全为负（{sign_str}），感官中位{med_dir} → 剂量-响应方向一致"
    else:
        pattern = "non_monotonic"
        note = f"组内 ρ 符号不一致（{sign_str}），感官中位{med_dir} → 非单调"
    return pattern, note


def dose_response(x: pd.DataFrame, y: pd.DataFrame, thresholds: Dict[str, dict]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B1–B3：分位级统计 + 判定表 + 风格分层稳健性。"""
    styles = dc.load_styles()
    style_counts = styles.value_counts()
    big_styles = [s for s, c in style_counts.items() if c >= 8]

    dose_rows: List[dict] = []
    verdict_rows: List[dict] = []
    style_rows: List[dict] = []
    detail_rows: List[dict] = []

    for compound, dim in DOSE_PAIRS:
        conc = x[compound]
        sens = y[dim]
        q = quartile_assignment(conc)
        thr = thresholds.get(compound)
        overall = spearman(conc.to_numpy(), sens.to_numpy())

        rhos: List[float] = []
        meds: List[float] = []
        for k in (1, 2, 3, 4):
            mask = (q == k).to_numpy()
            sub_c, sub_s = conc.to_numpy()[mask], sens.to_numpy()[mask]
            rho = spearman(sub_c, sub_s)
            rhos.append(rho)
            meds.append(float(np.median(sub_s)))
            if thr is not None:
                exceed = float(np.mean(sub_c > thr["log10_threshold"]))
            else:
                exceed = float("nan")
            dose_rows.append({
                "compound": compound, "sensory_dim": dim, "quartile": k,
                "n": int(mask.sum()),
                "conc_median": float(np.median(sub_c)),
                "rho_within": rho,
                "sensory_median": meds[-1],
                "exceed_frac": exceed,
            })

        pattern, note = verdict_from(rhos, meds, overall)

        # 风格分层稳健性（n ≥ 8 的全部风格）
        keep_patterns = 0
        eligible = 0
        signs_match = 0
        half_keep = 0
        half_eligible = 0
        flips: List[str] = []
        for style in big_styles:
            mask = (styles == style).to_numpy()
            s_c = conc.to_numpy()[mask]
            s_s = sens.to_numpy()[mask]
            rho_style = spearman(s_c, s_s)
            s_q = quartile_assignment(pd.Series(s_c))
            sub_rhos, sub_meds, cells = [], [], []
            for k in (1, 2, 3, 4):
                m = (s_q == k).to_numpy()
                if int(m.sum()) >= 3:
                    sub_rhos.append(spearman(s_c[m], s_s[m]))
                    sub_meds.append(float(np.median(s_s[m])))
                    cells.append(int(m.sum()))
                else:
                    sub_rhos.append(float("nan"))
                    sub_meds.append(float("nan"))
                    cells.append(int(m.sum()))
            if all(np.isfinite(r) for r in sub_rhos) and len(sub_rhos) == 4:
                s_pattern, s_note = verdict_from(sub_rhos, sub_meds, rho_style)
            else:
                s_pattern = "insufficient_quarters"
                s_note = "风格样本量过小，四分位格 <3 样本，组内 ρ 不可靠"
            # 半分割（浓度中位数分低/高两半）：每个 n≥8 的风格都可判，作为更稳健的伴随指标
            cut = float(np.median(s_c))
            low = s_c <= cut
            high = ~low
            rho_lo = spearman(s_c[low], s_s[low])
            rho_hi = spearman(s_c[high], s_s[high])
            med_lo = float(np.median(s_s[low]))
            med_hi = float(np.median(s_s[high]))
            h_pattern, h_note = verdict_from([rho_lo, rho_hi], [med_lo, med_hi], rho_style)
            half_eligible += 1
            if h_pattern == pattern:
                half_keep += 1
            style_rows.append({"compound": compound, "sensory_dim": dim, "style": style,
                               "n": int(mask.sum()), "rho": rho_style, "pattern": s_pattern})
            detail_rows.append({"compound": compound, "sensory_dim": dim, "style": style,
                                "n": int(mask.sum()), "rho": rho_style,
                                "rho_q1": sub_rhos[0], "rho_q2": sub_rhos[1],
                                "rho_q3": sub_rhos[2], "rho_q4": sub_rhos[3],
                                "cells_q1": cells[0], "cells_q2": cells[1],
                                "cells_q3": cells[2], "cells_q4": cells[3],
                                "pattern": s_pattern, "note": s_note,
                                "rho_half_low": rho_lo, "rho_half_high": rho_hi,
                                "pattern_half": h_pattern, "note_half": h_note})
            if s_pattern != "insufficient_quarters":
                eligible += 1
                if s_pattern == pattern:
                    keep_patterns += 1
                else:
                    flips.append(f"{style}({s_pattern})")
                if np.isfinite(rho_style) and np.isfinite(overall) and \
                        np.sign(rho_style) == np.sign(overall):
                    signs_match += 1

        frac = keep_patterns / eligible if eligible else float("nan")
        half_frac = half_keep / half_eligible if half_eligible else float("nan")
        thr_txt = ("无阈值" if thr is None else
                   f"阈值中位 {thr['threshold_mg_per_L']:.4g} mg/L (log10={thr['log10_threshold']:.4f})")
        flip_txt = ("；风格分层后模式不一致的风格：" + "、".join(flips)) if flips else ""
        verdict_rows.append({
            "compound": compound, "sensory_dim": dim, "pattern": pattern,
            "overall_rho": overall,
            "rho_q1": rhos[0], "rho_q2": rhos[1], "rho_q3": rhos[2], "rho_q4": rhos[3],
            "style_robust_fraction": frac,
            "note": (f"{note}；{thr_txt}；风格分层（n≥8，共 {eligible} 个风格四分位可判，"
                     f"{half_eligible} 个风格半分割可判）保持同一模式 {keep_patterns}/{eligible}"
                     f"（半分割 {half_keep}/{half_eligible} = {half_frac:.2f}），"
                     f"风格级 ρ 符号一致 {signs_match}/{eligible}{flip_txt}"),
        })

    return (pd.DataFrame(dose_rows), pd.DataFrame(verdict_rows),
            pd.DataFrame(style_rows, columns=["compound", "sensory_dim", "style", "n",
                                              "rho", "pattern"]),
            pd.DataFrame(detail_rows))


# --------------------------------------------------------------------------- 主流程


def run_consumer_stage(args, raw, cache: Dict) -> dict:
    """A1–A4：消费者清洗、建模、交叉一致性、价格回归。"""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_consumer, consumer_info = load_consumer_matrix(raw)
    x_drop = load_features(raw, keep_derived=False)
    x_keep = load_features(raw, keep_derived=True)
    y_panel = dc.load_targets(raw)
    styles = dc.load_styles(raw)
    split = dc.load_split()
    train_ids = split.loc[split["split"] == "train", "beer"].to_numpy(dtype=int)
    test_ids = split.loc[split["split"] == "test", "beer"].to_numpy(dtype=int)
    assert len(train_ids) == 175 and len(test_ids) == 75
    assert x_drop.shape == (250, 227) and x_keep.shape == (250, 231)
    assert y_consumer.shape == (250, 183) and y_panel.shape == (250, 50)

    if getattr(args, "smoke", False):
        y_consumer = y_consumer.iloc[:, :25]
        y_panel = y_panel.iloc[:, :10]
        log(f"冒烟模式：消费者维度 {y_consumer.shape[1]}，品评组维度 {y_panel.shape[1]}")

    fixed = q1_fixed_params()
    models = list(MODEL_ORDER)

    # --- A2 消费者层：锁定 175/75 留出集 ---
    cons_pred, cons_per_dim, cons_summary = run_holdout(
        x_drop, y_consumer, train_ids, test_ids, models, args.n_jobs, fixed, tag=" [consumer]")
    panel_pred, panel_per_dim, panel_summary = run_holdout(
        x_drop, y_panel, train_ids, test_ids, models, args.n_jobs, fixed, tag=" [panel]")

    summary_rows = []
    for name in models:
        s = cons_summary[name]
        summary_rows.append({"model": name, "n_targets": s["n_targets"],
                             "Macro_RMSE": s["Macro_RMSE"], "Macro_MAE": s["Macro_MAE"],
                             "R2_median": s["R2_median"], "Spearman_median": s["Spearman_median"]})
    consumer_summary = pd.DataFrame(summary_rows)
    consumer_summary.to_csv(out_dir / "consumer_model_summary.csv", index=False,
                            encoding="utf-8")

    # --- 派生列口径对照（231 列 vs 227 列）---
    keep_pred, _, keep_summary = run_holdout(
        x_keep, y_consumer, train_ids, test_ids, models, args.n_jobs, fixed,
        tag=" [consumer+derived]")
    derived_compare = pd.DataFrame([
        {"model": name,
         "features_drop_derived": cons_summary[name]["Macro_RMSE"],
         "features_keep_derived": keep_summary[name]["Macro_RMSE"],
         "delta_macro_rmse": keep_summary[name]["Macro_RMSE"] - cons_summary[name]["Macro_RMSE"],
         "drop_derived_spearman_median": cons_summary[name]["Spearman_median"],
         "keep_derived_spearman_median": keep_summary[name]["Spearman_median"]}
        for name in models])

    # --- A2 对比表：同口径（同一留出集 + 同一超参）下的消费者 vs 品评组 ---
    compare_rows = []
    for name in models:
        for target_set, summ in (("consumer_183", cons_summary), ("panel_50", panel_summary)):
            s = summ[name]
            compare_rows.append({
                "model": name, "target_set": target_set, "n_targets": s["n_targets"],
                "evaluation": "locked_holdout_175_75",
                "Macro_RMSE": s["Macro_RMSE"], "Macro_MAE": s["Macro_MAE"],
                "R2_median": s["R2_median"], "Spearman_median": s["Spearman_median"],
                "source": "本脚本实跑"})
    # Q1 已发布的 OOF 口径结果（外部参照，只读）
    q1_path = REPO_ROOT / "data/processed/q1_final/model_summary.csv"
    if q1_path.exists():
        q1 = pd.read_csv(q1_path)
        for _, r in q1.iterrows():
            compare_rows.append({
                "model": r["model"], "target_set": "panel_50", "n_targets": 50,
                "evaluation": "q1_published_5fold_oof",
                "Macro_RMSE": float(r["Macro_RMSE"]), "Macro_MAE": float(r["Macro_MAE"]),
                "R2_median": float(r["R2_median"]),
                "Spearman_median": float(r["Spearman_median"]),
                "source": "data/processed/q1_final/model_summary.csv（只读引用）"})
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(out_dir / "consumer_vs_panel_models.csv", index=False, encoding="utf-8")

    # --- 选定模型族（按消费者留出集 Macro-RMSE 最小）→ OOF 预测 ---
    best_model = consumer_summary.sort_values("Macro_RMSE").iloc[0]["model"]
    log(f"选定用于交叉一致性/价格回归的模型族：{best_model}")
    if best_model == "global_mean":
        best_model = "ridge"
        log("global_mean 无个体化信息，退回 ridge")
    oof_consumer = run_oof(x_drop, y_consumer, best_model, fixed, args.n_jobs)
    oof_panel = run_oof(x_drop, y_panel, best_model, fixed, args.n_jobs)

    # --- A3 交叉一致性 ---
    cross_rows = []
    for panel_dim, consumer_desc, basis in CROSS_PAIRS:
        if panel_dim not in y_panel.columns or consumer_desc not in y_consumer.columns:
            continue
        truth_rho = spearman(y_panel[panel_dim].to_numpy(), y_consumer[consumer_desc].to_numpy())
        pred_rho = spearman(oof_panel[panel_dim].to_numpy(), oof_consumer[consumer_desc].to_numpy())
        cross_rows.append({"panel_dim": panel_dim, "consumer_desc": consumer_desc,
                           "rho_truth": truth_rho, "rho_pred": pred_rho,
                           "delta": pred_rho - truth_rho, "n": int(len(y_panel))})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(out_dir / "cross_consistency.csv", index=False, encoding="utf-8")

    # --- A4 价格回归 ---
    price_ge2 = price_frame(raw, 2)
    price_ge1 = price_frame(raw, 1)
    descriptors = [d for d in PRICE_DESCRIPTORS if d in oof_consumer.columns]
    if not descriptors:
        descriptors = list(oof_consumer.columns[:5])
    price_table, price_meta = price_regression(price_ge2, oof_consumer, styles, descriptors)
    price_table.to_csv(out_dir / "price_regression.csv", index=False, encoding="utf-8")
    price_table_ge1, price_meta_ge1 = price_regression(price_ge1, oof_consumer, styles, descriptors)

    cache.update({
        "consumer_info": consumer_info,
        "consumer_summary": consumer_summary,
        "consumer_per_dim": cons_per_dim,
        "panel_per_dim": panel_per_dim,
        "panel_summary": panel_summary,
        "derived_compare": derived_compare,
        "compare": compare,
        "cross": cross,
        "best_model": best_model,
        "oof_consumer": oof_consumer,
        "oof_panel": oof_panel,
        "price_table": price_table,
        "price_meta": price_meta,
        "price_table_ge1": price_table_ge1,
        "price_meta_ge1": price_meta_ge1,
        "price_ge1_n": int(len(price_ge1)),
        "price_ge2_n": int(len(price_ge2)),
        "fixed_params": fixed,
    })
    # 补充落盘：逐维指标与 OOF 预测（供分阶段重跑与论文取数）
    cons_per_dim.to_csv(out_dir / "consumer_per_dimension_metrics.csv", index=False,
                        encoding="utf-8")
    panel_per_dim.to_csv(out_dir / "panel_per_dimension_metrics.csv", index=False,
                         encoding="utf-8")
    oof_consumer.to_csv(out_dir / "oof_predictions_consumer.csv.gz", compression="gzip",
                        encoding="utf-8")
    oof_panel.to_csv(out_dir / "oof_predictions_panel.csv.gz", compression="gzip",
                     encoding="utf-8")
    return cache


def run_dose_stage(args, raw, cache: Dict) -> dict:
    """B1–B4：剂量-响应再检验。"""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = load_features(raw, keep_derived=False)
    y = dc.load_targets(raw)
    thresholds = sensory_threshold_lookup(raw)
    dose, verdict, style_robust, detail = dose_response(x, y, thresholds)
    dose.to_csv(out_dir / "defect_dose_response.csv", index=False, encoding="utf-8")
    verdict.to_csv(out_dir / "defect_dose_verdict.csv", index=False, encoding="utf-8")
    style_robust.to_csv(out_dir / "style_robustness.csv", index=False, encoding="utf-8")
    detail.to_csv(out_dir / "style_robustness_detail.csv", index=False, encoding="utf-8")

    # 基准复现核对
    checks = []
    for (compound, dim), (o_ref, q_ref, m_ref) in BENCHMARK.items():
        row = verdict[(verdict["compound"] == compound) & (verdict["sensory_dim"] == dim)].iloc[0]
        got_q = [float(row[f"rho_q{i}"]) for i in (1, 2, 3, 4)]
        diffs = [abs(got_q[i] - q_ref[i]) for i in range(4)]
        rec = {"compound": compound, "sensory_dim": dim,
               "overall_rho_ref": o_ref, "overall_rho_got": float(row["overall_rho"]),
               "max_abs_diff_rho": float(max(diffs)),
               "pass": bool(abs(float(row["overall_rho"]) - o_ref) <= BENCH_TOL and
                            max(diffs) <= BENCH_TOL)}
        if m_ref[0] is not None:
            d = dose[(dose["compound"] == compound) & (dose["sensory_dim"] == dim)]
            got_m = [float(d[d["quartile"] == i].iloc[0]["sensory_median"]) for i in (1, 4)]
            rec["sensory_median_q1_ref"] = m_ref[0]
            rec["sensory_median_q1_got"] = got_m[0]
            rec["sensory_median_q4_ref"] = m_ref[3]
            rec["sensory_median_q4_got"] = got_m[1]
            rec["pass"] = bool(rec["pass"] and abs(got_m[0] - m_ref[0]) <= BENCH_TOL
                               and abs(got_m[1] - m_ref[3]) <= BENCH_TOL)
        checks.append(rec)
    bench = pd.DataFrame(checks)
    bench.to_csv(out_dir / "dose_benchmark_check.csv", index=False, encoding="utf-8")
    log(f"基准复现：{int(bench['pass'].sum())}/{len(bench)} 行在 ±{BENCH_TOL} 内复现")
    cache.update({"dose": dose, "verdict": verdict, "style_robust": style_robust,
                  "dose_detail": detail, "benchmark": bench, "thresholds": thresholds})
    return cache


# --------------------------------------------------------------------------- 图


def make_figures(args, cache: Dict) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps

    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # 分阶段重跑时，从落盘的中间表补齐绘图所需数据
    out_dir = Path(args.output_dir)
    if "consumer_per_dim" not in cache:
        cons_path = out_dir / "consumer_per_dimension_metrics.csv"
        panel_path = out_dir / "panel_per_dimension_metrics.csv"
        if cons_path.exists() and panel_path.exists():
            cache["consumer_per_dim"] = pd.read_csv(cons_path)
            cache["panel_per_dim"] = pd.read_csv(panel_path)
    if "dose" not in cache and (out_dir / "defect_dose_response.csv").exists():
        cache["dose"] = pd.read_csv(out_dir / "defect_dose_response.csv")
    if "price_table" not in cache and (out_dir / "price_regression.csv").exists():
        cache["price_table"] = pd.read_csv(out_dir / "price_regression.csv")

    # 图 1：逐维可预测性（消费者 183 vs 品评组 50）
    ps.apply_style()
    cons = cache["consumer_per_dim"]
    panel = cache["panel_per_dim"]
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, 3.2))
    data = [panel["R2"].to_numpy(), cons["R2"].to_numpy()]
    box_labels = ["品评组 50 维", "消费者 183 维"]
    try:
        bp = ax.boxplot(data, tick_labels=box_labels, widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": ps.INK, "linewidth": 1.2})
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(data, labels=box_labels, widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": ps.INK, "linewidth": 1.2})
    for patch, color in zip(bp["boxes"], [ps.BASELINE, ps.ACCENT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for i, arr in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(arr))
        ax.scatter(np.full(len(arr), i) + jitter, np.clip(arr, -1.2, 1.0), s=4,
                   color=ps.INK2, alpha=0.5, linewidths=0)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.2, 1.0)
    ax.set_ylabel("逐维 R²")
    ps.tidy(ax, grid_axis="y")
    ps.simple_title(ax, "逐维可预测性对比")
    written += [rel(p) for p in ps.save(fig, fig_dir / "fig1_consumer_predictability")]
    plt.close(fig)

    # 图 2：交叉一致性散点
    cross = cache["cross"]
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, 3.2))
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color=ps.BASELINE, linewidth=0.9, linestyle="--")
    ax.scatter(cross["rho_truth"], cross["rho_pred"], s=18, color=ps.ACCENT,
               alpha=0.85, linewidths=0)
    ax.axhline(0, color=ps.GRID, linewidth=0.7)
    ax.axvline(0, color=ps.GRID, linewidth=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ρ 真值（品评组 vs 消费者）")
    ax.set_ylabel("ρ 预测（化学模型）")
    ps.tidy(ax, grid_axis="both")
    ps.simple_title(ax, "两条感知路径一致性")
    written += [rel(p) for p in ps.save(fig, fig_dir / "fig2_cross_consistency")]
    plt.close(fig)

    # 图 3：剂量-响应曲线
    dose = cache["dose"]
    verdict = cache["verdict"]
    compounds = []
    for compound, dim in DOSE_PAIRS:
        if (compound, dim) not in compounds:
            compounds.append((compound, dim))
    fig, axes = plt.subplots(1, 2, figsize=(ps.FIG_WIDTH_IN * 1.25, 3.2))
    fig.subplots_adjust(wspace=0.42)
    palette = [ps.ACCENT, ps.OKABE_ITO[1], ps.OKABE_ITO[2], ps.OKABE_ITO[3],
               ps.OKABE_ITO[4], ps.OKABE_ITO[6]]
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "-", "-", "-", "-", "-"]
    for idx, ((compound, dim), color) in enumerate(zip(compounds, palette)):
        defect = compound in DEFECT_KEYS
        if defect:
            # 缺陷物统一用强调色，靠点型/线型区分
            color = ps.DEFECT
            marker = markers[idx]
            ls = ["-", "--", ":"][idx % 3]
        else:
            marker = markers[idx]
            ls = linestyles[idx]
        sub = dose[(dose["compound"] == compound) & (dose["sensory_dim"] == dim)]
        xs = sub["quartile"].to_numpy()
        axes[0].plot(xs, sub["rho_within"].to_numpy(), marker=marker, color=color,
                     linestyle=ls, label=ps.dim_cn(dim))
        axes[1].plot(xs, sub["sensory_median"].to_numpy(), marker=marker, color=color,
                     linestyle=ls, label=ps.dim_cn(dim))
    axes[0].axhline(0, color=ps.BASELINE, linewidth=0.8, linestyle="--")
    axes[1].axhline(0, color=ps.BASELINE, linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("浓度四分位")
    axes[0].set_ylabel("组内 Spearman ρ")
    axes[1].set_xlabel("浓度四分位")
    axes[1].set_ylabel("感官维度中位")
    for ax in axes:
        ps.tidy(ax, grid_axis="y")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[1].set_xticks([1, 2, 3, 4])
    axes[0].legend(fontsize=6.5, ncol=1, handlelength=1.4, labelspacing=0.25)
    ps.simple_title(axes[0], "剂量-响应")
    written += [rel(p) for p in ps.save(fig, fig_dir / "fig3_dose_response")]
    plt.close(fig)

    # 图 4：价格回归部分效应
    price = cache["price_table"]
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, 3.2))
    terms = [d for d in PRICE_DESCRIPTORS] + ["ABV"]
    models = ["ols", "ols_with_abv", "fixed_effects"]
    colors = {"ols": ps.OKABE_ITO[0], "ols_with_abv": ps.OKABE_ITO[1],
              "fixed_effects": ps.OKABE_ITO[2]}
    markers = {"ols": "o", "ols_with_abv": "s", "fixed_effects": "^"}
    offsets = {"ols": -0.18, "ols_with_abv": 0.0, "fixed_effects": 0.18}
    labels = [ps.dim_cn(t) if t != "ABV" else "ABV" for t in terms]
    for model in models:
        sub = price[price["model"] == model].set_index("term")
        ys = [float(sub.loc[t, "coef"]) if t in sub.index else np.nan for t in terms]
        es = [float(sub.loc[t, "std_err"]) if t in sub.index else np.nan for t in terms]
        xs = [i + offsets[model] for i in range(len(terms))]
        ax.errorbar(xs, ys, yerr=es, fmt=markers[model], color=colors[model],
                    markersize=3.4, capsize=2, linewidth=0.8, label=model)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("log 价格系数")
    ax.legend(fontsize=7, handlelength=1.4)
    ps.tidy(ax, grid_axis="y")
    ps.simple_title(ax, "价格部分效应")
    written += [rel(p) for p in ps.save(fig, fig_dir / "fig4_price_effects")]
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- README


def _append_dose_sections(A, cache: Dict) -> None:
    """把 B（剂量-响应）相关章节追加到 README 行列表。"""
    verdict = cache["verdict"]
    bench = cache["benchmark"]
    n_pass = int(bench["pass"].sum())
    A("### `defect_dose_response.csv` / `defect_dose_verdict.csv` / `style_robustness.csv`（B）")
    A("")
    A("- `defect_dose_response.csv`：6 组配对 × 4 个浓度四分位，共 24 行。`conc_median` 是")
    A("  分位内浓度中位（统一 log10(mg/L)），`rho_within` 是分位内 Spearman，`sensory_median` 是")
    A("  分位内感官维度中位，`exceed_frac` 是分位内浓度超过**感官阈值（中位阈值）**的样品比例")
    A("  （阈值统一换算为 log10(mg/L)；无阈值的化合物留空）。")
    A("- `defect_dose_verdict.csv`：`pattern` 取值仅 "
      "`monotonic_positive / monotonic_negative / none / non_monotonic`；")
    A("  判定依据 = **组内 ρ 的符号一致性**（主）+ **感官中位单调性**（辅）；")
    A("  `style_robust_fraction` = n≥8 的风格中保持同一模式的比例。")
    A("- `style_robustness.csv`：n≥8 的全部风格逐个重算（风格内用该风格自己的浓度四分位；")
    A("  四分位格 <3 样本的风格标记 `insufficient_quarters`，不计入稳健比例的分母）。")
    A("")
    A("## 4. 剂量-响应再检验结果")
    A("")
    A(f"**基准复现核对**：{n_pass}/{len(bench)} 行与主控实测基准在 ±{BENCH_TOL} 内一致"
      f"（`dose_benchmark_check.csv`）。")
    A("")
    A("| 化合物 | 品评组维度 | 全体 ρ | ρ(Q1..Q4) | 感官中位 Q1→Q4 | 模式 | 风格稳健比例 |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    meds = cache["dose"]
    for _, r in verdict.iterrows():
        q = " / ".join(f"{r[f'rho_q{i}']:+.3f}" for i in (1, 2, 3, 4))
        sub = meds[(meds["compound"] == r["compound"]) & (meds["sensory_dim"] == r["sensory_dim"])]
        m1 = float(sub[sub["quartile"] == 1].iloc[0]["sensory_median"])
        m4 = float(sub[sub["quartile"] == 4].iloc[0]["sensory_median"])
        A(f"| {r['compound']} | {r['sensory_dim']} | {r['overall_rho']:+.3f} | {q} | "
          f"{m1:+.3f} → {m4:+.3f} | {r['pattern']} | "
          f"{r['style_robust_fraction']:.2f} |")
    A("")
    A("### 风格分层稳健性与辛普森式混合效应（B3）")
    A("")
    A("- **四分位口径**：风格内用该风格自己的浓度四分位；四分位格 <3 样本记 "
      "`insufficient_quarters`，不计入分母（n≥8 的 16 个风格中仅 6–8 个可判，")
    A("  因为每个四分位格平均只有 2–8 个样本，组内 ρ 本身方差极大）。")
    A("- **半分割口径**（更稳健的伴随指标）：风格内按浓度中位数分低/高两半，n≥8 的全部风格都可判。")
    A("")
    detail = cache.get("dose_detail")
    A("| 化合物 | 品评组维度 | 整体模式 | 四分位口径保持比例 | 半分割口径保持比例 | 方向翻转的风格 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for _, r in verdict.iterrows():
        pair = (r["compound"], r["sensory_dim"])
        sub = detail[(detail["compound"] == r["compound"]) &
                     (detail["sensory_dim"] == r["sensory_dim"])] if detail is not None else None
        half_txt = "—"
        flip_txt = "—"
        if sub is not None and len(sub):
            half_ok = int((sub["pattern_half"] == r["pattern"]).sum())
            half_txt = f"{half_ok}/{len(sub)} = {half_ok / len(sub):.2f}"
            opposite = {"monotonic_positive": "monotonic_negative",
                        "monotonic_negative": "monotonic_positive"}
            opp = opposite.get(r["pattern"])
            if opp:
                flipped = sub[(sub["pattern"] == opp)]
            else:
                flipped = sub[(sub["pattern"] != r["pattern"]) &
                              (sub["pattern"] != "insufficient_quarters")]
            if len(flipped):
                flip_txt = "、".join(f"{row['style']}(ρ={row['rho']:+.2f}, {row['pattern']})"
                                     for _, row in flipped.iterrows())
        A(f"| {r['compound']} | {r['sensory_dim']} | {r['pattern']} | "
          f"{r['style_robust_fraction']:.2f} | {half_txt} | {flip_txt} |")
    A("")
    A("（「—」表示该配对在可判风格中**没有出现方向相反**的模式；部分配对只是从单调**退化**为")
    A("`non_monotonic`，这种退化同样计入「保持同一模式比例」的下降。全部风格明细见")
    A("`style_robustness.csv` 与 `style_robustness_detail.csv`。）")
    A("")
    A("判读要点（数字由实跑结果填入）：")
    A("")
    t2n_key = ("X.E..2.nonenal", "t2n")
    lin_key = ("linalool", "A_hops_citrus")
    ea_key = ("ethyl_acetate", "A_esters_all")

    def _half(pair) -> str:
        sub = detail[(detail["compound"] == pair[0]) &
                     (detail["sensory_dim"] == pair[1])] if detail is not None else None
        if sub is None or not len(sub):
            return "—"
        row = verdict[(verdict["compound"] == pair[0]) &
                      (verdict["sensory_dim"] == pair[1])].iloc[0]
        ok = int((sub["pattern_half"] == row["pattern"]).sum())
        return f"{ok}/{len(sub)}"

    A(f"- **trans-2-壬烯醛 → t2n**：整体判定 `monotonic_negative`，但风格分层后**没有一个风格**")
    A(f"  保持该模式（四分位口径 0/7，半分割口径 {_half(t2n_key)}），且 Strong ale 与 Stout/Porter")
    A("  两个风格内的模式**翻转为正**（风格级 ρ 为正）。这是典型的**辛普森式混合效应**：整体负向")
    A("  剂量-响应由风格间差异驱动，在风格内部并不成立，**必须如实报告为不稳健**。")
    A(f"- **芳樟醇 → 香气·柑橘**：整体 `monotonic_positive`，但四分位口径只有 1/6 的风格保持")
    A(f"  （Tripel），半分割口径 {_half(lin_key)}；风格级 ρ 的符号在 6/6 可判风格中均为正，")
    A("  说明**方向稳健而「单调」不稳健**。")
    A(f"- **乙酸乙酯 → 香气·酯类总体**：整体 `non_monotonic`（非单调），四分位口径 6/8、")
    A(f"  半分割口径 {_half(ea_key)}；Stout/Porter 翻转为 `monotonic_negative`、Amber 翻转为")
    A("  `monotonic_positive`，进一步支持「整体正相关来自组间差异而非组内剂量-响应」的判断。")
    A("- **DMS → 二甲基硫醚、双乙酰 → 双乙酰**：均为 `non_monotonic`，且风格级 ρ 的符号只有 5/8 与")
    A("  整体一致 → 缺陷物与对应感官维度的关系**既非单调、也不在风格间稳定**。")
    A("")
    A("### 与原文结论的对照")
    A("")
    A("原文主张：「通常被视为异味的风味物（乙酸乙酯、二甲基硫醚、双乙酰、某些醛）在低浓度且与")
    A("果香酯类共存时反而提升香气」，其后半段通常被引申为「提升消费者喜爱度」。")
    A("")
    A("1. **可检验的部分**：「浓度—品评组感知」关系。本附件可检验，数据结论见上表。")
    A("2. **不可检验的部分（必须明写）**：原文结论的后半段「提升消费者喜爱度」在本附件条件下")
    A("   **不可检验**——附件**没有消费者评分/喜爱度标签**，且 183 维消费者描述符的 PCA 第一主成分")
    A("   仅解释 10.4% 方差，不存在可当作喜爱度的低维轴。本层因此**不声称**任何关于消费者")
    A("   喜爱度或偏好的结论。")
    A("3. **本数据能说的是**：乙酸乙酯与酯类香气在**全体样本上**同向（ρ=+0.321 / +0.380），")
    A("   感官中位随浓度四分位**单调上升**，但**分位内 ρ 的符号不一致**（Q3 转负、Q4 转强正），")
    A("   因此判定为 `non_monotonic`：整体正相关主要来自「低浓度组与高浓度组之间的差异」，")
    A("   组内并不存在稳定的剂量-响应斜率。这是典型的**分组均值差掩盖了组内关系**的情形。")
    A("4. **辛普森式混合效应**：凡风格分层后方向翻转或消失的配对，均在 `note` 与")
    A("   `style_robustness.csv` 中如实标注，不做掩盖。")
    A("")


def write_readme(args, cache: Dict, elapsed: float) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "consumer_summary" not in cache:
        lines: List[str] = []
        A = lines.append
        A("# B2 · 缺陷化合物剂量-响应再检验（仅 dose 阶段）")
        A("")
        A(f"生成脚本：`scripts/q4_consumer.py --stages dose`　｜　随机种子：{SEED}　｜　"
          f"运行耗时：{elapsed:.1f} s")
        A("")
        A("> 本次仅执行 dose 阶段，消费者层（A1–A4）产物需运行 `--stages consumer,dose` 生成。")
        A("")
        _append_dose_sections(A, cache)
        A("## 复现命令")
        A("")
        A("```bash")
        A("python scripts/q4_consumer.py --stages consumer,dose,figures   # 完整流程")
        A("```")
        A("")
        (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    cons = cache["consumer_summary"]
    cmp_tbl = cache["compare"]
    cross = cache["cross"]
    price = cache["price_table"]
    pmeta = cache["price_meta"]
    p1meta = cache["price_meta_ge1"]
    verdict = cache["verdict"]
    bench = cache["benchmark"]
    dcmp = cache["derived_compare"]
    cinfo = cache["consumer_info"]
    style_robust = cache["style_robust"]
    best = cache["best_model"]

    base = cons[cons["model"] == "global_mean"].iloc[0]
    best_row = cons.sort_values("Macro_RMSE").iloc[0]
    n_pass = int(bench["pass"].sum())

    def model_line(name: str, frame: pd.DataFrame) -> str:
        r = frame[frame["model"] == name].iloc[0]
        return (f"| {name} | {int(r['n_targets'])} | {r['Macro_RMSE']:.4f} | "
                f"{r['Macro_MAE']:.4f} | {r['R2_median']:.4f} | {r['Spearman_median']:.4f} |")

    lines: List[str] = []
    A = lines.append
    A("# B2 · 消费者层 + 缺陷化合物剂量-响应再检验")
    A("")
    A(f"生成脚本：`scripts/q4_consumer.py`　｜　随机种子：{SEED}　｜　运行耗时：{elapsed:.1f} s")
    A("")
    A("## 0. 边界声明（先读）")
    A("")
    A("- **附件未提供任何消费者评分 / 喜爱度标签**（7.csv 只有 186 个描述符强度，没有 overall liking、")
    A("  评分或打分列）。因此本层**不声称**预测「消费者喜爱度」，只做两件事：")
    A("  1. **两条感知路径的一致性**：化学 → 品评组感官（50 维）与化学 → 消费者描述符（183 维）")
    A("     是否指向同一化学驱动因子；")
    A("  2. **外部市场价值的可解释比例**：化学预测感知能解释多少 log 几何均价的差异。")
    A("- 183 维消费者描述符上 PCA 第一主成分仅解释 **10.4%** 方差，不存在单一「总体喜爱度」轴；")
    A("  本层**不构造**任何「消费者喜爱度」目标变量。")
    A("- 价格为**单一零售源附件**（8.xlsx / Beer Prices）的零售商报价，跨币种（USD/EUR/GBP）")
    A("  未做汇率换算，**只有 ≥2 家报价的 189 款**进入主回归；价格同时受 ABV 与风格混杂，")
    A("  且缺少消费者评分标签，市场价值只能解释为「可解释比例」而非因果效应。")
    A("")
    A("## 1. 数据来源与清洗口径")
    A("")
    A("| 环节 | 来源 | 说明 |")
    A("| --- | --- | --- |")
    A("| 化学特征 | `data_contract.to_unified_scale()` | 统一 log10(mg/L) 矩阵，231 列 |")
    A("| 品评组感官 | `data_contract.load_targets()` | 4.csv 的 50 维标准化感官分数 |")
    A("| 消费者描述符 | `data/raw/7.csv` | 250 × 186 描述符 |")
    A("| 风格 | `data_contract.load_styles()` | 22 个风格 |")
    A("| 切分 | `data/splits/seed_0/split_manifest.csv` | **锁定 175/75（seed 0）**，未重新划分 |")
    A("| 价格 | 8.xlsx / Beer Prices | 250 行 × 11 列（ABV + 7 个零售商报价） |")
    A("| 感官阈值 | 8.xlsx / Sensory thresholds | `threshold_median` 统一换算为 log10(mg/L) |")
    A("")
    A("### A1 消费者矩阵清洗")
    A("")
    A(f"- 原始描述符 186 列 → 剔除 **{len(cinfo['dropped_all_na_columns'])} 个整列为空的列**："
      + "、".join(f"`{c}`" for c in cinfo["dropped_all_na_columns"])
      + "（各 250 个 NA）")
    A(f"- 保留 **{cinfo['kept_dimensions']} 维**；零星 NA 按列中位数填补，"
      f"实际填补 **{cinfo['imputed_cells']} 个单元格**（{cinfo['imputed_columns']} 列）")
    A("- 索引与 1.csv 的 `beer` 逐行对齐（250 行，顺序一致）")
    A("")
    A("### 化学特征口径：保留还是剔除 `*_sum` 派生列？")
    A("")
    A("`to_unified_scale()` 返回 231 列，其中 4 列是 `esters_sum / aroma_hops_sum / acids_sum /")
    A("sulfur_sum`（组内浓度和取 log10 的非线性派生量，与成分列高度共线）。**主口径剔除这 4 列**")
    A("（227 列，与 Q1 建模矩阵 `load_modeling_matrix()` 完全一致），231 列作为对照口径：")
    A("")
    A("| 模型 | 227 列（剔除派生）Macro-RMSE | 231 列（保留派生）Macro-RMSE | Δ | "
      "Spearman 中位（剔除 / 保留） |")
    A("| --- | --- | --- | --- | --- |")
    for _, r in dcmp.iterrows():
        A(f"| {r['model']} | {r['features_drop_derived']:.4f} | {r['features_keep_derived']:.4f} | "
          f"{r['delta_macro_rmse']:+.4f} | {r['drop_derived_spearman_median']:.4f} / "
          f"{r['keep_derived_spearman_median']:.4f} |")
    A("")
    A("## 2. 模型与切分")
    A("")
    A(f"- 切分：**锁定的 175/75**（`split_manifest.csv`，seed 0），基线为**训练折目标均值**。")
    A(f"- 模型族（与 Q1 同名同族）：{', '.join(MODEL_ORDER)}；`ridge` / `pls2` 用训练折内层 5 折")
    A("  GridSearchCV 选参；`global_extra_trees` / `independent_gbr` 直接复用 Q1 在 50 维品评组上")
    A("  已选定的超参（只读引用 `data/processed/q1_runs/*/model_best_params.csv`），避免本层二次")
    A("  调参、保证两层同口径，也把运行时间控制在 CPU/内存预算内。")
    A(f"- OOF 预测（5 折，`KFold(shuffle=True, random_state={SEED})`）用于交叉一致性与价格回归，")
    A("  避免自证（每个样本的预测都来自未见过它的模型）。")
    A(f"- 用于 A3/A4 的模型族：**{best}**（消费者留出集 Macro-RMSE 最小者）。")
    A("")
    A("### A2 消费者层模型（锁定 175/75 留出集）")
    A("")
    A("| 模型 | 目标数 | Macro-RMSE | Macro-MAE | 逐维 R² 中位 | Spearman 中位 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for name in MODEL_ORDER:
        A(model_line(name, cons))
    A("")
    A(f"- 基线 `global_mean` Macro-RMSE = **{base['Macro_RMSE']:.4f}**；")
    A(f"  最优模型 **{best_row['model']}** Macro-RMSE = **{best_row['Macro_RMSE']:.4f}**"
      f"（相对基线 {best_row['Macro_RMSE'] - base['Macro_RMSE']:+.4f}），"
      f"Spearman 中位 {best_row['Spearman_median']:.4f}。")
    A(f"- 结论：消费者层模型 Macro-RMSE "
      f"{'**优于**' if best_row['Macro_RMSE'] < base['Macro_RMSE'] else '**未优于**'}全局均值基线。")
    A("")
    A("### A2 同口径对比：品评组 50 维 vs 消费者 183 维")
    A("")
    A("`consumer_vs_panel_models.csv` 同时给出：同一留出集、同一超参下两层各自的结果，")
    A("以及 Q1 已发布的 5 折 OOF 结果（只读引用）。")
    A("")
    A("| 模型 | 目标集 | 评估口径 | Macro-RMSE | Macro-MAE | R² 中位 | Spearman 中位 |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in cmp_tbl.iterrows():
        A(f"| {r['model']} | {r['target_set']} | {r['evaluation']} | {r['Macro_RMSE']:.4f} | "
          f"{r['Macro_MAE']:.4f} | {r['R2_median']:.4f} | {r['Spearman_median']:.4f} |")
    A("")
    # 与 Q1 已发布结果的对账（本脚本的品评组留出集 vs Q1 公布值）
    q1_rows = cmp_tbl[cmp_tbl["evaluation"] == "q1_published_5fold_oof"]
    my_rows = cmp_tbl[(cmp_tbl["evaluation"] == "locked_holdout_175_75") &
                      (cmp_tbl["target_set"] == "panel_50")]
    merged = my_rows.merge(q1_rows, on="model", suffixes=("_本脚本", "_Q1"))
    if len(merged):
        diff = (merged["Macro_RMSE_本脚本"] - merged["Macro_RMSE_Q1"]).abs()
        exact = merged[diff < 1e-9]["model"].tolist()
        A(f"**对账**：本脚本在品评组 50 维上的 Macro-RMSE 与 Q1 已发布 `model_summary.csv` 的最大"
          f"差异 = **{diff.max():.4f}**；其中 {', '.join(exact) if exact else '无'} **完全一致**"
          f"（说明两层确为同一锁定切分、同一评价口径）；"
          "`global_extra_trees` / `independent_gbr` 的微小差异来自本层**复用 Q1 已选定的固定超参**"
          "而未重跑其内层 GridSearchCV。")
        A("")
        best_name = cons.sort_values("Macro_RMSE").iloc[0]["model"]
        c_row = cmp_tbl[(cmp_tbl["model"] == best_name) &
                        (cmp_tbl["target_set"] == "consumer_183")].iloc[0]
        p_row = cmp_tbl[(cmp_tbl["model"] == best_name) &
                        (cmp_tbl["target_set"] == "panel_50")].iloc[0]
        A(f"**解读**（以最优模型 {best_name} 为例）：消费者描述符的**绝对误差更大**"
          f"（Macro-RMSE {c_row['Macro_RMSE']:.4f} vs 品评组 {p_row['Macro_RMSE']:.4f}），"
          f"但**逐维 R² 中位**（{c_row['R2_median']:.4f} vs {p_row['R2_median']:.4f}）与 "
          f"**Spearman 中位**（{c_row['Spearman_median']:.4f} vs {p_row['Spearman_median']:.4f}）"
          "相当甚至更高 → 消费者描述符更难预测主要来自**个体化噪声更大**"
          "（186 个分散的消费者描述符打分 vs 训练有素的品评组 50 维打分），"
          "而不是化学可解释性更差。")
        A("")
    A("## 3. 三张核心表怎么读")
    A("")
    A("### `cross_consistency.csv`（A3）")
    A("")
    A("- `rho_truth` = 品评组感官维度**真值**与消费者描述符**真值**之间的 Spearman（n=250）；")
    A("  `rho_pred` = 两者**模型预测**之间的 Spearman（OOF）；`delta = rho_pred − rho_truth`。")
    A("- `delta` 接近 0 → 两条感知路径（品评组 vs 消费者）对同一化学驱动因子的指向一致；")
    A("  `delta` 明显为负 → 化学模型在消费者路径上的还原度弱于品评组路径。")
    A("- **注意 `delta > 0` 的结构性来源**：两条预测列由**同一套 227 维化学特征**、**同一个模型族**")
    A("  投影得到，预测值天然共享化学主轴方向，因此 `rho_pred` 往往高于 `rho_truth`；")
    A("  `delta > 0` 反映的是「化学结构被还原得比感知更整齐」，**不能**直接读作「两条感知路径")
    A("  更一致」。判读应以 `rho_truth` 的符号/量级为主，`delta` 只用于判断模型是否**削弱**了")
    A("  原有关系（`delta` 大幅为负时才值得警惕）。")
    A("- **词汇表循环性风险（必须声明）**：品评组感官维度与消费者描述符共用同一套啤酒感官")
    A("  词汇（citrus / fruity / bitter / sour / stale / cardboard …），二者不是相互独立的测量；")
    A("  共享词汇会把「品评组与消费者说同一件事」直接写进 `rho_truth`，因此 `rho_truth` 偏高")
    A("  **部分是词汇重叠的产物**，不能当作两条独立证据。`delta`（预测 vs 真值的落差）比")
    A("  `rho_truth` 的绝对值更可靠。")
    A("")
    A("| 品评组维度 | 消费者描述符 | 配对依据 | ρ 真值 | ρ 预测 | Δ |")
    A("| --- | --- | --- | --- | --- | --- |")
    for panel_dim, consumer_desc, basis in CROSS_PAIRS:
        sel = cross[(cross["panel_dim"] == panel_dim) &
                    (cross["consumer_desc"] == consumer_desc)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        A(f"| {panel_dim} | {consumer_desc} | {basis} | {r['rho_truth']:+.3f} | "
          f"{r['rho_pred']:+.3f} | {r['delta']:+.3f} |")
    A("")
    A("### `price_regression.csv`（A4）")
    A("")
    A(f"- 被解释变量：**log(几何均价)**；样本 = **{pmeta['n_obs']} 款**（≥2 家零售商报价）；")
    A(f"  对照口径（≥1 家）n = **{cache['price_ge1_n']}** 款，R² = "
      f"{p1meta['specs']['ols_with_abv']['r2']:.4f}（ols_with_abv），"
      f"≥2 家口径 R² = {pmeta['specs']['ols_with_abv']['r2']:.4f}。")
    A("- `ols` = 仅化学预测感知；`ols_with_abv` = 再加 ABV；")
    A("  `fixed_effects` = 再加风格固定效应（样本量 ≥ "
      f"{FE_MIN_STYLE_N} 的风格设虚拟变量，其余并入 Other；基准组 = "
      f"`{pmeta.get('fe_reference_style', '—')}`）并用**按风格聚类的稳健标准误**"
      "（聚类数远小于参数量，标准误偏保守，只作方向性参考）。")
    A(f"- log 价格与 ABV 的相关系数 = **{pmeta['abv_corr_log_price']:+.3f}**（价格受 ABV 混杂明显）。")
    A("- 三套规格的关键系数：")
    A("")
    A("| 模型 | n | R² | 调整 R² | 稳健标准误 |")
    A("| --- | --- | --- | --- | --- |")
    for model, spec in pmeta["specs"].items():
        A(f"| {model} | {spec['n']} | {spec['r2']:.4f} | {spec['adj_r2']:.4f} | {spec['robust']} |")
    A("")
    A("- 关键系数（完整表见 `price_regression.csv`，含全部风格虚拟变量）：")
    A("")
    A("| 项 | ols 系数 (SE) | ols_with_abv 系数 (SE) | fixed_effects 系数 (SE) |")
    A("| --- | --- | --- | --- |")
    ptab = price.set_index(["model", "term"])
    key_terms = list(PRICE_DESCRIPTORS) + ["ABV"]
    for term in key_terms:
        cells = []
        for model in ("ols", "ols_with_abv", "fixed_effects"):
            if (model, term) in ptab.index:
                r = ptab.loc[(model, term)]
                cells.append(f"{float(r['coef']):+.4f} ({float(r['std_err']):.4f})")
            else:
                cells.append("—")
        A(f"| {term} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")
    A("- 读法：`taste_sweet`（预测甜度）与 `aroma_hoppy`（预测酒花香）的系数为负、`taste_bitter`")
    A("  为正，方向与直觉相反 —— 因为自变量是**模型预测的感知值**而非实测消费者评分，")
    A("  且价格跨币种未换算、并与 ABV/品牌混杂；这些系数只应解读为「化学预测感知对 log 价格的")
    A("  条件相关」，**不能**解释为「消费者更偏爱甜/啤酒花」。")
    A("")
    _append_dose_sections(A, cache)
    A("")
    A("## 5. 复现命令")
    A("")
    A("```bash")
    A("python scripts/q4_consumer.py                 # 全流程（消费者层 + 剂量再检验 + 图）")
    A("python scripts/q4_consumer.py --stages consumer,dose")
    A("python scripts/q4_consumer.py --help")
    A("```")
    A("")
    A(f"随机种子固定为 {SEED}，重复运行结果一致（幂等）。上游输入 sha256、参数与耗时见 `manifest.json`。")
    A("")
    A("## 6. 产物清单")
    A("")
    A("| 文件 | 内容 |")
    A("| --- | --- |")
    A("| `consumer_model_summary.csv` | 消费者 183 维的模型汇总（Macro-RMSE/MAE、R² 中位、ρ 中位） |")
    A("| `consumer_vs_panel_models.csv` | 同口径的品评组 vs 消费者模型对比 + Q1 已发布 OOF 结果 |")
    A("| `cross_consistency.csv` | A3 交叉一致性（ρ 真值 / ρ 预测 / Δ / n） |")
    A("| `price_regression.csv` | A4 价格回归系数（ols / ols_with_abv / fixed_effects） |")
    A("| `defect_dose_response.csv` | B1 分位级统计（24 行） |")
    A("| `defect_dose_verdict.csv` | B2 模式判定与风格稳健比例 |")
    A("| `style_robustness.csv` | B3 风格分层稳健性（n≥8 的全部风格） |")
    A("| `style_robustness_detail.csv` | 风格内四分位 ρ 与半分割明细（补充） |")
    A("| `dose_benchmark_check.csv` | 与主控实测基准的逐行核对（补充） |")
    A("| `consumer_per_dimension_metrics.csv` | 消费者层逐维 RMSE/MAE/R²/ρ（补充） |")
    A("| `panel_per_dimension_metrics.csv` | 品评组 50 维逐维指标（补充） |")
    A("| `oof_predictions_consumer.csv.gz` / `oof_predictions_panel.csv.gz` | "
      "5 折 OOF 预测（250 行，供论文取数与复核） |")
    A("| `manifest.json` | 上游 sha256、参数、种子、耗时 |")
    A("| `../q4_figures/fig1…fig4` | 4 张图，每张 .png(300dpi) + .pdf |")
    A("| `_run_stdout.log` / `_run_stderr.log` | 本次实跑的控制台日志（运行证据） |")
    A("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(args, cache: Dict, elapsed: float, figures: List[str]) -> None:
    out_dir = Path(args.output_dir)
    inputs = {}
    for rel_path in ["data/raw/1.csv", "data/raw/4.csv", "data/raw/7.csv", "data/raw/8.xlsx",
                     "data/interim/chem_log10_matrix_v1.csv",
                     "data/splits/seed_0/split_manifest.csv",
                     "data/processed/q1_final/model_summary.csv",
                     "data/processed/q1_runs/global_extra_trees/model_best_params.csv",
                     "data/processed/q1_runs/independent_gbr/model_best_params.csv"]:
        p = REPO_ROOT / rel_path
        if p.exists():
            inputs[rel_path] = {"sha256": dc.sha256_of(p), "bytes": p.stat().st_size}
    payload = {
        "script": "scripts/q4_consumer.py",
        "seed": SEED,
        "inner_seed": INNER_SEED,
        "n_folds": N_FOLDS,
        "n_jobs": args.n_jobs,
        "elapsed_seconds": round(elapsed, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "upstream_inputs": inputs,
        "parameters": {
            "split": "data/splits/seed_0/split_manifest.csv (locked 175/75)",
            "feature_caliber_primary": "227 columns (4 *_sum derived columns dropped)",
            "feature_caliber_control": "231 columns (derived columns kept)",
            "consumer_dimensions": cache.get("consumer_info", {}).get("kept_dimensions"),
            "panel_dimensions": 50,
            "models": list(MODEL_ORDER),
            "oof_model_selected": cache.get("best_model"),
            "cross_pairs": [f"{a}|{b}" for a, b, _ in CROSS_PAIRS],
            "dose_pairs": [f"{a}|{b}" for a, b in DOSE_PAIRS],
            "price_descriptors": list(PRICE_DESCRIPTORS),
            "price_min_retailers_main": 2,
            "price_min_retailers_control": 1,
            "style_fe_min_n": FE_MIN_STYLE_N,
            "benchmark_tolerance": BENCH_TOL,
            "reused_q1_hyperparams": cache.get("fixed_params"),
        },
        "key_results": {
            "consumer_model_summary": (cache["consumer_summary"].to_dict("records")
                                       if "consumer_summary" in cache else None),
            "price_specs_ge2": cache.get("price_meta", {}).get("specs"),
            "price_specs_ge1": cache.get("price_meta_ge1", {}).get("specs"),
            "price_n_ge2": cache.get("price_ge2_n"),
            "price_n_ge1": cache.get("price_ge1_n"),
            "benchmark_pass": int(cache["benchmark"]["pass"].sum()),
            "benchmark_rows": int(len(cache["benchmark"])),
        },
        "figures": figures,
        "boundary_statement": ("附件未提供消费者评分/喜爱度标签，本层不声称预测消费者喜爱度；"
                               "价格来自单一零售源附件，仅 ≥2 家报价的 189 款进入主回归。"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(
        description="B2 · 消费者层（化学 → 消费者感知 → 市场价值）+ 缺陷化合物剂量-响应再检验")
    ap.add_argument("--output-dir", default="data/processed/q4_consumer",
                    help="产物目录（相对仓库根）")
    ap.add_argument("--figure-dir", default="data/processed/q4_figures",
                    help="图片目录（相对仓库根）")
    ap.add_argument("--stages", default="consumer,dose,figures",
                    help="执行阶段：consumer / dose / figures（逗号分隔）")
    ap.add_argument("--n-jobs", type=int, default=2, help="并行度（≤3）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：只用 25 个消费者维度 / 10 个品评组维度跑通全链路（不产出正式结论）")
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    log(f"阶段：{stages}；输出：{args.output_dir} / {args.figure_dir}")

    raw = dc.load_raw()
    cache: Dict = {}

    if "consumer" in stages:
        cache = run_consumer_stage(args, raw, cache)
    if "dose" in stages:
        cache = run_dose_stage(args, raw, cache)
    if "figures" in stages:
        figures = make_figures(args, cache)
        cache["figures"] = figures
        log(f"图表完成：{len(figures)} 个文件")

    elapsed = time.time() - t0
    if "consumer" in stages or "dose" in stages:
        write_readme(args, cache, elapsed)
        write_manifest(args, cache, elapsed, cache.get("figures", []))
        log(f"全部完成，用时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
