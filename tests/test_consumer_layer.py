# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：断言的列名与主控下发给 B2 的产物契约逐项对齐；183 维、189 款、
# 六行剂量配对等计数均来自本轮对附件的实测清点结果。
"""消费者层与剂量再检验的产物契约测试：守住「不虚构喜爱度」「口径可复核」两条底线。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

Q4 = ROOT / "data" / "processed" / "q4_consumer"

#: 7.csv 中三列整列为空的 palate 描述符（实测各 250 个 NA）
EMPTY_PALATE_COLUMNS = ["palate_acetaldehyde", "palate_acetate", "palate_dentist"]

#: 剂量再检验的配对（化合物列 → 品评组感官维度）
DOSE_PAIRS = [
    ("ethyl_acetate", "A_esters_all"),
    ("ethyl_acetate", "F_esters_all"),
    ("linalool", "A_hops_citrus"),
    ("DMS", "dms"),
    ("X2.3.butanedione", "diacetyl"),
    ("X.E..2.nonenal", "t2n"),
]

PATTERNS = {"monotonic_positive", "monotonic_negative", "none", "non_monotonic"}


def _csv(name: str) -> pd.DataFrame:
    path = Q4 / name
    assert path.exists(), f"缺少产物 {path}"
    return pd.read_csv(path, encoding="utf-8-sig")


# --------------------------------------------------------------- 数据层事实


def test_consumer_descriptors_have_183_valid_dimensions():
    """183 = 186 描述符 − 3 个整列为空的 palate 列。"""
    from data_contract import load_raw
    raw = load_raw()
    forum = raw.forum
    desc = [c for c in forum.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
    for col in EMPTY_PALATE_COLUMNS:
        assert col in desc, f"预期存在整列为空的描述符 {col}"
        assert forum[col].isna().all(), f"{col} 并非整列为空"
    valid = [c for c in desc if not forum[col].isna().all()]
    assert len(desc) == 186, f"描述符总数应为 186，实得 {len(desc)}"
    assert len(valid) == 183, f"有效描述符应为 183，实得 {len(valid)}"


def test_price_coverage_189_beers_with_two_retailers():
    """覆盖 ≥2 家零售商的 189 款可算几何均价。"""
    from data_contract import load_raw
    raw = load_raw()
    bp = raw.beer_prices
    price_cols = [c for c in bp.columns if str(c).startswith("Price_")]
    assert len(price_cols) == 7, f"价格列应为 7 个，实得 {len(price_cols)}"
    num = bp[price_cols].apply(pd.to_numeric, errors="coerce")
    covered = (num.notna().sum(axis=1) >= 2).sum()
    assert covered == 189, f"覆盖≥2 家的应为 189 款，实得 {covered}"


# --------------------------------------------------------------- 产物契约


def test_consumer_model_summary_present_and_beats_baseline():
    df = _csv("consumer_model_summary.csv")
    for col in ["model", "Macro_RMSE", "Macro_MAE", "R2_median", "Spearman_median"]:
        assert col in df.columns, f"consumer_model_summary 缺列 {col}"
    assert len(df) >= 3, "至少应有基线与两个模型"
    base = df[df["model"].astype(str).str.contains("mean", case=False)]
    assert not base.empty, "缺少全局均值基线行"
    best = df["Macro_RMSE"].min()
    assert best < base["Macro_RMSE"].min(), (
        f"最优模型 Macro-RMSE={best:.4f} 未优于基线 {base['Macro_RMSE'].min():.4f}")


def test_cross_consistency_at_least_10_pairs():
    df = _csv("cross_consistency.csv")
    for col in ["panel_dim", "consumer_desc", "rho_truth", "rho_pred", "delta"]:
        assert col in df.columns, f"cross_consistency 缺列 {col}"
    assert len(df) >= 10, f"语义配对仅 {len(df)} 组，要求 ≥10 组"
    for c in ["rho_truth", "rho_pred", "delta"]:
        v = pd.to_numeric(df[c], errors="coerce")
        assert v.notna().all(), f"{c} 存在缺失"
        assert v.between(-1, 1).all(), f"{c} 超出相关系数范围"


def test_price_regression_reports_abv_control_and_limits():
    df = _csv("price_regression.csv")
    for col in ["term", "coef", "model", "n"]:
        assert col in df.columns, f"price_regression 缺列 {col}"
    models = set(df["model"].astype(str))
    assert len(models) >= 2, "至少应有「控制 ABV 前 / 后」两个模型"
    n_vals = pd.to_numeric(df["n"], errors="coerce").dropna()
    assert n_vals.max() <= 189, f"价格回归样本量不应超过 189，实得 {n_vals.max()}"


def test_dose_response_covers_six_pairs():
    df = _csv("defect_dose_response.csv")
    for col in ["compound", "sensory_dim", "quartile", "n", "rho_within", "sensory_median"]:
        assert col in df.columns, f"defect_dose_response 缺列 {col}"
    got = set(zip(df["compound"], df["sensory_dim"]))
    for pair in DOSE_PAIRS:
        assert pair in got, f"缺少剂量配对 {pair}"
    q = pd.to_numeric(df["quartile"], errors="coerce")
    assert set(q.unique()) == {1, 2, 3, 4}, f"浓度分位应为 1..4，实得 {sorted(q.unique())}"


def test_dose_verdict_patterns_valid_and_reproduces_baseline():
    df = _csv("defect_dose_verdict.csv")
    for col in ["compound", "sensory_dim", "pattern", "overall_rho", "style_robust_fraction"]:
        assert col in df.columns, f"defect_dose_verdict 缺列 {col}"
    assert set(df["pattern"]) <= PATTERNS, f"pattern 取值非法：{set(df['pattern']) - PATTERNS}"
    # 复现主控实测的基准：乙酸乙酯→A_esters_all 全体 rho ≈ +0.321
    row = df[(df["compound"] == "ethyl_acetate") & (df["sensory_dim"] == "A_esters_all")]
    assert len(row) == 1, "缺少 ethyl_acetate→A_esters_all 判定行"
    rho = float(row.iloc[0]["overall_rho"])
    assert abs(rho - 0.321) < 0.02, f"全体 rho={rho} 与基准 0.321 不符（口径被改动？）"
    # 双乙酰的反向关系是本轮的核心发现，必须复现
    row2 = df[(df["compound"] == "X2.3.butanedione") & (df["sensory_dim"] == "diacetyl")]
    assert len(row2) == 1
    assert float(row2.iloc[0]["overall_rho"]) < 0, "双乙酰的负相关未复现"


def test_style_robustness_covers_large_styles():
    df = _csv("style_robustness.csv")
    for col in ["compound", "sensory_dim", "style", "n", "rho"]:
        assert col in df.columns, f"style_robustness 缺列 {col}"
    n = pd.to_numeric(df["n"], errors="coerce")
    assert (n >= 8).all(), "风格分层稳健性只应对 n≥8 的风格计算"
    assert df["style"].nunique() >= 10, f"覆盖的风格数偏少：{df['style'].nunique()}"


def test_readme_states_boundary_declarations():
    readme = Q4 / "README.md"
    assert readme.exists(), "缺少 q4_consumer/README.md"
    text = readme.read_text(encoding="utf-8")
    assert "喜爱度" in text, "README 必须声明不声称预测消费者喜爱度"
    assert ("单一" in text and "零售" in text) or "189" in text, \
        "README 必须声明价格数据的覆盖与来源限制"
    for bad in ["F:\\cs\\mothercup", "C:\\Users\\"]:
        assert bad not in text, f"README 泄漏本地路径：{bad}"
