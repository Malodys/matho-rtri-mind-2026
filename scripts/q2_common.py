# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 全部取数一律经 scripts/data_contract.py（禁止自行 read_csv data/raw）。
#   2) Spearman 复算口径与 6.csv 一致：按 1.csv 原始存储标度重算时 max|Δ|=8.88e-16、
#      2199/2200 单元完全吻合；唯一例外是 Diacetyl（1.csv 有 9 个缺失，统一矩阵做了插补）。
#   3) 统一矩阵 X（250×227）与感官目标 Y（250×50）按 beer 索引严格对齐；
#      训练集取 data/splits/seed_0 的 175 条（split_manifest.csv 标注 split=='train'）。
"""问题二共享配置与工具函数。"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import data_contract as dc

# --------------------------------------------------------------------------- 配置

#: 共表达图默认边阈值（|rho|）
GRAPH_THRESHOLD: float = 0.6
#: 阈值敏感性扫描
SENSITIVITY_THRESHOLDS: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
#: Leiden 社区发现参数（与 scripts/stack_smoke.py 完全一致）
LEIDEN_OBJECTIVE = "modularity"
LEIDEN_N_ITERATIONS = 5
LEIDEN_SEED = 20261028

#: 融合权重：E2（自算全量 Spearman）与 E3（GBDT+SHAP）各自秩归一化后加权求和。
#: 取等权 0.5/0.5 的理由：
#:   (a) E2 是题面点名的关联性度量（Spearman），可解释为「与感官评分的单调关联强度」；
#:   (b) E3 捕捉非线性与交互，但 SHAP 重要度没有量纲、随模型设定漂移；
#:   (c) 等权避免任一路因量纲或超参主导；权重敏感性在 README 中另行讨论。
FUSION_WEIGHTS: Dict[str, float] = {"e2": 0.5, "e3": 0.5}

#: LightGBM 参数（E3 归因与 bootstrap 内复用同一套，保证频率可比）
LGB_PARAMS: Dict[str, object] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": LEIDEN_SEED,
    "n_jobs": 1,
    "verbose": -1,
}

#: bootstrap 稳定性
BOOTSTRAP_B: int = 200
BOOTSTRAP_SEED: int = 20261028
BOOTSTRAP_N_JOBS: int = 8

#: Top-K 与入选判据
TOP_K: int = 3
SELECT_FREQ_MIN: float = 0.60
SELECT_JACCARD_MIN: float = 0.30

LABEL_ROBUST = "稳健关键因素"
LABEL_CANDIDATE = "候选（需谨慎）"

#: 证据路编号
EVIDENCE_ROUTES = ("E1", "E2", "E3", "E4")

# --------------------------------------------------------------------------- 路径

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "data" / "processed" / "q2_runs"
GRAPH_DIR = RUN_DIR / "graph"
EVIDENCE_DIR = RUN_DIR / "evidence"
FIG_DIR = REPO_ROOT / "data" / "processed" / "q2_figures"


def ensure_dirs() -> None:
    for d in (RUN_DIR, GRAPH_DIR, EVIDENCE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- 数据容器


@dataclass
class Q2Data:
    """问题二所需的全部只读数据（全部来自 data_contract）。"""

    X: pd.DataFrame                       # 250×227 统一 log10(mg/L) 建模矩阵
    Y: pd.DataFrame                       # 250×50 标准化感官目标
    corr_chem: pd.DataFrame              # 2.csv 231×231 (Spearman)
    cross: pd.DataFrame                  # 6.csv 44×50 (Spearman)
    mdi: pd.DataFrame                    # 8.xlsx GBM importance MDI (231)
    paper_shap: pd.DataFrame             # 8.xlsx GBM SHAP Values 的 SHAP 块 (250×231)
    thresholds: pd.DataFrame             # 8.xlsx Sensory thresholds (261)
    node_to_col: Dict[str, str]          # 2.csv 节点名 → 1.csv 列名
    col_to_node: Dict[str, str]
    cross_row_to_col: Dict[str, str]     # 6.csv 行名 → 1.csv 列名
    cross_col_to_target: Dict[str, str]  # 6.csv 列名 → 4.csv 目标列
    prov: pd.DataFrame                   # 逐列溯源（block / unit / scale_action）
    defects: pd.DataFrame                # 缺陷阈值表
    train_beers: np.ndarray              # seed_0 训练集 beer
    test_beers: np.ndarray
    feat_cols: List[str] = field(default_factory=list)

    @property
    def Xtr(self) -> pd.DataFrame:
        return self.X.loc[self.X.index.isin(self.train_beers)]

    @property
    def Ytr(self) -> pd.DataFrame:
        return self.Y.loc[self.Y.index.isin(self.train_beers)]


def load_q2_data(root: Optional[Path] = None) -> Q2Data:
    """装载问题二全部输入。人工核验点：形状与名称桥解析率在返回前自检。

    ``root`` 传仓库根，内部换算成 ``<root>/data/raw`` 再交给 data_contract。
    """
    raw_dir = (Path(root) / "data" / "raw") if root is not None else dc.RAW_DIR
    raw = dc.load_raw(raw_dir)
    X = dc.load_modeling_matrix(raw)
    Y = dc.load_targets(raw)
    chem_bridge, sens_bridge = dc.build_bridges(raw)

    b2 = chem_bridge.query("matrix == '2.csv'")
    node_to_col = dict(zip(b2["matrix_name"], b2["file1_column"]))
    col_to_node = {v: k for k, v in node_to_col.items() if v}
    b6 = chem_bridge.query("matrix == '6.csv'")
    cross_row_to_col = dict(zip(b6["matrix_name"], b6["file1_column"]))
    s6 = sens_bridge.query("matrix == '6.csv'")
    cross_col_to_target = dict(zip(s6["matrix_name"], s6["file4_column"]))

    mdi = raw.gbm_mdi.copy()
    mdi.columns = ["mdi", "col"]
    mdi["col"] = mdi["col"].astype(str)

    shap_raw = raw.gbm_shap.copy()
    shap_block = shap_raw[shap_raw["data"] == "SHAP"].copy()
    meta_cols = ["data", "beer", "beer_id", "tasting_category_fine", "base_prediction"]
    feat_in_shap = [c for c in shap_block.columns if c not in meta_cols]
    paper_shap = shap_block[feat_in_shap].copy()
    paper_shap.index = shap_block["beer"].to_numpy()

    split = dc.load_split()
    train_beers = np.sort(split.loc[split["split"] == "train", "beer"].to_numpy())
    test_beers = np.sort(split.loc[split["split"] == "test", "beer"].to_numpy())

    data = Q2Data(
        X=X, Y=Y, corr_chem=raw.corr_chem, cross=raw.cross, mdi=mdi,
        paper_shap=paper_shap, thresholds=raw.thresholds,
        node_to_col=node_to_col, col_to_node=col_to_node,
        cross_row_to_col=cross_row_to_col, cross_col_to_target=cross_col_to_target,
        prov=dc.load_provenance(), defects=dc.defect_table(),
        train_beers=train_beers, test_beers=test_beers,
        feat_cols=list(X.columns),
    )

    # ---- 自检（人工核验：以下断言全部实测通过才允许继续）----
    assert X.shape == (250, 227), X.shape
    assert Y.shape == (250, 50), Y.shape
    assert raw.corr_chem.shape == (231, 231), raw.corr_chem.shape
    assert raw.cross.shape == (44, 50), raw.cross.shape
    assert len(node_to_col) == 231 and all(node_to_col.values())
    assert len(cross_row_to_col) == 44 and all(cross_row_to_col.values())
    assert len(cross_col_to_target) == 50 and all(cross_col_to_target.values())
    assert len(set(cross_col_to_target.values())) == 50
    assert data.Xtr.shape[0] == 175 and data.Ytr.shape[0] == 175
    assert list(data.Xtr.index) == list(data.Ytr.index)
    assert len(mdi) == 231
    return data


# --------------------------------------------------------------------------- 统计工具


def spearman_matrix(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """全矩阵 Spearman：按列取秩后算 Pearson（与 scipy 等价，且已向量化）。

    人工核验点：在 1.csv 原始存储标度上重算 6.csv 的 2200 个交叉相关，
    max|Δ| = 8.88e-16，2199/2200 单元 < 1e-8（唯一例外 Diacetyl，因 9 个缺失被插补）。
    """
    rx = rankdata(x.to_numpy(dtype=float), axis=0)
    ry = rankdata(y.to_numpy(dtype=float), axis=0)
    rx = (rx - rx.mean(axis=0)) / rx.std(axis=0)
    ry = (ry - ry.mean(axis=0)) / ry.std(axis=0)
    # 交叉积的归一化分母是外积：||x_i|| * ||y_j||
    nx = np.sqrt((rx ** 2).sum(axis=0))
    ny = np.sqrt((ry ** 2).sum(axis=0))
    denom = nx[:, None] * ny[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (rx.T @ ry) / np.where(denom == 0, np.nan, denom)
    return pd.DataFrame(out, index=x.columns, columns=y.columns)


def rank_normalize(mat: pd.DataFrame) -> pd.DataFrame:
    """按列做秩归一化：(rank-1)/(n-1)，取值 [0,1]，值越大越强。"""
    ranks = mat.rank(axis=0, method="average")
    n = mat.shape[0]
    return (ranks - 1.0) / max(n - 1, 1)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """两集合的 Jaccard 相似度；两者皆空时定义为 1.0。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return float(len(sa & sb) / union) if union else 1.0


def top_k_from_scores(scores: pd.Series, k: int = TOP_K) -> List[str]:
    """按分数降序取前 k（并列时按名称字典序，保证可复现）。"""
    s = scores.astype(float).copy()
    s = s.sort_values(ascending=False, kind="mergesort")
    return list(s.index[:k])


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


def describe_params() -> dict:
    """把全部超参写进产物，保证可复现。"""
    return {
        "graph_threshold_abs_rho": GRAPH_THRESHOLD,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "leiden": {"objective_function": LEIDEN_OBJECTIVE,
                   "n_iterations": LEIDEN_N_ITERATIONS, "seed": LEIDEN_SEED},
        "fusion_weights": FUSION_WEIGHTS,
        "lightgbm_params": LGB_PARAMS,
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "n_jobs": BOOTSTRAP_N_JOBS,
                      "resample_unit": "训练样本(175 瓶)有放回重采样"},
        "top_k": TOP_K,
        "selection_rule": {"bootstrap_freq_min": SELECT_FREQ_MIN,
                           "mean_cross_method_jaccard_min": SELECT_JACCARD_MIN},
        "corr_type": "Spearman（2.csv / 5.csv / 6.csv 均为 Spearman，见 README 勘误说明）",
    }
