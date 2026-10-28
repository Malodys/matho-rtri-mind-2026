# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 输入矩阵列序不敏感：按工件记录的 feature_columns 重排，缺失列直接报错并列出，
#      *_sum 派生列自动剔除（与 data_contract.DERIVED_COLUMNS 口径一致）；
#   2) 输出索引与输入索引逐行一致、列恰为 50 个感官维度名，已用真实统一矩阵（250×231）
#      实跑核对：索引相同、列名与 4.csv 的 50 个目标完全一致、无 NaN；
#   3) 工件在全部 250 样本上重训，故本接口输出的是「全量重训模型的预测」，
#      与交付表 sensory_scores_0_6.csv 中的 5 折 OOF 预测口径不同，README 已写明；
#   4) 已实测「传入含 *_sum 派生列的 231 列矩阵」与「传入已剔除派生列的 227 列矩阵」输出逐元素相同。
"""问题一 · 对外推理接口（供问题三调用）。

用法：
    from q1_predict import predict_sensory
    scores = predict_sensory(compound_matrix)          # 使用最终选定模型
    scores = predict_sensory(compound_matrix, "ridge")  # 指定其它模型

命令行自检：
    python scripts/q1_predict.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "data" / "processed" / "q1_final"
DERIVED_SUFFIX = "_sum"

_ARTIFACT_CACHE: Dict[str, dict] = {}


def _default_model_name() -> str:
    """从 model_summary.csv 读取被选为最终模型的名称。"""
    summary_path = FINAL_DIR / "model_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        selected = summary.loc[summary.get("selected_as_final_model", False).astype(bool)]
        if len(selected):
            return str(selected.iloc[0]["model"])
    candidates = sorted(FINAL_DIR.glob("model_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"{FINAL_DIR} 下没有模型工件，请先运行 scripts/q1_pipeline.py")
    return candidates[0].name[len("model_"):-len(".joblib")]


def load_artifact(model: Optional[str] = None) -> dict:
    """读取模型工件（含特征列序、目标列序与拟合好的估计器）。"""
    name = model or _default_model_name()
    if name in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[name]
    import joblib

    path = FINAL_DIR / f"model_{name}.joblib"
    if not path.is_file():
        available = sorted(p.name[len("model_"):-len(".joblib")] for p in FINAL_DIR.glob("model_*.joblib"))
        raise FileNotFoundError(f"找不到模型工件 {path}；可用模型：{available}")
    artifact = joblib.load(path)
    _ARTIFACT_CACHE[name] = artifact
    return artifact


def _align_features(compound_matrix: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """把输入矩阵对齐到工件的特征列序（自动剔除 *_sum 派生列）。"""
    frame = compound_matrix.drop(columns=[c for c in compound_matrix.columns if c.endswith(DERIVED_SUFFIX)])
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"输入矩阵缺少 {len(missing)} 个特征列，例：{missing[:5]}")
    aligned = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if aligned.isna().any().any():
        bad = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(f"输入矩阵在以下特征列存在缺失/非数值：{bad[:5]}")
    return aligned


def predict_sensory(compound_matrix: pd.DataFrame, model: Optional[str] = None) -> pd.DataFrame:
    """预测 50 个感官维度的标准化连续值。

    参数
    ----
    compound_matrix : pd.DataFrame
        统一 log10(mg/L) 的化合物矩阵（可为 231 列含 ``*_sum`` 派生列，会自动剔除）；
        索引须为 beer 编号。
    model : str | None
        模型名；``None`` 表示使用 pipeline 选定的最终模型。

    返回
    ----
    pd.DataFrame
        索引与输入一致，列为 50 个感官维度名，值为标准化后的连续预测值。
    """
    artifact = load_artifact(model)
    features = _align_features(compound_matrix, artifact["feature_columns"])
    labels = list(artifact["target_columns"])
    kind = artifact["kind"]

    if kind == "constant":
        values = np.tile(np.asarray(artifact["estimators"]["constant"], dtype=float), (len(features), 1))
    elif kind == "global":
        values = np.asarray(artifact["estimators"]["estimator"].predict(features)).reshape(len(features), len(labels))
    elif kind == "grouped":
        values = np.zeros((len(features), len(labels)))
        group_members = artifact.get("group_members", {})
        for group, estimator in artifact["estimators"].items():
            members = list(group_members.get(group, []))
            if not members:
                raise KeyError(f"工件缺少分组 {group} 的成员清单")
            idx = [labels.index(m) for m in members]
            values[:, idx] = np.asarray(estimator.predict(features)).reshape(len(features), len(idx))
    elif kind == "independent":
        values = np.zeros((len(features), len(labels)))
        for j, label in enumerate(labels):
            values[:, j] = np.asarray(artifact["estimators"][label].predict(features)).reshape(-1)
    else:
        raise ValueError(f"未知工件类型：{kind}")

    out = pd.DataFrame(values, index=compound_matrix.index, columns=labels)
    if out.isna().any().any():
        raise ValueError("预测结果含缺失值")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="问题一 · 对外推理接口自检")
    ap.add_argument("--model", default=None)
    ap.add_argument("--self-test", action="store_true", help="用真实统一矩阵跑一次并打印摘要")
    args = ap.parse_args()

    if not args.self_test:
        ap.print_help()
        return 0

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from data_contract import load_modeling_matrix

    matrix = load_modeling_matrix()
    full = pd.read_csv(REPO_ROOT / "data/interim/chem_log10_matrix_v1.csv", index_col=0)
    full.index = full.index.astype(int)

    scores = predict_sensory(full, args.model)
    reduced = predict_sensory(matrix, args.model)
    same = np.allclose(scores.to_numpy(), reduced.to_numpy())
    print(f"模型：{args.model or _default_model_name()}")
    print(f"输入 231 列（含 *_sum）与 227 列输出一致：{same}")
    print(f"输出形状：{scores.shape}；索引与输入一致：{scores.index.equals(full.index)}")
    print(scores.iloc[:3, :5].to_string())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
