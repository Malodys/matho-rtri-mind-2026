# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 感官画像只调用问题 1 的 ``scripts/q1_predict.predict_sensory``，不自建感官模型
#      （避免与 Q1 重复建模、口径不一致）；接口缺失时优雅降级为 pending_q1，不报错退出。
#   2) 输入矩阵口径：统一 log10(mg/L) 的 231 列化合物矩阵（含 4 个 *_sum 派生列，
#      Q1 接口内部自行剔除）；输出索引与输入逐行一致、列恰为 50 个感官维度名，已核对。
#   3) 每个风格产出「有约束 / 无约束」两版感官画像，与两版化合物谱一一对应。
"""问题 3 Step 6：感官画像（生成化合物谱 → 50 维标准化感官分）。

依赖：问题 1 的 ``scripts/q1_predict.py``（接口 ``predict_sensory(compound_matrix, model=None)``）。

接口约定（供 Q1 侧对齐）
----------------------
* 输入：统一 log10(mg/L) 的化合物矩阵，**231 列**（含 ``esters_sum`` / ``aroma_hops_sum`` /
  ``acids_sum`` / ``sulfur_sum`` 四个派生列，Q1 接口内部按 ``*_sum`` 后缀自动剔除）；
  索引为样本 id（本脚本用合成 id 字符串，原样回传为输出索引）。
* 输出：DataFrame，索引与输入逐行一致，列为 50 个感官维度名，值为标准化连续分。
* 模型工件位于 ``data/processed/q1_final/model_<name>.joblib``；缺失时本脚本降级为 pending_q1。

产出
----
* ``data/processed/q3_runs/sensory_profile_<style>.csv``（有约束版）
* ``data/processed/q3_runs/sensory_profile_<style>_unconstrained.csv``（无约束版）
* 若 Q1 不可用：``data/processed/q3_runs/sensory_profile_pending.md``（说明与接口约定）
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RUN_DIR = REPO_ROOT / "data" / "processed" / "q3_runs"
GEN_DIR = RUN_DIR / "generation"

PENDING_MD = """# 感官画像：pending_q1

问题 3 的「生成化合物谱 → 50 维感官画像」依赖**问题 1 的感官预测接口**，
本次运行时该接口尚不可用，因此本步骤**未产出**感官画像（状态 `pending_q1`）。

## 依赖缺失的具体原因

{reason}

## 接口约定（问题 1 侧就绪后可直接跑通，无需改动问题 3 脚本）

调用方：`scripts/q3_sensory_profile.py`

```python
from q1_predict import predict_sensory
scores = predict_sensory(compound_matrix, model=None)   # model=None 用最终选定模型
```

* **输入** `compound_matrix`：统一 log10(mg/L) 标度的化合物矩阵，**231 列**
  （即 `data_contract.to_unified_scale()` 的全部列，**含** `esters_sum`、
  `aroma_hops_sum`、`acids_sum`、`sulfur_sum` 四个 `*_sum` 派生列；
  Q1 接口内部按 `*_sum` 后缀自动剔除，调用方无需自行处理）。
* **索引**：样本 id；合成谱用 `SYN_<style>_NNNN` / `SYNU_<style>_NNNN` 字符串 id，
  原样回传为输出索引。
* **输出** `scores`：DataFrame，索引与输入**逐行一致**，列为 **50 个感官维度名**，
  值为标准化后的连续分（4.csv 口径）。
* **模型工件**：需存在 `data/processed/q1_final/model_<name>.joblib`，
  并由 `data/processed/q1_final/model_summary.csv` 的 `selected_as_final_model` 标记最终模型。

## 恢复方式

```powershell
.\\.venv\\Scripts\\python.exe scripts\\q3_sensory_profile.py
```

接口与工件就绪后，本脚本会自动为每个风格产出
`sensory_profile_<style>.csv`（有约束版）与
`sensory_profile_<style>_unconstrained.csv`（无约束版），
并把 `generation_summary.csv` 的 `sensory_profile_status` 更新为 `ok`。

## 口径声明

附件 3.xlsx 为空白品评卡模板，附件中**没有任何原始 0–6 评分**；
4.csv 的感官分是标准化值。因此问题 3 的「感官画像」只能建立在**标准化尺度**上，
论文与图表须显式声明这一点。
"""


def slug(style: str) -> str:
    return (style.replace(" ", "_").replace("/", "-")).strip("_")


def load_q1_predict():
    """尝试加载 Q1 推理接口；返回 (模块或 None, 失败原因)。"""
    path = REPO_ROOT / "scripts" / "q1_predict.py"
    if not path.is_file():
        return None, f"未找到接口文件 {path}"
    try:
        module = importlib.import_module("q1_predict")
    except Exception as exc:                      # pragma: no cover
        return None, f"导入 q1_predict 失败：{exc!r}"
    if not hasattr(module, "predict_sensory"):
        return None, "q1_predict 模块缺少 predict_sensory 接口"
    return module, ""


def load_compound_matrix(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """读取生成谱 CSV，返回 (231 列化合物矩阵, 样本 id 列表)。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    ids = frame["beer"].astype(str).tolist()
    drop = [c for c in ("beer", "tasting_category_fine") if c in frame.columns]
    matrix = frame.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    matrix.index = pd.Index(ids, name="beer")
    return matrix, ids


CUTPOINT_PATH = (REPO_ROOT / "data" / "processed" / "q1_final"
                 / "proxy_grade_cutpoints.csv")


def load_cutpoints():
    """加载问题一的七分位分级切点（50 维 × 6 个边界）。

    行索引为 50 个感官维度名，列为 boundary_1..boundary_6。
    已核对：与 ``q1_evaluation.training_quantile_cutpoints(y_train)`` 重算结果
    最大偏差 9.7e-17（机器精度）。
    """
    if not CUTPOINT_PATH.is_file():
        return None
    cp = pd.read_csv(CUTPOINT_PATH, encoding="utf-8-sig", index_col=0)
    cp.index = cp.index.astype(str)
    return cp


def apply_cutpoints(values, cutpoints: np.ndarray) -> np.ndarray:
    """连续值 → 0–6 代理等级，口径与 ``q1_evaluation.apply_cutpoints`` 完全一致。

    ``np.digitize(values[:, j], cutpoints[j], right=True)``
    """
    values = np.asarray(values, dtype=float)
    grades = np.empty(values.shape, dtype=np.int8)
    for j in range(values.shape[1]):
        grades[:, j] = np.digitize(values[:, j], cutpoints[j], right=True)
    return grades


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题3 Step 6：生成谱 → 50 维感官画像（依赖问题 1 的 q1_predict 接口）")
    parser.add_argument("--styles", type=str, default="",
                        help="逗号分隔的风格列表；缺省为 generation/ 目录下全部风格")
    parser.add_argument("--model", type=str, default=None,
                        help="传给 Q1 接口的模型名；缺省用其最终选定模型")
    args = parser.parse_args(argv)

    module, reason = load_q1_predict()
    if module is None:
        print(f"[降级] Q1 接口不可用：{reason}")
        (RUN_DIR / "sensory_profile_pending.md").write_text(
            PENDING_MD.format(reason=reason), encoding="utf-8")
        print(f"[落盘] {RUN_DIR/'sensory_profile_pending.md'}")
        update_summary_status("pending_q1")
        return 0

    # 风格清单
    if args.styles.strip():
        styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    else:
        styles = []
        for p in sorted(GEN_DIR.glob("*_synthetic.csv")):
            stem = p.stem[: -len("_synthetic")]
            styles.append(stem.replace("_", " "))
    print(f"[风格] {styles}")

    ok_styles: List[str] = []
    failures: List[str] = []
    cutpoints = load_cutpoints()
    if cutpoints is None:
        print("[提示] 未找到分级切点表，只输出标准化尺度感官画像（不做 0–6 离散化）")
    for style in styles:
        for tag, suffix in (("constrained", "_synthetic.csv"),
                            ("unconstrained", "_unconstrained.csv")):
            path = GEN_DIR / f"{slug(style)}{suffix}"
            if not path.is_file():
                continue
            try:
                matrix, ids = load_compound_matrix(path)
                scores = module.predict_sensory(matrix, args.model)
                out = scores.copy()
                out.insert(0, "tasting_category_fine", style)
                out.insert(0, "variant", tag)
                name = (f"sensory_profile_{slug(style)}.csv" if tag == "constrained"
                        else f"sensory_profile_{slug(style)}_unconstrained.csv")
                out.to_csv(RUN_DIR / name, encoding="utf-8-sig")
                print(f"  [成功] {style:<12} {tag:<14} 输出 {out.shape} → {name}")

                # 0–6 代理等级：标准化预测 → 训练集七分位切点离散化
                if cutpoints is not None:
                    dims = [d for d in scores.columns if d in cutpoints.index]
                    cp = cutpoints.loc[dims].to_numpy(dtype=float)
                    grades = apply_cutpoints(scores[dims].to_numpy(dtype=float), cp)
                    gdf = pd.DataFrame(grades, index=scores.index, columns=dims)
                    gdf.insert(0, "tasting_category_fine", style)
                    gdf.insert(0, "variant", tag)
                    gname = (f"sensory_profile_{slug(style)}_0_6.csv" if tag == "constrained"
                             else f"sensory_profile_{slug(style)}_unconstrained_0_6.csv")
                    gdf.to_csv(RUN_DIR / gname, encoding="utf-8-sig")
                    print(f"  [成功] {style:<12} {tag:<14} 0–6 等级 {gdf.shape} → {gname}")
                if tag == "constrained":
                    ok_styles.append(style)
            except Exception as exc:
                failures.append(f"{style}/{tag}: {exc!r}")
                print(f"  [失败] {style:<12} {tag:<14} {exc!r}")

    if not ok_styles:
        detail = "；".join(failures) if failures else "所有风格均无生成谱文件"
        (RUN_DIR / "sensory_profile_pending.md").write_text(
            PENDING_MD.format(reason=f"Q1 接口存在但调用失败：{detail}"),
            encoding="utf-8")
        print(f"[降级] 全部风格失败 → {RUN_DIR/'sensory_profile_pending.md'}")
        update_summary_status("pending_q1")
        return 0

    # 汇总：生成谱 vs 真实感官分的风格均值对比
    try:
        import data_contract as dc
        raw = dc.load_raw()
        real = dc.load_targets(raw)
        styles_ser = dc.load_styles(raw)
        real = real.loc[styles_ser.index]
        profile_rows = []
        for style in ok_styles:
            prof = pd.read_csv(RUN_DIR / f"sensory_profile_{slug(style)}.csv",
                               encoding="utf-8-sig")
            dims = [c for c in prof.columns if c not in ("beer", "tasting_category_fine",
                                                          "variant")]
            style_real = real.loc[(styles_ser == style).to_numpy(), dims]
            grade_real = None
            if cutpoints is not None:
                cp = cutpoints.loc[dims].to_numpy(dtype=float)
                grade_real = apply_cutpoints(style_real.to_numpy(dtype=float), cp)
            for k, d in enumerate(dims):
                row = {
                    "style": style,
                    "sensory_dimension": d,
                    "generated_mean": round(float(prof[d].mean()), 6),
                    "generated_median": round(float(prof[d].median()), 6),
                    "generated_std": round(float(prof[d].std(ddof=1)), 6),
                    "real_mean": round(float(style_real[d].mean()), 6),
                    "real_median": round(float(style_real[d].median()), 6),
                    "real_std": round(float(style_real[d].std(ddof=1)), 6),
                    "mean_shift_generated_minus_real": round(
                        float(prof[d].mean() - style_real[d].mean()), 6),
                }
                if grade_real is not None:
                    row["generated_mean_grade_0_6"] = round(
                        float(apply_cutpoints(
                            prof[d].to_numpy(dtype=float).reshape(-1, 1),
                            cp[k].reshape(1, -1)).mean()), 4)
                    row["real_mean_grade_0_6"] = round(float(grade_real[:, k].mean()), 4)
                    row["grade_shift_generated_minus_real"] = round(
                        row["generated_mean_grade_0_6"] - row["real_mean_grade_0_6"], 4)
                profile_rows.append(row)
        pd.DataFrame(profile_rows).to_csv(
            RUN_DIR / "sensory_profile_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[落盘] {RUN_DIR/'sensory_profile_summary.csv'}")

        # 约束的净感官效应：有约束 − 无约束（同一候选池，唯一差别是缺陷约束）
        effect_rows = []
        defect_dims = {"dms", "X4vg", "diacetyl", "t2n"}
        for style in ok_styles:
            p_c = RUN_DIR / f"sensory_profile_{slug(style)}.csv"
            p_u = RUN_DIR / f"sensory_profile_{slug(style)}_unconstrained.csv"
            if not p_u.is_file():
                continue
            c = pd.read_csv(p_c, encoding="utf-8-sig")
            u = pd.read_csv(p_u, encoding="utf-8-sig")
            dims = [d for d in c.columns if d not in ("beer", "tasting_category_fine",
                                                      "variant")]
            gc = gu = None
            gc_path = RUN_DIR / f"sensory_profile_{slug(style)}_0_6.csv"
            gu_path = RUN_DIR / f"sensory_profile_{slug(style)}_unconstrained_0_6.csv"
            if cutpoints is not None and gc_path.is_file() and gu_path.is_file():
                gc = pd.read_csv(gc_path, encoding="utf-8-sig")
                gu = pd.read_csv(gu_path, encoding="utf-8-sig")
            for d in dims:
                row = {
                    "style": style,
                    "sensory_dimension": d,
                    "is_defect_dimension": d in defect_dims,
                    "constrained_mean_standardized": round(float(c[d].mean()), 6),
                    "unconstrained_mean_standardized": round(float(u[d].mean()), 6),
                    "delta_standardized": round(float(c[d].mean() - u[d].mean()), 6),
                }
                if gc is not None:
                    row["constrained_mean_grade_0_6"] = round(float(gc[d].mean()), 4)
                    row["unconstrained_mean_grade_0_6"] = round(float(gu[d].mean()), 4)
                    row["delta_grade_0_6"] = round(float(gc[d].mean() - gu[d].mean()), 4)
                effect_rows.append(row)
        if effect_rows:
            eff = pd.DataFrame(effect_rows)
            eff.to_csv(RUN_DIR / "sensory_constraint_effect.csv", index=False,
                       encoding="utf-8-sig")
            d4 = eff[eff["is_defect_dimension"]]
            o46 = eff[~eff["is_defect_dimension"]]
            print(f"[落盘] {RUN_DIR/'sensory_constraint_effect.csv'}")
            print(f"  约束净感官效应：缺陷 4 维 Δ标准化 中位="
                  f"{d4['delta_standardized'].median():+.4f}、其余 46 维 中位="
                  f"{o46['delta_standardized'].median():+.4f}")
    except Exception as exc:
        print(f"[提示] 感官画像汇总表生成失败（不影响主产物）：{exc!r}")

    update_summary_status("ok")
    return 0


def update_summary_status(status: str) -> None:
    """把感官画像状态写回 generation_summary.csv（若存在）。"""
    path = RUN_DIR / "generation_summary.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "sensory_profile_status" not in df.columns:
        df["sensory_profile_status"] = status
    else:
        df["sensory_profile_status"] = status
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[状态] generation_summary.csv 的 sensory_profile_status = {status}")


if __name__ == "__main__":
    raise SystemExit(main())
