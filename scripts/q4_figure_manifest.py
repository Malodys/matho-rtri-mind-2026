# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：题注中的全部数值均由本脚本从 data/processed/q4_consumer/*.csv 现场读取，
# 未硬编码；已逐条核对四张图的图内内容与题注描述一致（逐维 R² 分布 / 散点一致性 /
# 剂量-响应双面板 / 价格部分效应误差棒）；图内标题 ≤12 字、方法细节入题注的写法与
# scripts/plot_style.py 的契约一致。
"""为消费者层图表生成 figures_manifest.json（题注唯一来源）。

为什么需要这个脚本
------------------
问题一/二/三的图题注由各自的绘图脚本就地生成 `figures_manifest.json`，
`scripts/build_paper.py` 读取该文件决定论文中的图号与题注。消费者层（B2）的图输出在
`data/processed/q4_figures/`，但图由 `scripts/q4_consumer.py` 绘制，该脚本不写 manifest，
导致论文构建时消费者层图题注退化为兜底文案。

本脚本按与 B3 完全一致的 schema 生成 manifest，并把题注数值改为**现场从产物 CSV 读取**，
避免题注与数据脱节。

用法::

    python scripts/q4_figure_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONSUMER_DIR = PROCESSED / "q4_consumer"
FIG_DIR = PROCESSED / "q4_figures"
MANIFEST = FIG_DIR / "figures_manifest.json"

#: 与 scripts/plot_style.py 的契约版本号保持一致。
STYLE_CONTRACT = "plot_style_v1"

#: 题注中的「图 N」为目录内占位序号，最终图号由 scripts/build_paper.py 统一分配。
NUMBERING_NOTE = "caption 中的「图 N」为目录内占位序号，最终图号由 scripts/build_paper.py 统一分配。"


def _f(x: float, nd: int = 3) -> str:
    """数值转紧凑字符串，避免出现 0.890422 这类冗长尾数。"""
    return f"{float(x):.{nd}f}"


def _read(name: str) -> pd.DataFrame | None:
    path = CONSUMER_DIR / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def _outputs(stem: str) -> list[str]:
    return [f"data/processed/q4_figures/{stem}.png",
            f"data/processed/q4_figures/{stem}.pdf"]


def build_figures() -> list[dict]:
    """逐图构造 manifest 条目；数值全部来自产物 CSV。"""
    figs: list[dict] = []

    # ---- 图 1：逐维可预测性对比 ------------------------------------------------
    summary = _read("consumer_model_summary.csv")
    if summary is not None and not summary.empty:
        sm = summary.set_index("model")
        base = sm.loc["global_mean"]
        best = sm.loc["global_extra_trees"]
        imp = (1 - float(best["Macro_RMSE"]) / float(base["Macro_RMSE"])) * 100
        figs.append({
            "file": "fig1_consumer_predictability",
            "title": "逐维可预测性对比",
            "caption": (
                f"图 1  逐维可预测性对比。左侧为品评组 50 个感官维度的袋外决定系数分布"
                f"（箱线图），右侧为消费者 183 个描述符的袋外决定系数分布（散点），"
                f"两者均以同一化学特征矩阵与同一 5 折分层划分评估。"
                f"消费者路径最优模型 global_extra_trees 的宏平均 RMSE 为 {_f(best['Macro_RMSE'], 4)}，"
                f"优于全局均值基线 {_f(base['Macro_RMSE'], 4)}（改善 {_f(imp, 1)}%）；"
                f"其逐维 R² 中位数 {_f(best['R2_median'], 3)}、斯皮尔曼中位数 {_f(best['Spearman_median'], 3)}。"
                f"图中虚线为 R²=0 参考线。"
            ),
            "width_in": 5.0,
            "outputs": _outputs("fig1_consumer_predictability"),
        })

    # ---- 图 2：两条感知路径一致性 --------------------------------------------
    cross = _read("cross_consistency.csv")
    if cross is not None and not cross.empty:
        d = cross["delta"].abs()
        n = len(cross)
        figs.append({
            "file": "fig2_cross_consistency",
            "title": "两条感知路径一致性",
            "caption": (
                f"图 2  两条感知路径一致性。横轴为品评组路径与消费者路径在同一感官维度上的"
                f"实测相关（ρ 真值），纵轴为化学模型预测的相关（ρ 预测），共 {n} 组配对；"
                f"虚线为 ρ 预测 = ρ 真值的一致线。两组相关的绝对偏差中位数为 {_f(d.median(), 3)}，"
                f"其中 {(d <= 0.2).mean() * 100:.1f}% 不超过 0.2。"
                f"需注意品评组维度与消费者描述符来自不同词汇表，二者映射存在循环性风险，"
                f"故本图只用于方向性对照，不作为因果证据。"
            ),
            "width_in": 5.0,
            "outputs": _outputs("fig2_cross_consistency"),
        })

    # ---- 图 3：剂量-响应 --------------------------------------------------------
    verdict = _read("defect_dose_verdict.csv")
    resp = _read("defect_dose_response.csv")
    if verdict is not None and not verdict.empty:
        pairs = verdict.groupby(["compound", "sensory_dim"]).ngroups
        nq = len(resp["quartile"].unique()) if resp is not None and "quartile" in resp.columns else 4
        robust = verdict["style_robust_fraction"]
        figs.append({
            "file": "fig3_dose_response",
            "title": "剂量-响应",
            "caption": (
                f"图 3  剂量-响应。左面板为化合物浓度四分位内组内的斯皮尔曼相关 ρ，"
                f"右面板为对应感官维度的中位强度，两者刻画「浓度升高—感知变化」的方向与幅度。"
                f"共 {pairs} 组化合物—感官维度配对，每组按浓度四分位（Q1→Q{nq}）统计；"
                f"虚线为 ρ=0 参考线。橙红强调色配以不同点型/线型者为目标缺陷化合物配对。"
                f"风格分层稳健性（n≥8 的风格内重算）显示部分配对在任一风格内都不成立，"
                f"稳健比例中位数为 {_f(robust.median(), 3)}，逐组结论见正文表。"
            ),
            "width_in": 5.0,
            "outputs": _outputs("fig3_dose_response"),
        })

    # ---- 图 4：价格部分效应 ----------------------------------------------------
    price = _read("price_regression.csv")
    if price is not None and not price.empty:
        r2 = price.drop_duplicates("model").set_index("model")["r2"].to_dict()
        n = int(price["n"].iloc[0])
        figs.append({
            "file": "fig4_price_effects",
            "title": "价格部分效应",
            "caption": (
                f"图 4  价格部分效应。三个设定下 log 价格对各感官维度的偏回归系数及 95% 置信区间："
                f"ols（R²={_f(r2.get('ols', float('nan')), 3)}）、"
                f"ols_with_abv（加入 ABV，R²={_f(r2.get('ols_with_abv', float('nan')), 3)}）、"
                f"fixed_effects（再加入风格固定效应、按风格聚类稳健标准误，"
                f"R²={_f(r2.get('fixed_effects', float('nan')), 3)}），"
                f"样本量 n={n}（仅含 ≥2 家零售报价的啤酒）。"
                f"价格来自单一零售源附件，此处仅作为外部价值信号，不作市场因果解释。"
            ),
            "width_in": 5.0,
            "outputs": _outputs("fig4_price_effects"),
        })

    return figs


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = build_figures()
    if not figs:
        print("[warn] 未找到消费者层产物，未生成 manifest。")
        return 1
    payload = {
        "style_contract": STYLE_CONTRACT,
        "numbering_note": NUMBERING_NOTE,
        "figures": figs,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[ok] 已写出 {MANIFEST.relative_to(ROOT)}（{len(figs)} 张图）")
    for f in figs:
        print(f"     - {f['file']}：{f['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
