# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：断言中的图数量（17 张：q1=4 / q2=6 / q3=7）与目录实际清点结果一致；
# PNG/PDF 成对要求与 scripts/plot_style.py 的 save() 实现一致。
"""图表风格契约测试：守住「全文风格统一 + PNG/PDF 成对 + 标题简约」三条底线。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_style as ps  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FIGURE_DIRS = ["q1_figures", "q2_figures", "q3_figures"]
#: 各目录应有的图数（与改造前的既有产物一致，不得因风格统一而丢图）
EXPECTED_COUNTS = {"q1_figures": 4, "q2_figures": 6, "q3_figures": 7}


def test_contract_exports_present():
    for name in ["apply_style", "tidy", "simple_title", "save", "dim_cn", "class_cn",
                 "figsize"]:
        assert hasattr(ps, name), f"plot_style 缺少契约导出：{name}"
    assert ps.STYLE_CONTRACT_VERSION == "plot_style_v1"
    assert len(ps.OKABE_ITO) >= 8
    assert ps.FIG_WIDTH_IN <= 5.0 + 1e-9


def test_style_module_is_single_source_of_truth():
    """三个绘图脚本不得自行设置 rcParams 或自定义配色。"""
    import re
    for script in ["q1_figures.py", "q2_figures.py", "q3_figures.py"]:
        path = ROOT / "scripts" / script
        assert path.exists(), f"缺少绘图脚本 {script}"
        text = path.read_text(encoding="utf-8")
        offenders = [m for m in re.findall(r'plt\.rcParams\[[^\]]+\]\s*=', text)]
        assert not offenders, f"{script} 仍在自行设置 rcParams：{offenders}"
        assert "plot_style" in text, f"{script} 未使用统一风格模块"


@pytest.mark.parametrize("dirname", FIGURE_DIRS)
def test_figures_have_png_pdf_pairs(dirname):
    folder = PROCESSED / dirname
    assert folder.is_dir(), f"缺少图目录 {folder}"
    pngs = sorted(folder.glob("*.png"))
    assert pngs, f"{dirname} 没有 PNG 图"
    assert len(pngs) == EXPECTED_COUNTS[dirname], (
        f"{dirname} 图数为 {len(pngs)}，期望 {EXPECTED_COUNTS[dirname]}（风格统一不得丢图）")
    for png in pngs:
        pdf = png.with_suffix(".pdf")
        assert pdf.exists(), f"缺少矢量 PDF：{pdf.name}"
        assert png.stat().st_size > 2000, f"图文件过小，疑似空图：{png.name}"
        assert pdf.stat().st_size > 500, f"矢量 PDF 过小：{pdf.name}"


@pytest.mark.parametrize("dirname", FIGURE_DIRS)
def test_figures_manifest_captions(dirname):
    """每张图必须在 manifest 里有题注 caption（方法细节从图内移到了这里）。"""
    folder = PROCESSED / dirname
    manifest = folder / "figures_manifest.json"
    assert manifest.exists(), f"{dirname} 缺少 figures_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data.get("style_contract") == "plot_style_v1"
    figures = data.get("figures", [])
    assert len(figures) == EXPECTED_COUNTS[dirname], (
        f"{dirname} manifest 图数 {len(figures)} 与目录实际不符")
    for fig in figures:
        assert fig.get("caption"), f"图缺少题注：{fig.get('file')}"
        assert fig.get("title"), f"图缺少标题：{fig.get('file')}"
        assert len(str(fig.get("title"))) <= 12, (
            f"图内标题过长（应 ≤12 字）：{fig.get('title')}")
        assert fig.get("outputs"), f"图缺少输出文件记录：{fig.get('file')}"


def test_dim_cn_covers_panel_dimensions():
    from data_contract import load_targets
    targets = load_targets()
    missing = [c for c in targets.columns if ps.dim_cn(c) == c]
    assert not missing, f"以下感官维度缺中文名映射：{missing}"


def test_dim_cn_handles_consumer_descriptors():
    assert ps.dim_cn("aroma_citrus") == "香气·柑橘"
    assert ps.dim_cn("taste_stale") == "味道·陈化味"
    assert ps.dim_cn("aroma_citrus") == ps.dim_cn("A_hops_citrus")
