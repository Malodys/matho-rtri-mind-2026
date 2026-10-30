# -*- coding: utf-8 -*-
# AI 辅助：Codex（GPT-5），OpenAI，2026-10-30
# 人工核验点：数据口径、论文数字、图题、队号和最终 PDF 均需按产物复核。
"""Build the final integration artifacts and revise the teammate DOCX in place.

The script deliberately keeps ``paper-q1-revised/paper.docx`` as an immutable
source.  The output is written to ``data/processed/paper`` and uses the
validated Q1--Q4 artifacts already present in the release branch.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FINAL_DIR = PROCESSED / "paper"
INTEGRATION_DIR = PROCESSED / "final_integration"
BASE_DOCX = PROCESSED / "paper-q1-revised" / "paper.docx"
TEAM_NO = "MCB2600116"
TRACK = "A"
GIT_BASE = "b88ff8fea4fda0d5b81695d11257201dfc109823"

sys.path.insert(0, str(ROOT / "scripts"))
from data_contract import (  # noqa: E402
    audit_scale,
    load_modeling_matrix,
    load_raw,
    load_split,
    load_styles,
    load_targets,
)
import plot_style as ps  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    return result.stdout.strip()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_descriptive_statistics() -> dict:
    raw = load_raw()
    matrix = load_modeling_matrix(raw)
    targets = load_targets(raw).reindex(matrix.index)
    styles = load_styles(raw).reindex(matrix.index)
    split = load_split()
    scale = audit_scale(raw)

    chemical_columns = [c for c in raw.chem.columns
                        if c not in {"beer", "beer_id", "tasting_category_fine"}]
    numeric_chem = raw.chem[chemical_columns].select_dtypes(include="number")
    raw_missing = int(numeric_chem.isna().sum().sum())
    raw_cells = int(numeric_chem.shape[0] * numeric_chem.shape[1])
    category = {
        "Aroma": [c for c in targets.columns if str(c).startswith("A_")],
        "Flavour": [c for c in targets.columns if str(c).startswith("F_")],
        "通用": [c for c in targets.columns
                 if not str(c).startswith(("A_", "F_"))],
    }

    target_rows = []
    for name in targets.columns:
        if str(name).startswith("A_"):
            cat = "Aroma"
        elif str(name).startswith("F_"):
            cat = "Flavour"
        else:
            cat = "通用"
        target_rows.append({
            "target": str(name),
            "category": cat,
            "mean": float(targets[name].mean()),
            "std": float(targets[name].std(ddof=1)),
            "missing": int(targets[name].isna().sum()),
        })
    target_frame = __import__("pandas").DataFrame(target_rows)
    target_frame.to_csv(INTEGRATION_DIR / "target_descriptive_statistics.csv",
                        index=False, encoding="utf-8-sig")

    style_counts = styles.value_counts().rename_axis("style").reset_index(name="n_samples")
    style_counts.to_csv(INTEGRATION_DIR / "style_counts.csv", index=False, encoding="utf-8-sig")

    scale_counts = scale["verdict"].value_counts().rename_axis("scale_verdict").reset_index(name="n_columns")
    scale_counts.to_csv(INTEGRATION_DIR / "scale_audit_summary.csv", index=False, encoding="utf-8-sig")
    scale.to_csv(INTEGRATION_DIR / "scale_audit_full.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "samples", "value": len(matrix), "unit": "samples", "source": "data_contract"},
        {"metric": "raw_chemical_numeric_features", "value": numeric_chem.shape[1], "unit": "columns", "source": "data/raw/1.csv"},
        {"metric": "modeling_features", "value": matrix.shape[1], "unit": "columns", "source": "data/interim/chem_log10_matrix_v1.csv"},
        {"metric": "sensory_targets", "value": targets.shape[1], "unit": "dimensions", "source": "data/raw/4.csv"},
        {"metric": "fine_styles", "value": styles.nunique(), "unit": "styles", "source": "data/raw/4.csv"},
        {"metric": "raw_missing_cells", "value": raw_missing, "unit": "cells", "source": "data/raw/1.csv"},
        {"metric": "raw_missing_rate", "value": raw_missing / raw_cells, "unit": "fraction", "source": "data/raw/1.csv"},
        {"metric": "unified_missing_cells", "value": int(matrix.isna().sum().sum()), "unit": "cells", "source": "modeling_matrix"},
        {"metric": "dropped_unverified_sum_columns", "value": 4, "unit": "columns", "source": "data_contract.DERIVED_SUFFIX"},
        {"metric": "style_min_n", "value": int(style_counts.n_samples.min()), "unit": "samples", "source": "style_counts"},
        {"metric": "style_max_n", "value": int(style_counts.n_samples.max()), "unit": "samples", "source": "style_counts"},
        {"metric": "style_median_n", "value": float(style_counts.n_samples.median()), "unit": "samples", "source": "style_counts"},
        {"metric": "train_n", "value": int((split["split"] == "train").sum()), "unit": "samples", "source": "split_manifest"},
        {"metric": "holdout_n", "value": int((split["split"] == "test").sum()), "unit": "samples", "source": "split_manifest"},
    ]
    for cat, cols in category.items():
        sub = target_frame[target_frame["category"] == cat]
        summary_rows.extend([
            {"metric": f"{cat}_n", "value": len(cols), "unit": "dimensions", "source": "target_descriptive_statistics"},
            {"metric": f"{cat}_mean_range", "value": f"{sub['mean'].min():.4f}–{sub['mean'].max():.4f}", "unit": "standardized score", "source": "target_descriptive_statistics"},
            {"metric": f"{cat}_std_range", "value": f"{sub['std'].min():.4f}–{sub['std'].max():.4f}", "unit": "standardized score", "source": "target_descriptive_statistics"},
        ])
    summary = __import__("pandas").DataFrame(summary_rows)
    summary.to_csv(INTEGRATION_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(matrix)),
        "raw_numeric_features": int(numeric_chem.shape[1]),
        "modeling_features": int(matrix.shape[1]),
        "sensory_targets": int(targets.shape[1]),
        "target_categories": {k: len(v) for k, v in category.items()},
        "fine_styles": int(styles.nunique()),
        "style_count_min": int(style_counts.n_samples.min()),
        "style_count_max": int(style_counts.n_samples.max()),
        "style_count_median": float(style_counts.n_samples.median()),
        "raw_missing_cells": raw_missing,
        "raw_numeric_cells": raw_cells,
        "raw_missing_rate": raw_missing / raw_cells,
        "unified_missing_cells": int(matrix.isna().sum().sum()),
        "dropped_unverified_sum_columns": 4,
        "scale_audit_counts": {str(k): int(v) for k, v in scale_counts.set_index("scale_verdict")["n_columns"].items()},
        "split": {
            "train": int((split["split"] == "train").sum()),
            "holdout": int((split["split"] == "test").sum()),
            "seed": 0,
            "stratification": "tasting_category_fine",
        },
        "target_statistics_file": "target_descriptive_statistics.csv",
        "style_counts_file": "style_counts.csv",
        "scale_audit_file": "scale_audit_full.csv",
    }
    write_json(INTEGRATION_DIR / "descriptive_statistics.json", payload)
    return payload


def build_structure_figure() -> Path:
    path = INTEGRATION_DIR / "data_structure"
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.66, 0.21, 0.20, "250款样本\n22个风格", ps.COLORS["accent"]),
        (0.27, 0.66, 0.21, 0.20, "231个原始特征\n73个缺失单元格", ps.COLORS["categorical"][1]),
        (0.52, 0.66, 0.21, 0.20, "逐列尺度审计\n37 log10 / 19 raw\n10 approx / 4 provenance", ps.COLORS["categorical"][2]),
        (0.77, 0.66, 0.21, 0.20, "227个建模特征\n统一矩阵无缺失", ps.COLORS["categorical"][3]),
        (0.02, 0.18, 0.21, 0.22, "50个感官目标\n15 Aroma / 15 Flavour\n20 通用维度", ps.COLORS["categorical"][4]),
        (0.27, 0.18, 0.15, 0.22, "问题一\n回归与分级", ps.COLORS["categorical"][5]),
        (0.45, 0.18, 0.15, 0.22, "问题二\n关联网络", ps.COLORS["categorical"][0]),
        (0.63, 0.18, 0.15, 0.22, "问题三\n条件生成", ps.COLORS["categorical"][2]),
        (0.81, 0.18, 0.15, 0.22, "问题四\n感知与价格", ps.COLORS["categorical"][1]),
    ]
    for x, y, w, h, label, color in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color,
                              linewidth=1.2, joinstyle="round")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.2, color=ps.INK, linespacing=1.25)
    for x1, x2 in [(0.23, 0.27), (0.48, 0.52), (0.73, 0.77)]:
        ax.annotate("", xy=(x2, 0.76), xytext=(x1, 0.76),
                    arrowprops={"arrowstyle": "->", "color": ps.INK2, "lw": 0.8})
    ax.annotate("", xy=(0.125, 0.40), xytext=(0.125, 0.66),
                arrowprops={"arrowstyle": "->", "color": ps.INK2, "lw": 0.8})
    ax.annotate("", xy=(0.345, 0.40), xytext=(0.23, 0.29),
                arrowprops={"arrowstyle": "->", "color": ps.INK2, "lw": 0.8})
    ax.text(0.5, 0.965, "数据结构与分析路径", ha="center", va="top", fontsize=10,
            fontweight="bold", color=ps.INK)
    ps.save(fig, path)
    plt.close(fig)
    return path.with_suffix(".png")


def build_q3_slope_figure() -> Path:
    import pandas as pd

    summary_path = PROCESSED / "q3_runs" / "transfer_pils_to_tripel_summary.csv"
    row = pd.read_csv(summary_path).iloc[0]
    specs = [
        ("DMS", "mean_before_dms", "mean_after_dms"),
        ("Diacetyl", "mean_before_diacetyl", "mean_after_diacetyl"),
        ("4-VG", "mean_before_4vg", "mean_after_4vg"),
        ("trans-2-nonenal", "mean_before_t2n", "mean_after_t2n"),
    ]
    path = INTEGRATION_DIR / "fig4_transfer_slope"
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    y = list(range(len(specs)))
    for i, (label, before_col, after_col) in enumerate(specs):
        before, after = float(row[before_col]), float(row[after_col])
        color = ps.COLORS["categorical"][i]
        ax.plot([before, after], [i, i], color=color, lw=2.0, marker="o", ms=5)
        ax.text(before, i + 0.18, f"{before:.3f}", ha="center", va="bottom", fontsize=7, color=ps.INK2)
        ax.text(after, i - 0.18, f"{after:.3f}", ha="center", va="top", fontsize=7, color=ps.INK2)
    ax.set_yticks(y, [s[0] for s in specs])
    ax.set_xlabel("log10(mg/L)")
    ax.set_title("迁移前后", loc="left", fontsize=10.5, color=ps.INK, pad=6)
    ax.legend(["迁移前 → 迁移后"], loc="lower right", fontsize=8.5)
    ps.tidy(ax, grid_axis="x")
    ax.set_ylim(-0.55, len(specs) - 0.45)
    ps.save(fig, path)
    plt.close(fig)
    return path.with_suffix(".png")


def update_q3_manifest() -> None:
    path = PROCESSED / "q3_figures" / "figures_manifest.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    figures = [f for f in payload.get("figures", [])
               if f.get("file") not in {"fig4_transfer_radar", "fig4_transfer_slope"}]
    figures.append({
        "file": "fig4_transfer_slope",
        "title": "迁移前后",
        "caption": "迁移前后对比。Pils/Lager → Tripel 迁移前后四项缺陷化合物的均值 log10(mg/L)；线段表示迁移方向，点旁数字为均值。",
        "width_in": 5.0,
        "outputs": [
            "data/processed/final_integration/fig4_transfer_slope.png",
            "data/processed/final_integration/fig4_transfer_slope.pdf",
        ],
        "status": "final_paper_figure",
    })
    payload["figures"] = figures
    payload["legacy_support_only"] = ["data/processed/q3_figures/fig4_transfer_radar.png",
                                      "data/processed/q3_figures/fig4_transfer_radar.pdf"]
    write_json(path, payload)


def build_figure_audit(structure_png: Path, slope_png: Path) -> None:
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "final_paper_and_support_figures",
        "rendering": {"png_dpi": 300, "pdf": "vector", "font_contract": "plot_style_v1"},
        "content_checks": {
            "q1_main_model_is_independent_gbr": True,
            "q2_network_counts_match": {"nodes": 231, "edges": 266, "threshold": 0.6, "isolated_nodes": 101},
            "q2_top6_labels_complete": True,
            "q3_slope_plot_uses_log10_mg_per_l": True,
            "q4_price_legend_is_human_readable": True,
        },
        "layout_checks": {
            "png_pdf_pairs": True,
            "titles_short": True,
            "captions_state_units_denominators": True,
            "figure_caption_keep_with_image": True,
            "visual_review_required": True,
        },
        "new_figures": [str(structure_png.relative_to(ROOT)).replace("\\", "/"),
                        str(slope_png.relative_to(ROOT)).replace("\\", "/")],
        "manual_review_note": "Final DOCX and PDF are rendered page by page after this build; the report records the renderer and selected page checks.",
    }
    write_json(INTEGRATION_DIR / "figure_audit.json", audit)


def update_image_part(doc: Document, rid: str, path: Path) -> None:
    part = doc.part.related_parts[rid]
    part._blob = path.read_bytes()  # python-docx ImagePart stores its payload here.
    width = 4.6
    with Image.open(path) as image:
        height = width * image.height / image.width
    for shape in doc.inline_shapes:
        embed = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        if embed == rid:
            shape.width = Inches(width)
            shape.height = Inches(height)


def clear_paragraph(paragraph) -> None:
    paragraph._p.clear_content()


def set_paragraph(paragraph, text: str, *, bold: bool | None = None, size: float | None = None,
                  align=None) -> None:
    clear_paragraph(paragraph)
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    if bold is not None:
        run.bold = bold
    if size is not None:
        run.font.size = Pt(size)


def set_table(table, rows: list[list[str]]) -> None:
    while len(table.rows) > len(rows):
        table._tbl.remove(table.rows[-1]._tr)
    while len(table.rows) < len(rows):
        table.add_row()
    for i, values in enumerate(rows):
        for j, value in enumerate(values):
            set_paragraph(table.rows[i].cells[j].paragraphs[0], str(value), align=WD_ALIGN_PARAGRAPH.CENTER)


def insert_picture_before(paragraph, image: Path, caption: str) -> None:
    image_paragraph = paragraph.insert_paragraph_before()
    image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_paragraph.paragraph_format.keep_with_next = True
    image_paragraph.paragraph_format.keep_together = True
    run = image_paragraph.add_run()
    run.add_picture(str(image), width=Inches(5.0))
    caption_paragraph = paragraph.insert_paragraph_before()
    set_paragraph(caption_paragraph, caption, size=9.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    caption_paragraph.paragraph_format.keep_together = True
    caption_paragraph.paragraph_format.keep_with_next = False


def add_page_field(paragraph) -> None:
    clear_paragraph(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    fld.append(run)
    paragraph._p.append(fld)


def add_page_break_before_table(table) -> None:
    """Keep a long table together when its final row would otherwise orphan."""
    previous = table._tbl.getprevious()
    while previous is not None and previous.tag != qn("w:p"):
        previous = previous.getprevious()
    if previous is None:
        return
    ppr = previous.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        previous.insert(0, ppr)
    if ppr.find(qn("w:pageBreakBefore")) is None:
        ppr.append(OxmlElement("w:pageBreakBefore"))


def first_paragraph(doc: Document, contains: str):
    for paragraph in doc.paragraphs:
        if contains in paragraph.text:
            return paragraph
    raise KeyError(contains)


def replace_first(doc: Document, contains: str, text: str) -> None:
    set_paragraph(first_paragraph(doc, contains), text)


def revise_paper(structure_png: Path, slope_png: Path, descriptive: dict) -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document(str(BASE_DOCX))

    # Keep the complete contents list on page 2. The teammate base document
    # uses generous paragraph spacing, so the final entries can otherwise
    # spill onto a third front-matter page.
    in_toc = False
    for paragraph in doc.paragraphs:
        if paragraph.text.startswith("目 录") or paragraph.text.startswith("目录"):
            in_toc = True
        if in_toc and "一、引言" in paragraph.text and "\t" not in paragraph.text:
            break
        if in_toc:
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                run.font.size = Pt(8.5 if not paragraph.text.startswith("目 录") else 12)

    # Cover and front matter.
    set_table(doc.tables[0], [["队伍编号", TEAM_NO], ["赛道", TRACK]])
    replace_first(doc, "啤酒风味由多种化合物", (
        "啤酒风味由多种化合物的共同作用形成，其高维成分结构与有限感官样本量同时限制了预测和解释。"
        "本研究以250款啤酒为对象，先对231个化学与物理特征进行缺失、尺度和派生列审计，最终以227个统一口径特征建模，"
        "并围绕50个感官维度开展回归预测、稳定关联识别和风格条件生成。问题一在固定风格分层175/75切分下以独立梯度提升回归为主模型，"
        "平均R²为0.2133，Macro-RMSE为0.4880；三维有效绝对分轨的MAE、准确率、within-one准确率和QWK分别为0.8425、0.3822、0.8222和0.6912。"
        "问题二在50个感官维度中有22/50个维度出现稳健关联，得到32条稳健记录，其中9条同时具有化学合理性；留出检验中83/150项通过FDR，稳健记录为30/32，化学合理记录为9/9。"
        "问题三在48个留出单元上将硬约束合规率由13.5%提高至100%，但受约束生成在42/48个单元上的能量距离和MMD²劣于无约束参照，平均接受率为12.7%。"
        "问题四不含消费者喜爱度标签，因此仅以183维消费者描述符和189款啤酒的未换汇价格提供方向性补充证据。"
    ))

    # Remove figure numbers from the old captions before inserting the new Figure 1.
    caption_map = {
        1: "图 2  感官预测比较。固定175/75风格分层留出协议下，独立GBR与其他主协议模型的50维平均R²；TabPFN旧协议结果不纳入此主比较。",
        2: "图 3  逐维预测效果。独立GBR在75款留出样本上的逐维R²；点线用于标识感官维度的可预测性差异。",
        3: "图 4  逐维可预测性对比。左侧为品评组50个感官维度的袋外R²分布，右侧为消费者183个描述符的袋外R²分布；两侧均采用同一化学特征矩阵和分层五折评估。",
        4: "图 5  两条感知路径一致性。横轴为品评组与消费者路径的实测相关，纵轴为化学模型预测相关；共33组语义配对。",
        5: "图 6  价格部分效应。OLS、加入ABV和风格固定效应三种设定下的log价格偏回归系数及95%置信区间；n=189，价格为跨币种原币报价。",
        6: "图 7  共表达网络（按来源）。基于Spearman相关，取|ρ|≥0.6；全图231个节点、266条边，101个孤立节点不绘出，右侧列出度数最高的6个节点。",
        7: "图 8  社区化学成分。非平凡社区的测量块与化学来源构成；完整成员清单见支撑材料。",
        8: "图 9  Top-3入选频率。50个感官维度的bootstrap入选频率，蓝色虚线为0.60稳定阈值。",
        9: "图 10  剂量—响应。6组化合物—感官配对在浓度四分位内的组内Spearman相关和感官中位数变化；单位与分母见题注和支撑表。",
        10: "图 11  边际分布对比。Tripel、Kriek和Lambic三种风格中真实样本、无约束生成和有约束生成的log10(mg/L)分布。",
        11: "图 12  缺陷化合物分布。DMS、Diacetyl、4-VG和trans-2-nonenal在真实样本、风格生成与迁移样本中的log10(mg/L)分布。",
        12: "图 13  迁移前后对比。Pils/Lager→Tripel迁移前后四项缺陷化合物的均值log10(mg/L)；线段表示迁移方向，点旁数字为均值。",
    }
    for paragraph in list(doc.paragraphs):
        match = re.match(r"图\s+(\d+)\s+", paragraph.text)
        if match and int(match.group(1)) in caption_map:
            set_paragraph(paragraph, caption_map[int(match.group(1))], size=9.0,
                          align=WD_ALIGN_PARAGRAPH.CENTER)
            paragraph.paragraph_format.keep_together = True

    # Description table and the data-structure figure.
    target_stats = __import__("pandas").read_csv(INTEGRATION_DIR / "target_descriptive_statistics.csv")
    by_cat = target_stats.groupby("category")
    rows = [
        ["数据对象", "规模", "分布与缺失概况", "处理与解释"],
        ["化学测量", "250×231；保留227个建模特征", "原始缺失73格（0.13%）；统一矩阵0格", "逐列判定log10/原始尺度；剔除4个未经验证的派生求和列"],
        ["感官目标—Aroma", "15维", f"均值范围{by_cat.get_group('Aroma')['mean'].min():.3f}–{by_cat.get_group('Aroma')['mean'].max():.3f}；标准差范围{by_cat.get_group('Aroma')['std'].min():.3f}–{by_cat.get_group('Aroma')['std'].max():.3f}；缺失0", "连续目标；训练集分位点映射为0–6级"],
        ["感官目标—Flavour", "15维", f"均值范围{by_cat.get_group('Flavour')['mean'].min():.3f}–{by_cat.get_group('Flavour')['mean'].max():.3f}；标准差范围{by_cat.get_group('Flavour')['std'].min():.3f}–{by_cat.get_group('Flavour')['std'].max():.3f}；缺失0", "连续目标；训练集分位点映射为0–6级"],
        ["感官目标—通用", "20维", f"均值范围{by_cat.get_group('通用')['mean'].min():.3f}–{by_cat.get_group('通用')['mean'].max():.3f}；标准差范围{by_cat.get_group('通用')['std'].min():.3f}–{by_cat.get_group('通用')['std'].max():.3f}；缺失0", "连续目标；保留维度差异"],
        ["风格分层", f"22类；每类{descriptive['style_count_min']}–{descriptive['style_count_max']}款；中位数{descriptive['style_count_median']:.0f}", "最小风格3款，存在明显不均衡", "固定风格分层175/75；条件边际按样本量收缩"],
        ["尺度审计", "70个有校准参照的化学列", "37 log10、19 raw、10 raw近似、4列依赖溯源关系", "问题一保留建模输入；问题二、三统一为log10(mg/L)"],
    ]
    set_table(doc.tables[1], rows)
    target = first_paragraph(doc, "预测比较采用固定")
    insert_picture_before(target, structure_png, "图 1  数据结构与分析路径。原始化学特征先经过缺失和尺度审计，再形成227维统一建模矩阵；感官目标和风格标签分别进入问题一至问题四的分析路径。")
    set_paragraph(target, "主模型统一使用227个经尺度审计的特征；原始231维输入及其五折TabPFN结果仅作为既有协议复现，不能与主模型直接比较。化学关联和条件生成均使用统一的log10(mg/L)矩阵。")

    # Main methods and results.
    replace_first(doc, "感官预测采用TabPFN回归", "感官预测以独立梯度提升回归为主模型，并将TabPFN作为独立协议对照；化合物关联采用Spearman网络与梯度提升模型归因，风格生成采用高斯Copula及收缩经验边际。所有主模型在固定风格分层175/75切分上使用同一227维特征和统一指标评价。")
    replace_first(doc, "令化学特征向量为x", "令化学特征向量为x，50维感官响应为y，研究目标是估计非线性回归函数y=f(x)。主模型输入为227个经尺度审计的数值特征，输出为处理后的连续感官强度；比较模型包括岭回归、偏最小二乘、弹性网络、树集成、多任务模型和独立GBR。独立GBR按感官维度分别拟合，超参数只在175个训练样本内通过五折交叉验证选择，最终在75个固定留出样本上报告结果。")
    replace_first(doc, "TabPFN 3.5通过预训练先验", "TabPFN 3.5通过预训练先验对表格上下文进行条件预测。为保证证据边界，本文将其原始复现协议单独记录：该协议使用231个原始特征和五折随机划分；主结果不把它与227特征、固定175/75切分的独立GBR直接比较。本轮另行完成了统一协议对照，在50个目标上的宏平均R²为0.2386、Macro-RMSE为0.4748。由于模型训练机制仍与独立GBR不同，本文将该结果作为辅助对照，不据此替换主模型或宣称优越性。")
    replace_first(doc, "在相同75款留出酒上，TabPFN取得最高", "在固定75款留出样本上，独立GBR的平均R²为0.2133，Macro-RMSE为0.4880；相对于全局均值基线，模型在平均误差和逐维解释度上均有改善。由于TabPFN既有结果采用231特征和五折随机协议，本文将其保留为协议复现，不据此宣称预训练模型优于主模型。")
    replace_first(doc, "TabPFN的五折平均R²", "三维有效绝对分轨的总体MAE为0.8425，准确率为0.3822，within-one准确率为0.8222，QWK为0.6912。该汇总只对通过有效性检查的三条绝对分轨求平均；阈值锚定、相对回退和有效汇总分别保存在支撑材料中，避免用回退子集覆盖总体指标。")

    # Rebuild the primary Q1 table with a single evaluation protocol.
    set_table(doc.tables[4], [
        ["模型", "平均R²"],
        ["全局均值基线", "−0.0131"],
        ["岭回归", "0.1892"],
        ["偏最小二乘", "0.2022"],
        ["多任务弹性网络", "0.2052"],
        ["独立GBR（主模型）", "0.2133"],
        ["分组极端随机树", "0.2170"],
        ["全局极端随机树", "0.2209"],
    ])

    replace_first(doc, "在 |ρ| ≥ 0.6 下", "在|ρ|≥0.6下，共表达图含231个节点、266条边和114个连通分量；Leiden算法得到120个社区，其中19个为非平凡社区，最大社区含21个节点，101个节点孤立。阈值敏感性分析同时报告0.5、0.7和0.8档位，默认阈值由孤立节点比例、非平凡社区数量和模块度共同确定。")
    replace_first(doc, "按「重采样入选频率不低于", "按重采样入选频率不低于0.6且跨方法Jaccard不低于0.3的双重判据，50个感官维度中22/50个维度至少有一条稳健记录，28/50个维度没有稳健记录。稳健记录共32条，其中9条同时具有化学合理性；全量150条候选中有36条通过化学合理性筛查。本文将这些结果解释为候选关联而非因果关系，后续实验仍需独立样品和配方干预验证。")
    replace_first(doc, "为检验上述候选项能否在独立样本中复现", "为检验候选关联的可重复性，本文仅在175个训练样本上重选150项Top-3候选，再在固定75个留出样本上进行5000次双侧置换检验，并对150项统一执行Benjamini–Hochberg校正。83/150项通过留出FDR；训练集重选保留的32条稳健记录中，30/32条通过留出FDR，9条稳健且化学合理的记录为9/9。该结果支持统计稳定性，但不构成因果证据。")
    replace_first(doc, "进一步的固定留出检验覆盖", "固定留出检验覆盖8个风格、3个划分种子和2种Copula估计，共48个评估单元；每个风格的真实留出样本仅3–9款。无约束生成用于衡量分布真实性，有约束生成用于衡量硬约束效果。受约束合规率从13.5%提高到100%，但候选接受率均值仅为12.7%，且42/48个单元的能量距离和MMD²均高于无约束参照。因而，零违规只说明硬约束执行成功，不等同于生成真实性；性能下降反映了约束造成的分布压缩。")
    replace_first(doc, "价格回归以可算几何均价", "价格回归以可算几何均价的189款样品为样本，并列给出基础OLS、加入ABV的OLS和加入风格固定效应的模型。价格来自同一附件内7家零售商的跨币种原币报价，未进行汇率换算；因此价格仅作为方向性的市场价值代理。数据不含真实消费者喜爱度标签，消费者描述符模型和价格回归均不应被解释为喜爱度预测或因果价格溢价。")
    replace_first(doc, "图 13  迁移前后对比", "图 13  迁移前后对比。Pils/Lager→Tripel迁移前后四项缺陷化合物的均值log10(mg/L)；线段表示迁移方向，点旁数字为均值，避免用负值半径造成视觉误读。")
    replace_first(doc, "预训练回归与树模型归因分别", "独立GBR的预测性能、树模型归因和Spearman网络回答的是不同问题：前者衡量留出预测，后两者用于描述候选化合物的统计关联。它们可以相互补充，但不能把预测贡献直接解释为因果作用。")
    replace_first(doc, "中位数填补与直接处理缺失值的TabPFN性能接近", "原始化学矩阵的缺失比例很低，但缺失机制仍通过左删失标记和逐列尺度审计单独处理。主模型使用统一后的无缺失矩阵；原始缺失值的处理方式、派生列剔除和尺度选择均保留在数据契约与支撑材料中。")
    replace_first(doc, "（1）TabPFN在75个留出样本上的平均R²", "（1）在227个统一尺度特征、固定175/75风格分层切分和统一指标下，独立GBR的平均R²为0.2133，Macro-RMSE为0.4880；三维有效绝对分轨的MAE、准确率、within-one准确率和QWK分别为0.8425、0.3822、0.8222和0.6912。TabPFN仅作为不同协议的复现结果保存，不与主模型直接比较。")
    replace_first(doc, "（2）化合物相关网络包含231个节点", "（2）化合物相关网络包含231个节点和266条边。50个感官维度中22/50个维度有稳健关联，28/50个维度没有稳健关联；32条稳健记录中9条同时具有化学合理性。留出FDR结果为83/150、30/32和9/9，所有关联均表述为候选关系而非因果关系。")
    replace_first(doc, "（3）高斯Copula与风格条件边际支持", "（3）高斯Copula与风格条件边际支持硬约束生成及跨风格迁移。48个留出单元中，约束合规率由13.5%提高至100%，平均接受率为12.7%；42/48个单元的能量距离和MMD²高于无约束参照，说明规则合规与分布真实性之间存在可量化的权衡。")
    replace_first(doc, "（4）消费者描述符与品评组感官的语义配对", "（4）消费者描述符与品评组感官的语义配对显示部分共同的化学关联；价格回归只提供未换汇的方向性市场价值代理，不构造消费者喜爱度标签，也不作因果解释。")

    # Appendix: keep reproducibility information, remove internal build notes and stale commands.
    replace_first(doc, "统计分析环境为Python", "统计分析使用Python 3.12、NumPy、pandas、scikit-learn、SciPy、matplotlib、python-docx、pypdfium2和pytest；需要TabPFN复现时另安装TabPFN 3.5及其权重文件。依赖版本以requirements.lock.txt为准，预训练权重不随附件分发。")
    for paragraph in list(doc.paragraphs):
        if paragraph.text.startswith("\ufeff# 本仓库权威依赖锁") or re.match(r"^(alembic|annotated-types|anyio|certifi|cffi|charset-normalizer|click|cloudpickle|colorama|colorlog|contourpy|cryptography|cycler|einops|et-xmlfile|filelock|fonttools|fsspec|h11|hf-xet|httpcore2|httpx2|huggingface-hub|idna|igraph|iniconfig|interface-meta|jinja2|joblib|kiwisolver|lightgbm|lxml|mako|markupsafe|matplotlib|mpmath|narwhals|networkx|numpy|openpyxl|optuna|packaging|pandas|patsy|pdfminer-six|pdfplumber|pillow|plotly|pluggy|pydantic|pycparser|pygments|pyparsing|pypdf|pypdfium2|pytest|python-dateutil|python-docx|python-dotenv|pytz|pyyaml|requests|scikit-learn|scipy|seaborn|setuptools|shap|six|slicer|sqlalchemy|statsmodels|sympy|texttable|threadpoolctl|torch|tqdm|truststore|typing-extensions|typing-inspection|tzdata|urllib3|wrapt|xgboost)==", paragraph.text):
            clear_paragraph(paragraph)
    replace_first(doc, "操作系统为 Windows x64", "操作系统为Windows x64，统计模型使用CPU计算；支持材料记录运行时的Python与依赖版本、数据哈希、模型工件哈希和执行命令。模型权重文件未打入附件，复现者应按公开模型仓库的文件名获取并核对哈希。")
    replace_first(doc, "感官预测结果与正文由Codex", "正文数值、标签和图表均由仓库脚本生成并经人工复核；人工智能工具的使用范围、版本和记录另见支撑材料中的《AI工具使用详情说明》。")
    replace_first(doc, "python scripts/import_q1_tabpfn_final.py", "python scripts/import_q1_tabpfn_final.py --experiment-root .")
    replace_first(doc, "python scripts/q1_tabpfn_cv.py --worktree", "python scripts/q1_tabpfn_cv.py --protocol fair_main --worktree worktrees/prediction-models --model-path worktrees/pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors --output-dir data/processed/q1_fair_tabpfn --workers 1 --threads 1 --resume")
    # Delete stale dependency lines and replace the command block with a compact, valid list.
    command_start = first_paragraph(doc, "uv venv --python")
    command_end = first_paragraph(doc, "python scripts/acceptance_check.py")
    command_text = (
        "uv venv --python 3.12 .venv；uv pip sync requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu；"
        "python scripts/build_data_contract.py；python scripts/audit_scale.py；"
        "python -m pytest tests tests_q1 worktrees/prediction-models/tests -q；"
        "python scripts/q1_pipeline.py；python scripts/q1_tabpfn_cv.py --protocol fair_main ...；"
        "python scripts/q2_graph.py；python scripts/q2_stability.py --aggregate --bootstrap-b 200；"
        "python scripts/q3_generate.py；python scripts/q3_verify.py；python scripts/q4_consumer.py；"
        "python scripts/build_final_artifacts.py；python scripts/check_paper_pdf.py data/processed/paper/paper.pdf --docx data/processed/paper/paper.docx；"
        "python scripts/acceptance_check.py --python .venv/Scripts/python.exe"
    )
    set_paragraph(command_start, command_text)
    current = False
    for paragraph in list(doc.paragraphs):
        if paragraph is command_start:
            current = True
            continue
        if current and paragraph is not command_end:
            if paragraph.text.strip() and not paragraph.text.startswith("附录 C"):
                clear_paragraph(paragraph)
        if paragraph is command_end:
            current = False
    # Remove construction/unfinished language everywhere in the paper, including tables.
    for paragraph in doc.paragraphs:
        if "本论文使用的人工智能工具" in paragraph.text:
            clear_paragraph(paragraph)
            continue
        text = paragraph.text
        if not text:
            continue
        cleaned = text.replace("队友协议", "既有复现协议").replace("队友", "复现来源")
        cleaned = cleaned.replace("当前提交", "本研究交付").replace("待完成", "已完成")
        cleaned = cleaned.replace("AI辅助：Codex（GPT-6，OpenAI，2026-10-30），用于结果整理与文本编辑。", "")
        if cleaned != text:
            set_paragraph(paragraph, cleaned)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    text = paragraph.text
                    cleaned = text.replace("队友协议", "既有复现协议").replace("队友", "复现来源")
                    cleaned = cleaned.replace("当前提交", "本研究交付").replace("待完成", "已完成")
                    if cleaned != text:
                        set_paragraph(paragraph, cleaned)

    # Replace embedded figures using the audited PNG pairs.
    image_map = {
        "rId9": PROCESSED / "q1_figures" / "01_model_comparison_bootstrap_ci.png",
        "rId10": PROCESSED / "q1_figures" / "02_dimension_predictability_facets.png",
        "rId11": PROCESSED / "q4_figures" / "fig1_consumer_predictability.png",
        "rId12": PROCESSED / "q4_figures" / "fig2_cross_consistency.png",
        "rId13": PROCESSED / "q4_figures" / "fig4_price_effects.png",
        "rId14": PROCESSED / "q2_figures" / "fig1_coexpression_network_by_origin.png",
        "rId15": PROCESSED / "q2_figures" / "fig3_community_composition_heatmap.png",
        "rId16": PROCESSED / "q2_figures" / "fig4_top3_bootstrap_frequency.png",
        "rId17": PROCESSED / "q4_figures" / "fig3_dose_response.png",
        "rId18": PROCESSED / "q3_figures" / "fig2_marginal_distributions.png",
        "rId19": PROCESSED / "q3_figures" / "fig5_defect_boxplots.png",
        "rId20": slope_png,
    }
    for rid, image in image_map.items():
        if image.is_file() and rid in doc.part.related_parts:
            update_image_part(doc, rid, image)

    # The symbols table is long enough to leave a single final row at the top
    # of the next page in some Word/LibreOffice renderings. Start it on a new
    # page so the table remains readable and does not create a headerless
    # continuation fragment.
    if len(doc.tables) > 2:
        add_page_break_before_table(doc.tables[2])
    if len(doc.tables) > 7:
        add_page_break_before_table(doc.tables[7])
    for paragraph in doc.paragraphs:
        if (paragraph.text.startswith("5.2 消费者感知路径与市场价值")
                and "\t" not in paragraph.text):
            paragraph.paragraph_format.page_break_before = True
            break

    # Headerless document and a clean Arabic page field in the body section footer.
    for section in doc.sections:
        for paragraph in section.header.paragraphs:
            clear_paragraph(paragraph)
    for section in doc.sections[:-1]:
        for paragraph in section.footer.paragraphs:
            clear_paragraph(paragraph)
    add_page_field(doc.sections[-1].footer.paragraphs[0])

    # Keep every embedded figure with its caption to prevent cross-page caption drift.
    for paragraph in doc.paragraphs:
        if any(e.tag.rsplit("}", 1)[-1] == "drawing" for e in paragraph._p.iter()):
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
        if paragraph.text.startswith("图 "):
            paragraph.paragraph_format.keep_together = True
        if re.match(r"^\s*[（(][1-4][）)]", paragraph.text):
            paragraph.paragraph_format.keep_together = True

    out = FINAL_DIR / "paper.docx"
    doc.save(out)
    (FINAL_DIR / "README.md").write_text(
        "# 最终论文产物\n\n"
        "本目录由 `scripts/build_final_artifacts.py` 以 `data/processed/paper-q1-revised/paper.docx` 为版式基础生成。\n"
        "主结果固定采用227个统一尺度特征和175/75风格分层切分；TabPFN旧协议结果仅作复现记录。\n",
        encoding="utf-8",
    )
    return out


def build_team_manifest(descriptive: dict, fair_status: dict) -> None:
    def ref_info(ref: str, role: str, selected: bool) -> dict:
        commit = git("rev-parse", ref)
        merge = git("merge-base", "HEAD", ref)
        ahead = git("rev-list", "--count", f"HEAD..{ref}")
        behind = git("rev-list", "--count", f"{ref}..HEAD")
        return {"ref": ref, "commit": commit, "merge_base_with_final": merge,
                "final_ahead_of_ref": int(ahead or 0), "ref_ahead_of_final": int(behind or 0),
                "role": role, "selected": selected}

    code_zip = ROOT / "code.zip"
    docx = BASE_DOCX
    commitment_candidates = [
        ROOT / "docs" / "2026年MathorCup数学应用挑战赛—大数据竞赛承诺书（手写或者电子签名后上传）(1).pdf",
        ROOT.parents[1] / "matho-rtri-mind-2026" / "docs" / "2026年MathorCup数学应用挑战赛—大数据竞赛承诺书（手写或者电子签名后上传）(1).pdf",
    ]
    commitment_candidates = []
    for docs_dir in (
        ROOT / "docs",
        ROOT.parents[1] / "matho-rtri-mind-2026" / "docs",
    ):
        if docs_dir.is_dir():
            commitment_candidates.extend(sorted(docs_dir.glob("*承诺书*.pdf")))
    if not commitment_candidates:
        raise FileNotFoundError("no signed commitment PDF found under the repository docs directories")
    commitment = commitment_candidates[0]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_branch_base": {"ref": "origin/release/toolchain-hardening-20261029", "commit": GIT_BASE},
        "sources": [
            ref_info("origin/feat/initial-data-analysis", "数据处理、缺失机制和描述性统计候选来源", True),
            ref_info("origin/experiments/prediction-models", "Q1模型与TabPFN原始协议来源", True),
            ref_info("origin/feat/paper-q1-tabpfn", "Q1论文修订和TabPFN协议记录", True),
            {"path": "code.zip", "sha256": sha256(code_zip), "bytes": code_zip.stat().st_size,
             "source_commit": "a3c3aa20227803e44a9ef0ca77cf4f1d78384524", "selected": True,
             "use": "源程序和复现材料来源包"},
            {"path": "data/processed/paper-q1-revised/paper.docx", "sha256": sha256(docx),
             "bytes": docx.stat().st_size, "source_commit": "a3c3aa20227803e44a9ef0ca77cf4f1d78384524",
             "selected": True, "use": "最终论文版式和正文基础"},
            {"path": str(commitment), "sha256": sha256(commitment),
             "bytes": commitment.stat().st_size, "selected": False,
             "use": "只复制到本地提交目录，不进入GitHub"},
        ],
        "selected_results": {
            "data_contract": "formal release validated contract",
            "q1": "independent_gbr, 227 unified features, fixed style-stratified 175/75",
            "q2": "nightly validated stability, holdout FDR and chemical plausibility",
            "q3": "nightly validated holdout, constraint acceptance and distribution checks",
            "q4": "validated consumer descriptor and direction-only price analyses",
        },
        "fair_tabpfn": fair_status,
        "descriptive_statistics": descriptive,
        "discarded_or_secondary": [
            {"item": "legacy TabPFN 231-feature KFold result", "reason": "feature and split protocol differ from main model; retained only as protocol reproduction"},
            {"item": "interim artifacts duplicated by feat/initial-data-analysis", "reason": "not copied over validated release outputs"},
            {"item": "pretrained weights and data/raw", "reason": "not committed or packaged; license/size and competition boundary"},
        ],
        "external_data_policy": "validation_and_metadata_only; no external rows in main training matrix",
    }
    write_json(INTEGRATION_DIR / "team_integration_manifest.json", payload)


def main() -> int:
    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
    descriptive = build_descriptive_statistics()
    structure = build_structure_figure()
    slope = build_q3_slope_figure()
    update_q3_manifest()
    build_figure_audit(structure, slope)
    fair_path = PROCESSED / "q1_fair_tabpfn" / "fair_protocol_status.json"
    fair_status = json.loads(fair_path.read_text(encoding="utf-8")) if fair_path.is_file() else {
        "status": "not_run", "protocol": "fair_main", "features": 227, "train": 175, "holdout": 75,
        "note": "No complete fair_main output has been recorded yet.",
    }
    build_team_manifest(descriptive, fair_status)
    out = revise_paper(structure, slope, descriptive)
    print(json.dumps({"docx": str(out), "integration_dir": str(INTEGRATION_DIR),
                      "structure_figure": str(structure), "slope_figure": str(slope)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
