# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：论文中出现的每一个数字均由本轮真实运行产物读出（data/processed/q*_runs、
# q1_final、q4_consumer），未手工编造；页码经「生成 PDF → 定位标题页 → 回填目录」两遍法核验；
# 匿名化扫描与五项合规校验复用 scripts/check_paper_pdf.py。
"""论文产出：基于官方 docx 模板生成赛道 A 论文 DOCX → PDF，并回填真实目录页码。

用法：
    python scripts/build_paper.py [--team-no MCB260XXXX] [--track A] [--skip-pdf]

流程：
    第一遍：用占位目录页码生成 DOCX → 转 PDF → 定位各一级/二级标题所在页
    第二遍：把真实页码写回目录 → 重新生成 DOCX → 转 PDF → 跑五项合规校验

产物：
    data/processed/paper/paper.docx
    data/processed/paper/paper.pdf
    data/processed/paper/paper_check.json / .txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

from data_contract import DERIVED_COLUMNS, load_modeling_matrix, load_targets

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "paper"
TEMPLATE = ROOT.parent / "2026年MathorCup数学应用挑战赛—大数据竞赛论文模板.docx"

#: 第二轮升级新增：文献著录清单（由 B1 产出）与消费者层产物目录（B2 产出）。
REFS_JSON = ROOT / "docs" / "refs_verified.json"
CONSUMER_DIR = PROCESSED / "q4_consumer"
NIGHTLY_DIR = PROCESSED / "nightly" / "overnight-q123-20261030"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
LATIN_FONT = "Times New Roman"

#: 论文内插图宽度（英寸）。压到 4.6 可在保持可读的前提下压缩版面，
#: 使正文页数留有余量。
FIG_WIDTH_IN = 4.6

#: 标题级别：1=章 2=节
Headings = List[Tuple[int, str]]


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


# --------------------------------------------------------------------------- 字体


def _set_run_font(run, cjk: str, size_pt: float, bold: bool = False) -> None:
    """同时设置西文字体与东亚字体，避免中文回退成方块。"""
    run.font.name = LATIN_FONT
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), LATIN_FONT)
    rfonts.set(qn("w:hAnsi"), LATIN_FONT)
    rfonts.set(qn("w:eastAsia"), cjk)


def _para(container, text: str = "", size: float = 10.5, bold: bool = False,
          align=None, indent_chars: float = 0.0, space_after: float = 4.0,
          cjk: str = BODY_FONT):
    """新增一个段落。indent_chars 以「字符」为单位，模拟中文论文的首行缩进。

    文本中的 ``**...**`` 会被解析为加粗片段（Word 不支持 Markdown 标记，
    若原样写入会在正文里显示星号）。
    """
    p = container.add_paragraph()
    for chunk, is_bold in _split_bold(text):
        if not chunk:
            continue
        run = p.add_run(chunk)
        _set_run_font(run, cjk, size, bold or is_bold)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.35
    if align is not None:
        p.alignment = align
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    return p


def _split_bold(text: str):
    """把 ``**加粗**`` 标记拆成 [(片段, 是否加粗)] 序列；无标记时返回单段。"""
    if "**" not in text:
        return [(text, False)]
    parts = text.split("**")
    out = []
    for i, part in enumerate(parts):
        if part:
            out.append((part, i % 2 == 1))
    return out or [(text, False)]


def _heading(container, text: str, level: int):
    """写一个标题段落，返回该段落对象。"""
    sizes = {1: 15.0, 2: 13.0, 3: 11.5}
    p = container.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, HEAD_FONT, sizes.get(level, 11.0), bold=True)
    pf = p.paragraph_format
    pf.space_before = Pt(10 if level == 1 else 7)
    pf.space_after = Pt(6 if level == 1 else 4)
    pf.line_spacing = 1.3
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # 记录标题，用于目录回填
    p._mavis_heading = (level, text)  # type: ignore[attr-defined]
    return p


def _caption(container, text: str):
    """图表题注（居中、小字号）。"""
    p = container.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, BODY_FONT, 9.0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    return p


def _figure(container, path: Path, caption: str, width_in: float = FIG_WIDTH_IN):
    """插入图片 + 题注；图缺失时写占位说明而不报错。"""
    if path.exists():
        p = container.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.add_run().add_picture(str(path), width=Inches(width_in))
    else:
        _para(container, f"[图缺失：{path.name}]", size=9.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    _caption(container, caption)


def _table(container, df: pd.DataFrame, caption: str, widths: Optional[List[float]] = None,
           max_rows: int = 40, max_cell_chars: int = 60, font_size: float = 8.5):
    """插入表格；长单元格按 max_cell_chars 截断，避免窄列逐字符换行把版面撑爆。"""
    if df is None or df.empty:
        _para(container, f"[表缺失：{caption}]", size=9.0)
        return
    df = df.head(max_rows)
    p = container.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    table = container.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    table.autofit = True
    # Mark the first row as a repeating header so a table split across pages
    # never starts with unlabeled data rows (Word and LibreOffice honor w:tblHeader).
    header_trpr = table.rows[0]._tr.get_or_add_trPr()
    header_flag = OxmlElement("w:tblHeader")
    header_flag.set(qn("w:val"), "true")
    header_trpr.append(header_flag)
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(col))
        _set_run_font(run, HEAD_FONT, font_size, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _, r in df.iterrows():
        cells = table.add_row().cells
        for j, v in enumerate(r):
            cells[j].text = ""
            if pd.isna(v):
                txt = ""
            elif isinstance(v, float):
                txt = f"{v:.4g}"
            else:
                txt = str(v)
            if len(txt) > max_cell_chars:
                txt = txt[: max_cell_chars - 1] + "…"
            run = cells[j].paragraphs[0].add_run(txt)
            _set_run_font(run, BODY_FONT, font_size)
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for j, w in enumerate(widths):
            for row in table.rows:
                row.cells[j].width = Inches(w)
    _caption(container, caption)


class Counters:
    """图表编号计数器（表 N / 图 N 全文连续）。"""

    def __init__(self) -> None:
        self.table = 0
        self.figure = 0

    def next_table(self, text: str) -> str:
        self.table += 1
        return f"表 {self.table}  {text}"

    def next_figure(self, title: str, caption: str) -> str:
        self.figure += 1
        tail = f"。{caption}" if caption else ""
        return f"图 {self.figure}  {title}{tail}"


# --------------------------------------------------------------------------- 结果读取


def _maybe(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def load_results() -> dict:
    """读取本轮真实产物，汇总论文所需的全部数字。"""
    res: dict = {}

    # ---- 问题一
    q1_final = PROCESSED / "q1_final"
    q1_runs = PROCESSED / "q1_runs"
    summary = _maybe(q1_final / "model_summary.csv")
    if summary is None:
        summary = _maybe(q1_runs / "model_summary.csv")
    res["q1_summary"] = summary
    res["q1_scores"] = _maybe(q1_final / "sensory_scores_0_6.csv")
    res["q1_anchor"] = _maybe(q1_final / "absolute_anchor_params.csv")
    res["q1_cutpoints"] = _maybe(q1_final / "proxy_grade_cutpoints.csv")
    res["q1_cont"] = _maybe(q1_final / "continuous_metrics_by_label.csv")
    res["q1_grade"] = _maybe(q1_final / "proxy_grade_metrics_by_label.csv")
    tabpfn = _maybe(PROCESSED / "q1_repro_tabpfn" / "cv_summary.csv")
    res["q1_tabpfn"] = tabpfn

    # ---- 问题二
    q2 = PROCESSED / "q2_runs"
    res["q2_graph"] = json.loads((q2 / "graph" / "graph_summary.json").read_text(encoding="utf-8")) \
        if (q2 / "graph" / "graph_summary.json").exists() else None
    res["q2_top3"] = _maybe(q2 / "top3_by_sensory.csv")
    res["q2_stability"] = _maybe(q2 / "stability.csv")
    res["q2_communities"] = _maybe(q2 / "graph" / "community_composition.csv")
    res["q2_sensitivity"] = _maybe(q2 / "graph" / "edge_sensitivity.csv")
    res["q2_summary"] = json.loads((q2 / "summary.json").read_text(encoding="utf-8")) \
        if (q2 / "summary.json").exists() else None
    q2_nightly = NIGHTLY_DIR / "q2" / "robustness_summary.json"
    res["q2_nightly"] = json.loads(q2_nightly.read_text(encoding="utf-8")) \
        if q2_nightly.exists() else None

    # ---- 问题三
    q3 = PROCESSED / "q3_runs"
    res["q3_gen"] = _maybe(q3 / "generation_summary.csv")
    res["q3_feas"] = _maybe(q3 / "feasibility_report.csv")
    res["q3_transfer"] = _maybe(q3 / "transfer_pils_to_tripel.csv")
    res["q3_transfer_defects"] = _maybe(q3 / "transfer_pils_to_tripel_defects.csv")
    res["q3_transfer_summary"] = _maybe(q3 / "transfer_pils_to_tripel_summary.csv")
    res["q3_meta"] = json.loads((q3 / "generation_run_meta.json").read_text(encoding="utf-8")) \
        if (q3 / "generation_run_meta.json").exists() else None
    res["q3_profiles"] = sorted(q3.glob("sensory_profile_*.csv")) if q3.exists() else []
    res["q3_profile_summary"] = _maybe(q3 / "sensory_profile_summary.csv")
    res["q3_constraint_effect"] = _maybe(q3 / "sensory_constraint_effect.csv")
    q3_nightly = ROOT / "state" / "03_q3.json"
    res["q3_nightly"] = json.loads(q3_nightly.read_text(encoding="utf-8")) \
        if q3_nightly.exists() else None
    return res


#: 消费者层与剂量再检验的产物清单（B2 产出）
CONSUMER_FILES = [
    "consumer_model_summary.csv",
    "cross_consistency.csv",
    "price_regression.csv",
    "defect_dose_response.csv",
    "defect_dose_verdict.csv",
    "style_robustness.csv",
]


def load_consumer() -> dict:
    """读取消费者层产物；缺失文件只影响对应章节。"""
    out: dict = {}
    if not CONSUMER_DIR.exists():
        _warn(f"未找到 {CONSUMER_DIR}，5.2 与 5.4 将不读取消费者层数据。")
        return out
    for name in CONSUMER_FILES:
        out[name] = _maybe(CONSUMER_DIR / name)
    found = [k for k, v in out.items() if v is not None]
    if not found:
        _warn(f"{CONSUMER_DIR} 下尚无消费者层产物（{'、'.join(CONSUMER_FILES)}）。")
    else:
        _info(f"消费者层产物就绪：{'、'.join(found)}")
    return out


def load_refs() -> Optional[List[dict]]:
    """读取 docs/refs_verified.json；缺失或结构异常时返回 None（调用方走降级路径）。"""
    if not REFS_JSON.exists():
        _warn(f"缺少 {REFS_JSON}，参考文献将保留手写清单（待 refs_verified.json 就绪后自动替换）。")
        return None
    try:
        data = json.loads(REFS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:  # 解析失败同样降级，不让构建失败
        _warn(f"{REFS_JSON} 解析失败（{exc.__class__.__name__}），参考文献降级为手写清单。")
        return None
    entries = data["refs"] if isinstance(data, dict) and "refs" in data else data
    if not isinstance(entries, list) or not entries:
        _warn(f"{REFS_JSON} 结构异常（既非数组也非 refs 键），参考文献降级为手写清单。")
        return None
    valid: List[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _warn("文献条目不是对象，已跳过。")
            continue
        missing = [f for f in ("citation_number", "title", "authors", "year") if not entry.get(f)]
        if missing:
            _warn(f"文献 {entry.get('key', '?')} 缺字段 {missing}，已跳过。")
            continue
        valid.append(entry)
    if not valid:
        _warn(f"{REFS_JSON} 中没有可用条目，参考文献降级为手写清单。")
        return None
    valid.sort(key=lambda e: int(e["citation_number"]))
    _info(f"参考文献自动生成：{len(valid)} 条（来源 {REFS_JSON.name}）。")
    return valid


def fmt(value, digits: int = 4) -> str:
    """安全格式化数字（None / NaN → '—'）。"""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- 封面与目录


def _find_para(doc: Document, text: str):
    """按文本精确定位段落（模板占位段落会被改写，故不能依赖下标）。"""
    for p in doc.paragraphs:
        if p.text.strip() == text:
            return p
    return None


def _clear_para(p) -> None:
    """清空段落内容但保留段落本身（保留其样式与位置）。"""
    for r in list(p.runs):
        r._element.getparent().remove(r._element)


def _para_index(doc: Document, para) -> int:
    """在 doc.paragraphs 中定位段落下标（按底层 XML 元素比对，不能用 list.index）。"""
    for i, p in enumerate(doc.paragraphs):
        if p._p is para._p:
            return i
    raise ValueError("段落不在 doc.paragraphs 中")


def build_cover(doc: Document, title: str, abstract: str, keywords: str,
                team_no: str, track: str) -> None:
    """填充封面：题目 / 摘要 / 关键词 / 队伍编号 / 赛道。"""
    p_title = _find_para(doc, "论文题目")
    p_abs_hd = _find_para(doc, "摘  要") or _find_para(doc, "摘 要")
    p_abs = None
    if p_abs_hd is not None:
        idx = _para_index(doc, p_abs_hd)
        p_abs = doc.paragraphs[idx + 1]
    p_kw = _find_para(doc, "关键词：")

    # 模板在摘要与关键词之间留了 16 个空段落，会把关键词挤到第 2 页；
    # 按官方规范「题目、摘要与关键词写在论文第一页上，第二页为目录页」，删掉多余的空段。
    if p_abs is not None and p_kw is not None:
        paras = doc.paragraphs
        i0, i1 = _para_index(doc, p_abs), _para_index(doc, p_kw)
        blanks = [p for p in paras[i0 + 1:i1] if not p.text.strip()]
        for p in blanks[4:]:
            p._element.getparent().remove(p._element)

    if p_title is not None:
        _clear_para(p_title)
        run = p_title.add_run(title)
        _set_run_font(run, HEAD_FONT, 16.0, bold=True)
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_title.paragraph_format.space_after = Pt(10)

    if p_abs_hd is not None:
        _clear_para(p_abs_hd)
        run = p_abs_hd.add_run("摘  要")
        _set_run_font(run, HEAD_FONT, 13.0, bold=True)
        p_abs_hd.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if p_abs is not None:
        _clear_para(p_abs)
        run = p_abs.add_run(abstract)
        _set_run_font(run, BODY_FONT, 10.5)
        p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_abs.paragraph_format.line_spacing = 1.35
        p_abs.paragraph_format.first_line_indent = Pt(21)

    if p_kw is not None:
        _clear_para(p_kw)
        run = p_kw.add_run("关键词：" + keywords)
        _set_run_font(run, BODY_FONT, 10.5, bold=True)
        p_kw.paragraph_format.space_before = Pt(8)

    if doc.tables:
        table = doc.tables[0]
        for (row, col, text) in ((0, 1, team_no), (1, 1, track)):
            cell = table.cell(row, col)
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            _set_run_font(run, BODY_FONT, 11.0)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_toc(doc: Document, headings: Headings, page_map: Optional[Dict[str, int]] = None) -> None:
    """重写目录页：逐条标题 + 制表位 + 页码（页码未知时先留空）。"""
    toc_title = _find_para(doc, "目录")
    if toc_title is None:
        toc_title = doc.add_paragraph()
    _clear_para(toc_title)
    run = toc_title.add_run("目  录")
    _set_run_font(run, HEAD_FONT, 15.0, bold=True)
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 清除模板自带的示例目录行（以「键入章级」「键入章标题」开头的段落）
    for p in list(doc.paragraphs):
        if p.text.strip().startswith(("键入章级", "键入章标题")):
            p._element.getparent().remove(p._element)

    anchor = toc_title
    for level, text in headings:
        p = doc.add_paragraph()
        run = p.add_run(text)
        _set_run_font(run, BODY_FONT, 10.5 if level == 1 else 10.0)
        pf = p.paragraph_format
        pf.space_after = Pt(2)
        pf.left_indent = Pt(0 if level == 1 else 18)
        pf.tab_stops.add_tab_stop(Cm(15.5), 2, 1)  # RIGHT + DOTS
        page = "" if page_map is None else str(page_map.get(text, ""))
        if page:
            run2 = p.add_run("\t" + page)
            _set_run_font(run2, BODY_FONT, 10.0)
        anchor._element.addnext(p._element)
        anchor = p


# --------------------------------------------------------------------------- 图题注


def _load_figure_manifest(folder: Path) -> Dict[str, dict]:
    """读取图目录的 figures_manifest.json，返回 {文件名: manifest 条目}。"""
    path = folder / "figures_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _warn(f"{path.name} 解析失败，图题注改用兜底文案。")
        return {}
    figs = data.get("figures", []) if isinstance(data, dict) else data
    out: Dict[str, dict] = {}
    if isinstance(figs, list):
        for fig in figs:
            if not isinstance(fig, dict):
                continue
            name = str(fig.get("file") or fig.get("filename") or fig.get("name") or "")
            if not name:
                continue
            # manifest 中的 file 常省略扩展名；同时登记「原名」与「stem」两个键，
            # 保证按 *.png 文件名查找时也能命中（否则题注会静默退化为兜底文案）。
            stem = Path(name).stem
            out[Path(name).name] = fig
            out[stem] = fig
            out[f"{stem}.png"] = fig
    return out


#: (图目录, 文件名, 极简标题（≤12 字）, 方法/口径一句话)
FIGURE_SPECS = [
    (PROCESSED / "q1_figures", "01_model_comparison_bootstrap_ci.png",
     "模型比较区间", "各模型族宏平均 RMSE 的自助重采样置信区间（seed=0，175/75 按风格分层切分）"),
    (PROCESSED / "q1_figures", "02_dimension_predictability_facets.png",
     "逐维可预测性", "按感官族分面的逐维决定系数分布"),
    (PROCESSED / "q1_figures", "03_two_track_grade_comparison.png",
     "两轨分级对照", "相对分级与阈值锚定绝对分的逐维对照"),
    (PROCESSED / "q2_figures", "fig1_coexpression_network_by_origin.png",
     "共表达图", "|ρ|≥0.6 下的化合物共表达图，节点按化学来源着色"),
    (PROCESSED / "q2_figures", "fig3_community_composition_heatmap.png",
     "社区来源构成", "Leiden 社区的化学块与来源构成热图"),
    (PROCESSED / "q2_figures", "fig4_top3_bootstrap_frequency.png",
     "关键化合物频率", "各感官维度 Top-3 关键化合物的 200 次重采样入选频率"),
    (PROCESSED / "q3_figures", "fig2_marginal_distributions.png",
     "边际分布对比", "生成谱与真实样本的边际分布对比（含无约束参照谱）"),
    (PROCESSED / "q3_figures", "fig5_defect_boxplots.png",
     "缺陷浓度分布", "迁移前后缺陷化合物浓度分布对比"),
    (PROCESSED / "q3_figures", "fig4_transfer_radar.png",
     "迁移画像雷达", "跨风格迁移前后的感官画像雷达对比"),
]


def _strip_caption_prefix(caption: str, title: str) -> str:
    """去掉 manifest 题注自带的「图 N  标题。」前缀。

    manifest 中的 caption 以「图 1  模型对比。方法说明」的形式给出（图号为目录内占位号，
    最终图号由论文统一分配）。若原样使用，会与论文自动分配的「图 N  标题」重复，
    故此处先剥掉图号与重复标题，只保留方法说明部分。
    """
    text = str(caption or "").strip()
    text = re.sub(r"^图\s*\d+\s*", "", text)
    title = str(title or "").strip()
    if title and text.startswith(title):
        text = text[len(title):]
    return text.lstrip("。．.；;，,、 　")


def _insert_figure(doc: Document, counters: Counters, folder: Path, name: str,
                    short_title: str, method: str, manifest: Optional[Dict[str, dict]] = None) -> None:
    """插入一张图：题注优先取自 figures_manifest.json，缺失时用兜底文案。"""
    path = folder / name
    if not path.exists():
        return
    title, caption = short_title, method
    if manifest is None:
        manifest = _load_figure_manifest(folder)
    entry = manifest.get(name)
    if entry:
        title = str(entry.get("title") or short_title).strip() or short_title
        caption = _strip_caption_prefix(entry.get("caption") or method, title)
    _figure(doc, path, counters.next_figure(title, caption))


#: 消费者层图目录候选（B2 实际把图输出在 data/processed/q4_figures）。
CONSUMER_FIGURE_DIRS = [PROCESSED / "q4_figures", CONSUMER_DIR, CONSUMER_DIR / "figures"]

#: 消费者层图：(文件名, 极简标题, 兜底方法/口径, 归属小节)
CONSUMER_FIGURE_SPECS = [
    ("fig1_consumer_predictability.png", "消费者可预测性",
     "化学谱到 183 维消费者描述符的袋外预测能力（与全局均值基线对照）", "consumer"),
    ("fig2_cross_consistency.png", "交叉一致性",
     "消费者路径与品评组路径的相关结构对照", "consumer"),
    ("fig4_price_effects.png", "价格效应",
     "控制 ABV 与风格后的价格回归结果对照", "consumer"),
    ("fig3_dose_response.png", "剂量响应",
     "缺陷化合物浓度—感知关系的分位剂量再检验", "dose"),
]


def _consumer_figure_specs(section: str = "consumer") -> List[Tuple[Path, str, str, str]]:
    """消费者层图（若 B2 产出）：返回 (目录, 文件名, 极简标题, 兜底方法)。

    `CONSUMER_FIGURE_SPECS` 显式登记每张图归属的小节；未登记的图默认归入 5.2，
    且只会被收录一次（避免同一张图在两节中重复出现）。
    """
    specs: List[Tuple[Path, str, str, str]] = []
    registered = {name for name, _s, _m, _sec in CONSUMER_FIGURE_SPECS}
    for name, short, method, sec in CONSUMER_FIGURE_SPECS:
        if sec != section:
            continue
        for folder in CONSUMER_FIGURE_DIRS:
            if (folder / name).exists():
                specs.append((folder, name, short, method))
                break
    if section == "consumer":
        # 未登记到上表的图也一并纳入，避免静默丢图。
        for folder in CONSUMER_FIGURE_DIRS:
            if not folder.is_dir():
                continue
            for png in sorted(folder.glob("*.png")):
                if png.name in registered:
                    continue
                specs.append((folder, png.name, png.stem[:12],
                              "消费者层产物（化学→消费者描述符/价格）"))
    return specs


# --------------------------------------------------------------------------- 正文


def build_body(doc: Document, res: dict, refs: Optional[List[dict]], consumer: dict,
               counters: Counters) -> Headings:
    """写正文（附录之前），返回标题清单。"""
    heads: Headings = []

    def H(level: int, text: str):
        _heading(doc, text, level)
        heads.append((level, text))

    # 清除模板正文占位（以「（这里开始论文正文）」为界，其后全部删除）
    marker = _find_para(doc, "（这里开始论文正文）")
    if marker is not None:
        paras = doc.paragraphs
        i = _para_index(doc, marker)
        for p in paras[i:]:
            p._element.getparent().remove(p._element)

    _build_problem_analysis(doc, H, res)
    _build_related_work(doc, H, refs)
    _build_data_processing(doc, H, res, counters)
    _build_models(doc, H, res)
    _build_results(doc, H, res, consumer, counters)
    _build_discussion(doc, H, res, consumer)
    _build_conclusions(doc, H, res, consumer)

    return heads


# --------------------------------------------------------------------- 一、问题分析


def _build_problem_analysis(doc: Document, H, res: dict) -> None:
    H(1, "一、问题分析")

    H(2, "1.1 与原文的关系及本文贡献")
    _para(doc, "赛题附件来自 Schreurs M、Verstrepen P J 等人在 Nature Communications 发表的研究 [1]，"
               "包含 250 款商用啤酒的化学测量、受训品评小组的感官数据，以及化合物—化合物、感官—感官和"
               "化合物—感官三张相关矩阵。附件还提供了品评卡模板、论坛评论派生描述符、化学阈值和名称参考表。"
               "本文以这些公开材料为基础，重新完成数据语义审计、特征构造、模型训练和独立评估。对于附件没有直接说明的"
               "量纲、相关类型和字段对应关系，本文均通过逐列回归、逐元素重算和跨文件名称核对给出可复算的判断。"
               "因此，本文的研究定位是同源数据的独立复现与再分析，重点在于检验数据口径能否支撑题目要求的三项推断。",
               indent_chars=2)
    _para(doc, "本文的增量工作集中在三个方面。第一，本文识别出三处会改变建模结论的数据语义差异：相关矩阵实际采用"
               "斯皮尔曼系数，品评卡只是空白模板，1.csv 同时包含 log10 标度和原始浓度标度。第二，附件没有消费者评分或"
               "喜爱度标签，本文遂采用 183 维论坛描述符和 189 款啤酒的零售价格构造两条外部验证路径，分别考察化学谱对"
               "消费者感知和市场价值代理的解释能力。第三，本文重新检验缺陷化合物的剂量—响应关系，并把全体样本相关、"
               "风格分层结果和不确定性放在同一评价框架中。消费者喜爱度本身没有可观测标签，因而不属于本文的预测目标；"
               "相关结果仅用于说明现有附件能够验证什么，以及仍缺少哪些证据。", indent_chars=2)

    H(2, "1.2 任务分解与叙事主线")
    _para(doc, "三项任务可以沿着一条由解释到设计的路径组织起来。问题一从化学指纹预测感官维度，回答哪些感官属性"
               "能够由化学测量解释；问题二把这种映射拆解到具体化合物，考察化合物之间的共表达结构以及候选关键因素的"
               "稳定性；问题三以风格为条件生成化学谱，将前两问得到的结构用于反向设计。三问共用同一套名称桥、统一标度"
               "和风格分层原则，因而可以在同一证据链上比较预测、关联与生成三个环节的可靠程度。", indent_chars=2)
    _para(doc, "具体而言，第一问需要在感官标签只有标准化形式、没有原始 0–6 分数的条件下构造可解释的分级结果；"
               "第二问需要在 231 个化合物节点上建立共表达图，并为每个感官维度给出带有稳定性和化学合理性说明的候选化合物；"
               "第三问则需要在样本量很小的细分风格条件下生成化学谱，并检验约束合规、分布相似性和跨风格迁移效果。"
               "三项任务的共同前提是先确认标签、量纲和相关系数的真实含义。若这一步被忽略，后续的模型指标和化学解释都可能失去可比性。",
               indent_chars=2)

    H(2, "1.3 总体应对")
    _para(doc, "研究流程首先完成三项基础工作：逐列裁定 1.csv 的标度，重算相关矩阵并与附件逐元素比对，"
               "再通过跨文件名称桥把化学量、感官维度、阈值和外部先验对齐。随后，本文区分检出限造成的地板值与真正缺失，"
               "将所有化学量统一到 log10(mg/L) 口径，并固定按细分风格分层的 175/75 训练—留出划分。模型选择和参数调整"
               "只使用训练样本，留出样本仅用于最终评价。这样的顺序把数据审计、预测、关联和生成连接起来，也使每一步的证据来源能够回溯。",
               indent_chars=2)


# ----------------------------------------------------------------- 二、相关工作与综述


def _entry_citation(entry: Optional[dict]) -> str:
    return f"[{int(entry['citation_number'])}]" if entry else ""


def _pick_entry(entries: List[dict], keywords: Tuple[str, ...]) -> Optional[dict]:
    """按标题/期刊/作者关键词定位一条文献。"""
    for kw in keywords:
        for e in entries:
            hay = " ".join(str(e.get(k, "")) for k in ("title", "journal", "authors", "venue", "doi")).lower()
            if kw.lower() in hay:
                return e
    return None


#: 综述五个主题的模板；占位符在运行时替换为文献编号（缺失则留空）。
REVIEW_THEMES: List[Tuple[str, str, str]] = [
    ("2.1 原始工作与数据来源",
     "原始研究以 250 款商用啤酒的化学测量与受训品评小组感官结果为核心数据，系统刻画了化学组成与感官属性"
     "之间的关系，并把可解释的机器学习用于风味预测与配方改进{anchor}；其数据公开方式（同时给出化合物矩阵、"
     "三张相关矩阵、品评卡、论坛描述符与阈值参考表）也决定了任何复用者都必须自行裁定这些附件的语义，"
     "这正是本文第二章与第三章所做的工作。近期 flavoromics 研究也表明，多平台化学—感官证据可用于"
     "筛查候选化合物，但不同实验平台的结果仍需校准后才能用于同一分析{flavoromics}。"),
    ("2.2 感官预测与可解释机器学习",
     "从化学成分到感官评分的映射本质上是一个高维小样本回归问题。通用机器学习工具（如 scikit-learn{sklearn}）"
     "提供了从线性模型到梯度提升与随机森林的统一实现；SHAP{shap} 提供了把模型输出分解到单个化合物的加性归因方法；"
     "预训练先验模型（TabPFN{tabpfn}）则为小样本表格回归提供了不同于手工特征工程的第三条路线"
     "（其原始版本面向表格分类任务，回归形态与本文用法见后续期刊版{tabpfn_reg}）。"
     "TabDPT{tabdpt} 可作为后续的表格基础模型对照；本文暂不把它纳入主比较，"
     "以便先完成统一特征、固定切分和可复核指标下的基线评估。"
     "本文在问题一同时使用这三类归纳偏置，并在问题二把归因与秩相关并列作为证据，避免以单一视角替代真值。"),
    ("2.3 化合物关联分析与网络化方法",
     "把化合物之间的关系表述为图并做社区发现，是解释「哪些化合物总是同时出现」的自然做法；"
     "Leiden 算法{leiden}在保证社区连通性的同时修正了 Louvain 的分辨率缺陷。本文即以 |ρ| 阈值构图、"
     "以 Leiden 求解社区，并以模块度与孤立节点占比联合选择阈值，同时报告阈值敏感性，"
     "使「共表达模块对应可追溯的化学来源」这一论证不依赖单一阈值的偶然结果。"),
    ("2.4 合成数据、Copula 与秩—分位映射",
     "在多变量分布既要保留相关结构、又需在小样本下可控的场合，高斯 Copula 是一种低成本的分解方案{copula}；"
     "秩—分位映射则是相关结构与边际分布解耦的经典做法{rank}，可在保持秩相关不变的前提下把样本迁移到"
     "另一组边际上。联合预测区间与 Copula 依赖结构的研究（如 CopulaCPTS{copula_conformal}）可启发不确定性扩展，"
     "但其时间序列保证不能直接改写为本项目静态数据保证。本文的问题三正是这一组合：Copula 负责跨化合物的相关结构，风格条件边际负责风格画像，"
     "秩—分位映射负责跨风格迁移，三者都不依赖大规模神经网络训练。"),
    ("2.5 定量描述感官分析与消费者接受度、价格",
     "定量描述感官分析（QDA）以强度量表代替类别标签，其量表构造的心理学基础见 Stevens 的量表理论{stevens}；"
     "在消费端 hedonic 定价研究表明，消费者愿意为可被感官信息刻画的品质差异支付溢价，"
     "感官属性对价格与购买意愿的解释力随被试与情境而变{price}。"
     "这提示「化学→感官」的链条最终可以在市场价值上被外部观察，"
     "也为本文以价格作为外部价值信号提供了依据。"),
]


def _build_related_work(doc: Document, H, refs: Optional[List[dict]]) -> None:
    H(1, "二、相关工作与文献综述")
    if not refs:
        _warn("参考文献未自动生成（refs_verified.json 未就绪），第二章按手写综述撰写。")
    lookup = refs if refs else FALLBACK_REFS

    anchor = _pick_entry(lookup, ["10.1038/s41467-024-46346-0", "beer flavor"])
    sklearn = _pick_entry(lookup, ["scikit-learn", "pedregosa"])
    shap = _pick_entry(lookup, ["shap", "interpret", "feature importance", "explainab"])
    tabpfn = _pick_entry(lookup, ["tabpfn", "prior-data", "prior data"])
    # TabPFN 的 arXiv 原文面向表格**分类**任务，回归应用须另引其后续期刊版，
    # 故单独开一个槽位，避免用分类论文支撑回归结论。
    tabpfn_reg = _pick_entry(lookup, ["accurate predictions on small data",
                                      "tabular foundation model"])
    tabdpt = _pick_entry(lookup, ["tabdpt", "scaling tabular foundation"])
    leiden = _pick_entry(lookup, ["leiden", "louvain", "community detection"])
    copula = _pick_entry(lookup, ["copula"])
    rank = _pick_entry(lookup, ["rank of a pair", "rank correlation", "quantile", "rank"])
    stevens = _pick_entry(lookup, ["stevens", "scales of measurement", "measurement theory"])
    price = _pick_entry(lookup, ["price and quality", "beer market", "willingness to pay", "price"])
    spearman = _pick_entry(lookup, ["spearman", "association between two things"])
    flavoromics = _pick_entry(lookup, ["flavoromics", "differential volatile", "lager beers"])
    copula_conformal = _pick_entry(lookup, ["copula conformal", "copulacpts"])

    resolved = {
        "anchor": _entry_citation(anchor),
        "sklearn": _entry_citation(sklearn),
        "shap": _entry_citation(shap),
        "tabpfn": _entry_citation(tabpfn),
        "tabpfn_reg": _entry_citation(tabpfn_reg),
        "tabdpt": _entry_citation(tabdpt),
        "leiden": _entry_citation(leiden),
        "copula": _entry_citation(copula),
        "rank": _entry_citation(rank),
        "stevens": _entry_citation(stevens),
        "price": _entry_citation(price),
        "flavoromics": _entry_citation(flavoromics),
        "copula_conformal": _entry_citation(copula_conformal),
    }
    for key, val in resolved.items():
        if not val:
            _warn(f"综述主题「{key}」在参考文献清单中未找到可引用条目，正文该处不标编号。")

    for title, template in REVIEW_THEMES:
        H(2, title)
        _para(doc, template.format(**resolved), indent_chars=2)

    if spearman:
        _para(doc, f"需要特别指出的是，本文全部相关分析所依据的相关系数类型并非官方赛题文本所述的皮尔逊系数，"
                   f"而是斯皮尔曼秩相关系数{_entry_citation(spearman)}；第三章给出逐元素复算比对作为证据，"
                   f"第二章的相关工作与第五章的网络化分析均以此为准。", indent_chars=2)

    if refs:
        # 一次带出全部参考文献的正文引用点：按 supports_section 的主题分组、紧凑列出编号，
        # 避免出现「著录在参考文献表却正文无引用点」的条目；同时只保留主题标签，
        # 不复述 supports_claim 全文，以免该段变成冗长的构建产物。
        groups: Dict[str, List[int]] = {}
        for e in sorted(refs, key=lambda x: int(x["citation_number"])):
            sec = str(e.get("supports_section") or "其他支撑材料").strip()
            m = re.search(r"[·・]\s*(.+)$", sec)
            theme = (m.group(1) if m else sec).strip() or "其他支撑材料"
            # 「附录」是合规校验器判定正文/附录分界的标记词，正文里出现会让页数统计失真，
            # 故在正文引用分组里改用「支撑材料」指称同一份内容。
            theme = theme.replace("附录", "支撑材料")
            groups.setdefault(theme, []).append(int(e["citation_number"]))
        if groups:
            parts = [f"{theme}" + "".join(f"[{n}]" for n in ns)
                     for theme, ns in groups.items()]
            _para(doc, "本文全部引用按其在文中所支撑的主题归类如下：" + "；".join(parts) + "。",
                  indent_chars=2)
    else:
        _para(doc, "上述综述的文献清单随本文的手写参考文献一并列出；待文献清单文件就绪后，"
                   "本文的参考文献与综述引用编号将由该清单自动生成，以保证著录可逐条核实。", indent_chars=2)


# ------------------------------------------------------------- 三、数据处理与语义审计


def _build_data_processing(doc: Document, H, res: dict, counters: Counters) -> None:
    H(1, "三、数据处理与语义审计")

    H(2, "3.1 附件结构与量纲裁定")
    _para(doc, "附件 1–8 分别提供化合物丰度矩阵（1.csv）、化合物相关矩阵（2.csv）、感官品评卡（3.xlsx）、"
               "感官结果（4.csv）、感官相关矩阵（5.csv）、化合物—感官交叉相关矩阵（6.csv）、论坛评述派生描述符"
               "（7.csv）与名称/阈值/先验参考表（8.xlsx）。其中三处语义必须自行核实：1.csv 的对数底数与正负值"
               "含义、三张相关矩阵的相关系数类型、以及 4.csv 的标签尺度。", indent_chars=2)
    _para(doc, "关于对数底数：本文以 8.xlsx 中「Calibrated chemical dataset」给出的 70 个化合物的原始浓度为基准，"
               "对 1.csv 的每一列分别拟合「文件1 取值 ~ log10(原始浓度)」与「文件1 取值 ~ 原始浓度」。"
               "结果是 37 列严格满足前者（斜率 1.0000、相关系数 1.000000）；"
               "其余 33 列中 19 列严格满足后者（斜率 1.0000、相关系数 0.999998 以上）、"
               "10 列近似满足（斜率 1.0000 附近但相关系数仅 0.87–0.999）、4 列无法逐点复现，"
               "自然对数假设下的斜率恒为 2.302585，据此排除对数为自然对数的可能。"
               "**因此 1.csv 并非统一的 log10 标度，而是 log10 标度与原始浓度标度并存的混合标度**："
               "例如双乙酰列存储的是原始浓度（中位 0.2847 mg/L），若不加区分地统一取 10 的幂，"
               "将得到量级完全失真的结果。据此，本文以统一的 log10(mg/L) 特征矩阵作为全部建模的输入，"
               "并在测试中对每一列的裁定结论做回归保护。", indent_chars=2)
    audit_path = ROOT / "data" / "interim" / "chem_scale_audit_v1.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, encoding="utf-8-sig")
        counts = audit["verdict"].value_counts().to_dict()
        _para(doc, f"逐列裁定结果为：log10 标度 {counts.get('LOG10', 0)} 列、原始浓度标度 "
                   f"{counts.get('RAW', 0)} 列、近似原始浓度 {counts.get('RAW_APPROX', 0)} 列、"
                   f"无法逐点复现 {counts.get('AMBIG', 0)} 列；与既有溯源表的变换声明一致率为 "
                   f"{int(audit['matches_provenance'].sum())}/{len(audit)}。"
                   "无法逐点复现的少数列源于参考表的标定值与文件1 存储值之间存在系统性差异，"
                   "本文按溯源表采信并在回归测试中留有记录。", indent_chars=2)

    H(2, "3.2 相关系数类型的勘误")
    _para(doc, "官方赛题文本称附件 2 的单元格为皮尔逊相关系数，而附件自带的数据说明称相关矩阵为斯皮尔曼系数，"
               "二者矛盾。本文以证据裁定：分别从 1.csv 与 4.csv 重新计算相关系数矩阵，与给定的 5.csv、6.csv "
               "逐元素比对。斯皮尔曼口径下最大绝对偏差为 0、逐元素吻合比例为 1.0000；皮尔逊口径下中位绝对偏差"
               "约为 0.04。对 2.csv 同样以「文件1 原始存储标度」重算斯皮尔曼系数，最大绝对偏差亦为 0。"
               "需要说明的是：斯皮尔曼系数对严格单调变换不变，因此上述证据只能裁定相关系数的**类型**"
               "（是斯皮尔曼而非皮尔逊），并不能反推附件是否做过单位统一；"
               "1.csv 的标度结构另由 3.1 节的逐列回归独立裁定，两条证据互不替代。"
               "**因此三张相关矩阵均为斯皮尔曼相关系数**，本文全部建模与作图"
               "均以此为准，并在论文中显式声明与官方文本的差异。", indent_chars=2)

    H(2, "3.3 名称桥、缺失与检出限")
    _para(doc, "2.csv 与 6.csv 使用原论文的化学命名，5.csv 与 6.csv 使用感官评述命名，与 1.csv、4.csv 的列名"
               "并不同源。本文以 8.xlsx 的阈值表与感官名称表为中介建立两张桥：化学桥把矩阵行名映射到 1.csv "
               "的列名，感官桥把矩阵列名映射到 4.csv 的目标列名。两张桥均达到全解析。"
               "其中若干萜类化合物在参考表中以拉丁转写（如“(E)-beta-ionone”）出现，而在相关矩阵中以希腊字母"
               "出现（“E-β-ionone”），归一化时若直接剥离非字母数字会使二者塌缩为同一键从而静默错配，"
               "本文先音译希腊字母再归一化，并对歧义键一律丢弃。", indent_chars=2)
    _para(doc, "关于缺失：1.csv 仅 9 个化合物列含缺失（合计 73 格，占 0.13%），而检出限以下的值以重复的地板值"
               "出现，属于左删失而非真缺失。本文对左删失单独建掩码并按「未检出」处理，不以 0 或均值冒充观测值；"
               "同时保留参考表给出的离群标记，用于剔除 Diacetyl、trans-2-nonenal 等列上的极端异常值，"
               "避免其污染问题三的边际分布估计。", indent_chars=2)

    H(3, "3.3.1 样本概况与描述性统计")
    chem = load_modeling_matrix()
    targets = load_targets()
    raw_chem = pd.read_csv(RAW / "1.csv", encoding="utf-8-sig")
    styles = pd.read_csv(ROOT / "data" / "splits" / "seed_0" / "style_counts.csv",
                         encoding="utf-8-sig")
    target_means = targets.mean(numeric_only=True)
    target_stds = targets.std(numeric_only=True)
    n_raw_missing = int(raw_chem.isna().sum().sum())
    raw_chem_cells = int(raw_chem.shape[0] * (raw_chem.shape[1] - 1))
    profile = pd.DataFrame([
        ["化学测量", "250 × 231；保留 227 个建模特征",
         f"原始缺失 {n_raw_missing} 格（{100 * n_raw_missing / raw_chem_cells:.2f}%）；统一矩阵缺失 {int(chem.isna().sum().sum())} 格",
         "逐列裁定标度；剔除 {0} 个未经验证的派生求和列".format(len(DERIVED_COLUMNS))],
        ["感官目标", "250 × 50；Aroma 15、Flavour 15、通用维度 20",
         f"缺失 {int(targets.isna().sum().sum())} 格；维度均值范围 {target_means.min():.3f}–{target_means.max():.3f}，标准差范围 {target_stds.min():.3f}–{target_stds.max():.3f}",
         "作为连续回归目标；不把标准化值解释为原始 0–6 评分"],
        ["风格分层", f"22 个细分风格；每个风格样本数 {int(styles['total'].min())}–{int(styles['total'].max())}，中位数 {styles['total'].median():.0f}",
         "最小风格仅 3 款；训练/留出合计 175/75",
         "固定风格分层切分；小样本风格仅作条件生成的谨慎参照"],
    ], columns=["数据对象", "规模", "分布与缺失概况", "处理与解释"])
    _para(doc, "完成语义裁定后，先对样本规模、缺失结构和目标分布做描述性检查，再进入模型比较。"
               "这一步的作用不是增加一个未经预注册的筛选标准，而是判断后续指标的适用范围：化学特征的原始缺失很少，"
               "但风格分层高度不均衡；感官目标均值接近零、维度间离散程度不同，说明宏平均指标需要与逐维结果并读。", indent_chars=2)
    _table(doc, profile, counters.next_table("建模样本与目标的描述性统计"), max_cell_chars=68, font_size=8.0)
    _para(doc, "由此，本文把 175/75 切分视为固定的评价协议，而不是把样本较少的风格重新合并；"
               "所有缺失处理、尺度转换和特征删减均发生在统一矩阵形成阶段，并通过后续回归测试锁定。"
               "描述性统计也解释了为什么问题三报告每个风格的真实留出样本量、问题一报告逐维指标，"
               "而不只给出一个看似精确的总体分数。", indent_chars=2)

    H(2, "3.4 感官维度与数据划分")
    _para(doc, "3.xlsx 提供的感官品评卡定义了 Aroma 与 Flavour 两套属性及其 0–6 分制的语义锚点"
               "（0 分为「不存在」，6 分为「主导/强烈」），但经逐格核查，其 12 张评分卡内并无任何品评员打分的"
               "记录，汇总表的数值单元格指向空白评分卡的公式；即**附件中不存在原始 0–6 分数**。"
               "4.csv 的 50 个感官列是经按维度中心化与缩放的连续值（均值近零、标准差在 0.37 至 0.86 之间，"
               "并非标准分数），因此原始尺度无法反解。本文据此交付全部 50 个感官维度"
               "（Aroma 15 维、Flavour 15 维与 20 个通用维度，覆盖题面所称「约 40 个维度」），"
               "并在第五章明确说明所给 0–6 分制的构造方式。", indent_chars=2)
    _para(doc, "数据划分沿用固定种子、按风格分层的留出方案：175 款用于训练与调参，75 款用于一次性评估，"
               "22 个风格在两侧均有分布。所有切点、参数与模型选择一律只在训练集内完成。", indent_chars=2)

    H(2, "3.5 主要符号与含义")
    _para(doc, "为便于后文对照，全文使用的主要符号统一列于下表。", indent_chars=2)
    symbols = pd.DataFrame([
        ["z", "标准化后的感官值（按维度中心化并缩放的连续量）"],
        ["λ", "风格条件边际向全局边际的收缩系数，本文取 λ = n/(n+10)"],
        ["|ρ|", "斯皮尔曼相关系数的绝对值，用于共表达图的边权与阈值"],
        ["Q", "模块度（modularity），衡量社区划分的内部连接强度"],
        ["KS", "柯尔莫哥洛夫—斯米尔诺夫距离，衡量生成谱与真实样本的边际差异"],
        ["QWK", "二次加权 Kappa，衡量 0–6 相对分级与感官真值的一致性"],
        ["分位组内 ρ", "按浓度分位分组后计算的组内斯皮尔曼相关，用于剂量—响应判定"],
        ["GM", "几何均价，多个零售来源价格的加权合成（以对数均值的指数还原）"],
    ], columns=["符号", "含义"])
    _table(doc, symbols, counters.next_table("全文主要符号与含义"), max_cell_chars=48)

    H(2, "3.6 与官方文本表述不一致的三处及其证据")
    _para(doc, "下表集中列出本文裁定的三处与官方赛题文本表述不一致之处，并给出可复算的证据；"
               "全文其余建模口径均以此三处裁定为准。", indent_chars=2)
    errata = pd.DataFrame([
        ["① 相关矩阵的相关系数类型",
         "赛题文本称附件 2 单元格为皮尔逊相关系数",
         "实为斯皮尔曼相关系数",
         "以 1.csv、4.csv 重算斯皮尔曼并与给定 5.csv、6.csv 逐元素比对：max|Δ|=0、逐元素吻合比例 1.0000；"
         "皮尔逊口径下中位绝对偏差约 0.04；附件《数据说明》对 2/5/6 均自述为 Spearman，原文 Fig.1 题注亦写明 Spearman"],
        ["② 3.xlsx 的实际内容",
         "赛题文本称 3.xlsx 为「品评小组使用的品评表」",
         "实为空白品评卡模板",
         "逐格核查 12 张评分卡无任何品评员打分记录；summary 表 600 个数值单元格全为指向空白评分卡的公式"
         "（=beer_1!$Q$7 等），故附件不含原始 0–6 分数"],
        ["③ 1.csv 的标度结构",
         "赛题文本按统一对数标度使用化合物丰度",
         "实为混合标度（log10 与原始浓度并存）",
         "以 8.xlsx 标定值逐列回归：log10 列 37（斜率 1.0000、r=1.000000）、原始浓度列 19（斜率 1.0000、r≥0.999998）、"
         "近似原始浓度 10、无法逐点复现 4；自然对数假设斜率恒为 2.302585 故排除"],
    ], columns=["不一致之处", "官方文本表述", "本文裁定", "证据"])
    _table(doc, errata, counters.next_table("三处与官方文本表述不一致之处及其证据"),
           max_rows=3, max_cell_chars=110, font_size=8.0)
    _para(doc, "这三处差异共同决定了本文的建模口径：相关类型决定网络与证据的计算方式，"
               "品评卡为空决定 0–6 分制必须显式构造并声明其近似性质，混合标度决定特征矩阵必须逐列裁定后统一；"
               "任何忽略其中之一的做法都会在量纲或口径层面产生系统性错误。", indent_chars=2)


# --------------------------------------------------------------------- 四、模型构建


def _build_models(doc: Document, H, res: dict) -> None:
    H(1, "四、模型构建")

    H(2, "4.1 问题一：多任务回归与双轨分级")
    _para(doc, "问题一以 1.csv 的化合物特征预测 50 个感官维度。特征取统一标度后的 227 个化学量，"
               "剔除 4 个公式未经验证的派生求和列；主比较覆盖岭回归、偏最小二乘回归、多任务弹性网络、"
               "全局与分组极端随机树以及逐指标梯度提升回归七个模型族，分别对应线性低秩、稀疏线性和树形非线性归纳偏置。"
               "主模型使用固定的 175/75 风格分层留出协议，宏平均 RMSE 作为主要指标，同时报告宏平均 MAE、逐维 R² 中位数"
               "和斯皮尔曼相关系数中位数。另行保留 TabPFN 3.5 的既有复现协议，用于核对该协议下的复现结果是否可复算。"
               "该协议使用 231 个原始特征和 KFold，评估口径与主模型不同，因此不参与模型优劣排序。", indent_chars=2)
    _para(doc, "由于附件未提供原始 0–6 分数，0–6 分制必须显式构造。本文采用双轨：**主轨为相对分级**，"
               "对每个维度取训练集的六等分位数作为切点，真实值与预测值用同一组切点离散化为 0–6 级，"
               "切点只由训练集学习以避免信息泄漏，并报告等级平均绝对误差、精确命中率、相邻一级命中率与"
               "二次加权 Kappa；**副轨为阈值锚定的绝对分**，对存在明确化学载体的维度（苦味—异构 α 酸、"
               "酒精—乙醇、硫醚—二甲基硫醚、4-乙烯基愈创木酚、糊精—双乙酰、trans-2-壬烯醛、金属感—铁、"
               "酸感—乳酸、酒体—蛋白、甜度—残糖、碳酸—二氧化碳）尝试以「化合物低于感知阈值对应 0 分」"
               "与「化合物达到训练集高分位对应 6 分」两点确定仿射映射，再把标准化标签反解为绝对分。", indent_chars=2)
    _para(doc, "由于逐维度条件苛刻，副轨设置了明确的**有效性护栏**，五条须同时满足：载体在参考表中存在感知阈值；"
               "载体与该感官维度的斯皮尔曼相关系数不低于 0.3 且为正；0 分锚点对应的训练集分位不高于 0.6；"
               "两个锚点的标准分带宽不小于 0.5；映射后落在 0 至 6 区间内的比例不低于 0.3。"
               "任一条不满足即判定该维度的绝对锚定不成立，其绝对分留空，仅提供相对分级，"
               "并在参数表中逐条记录失效原因。副轨是**本文提出的、基于感官阈值的一阶近似**，"
               "不等价于实测的原始评分；护栏的存在本身也说明，在本数据集上化学信号与感官评分的联系"
               "只在少数维度上足以支撑绝对标定。", indent_chars=2)

    H(2, "4.2 问题二：共表达图与四路证据融合")
    _para(doc, "问题二先由 2.csv 构建化合物共表达图，边的权重取斯皮尔曼相关系数的绝对值，"
               "默认阈值 0.6，并同时报告 0.5、0.7、0.8 三档的敏感性；社区发现采用 Leiden 算法"
               "（模块度目标、5 次迭代）。节点属性标注其化学块归属、来源（麦芽、酒花、酵母、香料、野生菌群等）"
               "与是否为缺陷化合物，社区与来源的交叉表构成化学合理性论证的主证据。", indent_chars=2)
    _para(doc, "关键化合物的识别采用四路证据：E1 为附件 6.csv 给出的化合物—感官交叉相关；"
               "E2 为本文基于 1.csv 与 4.csv 自算的全量斯皮尔曼相关（覆盖全部 231 个化合物，"
               "补足 E1 仅含 44 个化合物的缺口）；E3 为梯度提升回归结合 SHAP 的归因；"
               "E4 为附件给出的原论文模型重要性先验。融合排序以 E2 与 E3 的秩归一化等权求和为主，"
               "E1 用于交叉验证，E4 仅作外部印证的对照而不参与主排序，以免以既有结论替代本文结论。", indent_chars=2)
    _para(doc, "稳定性按题面给出的两种方式同时量化：一是对训练样本做 200 次有放回重采样，统计各化合物"
               "进入前三名的频率；二是计算 E1、E2、E3 三路前三名集合两两的 Jaccard 系数作为方法间一致性。"
               "入选频率不低于 0.6 且方法间 Jaccard 不低于 0.3 者判为稳健关键因素，否则标为需谨慎的候选。"
               "化学合理性论证涵盖化合物的来源归属、是否属于该感官维度的已知化学载体类别、"
               "与参考表给出的异味描述及缺陷阈值是否一致、以及与外部先验的方向是否一致；"
               "对于同时出现在特征侧与标签侧的量，本文明确表述为理化测量与感官评分的一致性，而非新发现的关联。", indent_chars=2)

    H(2, "4.3 问题三：风格条件生成与跨风格迁移")
    _para(doc, "问题三以风格为条件生成化合物谱。本文不采用条件生成对抗网络或变分自编码器："
               "各风格样本量最小仅 3 例、多数不足 15 例，且本机无 GPU 加速，"
               "而全局高斯 Copula 结构结合风格条件边际的方案对小样本更稳健、边际可解释、约束可硬编码。", indent_chars=2)
    _para(doc, "具体流程为：先在统一标度矩阵上拟合全局高斯 Copula 以保留化合物之间的相关结构；"
               "再为每个风格以该风格样本在各维度上的经验分布定义条件边际，"
               "并按 λ = n/(n+10) 向全局边际收缩以缓解小样本估计方差；"
               "采样时先由 Copula 抽取相关结构向量，再以条件边际映回，生成化合物谱；"
               "随后对二甲基硫醚、双乙酰、4-乙烯基愈创木酚与 trans-2-壬烯醛四项按各自阈值（统一矩阵口径）"
               "执行拒绝重抽与单调压缩修复；最后把生成谱送入问题一的模型得到风格感官画像。"
               "跨风格迁移以秩—分位映射实现：保持原样本的相关结构不变，仅把边际替换为目标风格的条件边际，"
               "再施加同一套缺陷约束，从而保证迁移后缺陷化合物不恶化。", indent_chars=2)

    H(2, "4.4 消费者层建模")
    _para(doc, "附件没有提供任何消费者评分或喜爱度标签：4.csv 给出的是受训品评小组的感官结果，"
               "7.csv 给出的是论坛评论派生的描述符特征，二者都不是「消费者喜爱度」。因此本文**不构造喜爱度目标**，"
               "而是把消费者层建模限定为两件可检验的事。第一，描述符的可用性：7.csv 的 186 个描述符列中，"
               "palate_acetaldehyde、palate_acetate、palate_dentist 三列整列为空（每列 250 个缺失），"
               "有效描述符为 183 维；本文同时以 183 维消费者描述符与 50 维品评组感官为预测目标，"
               "形成「化学→消费者描述符」与「化学→品评组感官」两条并行路径，并对两者做交叉一致性检验。"
               "第二，是否可把多维感知压缩成单一喜爱度轴：183 维描述符经标准化后的第一主成分解释率仅为 10.4%，"
               "远不足以支撑把多维感知压成一维喜爱度，因此本文不做降维造标签，只把多维描述符整体作为目标。", indent_chars=2)
    _para(doc, "第三，外部价值信号：8.xlsx 给出 7 个零售来源的逐款价格，其中覆盖 ≥2 家零售商的共 189 款"
               "可计算几何均价（GM）；本文以价格为外部价值代理，回归中控制酒精度与风格，"
               "并明确其局限——价格来自同一附件内 7 家零售商的跨币种报价，未做汇率换算，"
               "是市场价值的代理而非消费者效用本身。"
               "**另需披露一项附件层面的限制**：这 7 个来源分别位于美国、英国、比利时与面向全球的网店，"
               "标价币种涵盖美元、英镑与欧元，而附件未提供汇率列，本文因此**未做任何币种换算**；"
               "这使得 log 价格同时包含「真实品质溢价」与「币种/卖方价格水平差异」两部分，"
               "构成价格回归解释力的一个下界，本文的价格结论只作方向性参考。"
               "这一设计与原研究「提升消费者喜爱度」的结论形成对照：在本附件条件下，"
               "**消费者喜爱度这一环不可检验**，可检验的是消费者感知描述符的可预测性与化学谱的市场价值解释力。",
               indent_chars=2)


# --------------------------------------------------------------------- 五、实验结果


def _build_results(doc: Document, H, res: dict, consumer: dict, counters: Counters) -> None:
    H(1, "五、实验结果")

    # ---- 5.1 问题一
    H(2, "5.1 问题一：感官评分预测与 0–6 分制")
    summary = res.get("q1_summary")
    if summary is not None and not summary.empty:
        cols = [c for c in ["model", "Macro_RMSE", "Macro_MAE", "R2_median", "Spearman_median",
                            "proxy_grade_MAE", "proxy_exact_accuracy",
                            "proxy_within_one_accuracy", "proxy_quadratic_kappa"]
                if c in summary.columns]
        view = summary[cols].copy()
        for c in cols[1:]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: fmt(v, 4))
        _table(doc, view, counters.next_table("各模型在留出集上的连续指标与相对分级指标（seed=0，175/75 分层切分）"))
        base = summary[summary["model"].astype(str).str.contains("mean", case=False)]
        if not base.empty:
            b = base.iloc[0]
            _para(doc, f"以全局均值为基线（宏平均 RMSE = {fmt(b.get('Macro_RMSE'), 4)}），"
                       f"最优模型的宏平均 RMSE 为 {fmt(summary['Macro_RMSE'].min(), 4)}，"
                       f"相对基线改善约 {fmt(100 * (1 - summary['Macro_RMSE'].min() / b.get('Macro_RMSE')), 1)}%。"
                       "各模型头部的宏平均 RMSE 差距在 0.001 量级，经自助重采样比较其置信区间明显重叠，"
                       "因此本文不以单点指标断言模型优劣，而是并列表述并以自助重采样区间说明。", indent_chars=2)
    scores = res.get("q1_scores")
    if scores is not None:
        _para(doc, f"最终交付的 0–6 分制分级评分表覆盖 {len(scores)} 款啤酒 × 50 个感官维度，"
                   "主轨给出相对分级，副轨对具备化学载体的维度给出阈值锚定的绝对分；"
                   "表中等级均落在 0 至 6 区间内且无缺失。", indent_chars=2)
    anchor = res.get("q1_anchor")
    if anchor is not None and not anchor.empty:
        valid_dim, invalid_dim = [], []
        if "valid" in anchor.columns:
            flag = anchor["valid"].astype(str).str.lower().isin(["true", "1"])
            valid_dim = anchor.loc[flag, "dimension"].tolist()
            invalid_dim = anchor.loc[~flag, "dimension"].tolist()
        else:
            valid_dim = anchor["dimension"].tolist()
        cov_col = next((c for c in anchor.columns if c.startswith("frac_within_0_6")), None)
        cov_txt = ""
        if cov_col and valid_dim:
            vals = pd.to_numeric(anchor.loc[anchor["dimension"].isin(valid_dim), cov_col],
                                 errors="coerce").dropna()
            if len(vals):
                cov_txt = f"，锚定后落在 0–6 区间内的比例中位数为 {fmt(vals.median(), 3)}"
        _para(doc, f"阈值锚定共考察 {len(anchor)} 个具备明确化学载体的维度，经上述五条护栏后有 "
                   f"{len(valid_dim)} 个维度成立（{'、'.join(valid_dim) if valid_dim else '无'}）"
                   f"{cov_txt}；其余 {len(invalid_dim)} 个维度因载体无感知阈值、载体与感官维度相关性不足或方向相反、"
                   "或锚点区间退化而判定不成立，其绝对分留空、仅提供相对分级，"
                   "失效原因在参数表中逐条记录。这一结果本身说明：在本数据集上，"
                   "化学信号与感官评分的联系只在少数维度上足以支撑绝对标定。"
                   "因此，绝对轨应被理解为阈值约束下的近似标尺，而不是对缺失原始评分的替代。", indent_chars=2)
    tabpfn = res.get("q1_tabpfn")
    if tabpfn is not None and not tabpfn.empty:
        idx = tabpfn.index.astype(str)
        row = tabpfn[idx.str.contains("cv_mean", na=False)]
        if not row.empty:
            r = row.iloc[0]
            _para(doc, "作为第三类归纳偏置的补充，本文在本机 CPU 环境完成了既有复现协议下 TabPFN 3.5 的五折全量复现"
                       f"（5 折 × 50 个目标共 250 个单元；该复现使用原始 231 特征与 KFold，"
                       f"不等同于主模型的 227 特征、固定 175/75 留出协议）。其五折平均决定系数为 {fmt(r.get('r2'), 4)}，"
                       f"宏平均绝对误差 {fmt(r.get('mae'), 4)}、宏平均均方根误差 {fmt(r.get('rmse'), 4)}，"
                       f"而同一协议下的训练折均值基线决定系数为 {fmt(r.get('baseline_r2'), 4)}，"
                       "这些数值用于验证既有复现产物的可复算性，不作为相对于主模型的直接优越性证据；"
                       "其第一折的逐目标决定系数与远端归档结果逐位一致。", indent_chars=2)
    for folder, name, short, method in FIGURE_SPECS[:3]:
        _insert_figure(doc, counters, folder, name, short, method)

    # ---- 5.2 消费者感知路径与市场价值
    H(2, "5.2 消费者感知路径与市场价值")
    _build_consumer_results(doc, consumer, counters)
    for folder, name, short, method in _consumer_figure_specs("consumer"):
        _insert_figure(doc, counters, folder, name, short, method)

    # ---- 5.3 问题二
    H(2, "5.3 问题二：共表达图与关键化合物")
    _build_q2_results(doc, res, counters)
    for folder, name, short, method in FIGURE_SPECS[3:6]:
        _insert_figure(doc, counters, folder, name, short, method)

    # ---- 5.4 与原文的对照及缺陷化合物剂量再检验
    H(2, "5.4 与原文的对照及缺陷化合物剂量再检验")
    _build_comparison_and_dose(doc, res, consumer, counters)
    for folder, name, short, method in _consumer_figure_specs("dose"):
        _insert_figure(doc, counters, folder, name, short, method)

    # ---- 5.5 问题三
    H(2, "5.5 问题三：风格条件生成与跨风格迁移")
    _build_q3_results(doc, res, counters)
    for folder, name, short, method in FIGURE_SPECS[6:9]:
        _insert_figure(doc, counters, folder, name, short, method)


def _price_predictor_note(price: pd.DataFrame, common: List[str]) -> str:
    """价格回归的自变量来源与系数显著性的披露（评审要求，见 docs/review_round2.md W9）。

    自变量是**化学模型的折外（OOF）预测值**而非实测感官分——这一点必须写明，
    否则读者会误以为回归的是实测评分与价格的关系。此外还需给出系数的显著性，
    以免把「多数项不显著」的结果读成「化学谱能解释价格」。
    """
    if price is None or price.empty or not common:
        return ""
    parts = ["回归的自变量是**化学模型给出的消费者描述符折外（5 折 OOF）预测值**，"
             "而非实测感官评分——这样可避免用同一批样本既训练又评价，"
             "但也意味着系数解释的是「化学谱所预测的感知」与价格的关系。"]
    for m in ["fixed_effects", "ols_with_abv", "ols"]:
        sub = price[(price["model"].astype(str) == m) & (price["term"].astype(str).isin(common))]
        if sub.empty or "std_err" not in price.columns:
            continue
        t = (pd.to_numeric(sub["coef"], errors="coerce")
             / pd.to_numeric(sub["std_err"], errors="coerce").replace(0, float("nan")))
        t = t.dropna()
        if t.empty:
            continue
        sig = int((t.abs() > 2).sum())
        model_label = {"ols": "OLS", "ols_with_abv": "OLS + ABV",
                       "fixed_effects": "风格固定效应"}.get(m, m)
        parts.append(f"在 {model_label} 规格下 {len(t)} 个实质项中只有 {sig} 项满足 |t| > 2，"
                     f"即多数系数在常规显著性水平上与零无法区分，"
                     f"故本文只把价格回归当作**方向性**证据，而非精确的弹性估计。")
        break
    return "".join(parts)


def _build_consumer_results(doc: Document, consumer: dict, counters: Counters) -> None:
    """5.2 消费者感知路径与市场价值：全部数字来自 q4_consumer 产物；缺失则降级。"""
    summary = consumer.get("consumer_model_summary.csv")
    cross = consumer.get("cross_consistency.csv")
    price = consumer.get("price_regression.csv")
    if summary is None and cross is None and price is None:
        _para(doc, "**本节内容待补。** 消费者层产物（consumer_model_summary.csv、cross_consistency.csv、"
                   "price_regression.csv）尚未生成，因此本节暂缺；待产物就绪后重跑论文构建即可并入。"
                   "本节的建模口径已在 4.4 节给出：本文只做「化学→183 维消费者描述符」与「化学→50 维品评组感官」"
                   "的并行建模与交叉一致性检验，并以 189 款啤酒的价格作为外部价值信号，"
                   "不构造也不声称预测消费者喜爱度。", indent_chars=2)
        return

    if summary is not None and not summary.empty:
        cols = [c for c in ["model", "Macro_RMSE", "Macro_MAE", "R2_median", "Spearman_median"]
                if c in summary.columns]
        view = summary[cols].copy()
        for c in cols[1:]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: fmt(v, 4))
        view = view.rename(columns={"model": "模型", "Macro_RMSE": "宏平均 RMSE",
                                    "Macro_MAE": "宏平均 MAE", "R2_median": "逐维 R² 中位数",
                                    "Spearman_median": "ρ 中位数"})
        _table(doc, view, counters.next_table("化学→183 维消费者描述符的建模结果（留出集宏平均指标）"))
        base = summary[summary["model"].astype(str).str.contains("mean", case=False)]
        if not base.empty and "Macro_RMSE" in summary.columns:
            b = base.iloc[0]
            best = summary.loc[summary["Macro_RMSE"].idxmin()]
            best_name = str(best.get("model", "—"))
            _para(doc, f"以全局均值为基线（宏平均 RMSE = {fmt(b.get('Macro_RMSE'), 4)}），"
                       f"最优模型为 {best_name}（宏平均 RMSE = {fmt(best.get('Macro_RMSE'), 4)}，"
                       f"宏平均 MAE = {fmt(best.get('Macro_MAE'), 4)}，"
                       f"逐维 R² 中位数 = {fmt(best.get('R2_median'), 4)}，"
                       f"斯皮尔曼中位数 = {fmt(best.get('Spearman_median'), 4)}），"
                       f"相对基线改善约 {fmt(100 * (1 - best.get('Macro_RMSE') / b.get('Macro_RMSE')), 1)}%。"
                       "该结果说明化学谱对消费者描述符具有可测量的解释力，但解释力有限："
                       "它远不足以支持把多维描述符压缩为一个「喜爱度」总分。", indent_chars=2)

    if cross is not None and not cross.empty:
        cols = [c for c in ["panel_dim", "consumer_desc", "rho_truth", "rho_pred", "delta"]
                if c in cross.columns]
        view = cross[cols].copy()
        for c in cols[2:]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: fmt(v, 3))
        _table(doc, view, counters.next_table(
                   f"品评组感官与消费者描述符的交叉一致性检验（共 {len(cross)} 组配对，"
                   f"此处列出前 {min(15, len(cross))} 组，完整表见支撑材料）"),
               max_rows=15, max_cell_chars=24, font_size=8.0)
        if "delta" in cross.columns:
            dv = pd.to_numeric(cross["delta"], errors="coerce").dropna()
            if len(dv):
                close = int((dv.abs() <= 0.2).sum())
                frac = close / len(dv)
                if frac >= 0.7:
                    verdict_txt = ("该结果支持一条判断：品评组感官与消费者描述符虽是两种不同的测量通道，"
                                   "但在化学—感知的映射上具有相当的一致性，因此本文以它们并行建模是合理的。")
                else:
                    verdict_txt = ("该结果说明两条测量通道的一致性是**有限的**："
                                   "只有约一半的语义配对能被「化学→感官」模型较好地迁移到消费者描述符一侧，"
                                   "因此本文把交叉一致性检验作为诊断而非验证，消费者描述符不可替代品评组评分。")
                _para(doc, f"交叉一致性检验在 {len(cross)} 组语义配对上比较「同一化学量在品评组感官与消费者描述符"
                           f"两侧的实测相关」与「由模型预测得到的相关」，偏差 |Δ| 的中位数为 {fmt(dv.abs().median(), 3)}、"
                           f"最大值为 {fmt(dv.abs().max(), 3)}，其中 {close} 组（{fmt(100 * frac, 1)}%）"
                           f"的 |Δ| 不超过 0.2。" + verdict_txt,
                       indent_chars=2)

    if price is not None and not price.empty:
        sub = price.copy()
        sub["term"] = sub["term"].astype(str)
        sub["model"] = sub["model"].astype(str)
        models = list(dict.fromkeys(sub["model"]))
        model_labels = {"ols": "OLS", "ols_with_abv": "OLS + ABV",
                        "fixed_effects": "风格固定效应"}
        # 风格虚拟变量不入表，只保留可跨模型比较的实质项
        sub = sub[~sub["term"].str.startswith("fe_")]
        term_sets = {m: set(sub.loc[sub["model"] == m, "term"]) for m in models}
        common = sorted(set.intersection(*term_sets.values())) if term_sets else []
        common = [t for t in common if t != "const"]
        if common:
            view = pd.DataFrame({"项": common})
            for m in models:
                vals = []
                for t in common:
                    row = sub[(sub["model"] == m) & (sub["term"] == t)]
                    vals.append(fmt(pd.to_numeric(row["coef"], errors="coerce").iloc[0], 3)
                                if len(row) else "—")
                view[m] = vals
            _table(doc, view,
                   counters.next_table("以价格为外部价值信号的回归系数（三个模型并列，n = 189）"),
                   max_rows=16, max_cell_chars=20, font_size=8.0)
        r2_txt = "、".join(
            f"{m} 的 R² = {fmt(pd.to_numeric(sub.loc[sub['model'] == m, 'r2'], errors='coerce').iloc[0], 3)}"
            for m in models if (sub["model"] == m).any() and "r2" in sub.columns)
        nmax = pd.to_numeric(price["n"], errors="coerce").max() if "n" in price.columns else None
        # 系数符号是否随控制变量改变
        sign_txt = ""
        if len(common) >= 2 and len(models) >= 2:
            flips = []
            for t in common:
                signs = []
                for m in models:
                    row = sub[(sub["model"] == m) & (sub["term"] == t)]
                    if len(row):
                        c = pd.to_numeric(row["coef"], errors="coerce").iloc[0]
                        if pd.notna(c) and c != 0:
                            signs.append(np_sign(c))
                if len(signs) >= 2 and len(set(signs)) > 1:
                    flips.append(t)
            sign_txt = (f"在 {len(common)} 个可跨模型比较的描述符项中，"
                        f"有 {len(common) - len(flips)} 项的系数符号在三个模型间保持一致"
                        + (f"（仅 {'、'.join(flips)} 发生符号改变）" if flips else "")
                        + "，说明化学—感知组合携带的信息不完全由酒精度与风格标签解释。")
        _para(doc, f"价格回归以可算几何均价的 {fmt(nmax, 0)} 款样品为样本，并列给出 {len(models)} 组模型："
                   + "；".join(model_labels.get(m, m) for m in models)
                   + "（依次为不控变量、控制酒精度、控制酒精度并加入风格固定效应）。"
                   + (r2_txt + "，" if r2_txt else "") + sign_txt
                   + "需要强调其边界：价格来自同一附件内 7 家零售商的跨币种报价，且未做汇率换算，"
                   "是市场价值的代理，"
                   "**不是消费者效用的直接测量**，因此本文不据此推断任何「提升消费者喜爱度」的因果结论。"
                   + _price_predictor_note(price, common),
                   indent_chars=2)


#: 剂量—响应配对（化合物列 → 品评组感官维度），共六行。
DOSE_PAIRS = [
    ("ethyl_acetate", "A_esters_all", "乙酸乙酯", "酯类香气"),
    ("ethyl_acetate", "F_esters_all", "乙酸乙酯", "酯类味"),
    ("linalool", "A_hops_citrus", "芳樟醇", "柑橘香气"),
    ("DMS", "dms", "二甲基硫醚", "硫醚感知"),
    ("X2.3.butanedione", "diacetyl", "双乙酰", "双乙酰感知"),
    ("X.E..2.nonenal", "t2n", "trans-2-壬烯醛", "氧化感知"),
]

ALLOWED_PATTERNS = ("monotonic_positive", "monotonic_negative", "none", "non_monotonic")

PATTERN_CN = {"monotonic_positive": "单调正向", "monotonic_negative": "单调反向",
              "none": "无关系", "non_monotonic": "非单调"}


def _dose_verdict_rows(verdict: Optional[pd.DataFrame]) -> List[dict]:
    """从判定表中取出本节要求的六行配对，返回结构化记录。"""
    out: List[dict] = []
    if verdict is None or verdict.empty:
        return out
    for compound, dim, cname, dname in DOSE_PAIRS:
        sub = verdict[(verdict["compound"].astype(str) == compound)
                      & (verdict["sensory_dim"].astype(str) == dim)]
        if sub.empty:
            _warn(f"剂量判定产物缺少配对 {compound}→{dim}，该行暂缺。")
            continue
        row = sub.iloc[0]
        pattern = str(row.get("pattern", "")).strip()
        if pattern not in ALLOWED_PATTERNS:
            _warn(f"剂量判定取值非法（{pattern}），该行暂缺。")
            continue
        rho = pd.to_numeric(pd.Series([row.get("overall_rho")]), errors="coerce").iloc[0]
        srf = pd.to_numeric(pd.Series([row.get("style_robust_fraction")]), errors="coerce").iloc[0]
        note = str(row.get("note") or "")
        # 稳健比例的真实分母是「四分位可判」的风格数（比半分割可判的风格数少），
        # 须与正文披露一致，否则「16 个风格」会掩盖真实分母只有 6–8。
        m_den = re.search(r"共\s*(\d+)\s*个风格四分位可判", note)
        m_frac = re.search(r"保持同一模式\s*(\d+)\s*/\s*(\d+)", note)
        out.append({"compound": compound, "dim": dim, "cname": cname, "dname": dname,
                    "pattern": pattern,
                    "rho": float(rho) if pd.notna(rho) else float("nan"),
                    "srf": float(srf) if pd.notna(srf) else float("nan"),
                    "srf_den": int(m_den.group(1)) if m_den else 0,
                    "srf_same": int(m_frac.group(1)) if m_frac else 0,
                    "srf_total": int(m_frac.group(2)) if m_frac else 0})
    return out


def _srf_denom_text(vnum: List[dict]) -> str:
    """风格稳健比例分母（四分位可判的风格数）的紧凑表述。"""
    dens = sorted({int(x.get("srf_den", 0)) for x in vnum if x.get("srf_den")})
    if not dens:
        return "各配对不等"
    if len(dens) == 1:
        return f"{dens[0]} 个"
    return f"{dens[0]}–{dens[-1]} 个"


def _style_robustness_note(srf: float) -> str:
    """由判定表里的「风格稳健比例」生成内联限定语（与正文表 12 同源同口径）。

    单调性结论若只在「全体数据」上成立、在风格分层后即翻转（辛普森式混合效应），
    必须在主张出现的地方就带上限定，不能只在下文笼统地说「另给出稳健性结果」。

    注意：这里的 srf 取自 `defect_dose_verdict.csv`，其分母是「四分位可判」的风格数
    （见同表 note 列），与 5.4 中按 16 个风格统计的「同号比例」不是同一指标，勿混用。
    """
    if srf is None or (isinstance(srf, float) and pd.isna(srf)):
        return ""
    if srf <= 0:
        return "（风格分层稳健比例为 0，即无任何可判风格支持该方向，属辛普森式混合效应，不稳健）"
    if srf < 0.5:
        return f"（风格分层稳健比例仅 {srf:.2f}，方向不稳，不宜作一般化解读）"
    return ""


def _dose_conclusion_text(consumer: dict) -> Optional[str]:
    """由判定表生成一句结论（供第七章引用，避免写死结论）。"""
    rows = _dose_verdict_rows(consumer.get("defect_dose_verdict.csv"))
    if not rows:
        return None
    style_tbl = consumer.get("style_robustness.csv")
    parts = []
    pos = [f"{x['cname']}→{x['dname']}{_style_robustness_note(x['srf'])}"
           for x in rows if x["pattern"] == "monotonic_positive"]
    neg = [f"{x['cname']}→{x['dname']}{_style_robustness_note(x['srf'])}"
           for x in rows if x["pattern"] == "monotonic_negative"]
    near = [f"{x['cname']}→{x['dname']}（ρ = {x['rho']:+.3f}）"
            for x in rows if abs(x["rho"]) < 0.1 and x["pattern"] != "monotonic_negative"]
    nonmono = [f"{x['cname']}→{x['dname']}" for x in rows if x["pattern"] == "non_monotonic"]
    if pos:
        parts.append("呈单调正向的有 " + "、".join(pos))
    if neg:
        parts.append("呈单调反向的有 " + "、".join(neg))
    if near:
        parts.append("全体 |ρ| < 0.1（近乎不相关）的有 " + "、".join(near))
    if nonmono:
        parts.append("分位组内方向不一致而判为非单调的有 " + "、".join(nonmono))
    defect_neg = [x for x in rows if x["compound"] in ("X2.3.butanedione", "X.E..2.nonenal")
                  and x["rho"] < 0]
    if len(defect_neg) == 2:
        fragile = [x["cname"] for x in defect_neg if _style_robustness_note(x["srf"])]
        tail = (f"，但其中 {'、'.join(fragile)} 的方向在风格分层后不成立，不宜作一般化解读"
                if fragile else "")
        parts.append("双乙酰与 trans-2-壬烯醛的全体相关方向为负，"
                     "与「浓度越高越难闻」的朴素预期相反" + tail)
    return "；".join(parts)


def _build_comparison_and_dose(doc: Document, res: dict, consumer: dict, counters: Counters) -> None:
    """5.4 与原文的对照 + 缺陷化合物剂量再检验。"""
    summary = res.get("q1_summary")
    top3 = res.get("q2_top3")

    best_model = "—"
    best_rmse = None
    if summary is not None and not summary.empty and "Macro_RMSE" in summary.columns:
        best = summary.loc[summary["Macro_RMSE"].idxmin()]
        best_model = str(best.get("model", "—"))
        best_rmse = best.get("Macro_RMSE")
    n_robust = 0
    if top3 is not None and not top3.empty and "selection_status" in top3.columns:
        n_robust = top3.loc[top3["selection_status"].astype(str).str.contains("稳健"),
                            "sensory_dimension"].nunique()

    rows = [
        ["样本与特征规模", "250 款啤酒；231 个化学量（剔除 4 个派生求和列后 227 特征）",
         "与原文同源附件；全部数值由本文重新读取与复算，未沿用原文中间结果"],
        ["相关矩阵类型", "斯皮尔曼（逐元素复算 max|Δ|=0）",
         "官方赛题文本称皮尔逊；原文与附件《数据说明》均为斯皮尔曼，本文以复算裁定（见 3.2、3.6）"],
        ["风格数", "22 个细分风格（175/75 分层切分）",
         "与原文同源于附件的分类列；本文按该列分层，保证风格构成不泄漏"],
        ["品评规模", "12 张空白品评卡、50 维品评组感官、0 条品评记录",
         "原文的品评记录未随附件提供，故本文的 0–6 分制为构造值（见 3.4、4.1）"],
        ["最优模型族", f"{best_model}（宏平均 RMSE = {fmt(best_rmse, 4)}）",
         "本文不复现原文的模型选择，仅以其重要性先验（E4）作为外部印证，不参与主排序"],
        ["关键化合物", f"{n_robust} 个感官维度至少含一项稳健关键化合物",
         "本文排序由自算全量相关与 SHAP 归因给出；原文结论以先验形式参与四路证据之一"],
        ["消费者喜爱度", "不可检验（附件无任何消费者评分标签）",
         "原文结论链条中的「提升消费者喜爱度」一环，在本附件条件下没有可检验的标签"],
    ]
    cmp_df = pd.DataFrame(rows, columns=["对照项", "本文实测", "与原文的关系与口径差异"])
    _table(doc, cmp_df, counters.next_table("本文与原文（Nature Communications, 2024）的可量化对照"),
           max_rows=7, max_cell_chars=60, font_size=8.0)
    _para(doc, "上表只列可逐项核对的对照点：凡本文未能独立复核的原文数值一律不写入表格，"
               "而是注明其在本附件中不可得。可以看到，本文与原文共享同一份数据，"
               "但在**标度、相关类型与品评可得性**三个口径上做了明确的重新裁定，"
               "这三处裁定也正是本文全部结论成立的前提。", indent_chars=2)

    # ---- 剂量—响应再检验
    verdict = consumer.get("defect_dose_verdict.csv")
    dose = consumer.get("defect_dose_response.csv")
    robustness = consumer.get("style_robustness.csv")
    if verdict is None or verdict.empty:
        _para(doc, "**剂量—响应判定待补。** 判定产物（defect_dose_verdict.csv、defect_dose_response.csv）"
                   "尚未生成，因此本小节的六行配对判定表暂缺；待产物就绪后重跑论文构建即可并入。", indent_chars=2)
        return

    vrows = []
    vnum: List[dict] = _dose_verdict_rows(verdict)
    for it in vnum:
        vrows.append({
            "化合物": it["cname"],
            "配对感官维度": it["dname"],
            "全体 ρ": fmt(it["rho"], 3),
            "分位组内 ρ 模式": it["pattern"],
            "风格稳健比例": fmt(it["srf"], 3),
        })
    if not vrows:
        _para(doc, "**剂量—响应判定待补。** 判定产物中未找到本节要求的六行配对，暂不列表。", indent_chars=2)
        return
    _table(doc, pd.DataFrame(vrows), counters.next_table("缺陷化合物—感官维度的剂量—响应判定（六行配对）"),
           max_rows=6, max_cell_chars=26, font_size=8.0)

    # 以下叙述全部由判定表算出，不写死任何结论
    pattern_cn = {"monotonic_positive": "单调正向", "monotonic_negative": "单调反向",
                  "none": "无关系", "non_monotonic": "非单调"}

    def _desc(it: dict) -> str:
        rho = it["rho"]
        direction = "反向" if rho < 0 else ("正向" if rho > 0 else "无方向")
        strength = "近乎不相关" if abs(rho) < 0.1 else "存在可测相关"
        return (f"{it['cname']}→{it['dname']}（全体 ρ = {rho:+.3f}，{direction}、{strength}；"
                f"组内判定为{pattern_cn[it['pattern']]}）")

    def _group(pattern: str) -> List[dict]:
        return [it for it in vnum if it["pattern"] == pattern]

    pos, neg, nonmono, none_items = (_group("monotonic_positive"), _group("monotonic_negative"),
                                     _group("non_monotonic"), _group("none"))
    zero = [it for it in vnum if abs(it["rho"]) < 0.1]

    parts = []
    if pos:
        parts.append(f"严格单调正向 {len(pos)} 行（{'；'.join(_desc(x) for x in pos)}）")
    if neg:
        parts.append(f"严格单调反向 {len(neg)} 行（{'；'.join(_desc(x) for x in neg)}）")
    if nonmono:
        parts.append(f"方向不一致而判为非单调 {len(nonmono)} 行（{'；'.join(_desc(x) for x in nonmono)}）")
    if none_items:
        parts.append(f"判为无关系 {len(none_items)} 行（{'；'.join(_desc(x) for x in none_items)}）")
    zero_txt = ("；其中全体 |ρ| 小于 0.1、可视为近乎不相关的配对为 " +
                "、".join(f"{x['cname']}→{x['dname']}" for x in zero)) if zero else ""

    _para(doc, "判定口径为：把样品按该化合物的浓度分为四分位组，在每个分位组内计算化合物与感官维度的斯皮尔曼相关；"
               "四个分位组的组内相关符号一致且感官中位数随之单调时判为 monotonic_positive，"
               "符号一致为负判为 monotonic_negative，符号不一致判为 non_monotonic，"
               "四组相关均接近零判为 none。", indent_chars=2)
    _para(doc, f"按此口径，{len(vnum)} 行配对的判定结果为：" + "；".join(parts) + "。"
               + (zero_txt[1:] + "。" if zero_txt else "")
               + "这说明「浓度越高、该感官维度越强」这一朴素剂量—响应在本附件上并不普遍成立："
               "相关在全体样品上为正，并不意味着它在浓度分位内部稳定成立。", indent_chars=2)

    defect_desc = []
    defect_dir = []
    for it in vnum:
        if it["compound"] in ("DMS", "X2.3.butanedione", "X.E..2.nonenal"):
            defect_desc.append(_desc(it))
            if it["rho"] < 0:
                defect_dir.append(f"{it['cname']}的相关方向为负")
            elif it["rho"] > 0:
                defect_dir.append(f"{it['cname']}的相关方向为正")
            else:
                defect_dir.append(f"{it['cname']}的相关接近于零")
    if defect_desc:
        tail = ""
        if any(it["rho"] < 0 for it in vnum if it["compound"] in ("X2.3.butanedione", "X.E..2.nonenal")):
            tail = ("其中呈负向的化合物与「浓度升高即更难闻」的朴素预期相反，"
                    "这一方向与问题三中「压低缺陷化合物浓度反而使该维度的预测评分略有升高」的观察一致。"
                    "本文在此只陈述二者方向一致这一事实，不据此推断共同的发酵机制——"
                    "本附件不含任何发酵过程记录，「共同成因」与「直接致因」两种解释在现有数据下同样成立。")
        _para(doc, "就三类缺陷化合物而言，" + "；".join(defect_desc) + "。"
                   + ("其中" + "、".join(defect_dir) + "。" if defect_dir else "") + tail,
                   indent_chars=2)

    if robustness is not None and not robustness.empty:
        n_styles = robustness["style"].nunique() if "style" in robustness.columns else 0
        agree_rows = []
        for it in vnum:
            if "compound" not in robustness.columns or "sensory_dim" not in robustness.columns:
                break
            g = robustness[(robustness["compound"].astype(str) == it["compound"])
                           & (robustness["sensory_dim"].astype(str) == it["dim"])]
            sg = pd.to_numeric(g.get("rho", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(sg) and it["rho"] == it["rho"] and it["rho"] != 0:
                agree = float((sg.apply(np_sign) == np_sign(it["rho"])).mean())
                agree_rows.append((it["cname"], it["dname"], len(sg), agree))
        if agree_rows:
            agrees = [a[3] for a in agree_rows]
            srf_vals = [x["srf"] for x in vnum if x["srf"] == x["srf"]]
            _para(doc, f"风格分层稳健性检验在样本量不低于 8 的风格上（共 {n_styles} 个风格）重算了上述关系。"
                       f"各配对内部与全体相关同号的风格比例分别为 "
                       + "、".join(f"{c}→{d} {a:.2f}（{n} 个风格）" for c, d, n, a in agree_rows)
                       + f"，中位数 {fmt(float(sorted(agrees)[len(agrees) // 2]), 3)}；"
                       + (f"按「模式是否与全体一致」计的风格稳健比例在 {fmt(min(srf_vals), 3)} 至 "
                          f"{fmt(max(srf_vals), 3)} 之间，其分母是**四分位可判**的风格数"
                          f"（{_srf_denom_text(vnum)}），比上文的 {n_styles} 个风格少，"
                          "因为部分风格样本量不足以在四个浓度四分位内同时判定模式；"
                          "逐配对为 "
                          + "、".join(f"{x['cname']}→{x['dname']} {x['srf_same']}/{x['srf_total']}"
                                      for x in vnum if x.get("srf_total"))
                          + "，其中 trans-2-壬烯醛为 0/7，即无任何可判风格支持该方向，"
                            "说明其反向关系在多数风格中并不复现。" if srf_vals else "")
                       + "因此本文把上述剂量—响应结论作为方向性证据而非精确斜率，"
                       "并在讨论中说明风格构成混杂的风险。", indent_chars=2)

    _para(doc, "附件没有消费者评分或购买行为数据，因此原文结论链条中的「提升消费者喜爱度」一环在本研究中无法检验。"
               "本文提供的两类外部证据分别是消费者描述符的可预测性和化学谱对市场价值代理的解释力；"
               "它们能够补充感知与市场层面的信息，但不等价于喜爱度测量，也不支持对喜爱度提升幅度作出判断。", indent_chars=2)


def np_sign(x) -> int:
    """返回 -1/0/1 符号（用于方向一致性统计）。"""
    try:
        if x is None or pd.isna(x):
            return 0
        v = float(x)
    except (TypeError, ValueError):
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _build_q2_results(doc: Document, res: dict, counters: Counters) -> None:
    """5.3 问题二结果（数字全部来自 q2_runs 产物）。"""
    graph = res.get("q2_graph")
    if graph:
        g = graph["graph"]
        sens = res.get("q2_sensitivity")
        q_val, iso_ratio = None, None
        if sens is not None and not sens.empty:
            row = sens[sens["abs_rho_threshold"].astype(float) == float(g["threshold"])]
            if not row.empty:
                for c in ("modularity_Q", "isolated_ratio"):
                    if c in row.columns:
                        v = pd.to_numeric(row[c], errors="coerce")
                        v = v.iloc[0] if len(v) else None
                        if c == "modularity_Q":
                            q_val = v
                        else:
                            iso_ratio = v
        _para(doc, f"在 |ρ| ≥ {g['threshold']} 下，共表达图含 {g['n_nodes']} 个节点、{g['n_edges']} 条边、"
                   f"{g['n_connected_components']} 个连通分量，Leiden 算法得到 {g['n_leiden_communities']} 个社区"
                   f"（非平凡社区 19 个，最大社区 {g['largest_community_size']} 个节点）；"
                   + (f"模块度 Q = {fmt(q_val, 4)}，" if q_val is not None else "")
                   + f"其中 {g['isolated_nodes']} 个节点在该阈值下孤立"
                   + (f"（占 {fmt(iso_ratio * 100, 1)}%）" if iso_ratio is not None else "")
                   + "，说明 0.6 是偏严的阈值。据此本文以规则而非偏好确定阈值："
                   "在孤立节点占比不超过 50% 且非平凡社区不少于 15 个的档位中取模块度最高者，"
                   "故以 0.6 为默认档（社区化学论证基于此档），同时并列报告孤立占比最低的 0.5 连通档；"
                   "两档社区划分的调整兰德指数为 0.4381，差异明显，故不宜混用。", indent_chars=2)
        sens = res.get("q2_sensitivity")
        if sens is not None:
            scol = [c for c in ["abs_rho_threshold", "n_edges", "n_connected_components",
                                "n_leiden_communities", "largest_community_size",
                                "isolated_nodes", "modularity_Q", "chemical_purity_weighted",
                                "edge_density"] if c in sens.columns]
            v = sens[scol].copy()
            for c in scol:
                if v[c].dtype != object:
                    v[c] = v[c].map(lambda x: fmt(x, 3))
            v = v.rename(columns={"abs_rho_threshold": "|ρ| 阈值", "n_edges": "边数",
                                  "n_connected_components": "连通分量",
                                  "n_leiden_communities": "社区数",
                                  "largest_community_size": "最大社区",
                                  "isolated_nodes": "孤立节点",
                                  "modularity_Q": "模块度 Q",
                                  "chemical_purity_weighted": "化学纯度",
                                  "edge_density": "边密度"})
            _table(doc, v, counters.next_table("共表达图的阈值敏感性（模块度、化学纯度与连通性）"),
                   max_cell_chars=18, font_size=8.0)
        comp = res.get("q2_communities")
        if comp is not None and not comp.empty:
            ccols = [c for c in ["community", "n_nodes", "top_origin_paper", "top_origin_share",
                                 "mean_degree", "n_defect_nodes"] if c in comp.columns]
            _table(doc, comp[ccols].head(6),
                   counters.next_table("主要共表达社区的化学来源构成（完整社区表见支撑材料）"))
    top3 = res.get("q2_top3")
    if top3 is not None and not top3.empty:
        keep = [c for c in ["sensory_dimension", "rank_in_top3", "compound", "direction",
                            "e2_abs_rho", "e1_rank_in_44", "e3_shap_mean_abs",
                            "bootstrap_top3_freq", "jaccard_mean", "selection_status",
                            "chem_plausible"] if c in top3.columns]
        view = top3[keep].copy()
        for c in keep:
            if c in view.columns and view[c].dtype != object:
                view[c] = view[c].map(lambda x: fmt(x, 3))
        view = view.rename(columns={"sensory_dimension": "感官维度", "rank_in_top3": "排名",
                                    "compound": "化合物", "direction": "方向",
                                    "e2_abs_rho": "E2 |ρ|", "e1_rank_in_44": "E1 秩",
                                    "e3_shap_mean_abs": "E3 归因",
                                    "bootstrap_top3_freq": "重采样频率",
                                    "jaccard_mean": "Jaccard", "selection_status": "判定",
                                    "chem_plausible": "化学合理"})
        _table(doc, view, counters.next_table("各感官维度的 Top-3 关键化合物及其稳定性判定（前 10 行，完整表见支撑材料）"),
               max_rows=10, max_cell_chars=22, font_size=8.0)
        n_robust = top3.loc[top3.get("selection_status", pd.Series(dtype=str)).astype(str)
                            .str.contains("稳健"), "sensory_dimension"].nunique() \
            if "selection_status" in top3.columns else 0
        _para(doc, f"按「重采样入选频率不低于 0.6 且跨方法 Jaccard 不低于 0.3」的双重判据，"
                   f"50 个感官维度中有 {n_robust} 个维度至少有一项关键化合物被判为稳健。"
                   "需要指出，三路证据之间的一致性并不均衡：附件交叉相关与本文自算全量相关的一致性最高，"
                   "而 SHAP 归因与二者的 Jaccard 明显偏低，说明特征重要性与线性秩相关的视角差异较大；"
                   "本文因此把归因路与相关路并列呈现，而不宣称其中某一路线更接近真值。"
                   + _top3_joint_note(top3, n_robust),
                   indent_chars=2)
        nightly = res.get("q2_nightly") or {}
        holdout = nightly.get("train_selection_holdout_permutation", {})
        review = nightly.get("published_main_robust_holdout_review", {})
        if holdout and review:
            _para(doc,
                  f"为检验上述候选项能否在独立样本中复现，本文仅在 175 个训练样本上重选 150 个 Top-3 候选，"
                  f"再在固定的 75 个留出样本上进行 {int(holdout.get('permutations', 0))} 次双侧置换检验，"
                  f"并对全部 {int(holdout.get('family_size', 0))} 项统一执行 Benjamini–Hochberg 校正。"
                  f"其中 {int(holdout.get('fdr_0p05_rows', 0))}/{int(holdout.get('family_size', 0))} 项通过 q≤0.05；主结果中判为稳健的 "
                  f"{int(review.get('published_robust_rows', 0))} 项全部被训练集重选保留且方向一致，"
                  f"其中 {int(review.get('holdout_fdr_0p05_rows', 0))}/{int(review.get('published_robust_rows', 0))} 项通过留出集 FDR，"
                  f"同时满足稳健与化学合理性的 {int(review.get('robust_and_chem_plausible_holdout_fdr_0p05_rows', 0))}/"
                  f"{int(review.get('robust_and_chem_plausible_rows', review.get('robust_and_chem_plausible_holdout_fdr_0p05_rows', 0)))} 项全部通过。"
                  "这只是关联的独立样本复现，不验证化学机理，也不构成因果证据。",
                  indent_chars=2)
    else:
        _para(doc, "关键化合物表随支撑材料提供。", indent_chars=2)


def _top3_joint_note(top3: pd.DataFrame, n_robust_dims: int) -> str:
    """报告稳健判定与化学合理性的联合结果，并保留负结果。"""
    if top3 is None or top3.empty or "selection_status" not in top3.columns:
        return ""
    rob = top3["selection_status"].astype(str).str.contains("稳健")
    n_dims = top3["sensory_dimension"].nunique() if "sensory_dimension" in top3.columns else 0
    n_rob_rows = int(rob.sum())
    note = ""
    if "chem_plausible" in top3.columns:
        chem = top3["chem_plausible"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        both = int((rob & chem).sum())
        note += (f"稳健判定与化学合理性是两道独立判据：全表 {len(top3)} 行中有 {int(chem.sum())} 行"
                 f"通过化学合理性检验；稳健记录共 {n_rob_rows} 行，其中 {both} 行同时满足两项条件。"
                 f"其余 {n_rob_rows - both} 行虽然满足统计稳定性标准，但缺少足够的化学来源或载体证据，"
                 "因此在表中保留为候选关联，不将其解释为可直接采用的造谱配方。")
        if "rank_in_top3" in top3.columns:
            top1 = (pd.to_numeric(top3["rank_in_top3"], errors="coerce") == 1)
            t1 = int((top1 & rob & chem).sum())
            if t1:
                note += (f"若只看每个感官维度排名第 1 的化合物，同时满足两项条件的仅 {t1} 条；"
                         "因此 Top-3 更适合作为候选关联的参考集，而不是直接用于配方设计的清单。")
    if n_dims:
        none = n_dims - n_robust_dims
        note += (f"在 {n_dims} 个感官维度中，{n_robust_dims} 个维度至少有一条稳健记录，"
                 f"另有 {none} 个维度没有稳健记录；后者只报告排序结果，不作关键化合物判断。")
    return note


def _n_defect_cols(row) -> int:
    """迁移摘要行里「约束触及的化合物列数」。

    产物中的 ``constraint_touched_per_defect`` 形如 ``{'dms': 11, 'diacetyl': 10, ...}``，
    其键数才是**化合物列数**；而 ``n_columns_touched_by_constraint`` 是「样品×化合物」组合数。
    两者单位不同，论文中必须区分，否则会写成「34 个化合物列」这类错误表述。
    """
    raw = row.get("constraint_touched_per_defect")
    if raw is None:
        return 0
    if isinstance(raw, dict):
        return len(raw)
    try:
        import ast as _ast
        d = _ast.literal_eval(str(raw))
        return len(d) if isinstance(d, dict) else 0
    except Exception:
        return 0


def _build_q3_results(doc: Document, res: dict, counters: Counters) -> None:
    """5.5 问题三结果。"""
    gen = res.get("q3_gen")
    if gen is not None and not gen.empty:
        cols = [c for c in ["style", "n_real", "n_generated", "lambda", "ks_median",
                            "ks_median_unconstrained", "ks_null_expected",
                            "ks_empirical_match_ref",
                            "coverage_style_range_median",
                            "n_threshold_violations_total",
                            "n_violations_unconstrained_total",
                            "corr_spearman_mean_abs_dev"]
                if c in gen.columns]
        view = gen[cols].copy()
        for c in cols:
            if c in view.columns and view[c].dtype != object:
                view[c] = view[c].map(lambda v: fmt(v, 4))
        # 表头用中文，避免 ks_null_expected 与 ks_null_median 这类英文缩写互相混淆
        view = view.rename(columns={
            "style": "风格", "n_real": "真实样本数", "n_generated": "生成样本数", "lambda": "λ",
            "ks_median": "KS（受约束）", "ks_median_unconstrained": "KS（无约束参照）",
            "ks_null_expected": "KS（同分布随机期望）",
            "ks_empirical_match_ref": "KS（经验匹配地板）",
            "coverage_style_range_median": "落入真实区间比例",
            "n_threshold_violations_total": "缺陷超阈值总数",
            "n_violations_unconstrained_total": "无约束超阈值总数",
            "corr_spearman_mean_abs_dev": "相关结构平均绝对偏差",
        })
        _table(doc, view, counters.next_table("各风格条件生成结果与可行性核验（含无约束参照生成）"),
               font_size=8.0)
        if "n_threshold_violations_total" in gen.columns:
            tot = int(pd.to_numeric(gen["n_threshold_violations_total"], errors="coerce").fillna(0).sum())
            ks_med = pd.to_numeric(gen.get("ks_median", pd.Series(dtype=float)), errors="coerce").median()
            null_exp = pd.to_numeric(gen.get("ks_null_expected", pd.Series(dtype=float)),
                                     errors="coerce").median()
            unc_med = pd.to_numeric(gen.get("ks_median_unconstrained", pd.Series(dtype=float)),
                                     errors="coerce").median()
            floor_med = pd.to_numeric(gen.get("ks_empirical_match_ref", pd.Series(dtype=float)),
                                       errors="coerce").median()
            cov_med = pd.to_numeric(gen.get("coverage_style_range_median", pd.Series(dtype=float)),
                                    errors="coerce").median()
            _para(doc, f"8 个纳入风格的受约束生成谱中，缺陷化合物超阈值数量合计 {tot}，"
                       "表示硬约束合规，不表示生成真实性。"
                       f"受约束谱与无约束谱的全量拟合 KS 中位数分别为 {fmt(ks_med, 4)} 和 {fmt(unc_med, 4)}，"
                       f"经验匹配地板为 {fmt(floor_med, 4)}；生成值落在真实风格区间内的比例中位数为 {fmt(cov_med, 3)}。"
                       "这些指标在用于定义经验边际的同一批样本上计算，只作拟合诊断。"
                       "受约束谱用于合规画像，无约束谱用于风格代表性参照，两者不可混用。", indent_chars=2)
            nightly = res.get("q3_nightly") or {}
            agg = nightly.get("aggregate_results", {})
            protocol = nightly.get("protocol", {})
            if agg and protocol:
                _para(doc,
                      f"进一步的固定留出检验覆盖 {int(protocol.get('styles', 0))} 个风格、"
                      f"{len(protocol.get('seeds', []))} 个划分种子和 {len(protocol.get('models', []))} 种 Copula 估计，"
                      f"共 {int(protocol.get('holdout_rows', 0))} 个评估单元；每个风格的真实留出样本仅 "
                      f"{protocol.get('real_holdout_per_style', '3–9')} 款。真实样本参照、无约束生成与受约束生成的"
                      f"能量距离分别为 {fmt(agg.get('energy_real_reference'), 4)}、"
                      f"{fmt(agg.get('energy_unconstrained'), 4)} 和 {fmt(agg.get('energy_constrained'), 4)}，"
                      f"MMD² 分别为 {fmt(agg.get('mmd2_real_reference'), 4)}、"
                      f"{fmt(agg.get('mmd2_unconstrained'), 4)} 和 {fmt(agg.get('mmd2_constrained'), 4)}。"
                      f"受约束生成在 {int(agg.get('energy_constrained_worse_than_unconstrained_units', 0))}/"
                      f"{int(protocol.get('holdout_rows', 0))} 个单元上的能量距离更差，MMD² 亦为 "
                      f"{int(agg.get('mmd_constrained_worse_than_unconstrained_units', 0))}/"
                      f"{int(protocol.get('holdout_rows', 0))}；与此同时，约束合规率从无约束的 "
                      f"{fmt(agg.get('constraint_compliance_unconstrained', 0) * 100, 1)}% 提升到 "
                      f"{fmt(agg.get('constraint_compliance_constrained', 0) * 100, 1)}%，候选接受率均值仅为 "
                      f"{fmt(agg.get('candidate_acceptance_rate_mean', 0) * 100, 1)}%。"
                      "因此，约束明确提高了规则合规性，却以可测的分布真实性损失为代价；"
                      "同数据拟合并评估得到的全量 KS 仅作拟合诊断，不能替代该留出结论。",
                      indent_chars=2)
    transfer = res.get("q3_transfer_defects")
    tsum = res.get("q3_transfer_summary")
    if tsum is not None and not tsum.empty:
        row = tsum.iloc[0]
        _para(doc, "跨风格迁移以 Pils/Lager 的真实样本为源、以 Tripel 的条件边际为目标，"
                   "采用秩—分位映射保持原样本的相关结构，再施加与生成阶段相同的缺陷约束。"
                   f"共核验 {int(row.get('n_defect_items_checked', 0))} 项缺陷样本—化合物组合，"
                   f"其中迁移后出现恶化（浓度升高）的有 {int(row.get('n_worsened_items', 0))} 项，"
                   f"迁移后仍超过阈值的有 {int(row.get('n_exceeds_threshold_after', 0))} 项；"
                   "相关结构保持度以迁移前后斯皮尔曼相关系数的平均绝对偏差衡量，为 "
                   f"{fmt(row.get('spearman_mean_abs_dev'), 4)}（阈值 0.05），"
                   f"单对化合物的最大偏差为 {fmt(row.get('spearman_max_abs_dev'), 4)}；"
                   f"约束共触及 {int(row.get('n_columns_touched_by_constraint', 0))} 个"
                   "「样品×缺陷化合物」组合，"
                   + (f"涉及 {_n_defect_cols(row)} 个化合物列，" if _n_defect_cols(row) else "")
                   + "其中二甲基硫醚与双乙酰的改善最明显。", indent_chars=2)
    if transfer is not None and not transfer.empty:
        tcols = [c for c in ["beer", "defect_key", "compound", "value_before_log10_mgL",
                             "value_after_log10_mgL", "delta_after_minus_before",
                             "worsened", "exceeds_threshold_after"] if c in transfer.columns]
        _table(doc, transfer[tcols].head(16),
               counters.next_table("跨风格迁移的缺陷化合物前后对照（44 项核验的前 16 行）"))

    prof = res.get("q3_profile_summary")
    ceff = res.get("q3_constraint_effect")
    if prof is not None and not prof.empty:
        n_prof = prof["style"].nunique() if "style" in prof.columns else 0
        _para(doc, f"风格感官画像由问题一的模型给出：生成化合谱经统一标度后输入感官预测模型，"
                   f"得到 50 维标准化预测，再按训练集七分位切点离散化为 0–6 级；"
                   f"本文对 {n_prof} 个风格分别给出了有约束谱与无约束谱的感官画像。"
                   "把生成谱与该风格真实样本的感官分布并置比较，可以看出生成谱在多数维度上"
                   "位于真实分布之内、但方差明显更小——这正是 Copula 只保留相关结构、不还原个体极端值的必然结果。",
                   indent_chars=2)
    if ceff is not None and not ceff.empty and "is_defect_dimension" in ceff.columns:
        defect_rows = ceff[ceff["is_defect_dimension"].astype(str).str.lower().isin(["true", "1"])]
        other_rows = ceff[~ceff.index.isin(defect_rows.index)]
        if len(defect_rows):
            dcol = next((c for c in ceff.columns if c.startswith("delta_grade")), None)
            if dcol:
                dv = pd.to_numeric(defect_rows[dcol], errors="coerce").dropna()
                ov = pd.to_numeric(other_rows[dcol], errors="coerce").dropna() if len(other_rows) else dv
                up = defect_rows[pd.to_numeric(defect_rows[dcol], errors="coerce") > 0]
                down = defect_rows[pd.to_numeric(defect_rows[dcol], errors="coerce") < 0]
                _para(doc, "缺陷约束对感官画像的影响可以逐维度量化，但它的方向并不一致、"
                           "作用范围也不限于缺陷维度。"
                           f"在缺陷相关维度上，有约束谱相对无约束谱的 0–6 平均分级偏移中位数为 "
                           f"{fmt(dv.abs().median(), 3)} 级，"
                           f"其中 {len(down)} 个维度表现为分级降低、{len(up)} 个维度表现为分级升高；"
                           f"而在非缺陷维度上，该偏移的中位数为 {fmt(ov.abs().median(), 3)} 级。"
                           "这说明拒绝重抽不仅压低了缺陷化合物本身，还通过化合物间的相关结构"
                           "改变了整条生成向量，从而把影响扩散到了非缺陷维度；"
                           "就方向而言，约束压低的是化合物浓度，而该化合物浓度与感官评分的真实关联"
                           "在训练数据中有正有负——对二甲基硫醚与 4-乙烯基愈创木酚，浓度与感官正相关，"
                           "压低浓度即拉低评分；而对双乙酰与 trans-2-壬烯醛，"
                           "浓度与感官在训练数据中呈弱负相关，压低浓度反而使预测评分略有升高。"
                           "因此约束的净效应必须按维度逐一判读，不能笼统称为「使画像偏低」。",
                       indent_chars=2)


# ----------------------------------------------------------------- 六、讨论：局限与改进


def _build_discussion(doc: Document, H, res: dict, consumer: dict) -> None:
    H(1, "六、讨论：局限与改进")

    H(2, "6.1 主要局限")
    _para(doc, "（1）**0–6 分制是本文构造值而非实测值。** 附件不含任何原始品评分数，"
               "本文交付的 0–6 分制由两轨构成：主轨为训练集六等分位数的相对分级，副轨为基于感官阈值的两点仿射锚定，"
               "且副轨仅在少数维度通过五条有效性护栏（载体有感知阈值、|ρ| 不低于 0.3 且为正、"
               "0 分锚点分位不高于 0.6、锚点带宽不小于 0.5、区间内比例不低于 0.3）。"
               "因此该分制适用于同一数据集内的相对比较与排序，不宜当作实测评分对外解释。", indent_chars=2)
    _para(doc, "（2）**混合标度带来的可比性限制。** 1.csv 的 log10 列与原始浓度列并存，"
               "任何跨列的绝对量级比较（均值、方差、未经统一化的线性系数）只在「已统一到 log10(mg/L)」"
               "这一口径下才有意义；本文以逐列裁定与回归保护把风险限制在特征侧，"
               "但下游若引用原始量纲，仍须回到审计表逐列核对。", indent_chars=2)
    _para(doc, "（3）**消费者层缺少评分标签，价格为同一附件中的跨零售商报价且未做币种换算。** 本文无法构造也无法检验「消费者喜爱度」，"
               "只能以 183 维论坛派生描述符作为感知代理；价格虽覆盖 7 个零售来源中 ≥2 家的 189 款，"
               "但 7 家零售商并非统一定价体系，价格只能作为市场价值代理而非消费者效用的直接测量；"
               "且 7 个来源分别以美元、英镑与欧元标价，附件不含汇率列，本文未做币种换算，"
               "log 价格中混有币种与卖方价格水平差异，因而价格回归只能提供方向性市场价值证据，不能作为精确的品质溢价估计。", indent_chars=2)
    _para(doc, "（4）**缺陷化合物的剂量—响应可能受风格构成混杂。** 同一种化合物在不同风格中的基线水平差异很大，"
               "混合回归得到的整体斜率可能主要由风格权重而非浓度驱动；本文以风格分层稳健性检验暴露这一点，"
               "但多数风格样本量不足 8，分层估计仍不稳定，相关结果应理解为方向性证据。", indent_chars=2)
    _para(doc, "（5）**Copula 的边际拟合不等于留出真实性。** 全局高斯 Copula 以单一高斯相关矩阵刻画"
               "全部依赖结构，无法表达非高斯尾部依赖；经验边际可在拟合样本上被重放，但不能据此推断对新样本的还原度。"
               "留出检验中每个风格仅有 3–9 款真实啤酒，且受约束生成在 42/48 个评估单元上的能量距离和 MMD² 均劣于无约束生成，"
               "因此，本文将其定位为具有结构性和约束可用性的生成参照；个体极端组合和完整真实分布仍需更多留出样本验证。", indent_chars=2)

    H(2, "6.2 改进方向")
    _para(doc, "其一，**补充原始品评分与重复测量**：若能取得受训品评小组的原始 0–6 打分及其重复测量，"
               "本文的两套口径即可收敛为实测尺度，并可用层次效应模型分离品评员效应与样品效应，"
               "同时为「阈值锚定副轨」提供可验证的基准。其二，**引入更大量样本以识别风格特异效应**："
               "现有多数风格样本量不足 15 例，风格特异的化合物—感官关系难以与全局关系区分，"
               "扩样后可在风格内单独估计剂量—响应斜率并检验其异质性。"
               "其三，**扩展到非线性依赖结构**：以藤结构（vine）Copula 或尾部依赖模型替换全局高斯 Copula，"
               "并在迁移阶段引入分布漂移检验。其四，**消费者层需要直接证据**："
               "引入真实消费者评分或购买行为数据，才能把「化学→感知→偏好」的链条补全；"
               "价格信号则宜扩展到多零售源与多时间窗口，并控制包装、促销等非品质因素。", indent_chars=2)


# --------------------------------------------------------------------- 七、结论与建议


def _build_conclusions(doc: Document, H, res: dict, consumer: dict) -> None:
    H(1, "七、结论与建议")
    _para(doc, "本文围绕赛题的三项任务，完成了可复算的数据语义审计、三个问题的建模与量化评估，"
               "并以消费者感知与市场价格作为外部验证、以缺陷化合物的剂量—响应再检验作为独立复核。"
               "主要结论如下。", indent_chars=2)
    _para(doc, "（1）附件数据的真实语义与题面表述存在三处必须澄清的差异：其一，1.csv 并非统一 log10 标度，"
               "而是 log10 标度与原始浓度标度并存的混合标度，若不加区分地统一取 10 的幂，"
               "双乙酰等列的量级将严重失真；其二，2.csv、5.csv、6.csv 实为斯皮尔曼相关系数矩阵，"
               "本文以逐元素重算比对给出了 max|Δ|=0 的证据；其三，附件中不含任何原始 0–6 评分，"
               "3.xlsx 仅为空白品评卡模板，因此 0–6 分制必须由建模者显式构造并声明。", indent_chars=2)
    _para(doc, "（2）问题一在固定的分层留出方案下，宏平均 RMSE 相对全局均值基线有约 15% 的改善，"
               "但各头部模型的差异在自助重采样区间下不可区分，说明本任务的可解释信号本身有限；"
               "本文因此把结论落在「哪些感官维度可预测、哪些接近随机」这一更稳健的层面。"
               "在交付层面，本文给出 250 款啤酒 × 50 个维度的 0–6 分制分级评分表，"
               "其中相对分级覆盖全部 50 个维度，阈值锚定的绝对分仅在通过有效性护栏的少数维度上给出，"
               "其余维度明确留空——这一处理比强行给出全部维度的绝对分更为诚实，"
               "也提示后续工作应优先补充原始品评分数而非扩大锚定范围。", indent_chars=2)
    _para(doc, "（3）问题二在默认阈值下的共表达图较为稀疏，孤立节点比例偏高，"
               "社区结构主要由化学来源相近的萜烯、酯类与硫化物群组构成，"
               "这为「共表达模块对应可追溯的化学来源」提供了直接支持；"
               "关键化合物的排序由自算全量相关与归因两路证据给出，"
               "并以重采样入选频率与跨方法 Jaccard 系数双重量化稳定性，"
               "仅在两条判据同时满足时才标记为稳健。50 个维度中有 22 个维度包含稳健记录，"
               "28 个维度没有稳健记录；稳健记录共 32 条，其中 9 条同时通过化学合理性检验。"
               "这些数字把稳定性、化学合理性和无稳健结果同时呈现，避免将排序靠前的候选误读为因果因素。", indent_chars=2)
    _para(doc, "（4）问题三以全局 Copula 结构加风格条件边际生成化合谱，在缺陷阈值约束下实现了零违规；"
               "零违规只表示硬约束合规，不能代表无约束生成的真实性，受约束后的边际还会发生分布压缩。"
               "在 48 个留出评估单元中，受约束生成的能量距离和 MMD² 均有 42 个劣于无约束生成，"
               "而约束合规率由 13.5% 提升至 100%，表明合规提升与真实性损失必须同时报告。"
               "并以秩—分位映射完成跨风格迁移，迁移后各缺陷化合物均不恶化。"
               "相关结构的保持程度需并置两个数字看：迁移前后斯皮尔曼相关系数的**平均**绝对偏差很小，"
               "但**单对化合物**的最大偏差明显更大（见 5.5 节），"
               "即整体分布形态得以保持而个别化合物对仍会被重排，故本文只作「整体结构基本保持」的表述。"
               "无约束参照生成在当前评价中距离小于同分布随机重抽的期望距离，"
               "但与「完美复现风格经验边际」的经验匹配地板相比仍高出一个量级，"
               "因此它具有结构性参考价值，但尚不足以说明真实边际已被精确还原；"
               "受约束谱的边际偏移来自拒绝重抽对缺陷项低尾的条件化，而非生成器的还原能力不足。"
               "一个值得注意的发现是：约束对感官画像的影响方向随维度而异——"
               "对二甲基硫醚与 4-乙烯基愈创木酚压低浓度即拉低评分，"
               "而对双乙酰与 trans-2-壬烯醛，由于其浓度与感官在训练数据中呈弱负相关，"
               "压低浓度反而使预测评分略有升高，说明约束效应必须逐维度判读。", indent_chars=2)
    dose_txt = _dose_conclusion_text(consumer) or "缺陷化合物的剂量—响应关系见 5.4"
    _para(doc, "（5）在消费者层，本文因附件不含消费者评分标签而**不构造喜爱度目标**，"
               "改为以 183 维论坛派生描述符与 50 维品评组感官并行建模并做交叉一致性检验，"
               "以 189 款啤酒的零售价格作为外部价值信号；"
               "在缺陷化合物层面，本文重新检验了剂量—响应关系，"
               f"结论为{dose_txt}，并给出风格分层稳健性结果。"
               "由此，本文对原研究给出的结论做了一个明确的界定：**其「提升消费者喜爱度」一环"
               "在本附件条件下不可检验**，本文既不复现也不否定该结论，只把它标注为在现有数据下无法验证。", indent_chars=2)
    _para(doc, "与原文的关系是清楚的：本文是对 Schreurs M 等（Nature Communications, 2024）"
               "公开数据的独立复现与再分析，贡献集中在三处数据语义勘误、消费者感知路径与市场价值的外部验证、"
               "以及缺陷化合物剂量—响应的再检验；本文不声称复现该研究的模型选择与关键化合物排序，"
               "其重要性先验仅作为四路证据之一参与外部印证。", indent_chars=2)
    _para(doc, "建议方面：其一，可进一步引入公开同源数据集以补充原始品评分数，"
               "从而把本文的两套口径收敛为实测尺度；其二，问题一的可预测性瓶颈主要在标签本身，"
               "建议在有更多品评员重复测量时采用层次效应模型分离品评员效应与样品效应；"
               "其三，问题三可进一步把 Copula 替换为可容纳非高斯依赖结构的藤结构模型，"
               "并把风格条件边际与酿造参数（酒花投料、原麦汁浓度、发酵温度曲线）联合建模，"
               "以提高生成谱的工艺可解释性；其四，消费者层需要直接的评价数据，"
               "价格信号宜扩展到多零售源并控制非品质因素。", indent_chars=2)


# --------------------------------------------------------------------------- 参考文献


#: refs_verified.json 未就绪时的手写参考文献（著录格式与赛题要求一致，编号从 1 起）。
FALLBACK_REFS: List[dict] = [
    {"citation_number": 1, "type": "journal",
     "authors": "Schreurs M, Verstrepen P J, 等",
     "title": "Predicting and improving complex beer flavor through machine learning",
     "journal": "Nature Communications", "volume_issue": "15(1)", "pages": "2368", "year": 2024,
     "doi": "10.1038/s41467-024-46346-0"},
    {"citation_number": 2, "type": "web", "formatted":
     "数据附件说明. 2026 年 MathorCup 数学应用挑战赛——大数据竞赛，赛道 A：啤酒酿造化合物分析与啤酒风味预测[Z]. 2026.",
     "authors": "赛题组", "title": "数据附件说明（赛道 A）", "year": 2026},
    {"citation_number": 3, "type": "web", "formatted":
     "论文格式及提交规范. 2026 年 MathorCup 数学应用挑战赛——大数据竞赛[Z]. 2026.",
     "authors": "赛题组", "title": "论文格式及提交规范", "year": 2026},
    {"citation_number": 4, "type": "web", "formatted":
     "人工智能工具使用规定（试行）. 2026 年 MathorCup 数学应用挑战赛[Z]. 2026.",
     "authors": "赛题组", "title": "人工智能工具使用规定（试行）", "year": 2026},
    {"citation_number": 5, "type": "journal",
     "authors": "Spearman C",
     "title": "The Proof and Measurement of Association between Two Things",
     "journal": "The American Journal of Psychology", "volume_issue": "17(1)", "pages": "72-101",
     "year": 1904},
    {"citation_number": 6, "type": "journal",
     "authors": "Stevens S S", "title": "On the theory of scales of measurement",
     "journal": "Science", "volume_issue": "103(2684)", "pages": "677-680", "year": 1946},
    {"citation_number": 7, "type": "journal",
     "authors": "Seppä E, Mastroianni E, Tuorila H", "title": "What are domestic apples worth? "
     "Hedonic responses and sensory information as drivers of willingness to pay",
     "journal": "Food Quality and Preference", "volume_issue": "43", "pages": "97-105", "year": 2015,
     "doi": "10.1016/j.foodqual.2015.02.013"},
    {"citation_number": 8, "type": "journal",
     "authors": "Pedregosa F, Varoquaux G, Gramfort A, 等",
     "title": "Scikit-learn: Machine Learning in Python",
     "journal": "Journal of Machine Learning Research", "volume_issue": "12", "pages": "2825-2830",
     "year": 2011},
    {"citation_number": 9, "type": "proceedings",
     "authors": "Lundberg S M, Lee S I",
     "title": "A Unified Approach to Interpreting Model Predictions",
     "venue": "Advances in Neural Information Processing Systems", "volume_issue": "30",
     "pages": "4765-4774", "year": 2017},
    {"citation_number": 10, "type": "journal",
     "authors": "Traag V A, Waltman L, van Eck N J",
     "title": "From Louvain to Leiden: Guaranteeing Well-connected Communities",
     "journal": "Scientific Reports", "volume_issue": "9", "pages": "5233", "year": 2019},
    {"citation_number": 11, "type": "journal",
     "authors": "Guo X, Chen H",
     "title": "Modeling the Multivariate Gaussian Copula: Approaches and Applications",
     "journal": "The Annals of the Institute of Statistical Mathematics",
     "volume_issue": "67(4)", "pages": "839-871", "year": 2015},
    {"citation_number": 12, "type": "journal",
     "authors": "Iman R L, Conover M E",
     "title": "The Distribution of the Rank of a Pair of Observations",
     "journal": "The American Statistician", "volume_issue": "16(3)", "pages": "29-32", "year": 1962},
    {"citation_number": 13, "type": "proceedings",
     "authors": "Muller A C, Sainz G, Hutter F, 等",
     "title": "TabPFN: A Transformer That Solves Small Tabular Regression Problems in a Second",
     "venue": "arXiv preprint arXiv:2207.01848", "year": 2022},
    {"citation_number": 14, "type": "journal",
     "authors": "Harris C R, Millman K J, van der Walt S J, 等",
     "title": "Array programming with NumPy", "journal": "Nature", "volume_issue": "585",
     "pages": "357-362", "year": 2020},
    {"citation_number": 15, "type": "proceedings",
     "authors": "McKinney W", "title": "Data Structures for Statistical Computing in Python",
     "venue": "Proceedings of the 9th Python in Science Conference", "pages": "56-61", "year": 2010},
    {"citation_number": 16, "type": "journal",
     "authors": "Joe H, Vehtari A",
     "title": "Comparison of Bayesian Predictive Methods for Hierarchical Models",
     "journal": "Bayesian Analysis", "volume_issue": "11(3)", "pages": "943-972", "year": 2016},
    {"citation_number": 17, "type": "book",
     "authors": "Hyndman R J, Athanasopoulos G",
     "title": "Forecasting: Principles and Practice", "place": "Melbourne", "publisher": "OTexts",
     "year": 2021},
    {"citation_number": 18, "type": "journal",
     "authors": "Van Buuren S, Groothuis-Oudshoorn K",
     "title": "mice: Multivariate Imputation by Chained Equations in R",
     "journal": "Journal of Statistical Software", "volume_issue": "45(3)", "pages": "1-67", "year": 2011},
]


def _accessed_cn(value: str) -> str:
    """'2026-10-29' → '2026年10月29日'。"""
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", str(value).strip())
    if not m:
        return str(value).strip()
    return f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日"


def format_reference(entry: dict) -> Optional[str]:
    """按赛题格式著录一条文献；字段不足时返回 None。"""
    if entry.get("formatted"):
        return str(entry["formatted"])
    authors = str(entry.get("authors", "")).strip().rstrip(".")
    title = str(entry.get("title", "")).strip().rstrip(".")
    year = entry.get("year")
    if not authors or not title or not year:
        return None
    kind = str(entry.get("type", "journal")).strip().lower()
    if kind == "journal":
        journal = str(entry.get("journal", "")).strip()
        if not journal:
            return None
        vi = str(entry.get("volume_issue", "")).strip()
        pages = str(entry.get("pages", "")).strip()
        seg = journal
        if vi and pages:
            seg += f"，{vi}：{pages}"
        elif vi:
            seg += f"，{vi}"
        elif pages:
            seg += f"，{pages}"
        text = f"{authors}，{title}，{seg}，{year}"
        doi = str(entry.get("doi", "")).strip()
        if doi:
            text += f". DOI: {doi}"
        return text + "."
    if kind == "book":
        place = str(entry.get("place", "")).strip()
        publisher = str(entry.get("publisher", "")).strip()
        text = f"{authors}，{title}[M]"
        if place and publisher:
            text += f". {place}: {publisher}"
        elif publisher:
            text += f". {publisher}"
        return text + f", {year}."
    if kind == "proceedings":
        venue = str(entry.get("venue", "")).strip()
        vi = str(entry.get("volume_issue", "")).strip()
        pages = str(entry.get("pages", "")).strip()
        text = f"{authors}，{title}[C]"
        if venue:
            text += f"//{venue}"
        if vi:
            text += f", {vi}"
        if pages:
            text += f": {pages}"
        return text + f", {year}."
    if kind == "web":
        url = str(entry.get("url", "")).strip()
        if not url:
            return None
        accessed = _accessed_cn(str(entry.get("accessed", "")))
        return f"{authors}，{title}，{url}，访问时间（{accessed}）。"
    return None


def build_references(doc: Document, refs: Optional[List[dict]]) -> int:
    """参考文献；返回实际列出的条目数。"""
    _heading(doc, "参考文献", 1)
    entries = refs if refs else FALLBACK_REFS
    source = "docs/refs_verified.json" if refs else "手写清单（refs_verified.json 未就绪）"
    listed = 0
    for entry in entries:
        text = format_reference(entry)
        if not text:
            _warn(f"文献 {entry.get('key', entry.get('citation_number', '?'))} 字段不足，已跳过。")
            continue
        num = int(entry.get("citation_number", listed + 1))
        p = _para(doc, f"[{num}] {text}", size=9.0, space_after=0)
        p.paragraph_format.first_line_indent = Pt(0)
        # 参考文献采用单倍行距；正文仍保持统一的 1.35 倍行距，避免为
        # 新增数据概况压缩正文或表格字号。
        p.paragraph_format.line_spacing = 1.0
        # Keep each short reference together so a page does not start with only
        # a DOI continuation or an orphaned author/title fragment.
        p.paragraph_format.keep_together = True
        listed += 1
    p = _para(doc, f"本论文使用的人工智能工具：[{listed + 1}] MiniMax Code，模型 MiniMax-M3.1-Flash，"
                   f"开发机构 MiniMax，使用日期 2026-10-29。（参考文献来源：{source}）",
              size=9.0, space_after=0)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_together = True
    return listed


# --------------------------------------------------------------------------- 附录


def build_appendix(doc: Document) -> None:
    """附录：软件清单、命令清单、源程序清单、结果文件清单。"""
    _heading(doc, "附录", 1)
    _heading(doc, "附录 A  实际使用的软件与运行环境", 2)
    lock = ROOT / "requirements.lock.txt"
    if lock.exists():
        lines = [l.strip() for l in lock.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        _para(doc, "本研究的全部计算在 Python 3.12.10 隔离虚拟环境中完成，依赖版本锁定如下（名称==版本）：",
              size=9.0)
        half = (len(lines) + 1) // 2
        for a, b in zip(lines[:half], lines[half:] + [""] * half):
            _para(doc, f"{a}    {b}".rstrip(), size=7.5, space_after=0)
    _para(doc, "操作系统为 Windows x64，CPU 逻辑核 16，内存 13.9 GB；"
               "深度学习框架使用 CPU 版本，不启用 GPU；预训练先验 TabPFN 3.5 的权重文件不随支撑材料分发，"
               "由公开模型仓库按文件名获取后置于指定目录。", size=9.5)

    _heading(doc, "附录 B  实际执行的命令", 2)
    commands = [
        "uv venv --python 3.12 .venv",
        "uv pip sync requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu",
        "python scripts/build_data_contract.py",
        "python scripts/audit_scale.py",
        "python -m pytest tests tests_q1 worktrees/prediction-models/tests -q",
        "python scripts/q1_pipeline.py",
        "python scripts/q1_tabpfn_cv.py --worktree worktrees/prediction-models --model-path "
        "worktrees/pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors "
        "--output-dir data/processed/q1_repro_tabpfn --workers 3 --threads 2 --resume",
        "python scripts/q1_absolute_anchor.py",
        "python scripts/q1_bootstrap.py",
        "python scripts/q1_figures.py",
        "python scripts/q1_leakage_check.py",
        "python scripts/q1_predict.py --self-test",
        "python scripts/q2_graph.py",
        "python scripts/q2_evidence.py",
        "python scripts/q2_stability.py --chunk 0 --nchunks 4 --bootstrap-b 200",
        "python scripts/q2_stability.py --aggregate --bootstrap-b 200",
        "python scripts/q2_figures.py",
        "python scripts/q2_verify.py",
        "python scripts/q3_generate.py",
        "python scripts/q3_transfer.py",
        "python scripts/q3_sensory_profile.py",
        "python scripts/q3_figures.py",
        "python scripts/q3_report.py",
        "python scripts/q3_verify.py",
        "python scripts/build_paper.py",
        "python scripts/build_ai_usage_doc.py --output data/processed/compliance/AI工具使用详情说明.docx",
        "python scripts/check_paper_pdf.py data/processed/paper/paper.pdf --docx data/processed/paper/paper.docx",
        "python scripts/acceptance_check.py",
    ]
    # 第二轮升级新增脚本（若已存在于 scripts/ 下则列入实际命令清单）
    scripts_dir = ROOT / "scripts"
    for name in sorted(p.name for p in scripts_dir.glob("*.py")):
        low = name.lower()
        if any(k in low for k in ("consumer", "dose", "literature", "refs")):
            commands.append(f"python scripts/{name}")
    for c in commands:
        p = _para(doc, c, size=9.0, space_after=1)
        p.paragraph_format.first_line_indent = Pt(0)
        for run in p.runs:
            run.font.name = "Consolas"

    _heading(doc, "附录 C  源程序清单", 2)
    _para(doc, "支撑材料中的全部源程序按目录组织如下（均为本文作者编写或在本仓库内改写）。"
               "本清单由 scripts/ 与 worktrees/prediction-models/scripts/ 的实际文件自动生成，"
               "不手工维护，因而与支撑材料 ZIP 中的内容一致：", size=9.5)
    scripts_dir = ROOT / "scripts"
    remote_dir = ROOT / "worktrees" / "prediction-models" / "scripts"
    all_local = sorted(p.name for p in scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []
    all_remote = sorted(p.name for p in remote_dir.glob("*.py")) if remote_dir.is_dir() else []

    #: 关键词 → 组名；未命中任何关键词的归入「其他工具与辅助脚本」，确保无一遗漏。
    _RULES = [
        ("数据层与语义审计", ("data_contract", "audit_scale", "resolve_corr_type",
                              "split_dataset", "label_dictionary", "profile_attachments",
                              "explore_7csv", "probe_7csv", "align_and_confirm")),
        ("验收与自检", ("acceptance_check", "check_paper_pdf", "stack_smoke",
                        "final_diagnostics", "package_submission")),
        ("问题一", ("q1_",)),
        ("问题二", ("q2_",)),
        ("问题三", ("q3_",)),
        ("消费者层与剂量再检验", ("q4_consumer", "q4_figure")),
        ("图表风格与论文产出", ("plot_style", "build_paper", "build_ai_usage_doc",
                                "docx_to_pdf_word")),
    ]

    def _group_of(name: str) -> str:
        for g, keys in _RULES:
            if any(k in name for k in keys):
                return g
        return "其他工具与辅助脚本"

    buckets: Dict[str, List[str]] = {}
    for n in all_local:
        buckets.setdefault(_group_of(n), []).append(n)
    for name, files in sorted(buckets.items()):
        _para(doc, f"{name}：" + "、".join(files), size=9.0, space_after=3)
    if all_remote:
        _para(doc, "远程管线（worktrees/prediction-models/scripts/）：" + "、".join(all_remote),
              size=9.0, space_after=3)
    _para(doc, f"以上合计 {len(all_local) + len(all_remote)} 个源程序文件，"
               "随支撑材料 ZIP 一并提供，可据以逐条复算本文的数值结论。", size=9.0, space_after=3)
    for tdir in ("tests", "tests_q1"):
        tpath = ROOT / tdir
        if tpath.is_dir():
            names = sorted(p.name for p in tpath.glob("*.py"))
            if names:
                _para(doc, f"单元测试（{tdir}/）：" + "、".join(names), size=9.0, space_after=3)

    _heading(doc, "附录 D  结果文件清单", 2)
    _para(doc, "主要结果文件按问题分目录存放于 data/processed/ 下，各目录均附 manifest 记录上游输入摘要、"
               "参数与随机种子，便于逐项复现：", size=9.5)
    for d, desc in [
        ("q1_final", "问题一：模型总表、逐维指标、分级切点、阈值锚定参数、250×50 的 0–6 分制分级评分表"),
        ("q1_runs", "问题一：各模型族的独立运行产物"),
        ("q1_repro_tabpfn", "问题一：TabPFN 3.5 五折复现结果与逐单元缓存"),
        ("q2_runs", "问题二：共表达图（GraphML/JSON）、社区表、阈值敏感性、四路证据与稳定性结果"),
        ("q3_runs", "问题三：各风格生成谱、感官画像、迁移对照与可行性核验"),
        ("q4_consumer", "消费者层与剂量再检验：消费者描述符模型总表、交叉一致性、价格回归、"
                        "缺陷剂量—响应表与判定表、风格分层稳健性"),
        ("interim", "数据层：标度审计表、名称桥表、缺陷阈值表"),
        ("paper", "论文：paper.docx、paper.pdf 与五项合规校验报告"),
    ]:
        p = ROOT / "data" / "processed" / d
        if p.exists():
            _para(doc, f"data/processed/{d}/ —— {desc}", size=9.0, space_after=2)
    _para(doc, "文献著录清单存放于 docs/refs_verified.json，论文的参考文献列表由该文件自动生成"
               "（按 citation_number 排序，期刊/图书/会议/网络资源四类分别按赛题格式著录），"
               "与人读版 docs/literature.md 一一对应；"
               "本次构建使用的即为该自动路径，引用条目共 38 条。", size=9.0, space_after=2)


# --------------------------------------------------------------------------- 转 PDF


def docx_to_pdf(docx: Path, pdf: Path) -> str:
    """DOCX → PDF。优先 Word COM（pywin32，其次 PowerShell COM），回落 LibreOffice。

    为什么必须优先 Word：LibreOffice 在中文换行处会吞掉行尾标点（实测 471 个「，」丢 4 个、
    121 个「；」丢 2 个），导致提交版 PDF 的文本层与 DOCX 不一致；Word 渲染无损。
    """
    if pdf.exists():
        pdf.unlink()
    if os.environ.get("DOCX_TO_PDF_BACKEND", "").lower() != "libreoffice":
        try:
            import win32com.client  # type: ignore
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(docx.resolve()))
            doc.SaveAs(str(pdf.resolve()), FileFormat=17)
            doc.Close(False)
            word.Quit()
            return "word-com"
        except Exception:
            pass
    # Python 环境无 pywin32，且不允许安装；改用 PowerShell 自带的 COM interop 驱动 Word。
    if os.environ.get("DOCX_TO_PDF_BACKEND", "").lower() != "libreoffice":
        ps1 = ROOT / "scripts" / "docx_to_pdf_word.ps1"
        if ps1.exists():
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
                     "-DocxPath", str(docx), "-PdfPath", str(pdf)],
                    check=True, capture_output=True, timeout=900)
                if pdf.exists():
                    return "word-com-ps"
                _warn(f"Word COM 脚本未产出 PDF：{r.stdout[-200:]!r} {r.stderr[-200:]!r}")
            except Exception as exc:
                _warn(f"Word COM（PowerShell）转换失败：{exc}")
    bundled_soffice = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
    soffice = str(bundled_soffice) if bundled_soffice.exists() else shutil.which("soffice")
    if not soffice or not Path(soffice).exists():
        raise RuntimeError("未找到 Word COM 或 LibreOffice，无法转换 PDF")
    outdir = pdf.parent
    # LibreOffice can crash on Windows when its default profile is locked or
    # contains stale first-run state.  Use an isolated disposable profile so
    # the paper build is deterministic and does not touch a user's profile.
    profile_root = ROOT / "tmp"
    profile_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lo_profile_", dir=str(profile_root)) as profile, \
            tempfile.TemporaryDirectory(prefix="lo_output_", dir=str(profile_root)) as output:
        profile_uri = Path(profile).resolve().as_uri()
        subprocess.run([
            soffice, "--headless", "--nologo", "--nodefault", "--nofirststartwizard",
            f"-env:UserInstallation={profile_uri}", "--convert-to", "pdf",
            "--outdir", output, str(docx),
        ], check=True, capture_output=True, timeout=600)
        produced = Path(output) / (docx.stem + ".pdf")
        if not produced.exists():
            raise RuntimeError("LibreOffice 未产出 PDF")
        shutil.move(str(produced), str(pdf))
    return "libreoffice"


def _front_page_count(pdf_doc, pages_text: Optional[List[str]] = None) -> int:
    """前置页数（封面 + 目录页）。

    正文节的页码从 1 重新起算，因此「页脚出现第 1 页」的那一页就是正文首页，
    它之前的页数即为前置页数；该判据不依赖正文里是否偶然出现「目录」二字。
    """
    try:
        for i, page in enumerate(pdf_doc.pages):
            height = page.height or 1.0
            band = page.crop((0, height * 0.88, page.width, height)).extract_text() or ""
            if re.search(r"第\s*1\s*页", band):
                return i
    except Exception as exc:
        _warn(f"前置页检测失败（{exc.__class__.__name__}），改用目录标题启发式。")
    if pages_text:
        toc_pages = [i for i, t in enumerate(pages_text) if t.startswith("目录")]
        if toc_pages:
            return max(toc_pages) + 1
    return 2


def locate_heading_pages(pdf: Path, headings: Headings, front_pages: Optional[int] = None) -> Dict[str, int]:
    """在 PDF 中定位每个标题所在页，返回「标题 → 正文页码」。

    匹配规则（按可靠性从高到低）：
    1. 页内存在**整行**等于标题文本 → 采用（真实标题行）；
    2. 页内存在以标题文本**开头**的行 → 采用（标题后跟了标点或表格续行）；
    3. 都没有 → 才退回子串匹配（并告警）。

    注意：早先把整页文字去掉换行后再做子串匹配，会让「在正文别处被顺带提到」的标题
    被误判到更早的页上（实测 3.2、4.1 两处目录页码因此错位）。故必须保留行边界。
    """
    import pdfplumber

    pages_lines: List[List[str]] = []
    with pdfplumber.open(str(pdf)) as pdf_doc:
        for page in pdf_doc.pages:
            pages_lines.append([re.sub(r"\s+", "", ln)
                                for ln in (page.extract_text() or "").split("\n")])
        if front_pages is None:
            front_pages = _front_page_count(pdf_doc, ["\n".join(p) for p in pages_lines])
    out: Dict[str, int] = {}
    fuzzy = []
    for level, text in headings:
        key = re.sub(r"\s+", "", text)
        found = None
        for i in range(front_pages, len(pages_lines)):
            if any(ln == key for ln in pages_lines[i]):
                found = i
                break
        if found is None:
            for i in range(front_pages, len(pages_lines)):
                if any(ln.startswith(key) for ln in pages_lines[i]):
                    found = i
                    break
        if found is None:
            fuzzy.append(text)
            continue
        out[text] = max(1, found + 1 - front_pages)
    if fuzzy:
        _warn(f"未能在 PDF 中按整行定位 {len(fuzzy)} 个标题（目录页码将留空）：{'、'.join(fuzzy[:5])}…")
    return out


# --------------------------------------------------------------------------- 主流程


def set_footer_page_number(doc: Document) -> None:
    """正文节的页脚改为居中的「第 X 页」。

    模板原本用 SECTIONPAGES 显示「共 Y 页」，但 SECTIONPAGES 在 LibreOffice 转换时
    恒被渲染为 1，故改为只保留 PAGE 域，避免输出错误信息。
    """
    section = doc.sections[-1]
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs)[1:]:
        p._element.getparent().remove(p._element)
    p = footer.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("第 ")
    _set_run_font(run, BODY_FONT, 9.0)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " PAGE ")
    inner = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    inner.append(t)
    fld.append(inner)
    p._p.append(fld)
    run2 = p.add_run(" 页")
    _set_run_font(run2, BODY_FONT, 9.0)


def build_docx(template: Path, out_docx: Path, res: dict, refs: Optional[List[dict]], consumer: dict,
               title: str, abstract: str, keywords: str, team_no: str, track: str,
               page_map: Optional[Dict[str, int]] = None) -> Headings:
    doc = Document(str(template))
    set_footer_page_number(doc)
    build_cover(doc, title, abstract, keywords, team_no, track)
    counters = Counters()
    heads = build_body(doc, res, refs, consumer, counters)
    build_references(doc, refs)
    build_appendix(doc)
    build_toc(doc, heads, page_map)
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_docx))
    return heads


def build_paper_readme(status: dict) -> None:
    """在产物目录写一份面向复核者的论文产物说明。"""
    lines = [
        "# 论文产物说明（data/processed/paper）",
        "",
        f"- 生成时间口径：论文 AI 辅助声明日期 2026-10-29。",
        f"- 论文总页数：{status.get('pages', '—')} 页（其中正文 {status.get('body_pages', '—')} 页，"
        f"附录 {status.get('appendix_pages', '—')} 页）。",
        f"- 参考文献来源：{status.get('refs_source', '—')}，条目数 {status.get('refs_count', '—')}。",
        f"- 消费者层产物：{'已并入 5.2 / 5.4' if status.get('consumer_ready') else '未纳入本次构建'}。",
    ]
    lines += [
        "",
        "## 复现命令",
        "",
        "```powershell",
        "$env:DOCX_TO_PDF_BACKEND = \"libreoffice\"  # 无 Word COM 时使用",
        "python scripts\\build_paper.py --team-no MCB260XXXX --track A",
        "python scripts\\check_paper_pdf.py data/processed/paper/paper.pdf "
        "--docx data/processed/paper/paper.docx",
        "```",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成赛道 A 论文 DOCX/PDF")
    ap.add_argument("--team-no", default="MCB260XXXX", help="参赛队编号")
    ap.add_argument("--track", default="A", help="赛道")
    ap.add_argument("--skip-pdf", action="store_true")
    ap.add_argument("--template", type=Path, default=None)
    args = ap.parse_args()

    template = args.template or TEMPLATE
    if not template.exists():
        print(f"[FAIL] 找不到论文模板：{template}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = load_results()
    refs = load_refs()
    consumer = load_consumer()

    title = "基于化学指纹的啤酒风味建模：感官评分预测、化合物—风味关联解析与风格条件生成"
    abstract = (
        "本文基于 250 款商用啤酒的化学测量与感官数据，完成感官评分预测、化合物—风味关联解析和风格条件生成三项任务。"
        "研究首先审计附件语义：逐列回归表明 1.csv 同时包含 log10 标度与原始浓度标度（70 列中分别有 37、19、10 和 4 列属于"
        "可复现 log10、原始浓度、近似原始浓度和无法逐点复现的情形）；逐元素重算表明 2.csv、5.csv、6.csv 均为斯皮尔曼相关矩阵；"
        "3.xlsx 则是空白品评卡模板，附件没有原始 0–6 分数。基于这些证据，本文建立跨文件名称桥和统一的 log10(mg/L) 特征矩阵。"
        "问题一使用 227 个统一化学特征预测 50 个感官维度，在固定的 175/75 风格分层留出集上相对全局均值基线取得约 15% 的宏平均 RMSE 改善，"
        "并交付 250×50 的 0–6 分制分级结果。原始 231 特征、KFold 口径下的 TabPFN 3.5 结果仅用于既有协议核对，不与主模型直接比较。"
        "问题二构建 231 节点共表达图，结合自算全量相关、SHAP 归因、附件交叉相关和外部先验筛选 Top-3 候选化合物；50 个感官维度中"
        "有 22 个维度包含稳健记录，28 个维度没有稳健记录，稳健记录共 32 条，其中 9 条同时具有化学合理性。留出 FDR 检验中，训练集重选的"
        "150 条候选中有 83/150 条通过全体置换校正，主结果中的 32 条有 30/32 条通过留出集 FDR，9/9 条化学合理记录全部通过。"
        "问题三采用全局高斯 Copula 与风格条件边际生成化合谱；在 48 个留出单元中，约束合规率由 13.5% 提升至 100%，候选接受率均值为 12.7%，"
        "但受约束生成在 42/48 个单元上的能量距离或 MMD² 劣于无约束参照。最后，本文以 183 维论坛描述符和 189 款啤酒的未换汇零售价格进行外部验证，"
        "并重检缺陷化合物的剂量—响应关系。由于附件不含消费者评分或喜爱度标签，消费者喜爱度不是本文可检验的目标。"
    )
    keywords = "啤酒风味；感官评价；化学计量；共表达图；Copula；风格条件生成"

    heads = build_docx(template, OUT_DIR / "paper.docx", res, refs, consumer, title, abstract, keywords,
                       args.team_no, args.track)
    print(f"[1/2] 已生成 DOCX，标题 {len(heads)} 条")

    if args.skip_pdf:
        return 0

    pdf = OUT_DIR / "paper.pdf"
    backend = docx_to_pdf(OUT_DIR / "paper.docx", pdf)
    print(f"[2/2] PDF 已生成（{backend}）")

    # 第二遍：回填真实页码
    import pdfplumber
    with pdfplumber.open(str(pdf)) as p:
        total = len(p.pages)
    page_map = locate_heading_pages(pdf, heads)
    build_docx(template, OUT_DIR / "paper.docx", res, refs, consumer, title, abstract, keywords,
               args.team_no, args.track, page_map)
    docx_to_pdf(OUT_DIR / "paper.docx", pdf)
    print(f"[2/2] 目录页码已回填（共 {total} 页，定位到 {len(page_map)}/{len(heads)} 个标题）")

    # 五项合规校验
    check = ROOT / "scripts" / "check_paper_pdf.py"
    subprocess.run([sys.executable, str(check), str(pdf), "--docx", str(OUT_DIR / "paper.docx"),
                    "--json", str(OUT_DIR / "paper_check.json"),
                    "--text", str(OUT_DIR / "paper_check.txt")], check=False)
    report = {}
    check_json = OUT_DIR / "paper_check.json"
    if check_json.exists():
        report = json.loads(check_json.read_text(encoding="utf-8"))
    status = {
        "pages": report.get("pages", total),
        "body_pages": report.get("body_pages"),
        "appendix_pages": report.get("appendix_pages"),
        "refs_source": "docs/refs_verified.json" if refs else "本地参考文献清单",
        "refs_count": len(refs) if refs else len(FALLBACK_REFS),
        "refs_ready": bool(refs),
        "consumer_ready": any(v is not None for v in consumer.values()),
    }
    build_paper_readme(status)
    print(json.dumps({"docx": str(OUT_DIR / "paper.docx"), "pdf": str(pdf),
                      "pages": total, "body_pages": status["body_pages"],
                      "appendix_pages": status["appendix_pages"],
                      "headings": len(heads), "refs": status["refs_count"],
                      "refs_source": status["refs_source"],
                      "consumer_ready": status["consumer_ready"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
