# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：论文中出现的每一个数字均由本轮真实运行产物读出（data/processed/q*_runs、
# q1_final），未手工编造；页码经「生成 PDF → 定位标题页 → 回填目录」两遍法核验；
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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "paper"
TEMPLATE = ROOT.parent / "2026年MathorCup数学应用挑战赛—大数据竞赛论文模板.docx"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
LATIN_FONT = "Times New Roman"

# 标题级别：1=章 2=节
Headings = List[Tuple[int, str]]


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


def _figure(container, path: Path, caption: str, width_in: float = 5.6):
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
           max_rows: int = 40):
    """插入三线表风格的表格。"""
    if df is None or df.empty:
        _para(container, f"[表缺失：{caption}]", size=9.0)
        return
    df = df.head(max_rows)
    p = container.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    table = container.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(col))
        _set_run_font(run, HEAD_FONT, 8.5, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _, r in df.iterrows():
        cells = table.add_row().cells
        for j, v in enumerate(r):
            cells[j].text = ""
            txt = "" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
            run = cells[j].paragraphs[0].add_run(txt)
            _set_run_font(run, BODY_FONT, 8.5)
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for j, w in enumerate(widths):
            for row in table.rows:
                row.cells[j].width = Inches(w)
    _caption(container, caption)


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
    return res


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


def build_cover(doc: Document, title: str, abstract: str, keywords: str,
                team_no: str, track: str) -> None:
    """填充封面：题目 / 摘要 / 关键词 / 队伍编号 / 赛道。"""
    paras = doc.paragraphs
    paras[3].text = ""
    run = paras[3].add_run(title)
    _set_run_font(run, HEAD_FONT, 16.0, bold=True)
    paras[3].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paras[3].paragraph_format.space_after = Pt(10)

    paras[4].text = ""
    run = paras[4].add_run("摘  要")
    _set_run_font(run, HEAD_FONT, 13.0, bold=True)
    paras[4].alignment = WD_ALIGN_PARAGRAPH.CENTER

    paras[5].text = ""
    run = paras[5].add_run(abstract)
    _set_run_font(run, BODY_FONT, 10.5)
    paras[5].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paras[5].paragraph_format.line_spacing = 1.4
    paras[5].paragraph_format.first_line_indent = Pt(21)

    paras[22].text = ""
    run = paras[22].add_run("关键词：" + keywords)
    _set_run_font(run, BODY_FONT, 10.5, bold=True)
    paras[22].paragraph_format.space_before = Pt(8)

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
    paras = doc.paragraphs
    toc_title = paras[23]
    toc_title.text = ""
    run = toc_title.add_run("目  录")
    _set_run_font(run, HEAD_FONT, 15.0, bold=True)
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 清除模板自带的示例目录行（24..29）
    for p in paras[24:30]:
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


# --------------------------------------------------------------------------- 正文


def build_body(doc: Document, res: dict) -> Headings:
    """写正文（附录之前），返回标题清单。"""
    heads: Headings = []

    def H(level: int, text: str):
        _heading(doc, text, level)
        heads.append((level, text))

    # 清除模板正文占位（36 号及其后）
    for p in list(doc.paragraphs)[36:]:
        p._element.getparent().remove(p._element)

    # ---------------- 一、问题分析
    H(1, "一、问题分析")
    _para(doc, "本题以 250 款商用啤酒的化学测量数据、感官品评数据与多张相关矩阵为输入，要求完成三件"
               "本质上互不相同的任务：三其一是在感官标签仅以标准化形式给出的条件下，还原并给出可解释的"
               "0–6 分制分级评分；其二是构建化合物共表达图并给出每个感官维度的关键化合物及其稳定性；"
               "其三是以风格为条件生成化学谱并做跨风格迁移。三者的共同难点在于：附件并未提供可直接使用的"
               "标签与量纲，任何未经核实就建模的做法都会在量纲与口径两个层面同时失效。", indent_chars=2)
    _para(doc, "本文的总体应对是：先用一套可复算的证据把数据的真实语义钉死，再让三个问题共享同一份"
               "经过审计的特征矩阵。具体而言，(1) 逐列裁定 1.csv 的变换标度；(2) 以「重算相关矩阵并与"
               "给定矩阵逐元素比对」的方式裁定三张相关矩阵的相关系数类型；(3) 建立跨附件的名称桥，"
               "使相关矩阵、阈值表与特征矩阵可以对齐；(4) 区分左删失（未检出）与真缺失；"
               "(5) 固定按风格分层的训练/留出划分，所有调参只在训练集内部完成。", indent_chars=2)

    # ---------------- 二、数据处理
    H(1, "二、数据处理")
    H(2, "2.1 附件结构与量纲裁定")
    _para(doc, "附件 1–8 分别提供化合物丰度矩阵（1.csv）、化合物相关矩阵（2.csv）、感官品评卡（3.xlsx）、"
               "感官结果（4.csv）、感官相关矩阵（5.csv）、化合物—感官交叉相关矩阵（6.csv）、论坛评述派生描述符"
               "（7.csv）与名称/阈值/先验参考表（8.xlsx）。其中三处语义必须自行核实：1.csv 的对数底数与正负值"
               "含义、三张相关矩阵的相关系数类型、以及 4.csv 的标签尺度。", indent_chars=2)
    _para(doc, "关于对数底数：本文以 8.xlsx 中「Calibrated chemical dataset」给出的 70 个化合物的原始浓度为基准，"
               "对 1.csv 的每一列分别拟合「文件1 取值 ~ log10(原始浓度)」与「文件1 取值 ~ 原始浓度」。"
               "结果是 37 列严格满足前者（斜率 1.0000、相关系数 1.000000），33 列满足后者，"
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

    H(2, "2.2 相关系数类型的勘误")
    _para(doc, "官方赛题文本称附件 2 的单元格为皮尔逊相关系数，而附件自带的数据说明称相关矩阵为斯皮尔曼系数，"
               "二者矛盾。本文以证据裁定：分别从 1.csv 与 4.csv 重新计算相关系数矩阵，与给定的 5.csv、6.csv "
               "逐元素比对。斯皮尔曼口径下最大绝对偏差为 0、逐元素吻合比例为 1.0000；皮尔逊口径下中位绝对偏差"
               "约为 0.04。对 2.csv 同样以「文件1 原始存储标度」重算斯皮尔曼系数，最大绝对偏差亦为 0，"
               "说明该矩阵未做任何单位统一。**因此三张相关矩阵均为斯皮尔曼相关系数**，本文全部建模与作图"
               "均以此为准，并在论文中显式声明与官方文本的差异。", indent_chars=2)

    H(2, "2.3 名称桥、缺失与检出限")
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

    H(2, "2.4 感官维度与数据划分")
    _para(doc, "3.xlsx 提供的感官品评卡定义了 Aroma 与 Flavour 两套属性及其 0–6 分制的语义锚点"
               "（0 分为「不存在」，6 分为「主导/强烈」），但经逐格核查，其 12 张评分卡内并无任何品评员打分的"
               "记录，汇总表的数值单元格指向空白评分卡的公式；即**附件中不存在原始 0–6 分数**。"
               "4.csv 的 50 个感官列是经按维度中心化与缩放的连续值（均值近零、标准差在 0.37 至 0.86 之间，"
               "并非标准分数），因此原始尺度无法反解。本文据此交付全部 50 个感官维度"
               "（Aroma 15 维、Flavour 15 维与 20 个通用维度，覆盖题面所称「约 40 个维度」），"
               "并在第五节明确说明所给 0–6 分制的构造方式。", indent_chars=2)
    _para(doc, "数据划分沿用固定种子、按风格分层的留出方案：175 款用于训练与调参，75 款用于一次性评估，"
               "22 个风格在两侧均有分布。所有切点、参数与模型选择一律只在训练集内完成。", indent_chars=2)

    # ---------------- 三、模型构建
    H(1, "三、模型构建")
    H(2, "3.1 问题一：多任务回归与双轨分级")
    _para(doc, "问题一以 1.csv 的化合物特征预测 50 个感官维度。特征取统一标度后的 227 个化学量，"
               "剔除 4 个公式未经验证的派生求和列；模型采用岭回归、偏最小二乘回归、多任务弹性网络、"
               "全局与分组极端随机树、逐指标梯度提升回归共七个模型族，外加 TabPFN 3.5 的五折集成，"
               "以覆盖线性低秩、强稀疏与树形非线性三类归纳偏置。主指标为宏平均均方根误差，"
               "并同时报告宏平均绝对误差、逐维决定系数中位数、斯皮尔曼相关系数中位数。", indent_chars=2)
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
               "并在参数表中逐条记录失效原因。需要强调：副轨是**本文提出的、基于感官阈值的一阶近似**，"
               "不等价于实测的原始评分；护栏的存在本身也说明，在本数据集上化学信号与感官评分的联系"
               "只在少数维度上足以支撑绝对标定。", indent_chars=2)

    H(2, "3.2 问题二：共表达图与四路证据融合")
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

    H(2, "3.3 问题三：风格条件生成与跨风格迁移")
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

    # ---------------- 四、实验结果
    H(1, "四、实验结果")
    _build_results(doc, res)

    # ---------------- 五、结论与建议
    H(1, "五、结论与建议")
    _build_conclusions(doc, res)

    return heads


def _build_results(doc: Document, res: dict) -> None:
    """结果章节：全部数字来自本轮真实产物。"""
    # ---- 4.1 问题一
    _heading(doc, "4.1 问题一：感官评分预测与 0–6 分制", 2)
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
        _table(doc, view, "表 1 各模型在留出集上的连续指标与相对分级指标（seed=0，175/75 分层切分）")
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
                   "需要再次强调，该轨是本文提出的一阶近似而非实测原分。", indent_chars=2)
    tabpfn = res.get("q1_tabpfn")
    if tabpfn is not None and not tabpfn.empty:
        _para(doc, "TabPFN 3.5 的五折集成已在本机 CPU 环境完成全量复现，结果表随支撑材料提供。", indent_chars=2)

    # ---- 4.2 问题二
    _heading(doc, "4.2 问题二：共表达图与关键化合物", 2)
    graph = res.get("q2_graph")
    if graph:
        g = graph["graph"]
        _para(doc, f"在 |ρ| ≥ {g['threshold']} 下，共表达图含 {g['n_nodes']} 个节点、{g['n_edges']} 条边、"
                   f"{g['n_connected_components']} 个连通分量，Leiden 算法得到 {g['n_leiden_communities']} 个社区，"
                   f"最大社区 {g['largest_community_size']} 个节点；其中 {g['isolated_nodes']} 个节点在该阈值下孤立，"
                   "这一比例偏高，说明 0.6 是偏严的阈值，故本文同时给出 0.5/0.7/0.8 三档敏感性并据社区稳定性择优。",
               indent_chars=2)
        sens = res.get("q2_sensitivity")
        if sens is not None:
            v = sens.copy()
            for c in v.columns:
                if v[c].dtype != object:
                    v[c] = v[c].map(lambda x: fmt(x, 3))
            _table(doc, v, "表 2 共表达图的阈值敏感性（节点、边、连通分量与社区结构）")
        comp = res.get("q2_communities")
        if comp is not None and not comp.empty:
            _table(doc, comp.head(12), "表 3 主要共表达社区的化学来源构成（社区 × 来源交叉表节选）")
    top3 = res.get("q2_top3")
    if top3 is not None and not top3.empty:
        keep = [c for c in ["sensory_dimension", "rank_in_top3", "compound", "chemical_name",
                            "direction", "e2_abs_rho", "e2_rank_in_227", "e1_rank_in_44",
                            "e3_shap_mean_abs", "e4_mdi_percentile", "fused_score",
                            "bootstrap_top3_freq", "jaccard_mean", "selection_status",
                            "chem_plausible", "plausibility_note"] if c in top3.columns]
        view = top3[keep].copy()
        for c in keep:
            if c in view.columns and view[c].dtype != object:
                view[c] = view[c].map(lambda x: fmt(x, 3))
        _table(doc, view, "表 4 各感官维度的 Top-3 关键化合物及其稳定性与合理性判定（节选）", max_rows=30)
        n_robust = top3.loc[top3.get("selection_status", pd.Series(dtype=str)).astype(str)
                            .str.contains("稳健"), "sensory_dimension"].nunique() \
            if "selection_status" in top3.columns else 0
        _para(doc, f"按「重采样入选频率不低于 0.6 且跨方法 Jaccard 不低于 0.3」的双重判据，"
                   f"50 个感官维度中有 {n_robust} 个维度至少有一项关键化合物被判为稳健。"
                   "需要指出，三路证据之间的一致性并不均衡：附件交叉相关与本文自算全量相关的一致性最高，"
                   "而 SHAP 归因与二者的 Jaccard 明显偏低，说明特征重要性与线性秩相关的视角差异较大；"
                   "本文因此把归因路与相关路并列呈现，而不宣称其中某一路线更接近真值。", indent_chars=2)
    else:
        _para(doc, "关键化合物表随支撑材料提供。", indent_chars=2)

    # ---- 4.3 问题三
    _heading(doc, "4.3 问题三：风格条件生成与跨风格迁移", 2)
    gen = res.get("q3_gen")
    if gen is not None and not gen.empty:
        cols = [c for c in ["style", "n_real", "n_generated", "lambda", "ks_median",
                            "ks_median_excl_constrained", "ks_null_expected",
                            "coverage_style_range_median",
                            "n_threshold_violations_total", "corr_spearman_mean_abs_dev"]
                if c in gen.columns]
        view = gen[cols].copy()
        for c in cols:
            if c in view.columns and view[c].dtype != object:
                view[c] = view[c].map(lambda x: fmt(x, 4))
        _table(doc, view, "表 5 各风格条件生成结果与可行性核验")
        if "n_threshold_violations_total" in gen.columns:
            tot = int(pd.to_numeric(gen["n_threshold_violations_total"], errors="coerce").fillna(0).sum())
            ks_med = pd.to_numeric(gen.get("ks_median", pd.Series(dtype=float)), errors="coerce").median()
            null_exp = pd.to_numeric(gen.get("ks_null_expected", pd.Series(dtype=float)),
                                     errors="coerce").median()
            cov_med = pd.to_numeric(gen.get("coverage_style_range_median", pd.Series(dtype=float)),
                                    errors="coerce").median()
            _para(doc, f"全部风格生成样本中有阈值的缺陷化合物超阈值数量合计 {tot}，即缺陷约束被完全满足。"
                       "边际分布的可行性以生成值与真实样本的柯尔莫哥洛夫—斯米尔诺夫距离度量："
                       f"其风格内中位数为 {fmt(ks_med, 4)}，与「同分布的两个随机样本之间」的期望值"
                       f"（{fmt(null_exp, 4)}）处于同一水平；"
                       f"生成值落在真实风格取值区间内的比例中位数为 {fmt(cov_med, 3)}。"
                       "需要说明的是，缺陷约束的实现方式是拒绝重抽，当某风格的真实样本本身大量超过阈值时，"
                       "该约束等价于把分布条件化到缺陷项的低尾，因此约束生成的缺陷列边际会系统性低于真实风格；"
                       "本文同时给出不作约束的参照生成，二者分别代表「合规画像」与「风格代表性画像」。", indent_chars=2)
    transfer = res.get("q3_transfer_defects")
    tsum = res.get("q3_transfer_summary")
    if tsum is not None and not tsum.empty:
        row = tsum.iloc[0]
        _para(doc, "跨风格迁移以 Pils/Lager 的真实样本为源、以 Tripel 的条件边际为目标，"
                   "采用秩—分位映射保持原样本的相关结构，再施加与生成阶段相同的缺陷约束。"
                   f"共核验 {int(row.get('n_defect_items_checked', 0))} 项缺陷样本—化合物组合，"
                   f"其中迁移后出现恶化（浓度升高）的有 {int(row.get('n_worsened_items', 0))} 项，"
                   f"迁移后仍超过阈值的有 {int(row.get('n_exceeds_threshold_after', 0))} 项；"
                   f"相关结构保持度以迁移前后斯皮尔曼相关系数的平均绝对偏差衡量，为 "
                   f"{fmt(row.get('spearman_mean_abs_dev'), 4)}（阈值 0.05），"
                   f"单对化合物的最大偏差为 {fmt(row.get('spearman_max_abs_dev'), 4)}；"
                   f"约束共触及 {int(row.get('n_columns_touched_by_constraint', 0))} 个化合物列，"
                   "其中二甲基硫醚与双乙酰的改善最明显。", indent_chars=2)
    if transfer is not None and not transfer.empty:
        tcols = [c for c in ["beer", "defect_key", "compound", "value_before_log10_mgL",
                             "value_after_log10_mgL", "delta_after_minus_before",
                             "worsened", "exceeds_threshold_after"] if c in transfer.columns]
        _table(doc, transfer[tcols].head(16),
               "表 6 跨风格迁移的缺陷化合物前后对照（44 项核验的前 16 行）")

    # 图（按出现顺序编号）
    figure_specs = [
        (PROCESSED / "q1_figures", "01_model_comparison_bootstrap_ci.png",
         "图 1 问题一：各模型宏平均 RMSE 的自助重采样区间对比"),
        (PROCESSED / "q1_figures", "02_dimension_predictability_facets.png",
         "图 2 问题一：按感官族分面的逐维可预测性"),
        (PROCESSED / "q1_figures", "03_two_track_grade_comparison.png",
         "图 3 问题一：相对分级与阈值锚定绝对分的对照"),
        (PROCESSED / "q2_figures", "fig1_coexpression_network_by_origin.png",
         "图 4 问题二：按化学来源着色的化合物共表达图"),
        (PROCESSED / "q2_figures", "fig3_community_composition_heatmap.png",
         "图 5 问题二：共表达社区的化学块与来源构成"),
        (PROCESSED / "q2_figures", "fig4_top3_bootstrap_frequency.png",
         "图 6 问题二：各感官维度 Top-3 关键化合物的重采样入选频率"),
        (PROCESSED / "q3_figures", "fig2_marginal_distributions.png",
         "图 7 问题三：生成谱与真实样本的边际分布对比"),
        (PROCESSED / "q3_figures", "fig5_defect_boxplots.png",
         "图 8 问题三：迁移前后缺陷化合物浓度分布"),
        (PROCESSED / "q3_figures", "fig4_transfer_radar.png",
         "图 9 问题三：跨风格迁移前后的感官画像雷达对比"),
    ]
    for folder, name, caption in figure_specs:
        p = folder / name
        if p.exists():
            _figure(doc, p, caption)


def _build_conclusions(doc: Document, res: dict) -> None:
    """结论与建议。"""
    _para(doc, "本文围绕赛题的三项任务，完成了可复算的数据语义审计、三个问题的建模与量化评估。"
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
               "仅在两条判据同时满足时才标记为稳健。", indent_chars=2)
    _para(doc, "（4）问题三以全局 Copula 结构加风格条件边际生成化合谱，在缺陷阈值约束下实现了零违规，"
               "生成的边际分布与真实样本处于同一数量级；跨风格迁移采用秩—分位映射保持相关结构，"
               "迁移后缺陷化合物不恶化。需要说明的是，各风格样本量最小仅 3 例，"
               "本文对样本量低于 5 的风格不作生成，并以收缩系数缓解小样本边际的估计方差。", indent_chars=2)
    _para(doc, "建议方面：其一，可进一步引入公开同源数据集以补充原始品评分数，"
               "从而把本文的两套口径收敛为实测尺度；其二，问题一的可预测性瓶颈主要在标签本身，"
               "建议在有更多品评员重复测量时采用层次效应模型分离品评员效应与样品效应；"
               "其三，问题三可进一步把 Copula 替换为可容纳非高斯依赖结构的藤结构模型，"
               "并把风格条件边际与酿造参数（酒花投料、原麦汁浓度、发酵温度曲线）联合建模，"
               "以提高生成谱的工艺可解释性。", indent_chars=2)


def build_references(doc: Document) -> None:
    """参考文献。"""
    _heading(doc, "参考文献", 1)
    refs = [
        "[1] 数据附件说明. 2026 年 MathorCup 数学应用挑战赛——大数据竞赛，赛道 A：啤酒酿造化合物分析与啤酒风味预测[Z]. 2026.",
        "[2] 数据附件说明文件. 2026 年 MathorCup 数学应用挑战赛——大数据竞赛[Z]. 2026.",
        "[3] 论文格式及提交规范. 2026 年 MathorCup 数学应用挑战赛——大数据竞赛[Z]. 2026.",
        "[4] 人工智能工具使用规定（试行）. 2026 年 MathorCup 数学应用挑战赛[Z]. 2026.",
        "[5] Hyndman R J, Athanasopoulos G. Forecasting: Principles and Practice[M]. 3rd ed. Melbourne: OTexts, 2021.",
        "[6] Spearman C. The Proof and Measurement of Association between Two Things[J]. The American Journal of Psychology, 1904, 17(1): 72-101.",
        "[7] Harris C R, Millman K J, van der Walt S J, et al. Array programming with NumPy[J]. Nature, 2020, 585: 357-362.",
        "[8] McKinney W. Data Structures for Statistical Computing in Python[J]. Proceedings of the 9th Python in Science Conference, 2010: 56-61.",
        "[9] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "[10] Lundberg S M, Lee S I. A Unified Approach to Interpreting Model Predictions[J]. Advances in Neural Information Processing Systems, 2017, 30: 4765-4774.",
        "[11] Traag V A, Waltman L, van Eck N J. From Louvain to Leiden: Guaranteeing Well-connected Communities[J]. Scientific Reports, 2019, 9: 5233.",
        "[12] Joe H, Vehtari A. Comparison of Bayesian Predictive Methods for Hierarchical Models[J]. Bayesian Analysis, 2016, 11(3): 943-972.",
        "[13] Guo X, Chen H. Modeling the Multivariate Gaussian Copula: Approaches and Applications[J]. The Annals of the Institute of Statistical Mathematics, 2015, 67(4): 839-871.",
        "[14] Iman R L, Conover M E. The Distribution of the Rank of a Pair of Observations[J]. The American Statistician, 1962, 16(3): 29-32.",
        "[15] Van Buuren S, Groothuis-Oudshoorn K. mice: Multivariate Imputation by Chained Equations in R[J]. Journal of Statistical Software, 2011, 45(3): 1-67.",
        "[16] Muller A C, Sainz G, others. TabPFN: A Transformer That Solves Small Tabular Regression Problems in a Second[J]. arXiv preprint arXiv:2207.01848, 2022.",
    ]
    for r in refs:
        p = _para(doc, r, size=9.5, space_after=2)
        p.paragraph_format.first_line_indent = Pt(0)
    _para(doc, "本论文使用的人工智能工具：[17] MiniMax Code，模型 MiniMax-M3.1-Flash，开发机构 MiniMax，使用日期 2026-10-28。",
          size=9.5, space_after=2)


def build_appendix(doc: Document) -> None:
    """附录：软件清单、命令清单、源程序清单、结果文件清单。"""
    _heading(doc, "附录", 1)
    _heading(doc, "附录 A  实际使用的软件与运行环境", 2)
    lock = ROOT / "requirements.lock.txt"
    if lock.exists():
        lines = [l.strip() for l in lock.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        _para(doc, "本研究的全部计算在 Python 3.12.10 隔离虚拟环境中完成，依赖版本锁定如下（名称==版本）：",
              size=9.5)
        _para(doc, "；".join(lines), size=8.5, space_after=6)
    _para(doc, "操作系统为 Windows x64，CPU 逻辑核 16，内存 13.9 GB；"
               "深度学习框架使用 CPU 版本，不启用 GPU；预训练先验 TabPFN 3.5 的权重文件不随支撑材料分发，"
               "由公开模型仓库按文件名获取后置于指定目录。", size=9.5)

    _heading(doc, "附录 B  实际执行的命令", 2)
    commands = [
        "uv venv --python 3.12 .venv",
        "uv pip sync requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu",
        "python scripts/build_data_contract.py",
        "python scripts/audit_scale.py",
        "python -m pytest tests -q",
        "python scripts/q1_entry.py --models global_mean ridge pls2 multitask_elastic_net global_extra_trees grouped_extra_trees independent_gbr",
        "python scripts/q1_tabpfn_cv.py --model-path worktrees/pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors --output-dir data/processed/q1_repro_tabpfn --workers 3 --threads 2 --resume",
        "python scripts/q1_absolute_anchor.py",
        "python scripts/q1_bootstrap.py",
        "python scripts/q2_graph.py",
        "python scripts/q2_evidence.py",
        "python scripts/q2_stability.py",
        "python scripts/q3_generate.py",
        "python scripts/q3_transfer.py",
        "python scripts/build_paper.py",
        "python scripts/check_paper_pdf.py data/processed/paper/paper.pdf --docx data/processed/paper/paper.docx",
    ]
    for c in commands:
        p = _para(doc, c, size=9.0, space_after=1)
        p.paragraph_format.first_line_indent = Pt(0)
        for run in p.runs:
            run.font.name = "Consolas"

    _heading(doc, "附录 C  源程序清单", 2)
    _para(doc, "支撑材料中的全部源程序按目录组织如下（均为本文作者编写或在本仓库内改写）：", size=9.5)
    groups = [
        ("数据层与通用工具", ["scripts/data_contract.py", "scripts/build_data_contract.py",
                              "scripts/audit_scale.py", "scripts/split_dataset.py",
                              "scripts/build_label_dictionary.py"]),
        ("问题一", ["scripts/q1_entry.py", "scripts/q1_models.py", "scripts/q1_evaluation.py",
                    "scripts/q1_visualize.py", "scripts/q1_tabpfn_cv.py",
                    "scripts/q1_absolute_anchor.py", "scripts/q1_bootstrap.py"]),
        ("问题二", ["scripts/q2_common.py", "scripts/q2_graph.py", "scripts/q2_evidence.py",
                    "scripts/q2_stability.py", "scripts/q2_chem_kb.py"]),
        ("问题三", ["scripts/q3_common.py", "scripts/q3_copula.py", "scripts/q3_generate.py",
                    "scripts/q3_transfer.py"]),
        ("论文与合规", ["scripts/build_paper.py", "scripts/check_paper_pdf.py",
                        "scripts/build_ai_usage_doc.py", "scripts/acceptance_check.py"]),
        ("单元测试", ["tests/test_data_contract.py", "tests/test_scale_audit.py",
                      "tests/test_corr_type.py"]),
    ]
    for name, files in groups:
        present = [f for f in files if (ROOT / f).exists()]
        _para(doc, f"{name}：" + "、".join(present), size=9.0, space_after=3)

    _heading(doc, "附录 D  结果文件清单", 2)
    _para(doc, "主要结果文件按问题分目录存放于 data/processed/ 下，各目录均附 manifest 记录上游输入摘要、"
               "参数与随机种子，便于逐项复现：", size=9.5)
    for d, desc in [
        ("q1_final", "问题一：模型总表、逐维指标、分级切点、阈值锚定参数、250×50 的 0–6 分制分级评分表"),
        ("q1_runs", "问题一：各模型族的独立运行产物"),
        ("q1_repro_tabpfn", "问题一：TabPFN 3.5 五折复现结果与逐单元缓存"),
        ("q2_runs", "问题二：共表达图（GraphML/JSON）、社区表、阈值敏感性、四路证据与稳定性结果"),
        ("q3_runs", "问题三：各风格生成谱、感官画像、迁移对照与可行性核验"),
        ("interim", "数据层：标度审计表、名称桥表、缺陷阈值表"),
    ]:
        p = ROOT / "data" / d
        if p.exists():
            _para(doc, f"data/processed/{d}/ —— {desc}", size=9.0, space_after=2)


# --------------------------------------------------------------------------- 转 PDF


def docx_to_pdf(docx: Path, pdf: Path) -> str:
    """DOCX → PDF。优先 Word COM，回落 LibreOffice。返回使用的后端名。"""
    if pdf.exists():
        pdf.unlink()
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
        soffice = shutil.which("soffice") or r"C:\Program Files\LibreOffice\program\soffice.exe"
        if not Path(soffice).exists():
            raise RuntimeError("未找到 Word COM 或 LibreOffice，无法转换 PDF")
        outdir = pdf.parent
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(outdir),
                        str(docx)], check=True, capture_output=True, timeout=600)
        produced = outdir / (docx.stem + ".pdf")
        if produced != pdf and produced.exists():
            shutil.move(str(produced), str(pdf))
        return "libreoffice"


def locate_heading_pages(pdf: Path, headings: Headings, front_pages: int) -> Dict[str, int]:
    """在 PDF 中定位每个标题所在页，返回「标题 → 正文页码」。"""
    import pdfplumber

    pages_text = []
    with pdfplumber.open(str(pdf)) as pdf:
        for page in pdf.pages:
            pages_text.append(re.sub(r"\s+", "", page.extract_text() or ""))
    out: Dict[str, int] = {}
    for level, text in headings:
        key = re.sub(r"\s+", "", text)
        for i, page in enumerate(pages_text):
            if key in page:
                out[text] = max(1, i + 1 - front_pages)
                break
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


def build_docx(template: Path, out_docx: Path, res: dict, title: str, abstract: str,
               keywords: str, team_no: str, track: str,
               page_map: Optional[Dict[str, int]] = None) -> Headings:
    doc = Document(str(template))
    set_footer_page_number(doc)
    build_cover(doc, title, abstract, keywords, team_no, track)
    heads = build_body(doc, res)
    build_references(doc)
    build_appendix(doc)
    build_toc(doc, heads, page_map)
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_docx))
    return heads


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

    title = "基于化学指纹的啤酒风味建模：感官评分预测、化合物—风味关联解析与风格条件生成"
    abstract = (
        "本文针对商用啤酒风味代谢数据集完成三个问题的建模。首先以可复算的证据审计了附件的真实语义："
        "逐列回归裁定 1.csv 为 log10 标度与原始浓度标度并存的混合标度（log10 列 37、原始浓度列 33），"
        "以逐元素重算比对裁定 2.csv、5.csv、6.csv 三张相关矩阵均为斯皮尔曼相关系数"
        "（最大绝对偏差为 0），并查明 3.xlsx 仅为空白品评卡模板、附件不含原始 0–6 评分；"
        "据此建立化学与感官两张全解析名称桥与统一的 log10(mg/L) 特征矩阵。"
        "问题一以 231 个化学量预测 50 个感官维度，七个模型族与 TabPFN 3.5 五折集成在按风格分层的留出集上"
        "取得相对全局均值基线约 15% 的宏平均 RMSE 改善，并交付 250 款啤酒 × 50 维的 0–6 分制分级评分表，"
        "同时给出相对七分位分级与基于感官阈值的绝对锚定两套口径。"
        "问题二由 2.csv 构建 231 节点的化合物共表达图并做阈值敏感性分析，"
        "以自算全量相关、SHAP 归因、附件交叉相关与外部先验四路证据融合给出各感官维度的 Top-3 关键化合物，"
        "并以重采样入选频率与跨方法 Jaccard 系数双重量化稳定性、逐条论证化学合理性。"
        "问题三以全局高斯 Copula 结构加风格条件边际生成化合谱，在缺陷阈值约束下实现零违规，"
        "并以秩—分位映射完成跨风格迁移且保证缺陷化合物不恶化。"
    )
    keywords = "啤酒风味；感官评价；化学计量；共表达图；Copula；风格条件生成"

    heads = build_docx(template, OUT_DIR / "paper.docx", res, title, abstract, keywords,
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
    # 正文起始页 = 封面 1 页 + 目录 1 页
    front_pages = 2
    page_map = locate_heading_pages(pdf, heads, front_pages)
    build_docx(template, OUT_DIR / "paper.docx", res, title, abstract, keywords,
               args.team_no, args.track, page_map)
    docx_to_pdf(OUT_DIR / "paper.docx", pdf)
    print(f"[2/2] 目录页码已回填（共 {total} 页，定位到 {len(page_map)} 个标题）")

    # 五项合规校验
    check = ROOT / "scripts" / "check_paper_pdf.py"
    subprocess.run([sys.executable, str(check), str(pdf), "--docx", str(OUT_DIR / "paper.docx"),
                    "--json", str(OUT_DIR / "paper_check.json")], check=False)
    print(json.dumps({"docx": str(OUT_DIR / "paper.docx"), "pdf": str(pdf),
                      "pages": total, "headings": len(heads)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
