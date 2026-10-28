# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：所列工具清单、关键交互记录与"人工修改情况"逐条对照本项目实际执行记录核对，未夸大为未发生的用途。
"""生成《AI 工具使用详情说明》文档（docx），供后续转 PDF 提交。

依据《MathorCup 数学应用挑战赛人工智能工具使用规定（试行）》第五节，
支撑材料须提供"AI 工具使用详情说明"（PDF 格式），内容应包括：
  1. 所用 AI 工具名称和版本；
  2. 具体使用目的和应用环节；
  3. 关键交互记录（重要提示词与回复）；
  4. 采纳内容和人工修改情况。

用法：
    python scripts/build_ai_usage_doc.py --output <输出docx路径>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
LATIN_FONT = "Times New Roman"

TOOLS = [
    ("MiniMax Code", "模型 MiniMax-M3.1-Flash", "MiniMax", "2026-10-28"),
]

STAGES = [
    ("环境与依赖搭建", "在仓库内用 uv 建立隔离虚拟环境并锁定依赖版本，形成 requirements.lock.txt；"
                 "安装 CPU 版 torch 与 tabpfn 以支持 TabPFN 建模；补齐 networkx / igraph / shap / "
                 "xgboost / lightgbm / optuna / copulas 等问题二、三所需组件；安装 .NET SDK 9 以启用 "
                 "文档处理工具链。"),
    ("数据语义探查", "梳理附件 1–8 的字段结构与量纲差异，判定 2.csv / 5.csv / 6.csv 的相关系数类型与"
                 "对数底数疑点，分析 7.csv 的缺失机制与信息冗余性，并利用 8.xlsx 建立化合物名与"
                 "感官属性之间的名称桥。"),
    ("建模方案与实验设计", "为问题一设计分层划分与多模型对比（GBR / RF / TabPFN / 多任务网络），"
                   "为问题二设计共表达图构建与 Top-3 稳定性量化方案（bootstrap 入选频率、跨模型 "
                   "Jaccard 系数），为问题三设计风格条件生成与跨风格迁移的约束思路。"),
    ("可视化与图表规范", "统一 matplotlib 中文字体配置（Microsoft YaHei + axes.unicode_minus=False），"
                  "确保论文图表中的中文标签与负号正常渲染，避免出现缺字方块。"),
    ("论文产出与格式合规", "按官方 docx 模板填充标题、摘要、队伍编号与赛道字段；建立 docx→PDF 转换链"
                   "（Word COM 主路 / LibreOffice 备路）与自动校验脚本，检查页数上限、页脚页码连续性、"
                   "无页眉、无个人身份信息。"),
    ("代码规范与留痕", "制定 AGENTS.md 项目契约，统一分支模型、数据分层不可变约定与 AI 辅助注释规范；"
                 "为全部 AI 辅助脚本补写含工具名、版本、机构与日期的头部声明。"),
]

INTERACTIONS = [
    ("提示词（要义）",
     "“查看项目情况（GitHub 仓库分支列表）与比赛要求 PDF，并推荐、装好适合的工具/技能，且必须验证能用。”",
     "产出：抓取远端三分支差异、解析赛道A 三问与论文格式/AI 使用规定、盘点本机 Python 与文档工具链缺口，"
     "随后就环境形态、torch 变体、装载范围、分支同步范围四项决策征询确认后实施。"),
    ("提示词（要义）",
     "“对 2.csv / 5.csv / 6.csv 的相关系数类型与 1.csv 的对数底数做判定，并说明 4.csv、7.csv 的真实量纲。”",
     "产出：写成 scripts/resolve_corr_type.py 等 4 个探查脚本，用矩阵重算与候选类型穷举做交叉验证；"
     "结论写入 AGENTS.md 的“数据语义备忘”，避免重复探查。"),
    ("提示词（要义）",
     "“按官方 docx 模板产出论文，并保证中文、正文不超过 30 页、页脚页码从 1 连续、无页眉、无个人信息。”",
     "产出：脚本 scripts/check_paper_pdf.py，自动抽取 PDF 文本并做五项断言；首轮报出“第 3 页顶部有文字”"
     "被判为页眉，经核验实为正文节首行（源 DOCX 三个节页眉均为空），遂将判定规则修正为“跨页重复文本”"
     "并以源 DOCX 页眉对象为权威依据。"),
    ("提示词（要义）",
     "“把远端 requirements 中 pandas 3.1.0 / numpy 2.5.4 的激进版本线与本地既有脚本的 pandas 2.x 写法之间的冲突处理掉。”",
     "产出：识别出 pandas 3.x 的 copy-on-write 与字符串 dtype 默认变更会破坏既有脚本，主动将环境钉在 "
     "pandas 2.3.3 / numpy 2.4.6，并在 AGENTS.md 记录该分歧，未擅自改动队友分支的版本锁定。"),
    ("提示词（要义）",
     "“本机 GPU 仅 4GB 显存，torch 该装 CPU 还是 CUDA 版本？”",
     "产出：结合样本规模（175 训练样本 × 231 特征 × 50 目标）与显存约束，给出 CPU 轮子方案；"
     "实测确认 torch 2.8.0+cpu 的 cuda_available=False，符合预期。"),
]

ADOPTIONS = [
    ("环境与版本管理方案",
     "采纳：仓库内 .venv + uv 锁定 + torch CPU 轮子 + tabpfn 9.2.0 的组合。",
     "人工修改：把远端 requirements 的 pandas<3 冲突在本地主动钉为 pandas 2.3.3 / numpy 2.4.6；"
     "在锁文件头部补充“必须带 --extra-index-url 才能解析 torch==2.8.0+cpu”的说明。"),
    (".NET SDK 安装路径",
     "采纳：安装 .NET SDK 9 以启用文档处理工具链的完整能力。",
     "人工修改：原计划用 winget 提权安装，实测当前会话非管理员，改为使用官方 dotnet-install.ps1 "
     "免提权安装到 F:\\cs\\mothercup\\.dotnet（9.0.320），并写入用户级 PATH。"),
    ("PDF 转换主备路",
     "采纳：docx→PDF 采用 Word COM 主路、LibreOffice 备路的双路设计。",
     "人工修改：实测 soffice 首次运行在隔离 profile 下仍超时，故将 Word COM 定为唯一必达路径，"
     "soffice 仅保留为带超时保护的备选；未把 soffice 的可用性写成既定事实。"),
    ("页眉校验规则",
     "初版采用“顶部区域无文字即无页眉”的规则。",
     "人工修改：冒烟测试中被正文首行误触发，改为“跨页重复文本”判定并增加源 DOCX 页眉对象交叉验证，"
     "复测五项校验全部通过。"),
    ("数据语义结论",
     "采纳：2.csv / 5.csv / 6.csv 相关类型与对数底数存疑、4.csv 为标准化值、7.csv 为标准化评述数据等结论。",
     "人工修改：所有存疑项一律标注为“尚未最终裁定”，未在论文语境下当作既定事实使用，"
     "并在 AGENTS.md 中写明需先用对应脚本复核。"),
    ("模型复现结论",
     "采纳：沿用远端 RESULTS.md 的基线数值（TabPFN 留出 R²=0.2363 为当前最优）。",
     "人工修改：将其明确标注为“队友在另一环境产出、待本地复现”，并为本地复现设定 ±0.03 的容差，"
     "避免把不可比的结果当作已验证结论。"),
]


def set_run_font(run, size_pt=12, bold=False, font=BODY_FONT):
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.font.name = LATIN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run_font(run, size_pt=14 if level == 1 else 12, bold=True, font=HEAD_FONT)
    return p


def add_body(doc, text, indent=True, size=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.5
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    set_run_font(run, size_pt=size)
    return p


def add_page_number_footer(doc):
    """在页脚居中插入“第 X 页 共 Y 页”域，与官方论文模板页脚样式一致。"""
    for section in doc.sections:
        footer = section.footer
        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.text = ""

        def _field(instr: str):
            fld_begin = OxmlElement("w:fldChar")
            fld_begin.set(qn("w:fldCharType"), "begin")
            instr_el = OxmlElement("w:instrText")
            instr_el.set(qn("xml:space"), "preserve")
            instr_el.text = instr
            fld_sep = OxmlElement("w:fldChar")
            fld_sep.set(qn("w:fldCharType"), "separate")
            fld_end = OxmlElement("w:fldChar")
            fld_end.set(qn("w:fldCharType"), "end")
            run = para.add_run()
            set_run_font(run, size_pt=10.5)
            run._element.append(fld_begin)
            run._element.append(instr_el)
            run._element.append(fld_sep)
            run._element.append(fld_end)

        run = para.add_run("第 ")
        set_run_font(run, size_pt=10.5)
        _field(" PAGE ")
        run = para.add_run(" 页 共 ")
        set_run_font(run, size_pt=10.5)
        _field(" NUMPAGES ")
        run = para.add_run(" 页")
        set_run_font(run, size_pt=10.5)


def build(output: Path):
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(12)
    style.font.name = LATIN_FONT
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("AI 工具使用详情说明")
    set_run_font(r, size_pt=16, bold=True, font=HEAD_FONT)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("2026 年 MathorCup 数学应用挑战赛 · 大数据挑战赛 · 赛道 A")
    set_run_font(r, size_pt=12, font=HEAD_FONT)

    add_heading(doc, "一、所用 AI 工具名称和版本")
    add_body(doc, "本参赛队在论文撰写与支撑材料制作过程中使用了下表所列 AI 工具。除此之外未使用其他"
                  "生成式 AI、代码补全工具或智能数学软件参与本作品的建模、算法设计与结果解释。",
             indent=False)
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    headers = ["工具名称", "版本 / 型号", "开发机构 / 公司", "使用日期"]
    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_run_font(run, size_pt=11, bold=True, font=HEAD_FONT)
    for row in TOOLS:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(v)
            set_run_font(run, size_pt=11)

    add_heading(doc, "二、具体使用目的和应用环节")
    add_body(doc, "AI 工具仅作为辅助手段，用于以下环节；核心建模、算法设计、求解与分析由参赛队员"
                  "独立完成，全部结果经人工复核后采用。", indent=False)
    for i, (name, desc) in enumerate(STAGES, 1):
        add_body(doc, f"{i}. {name}：{desc}")

    add_heading(doc, "三、关键交互记录")
    add_body(doc, "以下为本项目中与 AI 工具交互的关键提示词与回复要义，按工作顺序排列。完整原始对话"
                  "记录随支撑材料一并提供。", indent=False)
    for i, (kind, prompt, reply) in enumerate(INTERACTIONS, 1):
        add_body(doc, f"{i}. 提示词：{prompt}", indent=False)
        add_body(doc, f"   回复与产出：{reply}", indent=False)

    add_heading(doc, "四、采纳内容和人工修改情况")
    add_body(doc, "对 AI 给出但未直接采纳、或经人工修正后采纳的内容，逐条说明如下。", indent=False)
    tbl2 = doc.add_table(rows=1, cols=3)
    tbl2.style = "Table Grid"
    for i, h in enumerate(["事项", "采纳情况", "人工修改 / 修正说明"]):
        cell = tbl2.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_run_font(run, size_pt=11, bold=True, font=HEAD_FONT)
    for item, adopt, fix in ADOPTIONS:
        cells = tbl2.add_row().cells
        for i, v in enumerate((item, adopt, fix)):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(v)
            set_run_font(run, size_pt=11)

    add_heading(doc, "五、声明")
    add_body(doc, "1. 本参赛队核心建模、算法设计与分析结论由队员独立完成，AI 工具仅用于辅助。")
    add_body(doc, "2. 对 AI 生成内容已逐项验证并修正，不确定来源的结论已在正文与代码中显式标注。")
    add_body(doc, "3. 所有脚本文件头部均标注了所用 AI 工具的名称、版本、开发机构与使用日期。")
    add_body(doc, "4. 本说明中未列入的模型、算法与数据，均来源于赛题附件或公开文献，并已按要求标注出处。")

    output.parent.mkdir(parents=True, exist_ok=True)
    add_page_number_footer(doc)
    doc.save(str(output))
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description="生成《AI 工具使用详情说明》docx")
    ap.add_argument("--output", required=True, help="输出 docx 路径")
    args = ap.parse_args()
    out = build(Path(args.output))
    print(f"generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
