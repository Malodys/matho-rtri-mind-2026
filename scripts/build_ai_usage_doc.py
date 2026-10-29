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
    ("数据语义裁定与共享数据层", "以可复算的证据最终裁定三项语义：一是用附件 8 的原始浓度对每一列回归，"
                          "判定 1.csv 为 log10 标度与原始浓度标度并存的混合标度；二是以「重算相关矩阵并与"
                          "给定矩阵逐元素比对」判定三张相关矩阵均为斯皮尔曼相关系数（与赛题 PDF 表述不一致）；"
                          "三是查明 3.xlsx 仅为空白品评卡模板、附件不含原始 0–6 分数。据此建立"
                          "scripts/data_contract.py 作为三问唯一取数入口，并配套 20 项回归测试固化上述裁定。"),
    ("建模方案与实验设计", "为问题一设计分层划分与多模型对比（GBR / RF / TabPFN / 多任务网络），"
                   "为问题二设计共表达图构建与 Top-3 稳定性量化方案（bootstrap 入选频率、跨模型 "
                   "Jaccard 系数），为问题三设计风格条件生成与跨风格迁移的约束思路。"),
    ("问题一实现与复现", "在本机重跑七个模型族并与队友既有结果逐模型对照（最大宏平均均方根误差偏差 "
                   "3.9e-4，多数为机器精度级一致）；实现以训练集七分位切点为主的相对分级与以感官阈值为锚的"
                   "绝对分级双轨口径；补做 TabPFN 3.5 五折全量复现。"),
    ("问题二实现", "构建 231 节点共表达图并做四档阈值敏感性；实现四路证据（自算全量相关、SHAP 归因、"
                "附件交叉相关、外部先验）的融合排序；以重采样入选频率与跨方法 Jaccard 系数双重量化稳定性；"
                "为每个入选化合物给出化学来源归属、载体类别、异味描述与阈值一致性四层合理性论证。"),
    ("问题三实现", "实现全局高斯 Copula 结构、风格条件边际（含小样本收缩）、缺陷阈值拒绝重抽与单调压缩、"
                "秩—分位映射式跨风格迁移；并补充无约束参照生成以区分「生成器还原度」与「约束对边际的压平作用」。"),
    ("可视化与图表规范", "统一 matplotlib 中文字体配置（Microsoft YaHei + axes.unicode_minus=False），"
                  "确保论文图表中的中文标签与负号正常渲染，避免出现缺字方块。"),
    ("论文产出与格式合规", "按官方 docx 模板填充标题、摘要、队伍编号与赛道字段；建立 docx→PDF 转换链"
                   "（Word COM 主路 / LibreOffice 备路）与自动校验脚本，检查页数上限、页脚页码连续性、"
                   "无页眉、无个人身份信息。"),
    ("提交版 PDF 保真转换", "实测发现 LibreOffice 转换会在中文换行处吞掉行尾标点（471 个「，」丢 4 个、"
                     "121 个「；」丢 2 个），使提交版 PDF 的文本层与源 DOCX 不一致；"
                     "在无 pywin32 且不允许安装包的约束下，改用 PowerShell 自带 COM interop 驱动本机 "
                     "Word 16.0 完成转换，标点零损失、页数一致，"
                     "并把该路径固化为独立脚本与构建脚本的首选后端。"),
    ("代码规范与留痕", "制定 AGENTS.md 项目契约，统一分支模型、数据分层不可变约定与 AI 辅助注释规范；"
                 "为全部 AI 辅助脚本补写含工具名、版本、机构与日期的头部声明。"),
    ("文献锚定与溯源", "检索并核实本题数据的原始出处文献，确认附件为 Nature Communications 2024 年"
                  "《Predicting and improving complex beer flavor through machine learning》的公开数据与补充材料；"
                  "逐条核实该文可引用的具体数字（品评组规模、消费者评论条数、模型清单、关键化合物结论的原始措辞），"
                  "并扩充方法学与领域综述文献，形成可核实的著录清单。"),
    ("消费者数据层建模", "清洗 RateBeer 消费者描述符（剔除 3 个整列为空的列，得 183 维），"
                    "建立「化学→消费者描述符」模型并与「化学→品评组感官」模型做交叉一致性检验；"
                    "以多零售商定价的几何均价为外部价值信号，控制酒精度与风格后回归。"),
    ("缺陷化合物剂量再检验", "对原文「低浓度异味物可能提升香气」的结论做独立再分析：以浓度四分位分组"
                     "计算组内秩相关与感官中位，并做风格分层稳健性检查以排除辛普森式混合效应；"
                     "同时明确声明原文结论中「提升消费者喜爱度」一环在本附件条件下不可检验。"),
    ("图表风格统一", "抽取统一的绘图风格契约模块（字体、字号、色盲友好配色、线条、底色、"
                 "PNG 与矢量 PDF 双输出、中文标签映射），改造三处既有绘图脚本，"
                 "将冗长的图内标题压缩为极简标题并把方法细节移入论文题注。"),
    ("图表题注与论文插图修复", "定位并修复图题注链路上的三处缺陷：题注清单的键名与插图查找用的文件名不一致导致"
                      "方法说明整体失效、题注自带「图 N 标题。」前缀与论文统一编号重复、"
                      "消费者层图未纳入插图路径而在两节中重复出现；"
                      "另生成消费者层题注清单脚本，题注数值现场从产物 CSV 读取，避免题注与数据脱节。"),
    ("结论稳健性限定内联", "复核剂量再检验结论的表述，发现结论段把风格分层稳健比例为 0 的配对（trans-2-壬烯醛）"
                    "与稳健配对并列陈述；改为在主张出现处内联风格分层稳健性限定，"
                    "并与正文表格采用同一口径，消除结论与证据之间的措辞落差。"),
    ("论文结构与表达", "在保留三问骨架的前提下补充相关工作与文献综述、符号表、消费者层与原文对照子节、"
                 "讨论与局限章节，使分析过程与结论边界对评审可核可查。"),
    ("独立同行评审", "以独立 Agent 按学术论文评审框架对自身论文做结构化审查（方法论六维评分、优点与缺点清单、"
                 "文献定位与可执行改进建议），评审方独立重跑数值验证并给出 22 条编号问题；"
                 "作者逐条记录采纳与否并落实到正文。其中评审方复算发现摘要的标度列数相加不等于 70、"
                 "特征数应为 227 而非 231、斯皮尔曼系数对单调变换不变故不能反推单位统一、"
                 "价格数据跨美元/英镑/欧元而附件未提供汇率等实质问题，均已据实修正。"),
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
    ("提示词（要义）",
     "“以 1.csv 为基础给出 3.xlsx 定义的 Aroma 与 Flavour 两套约 40 个感官维度的 0–6 分制分级评分结果；"
     "请给出 0–6 分制的构造口径、交付范围与多 Agent 分工方式的建议。”",
     "产出：给出双轨口径（相对七分位分级为主、感官阈值锚定为辅）、交付范围与「按问题并行 + 独立复核」"
     "的分工拓扑三项建议，均经人工确认后执行。"),
    ("提示词（要义）",
     "“对 1.csv 的对数底数与正负值含义、对三张相关矩阵的相关系数类型、3.xlsx 是否含真实评分数据做证据级裁定。”",
     "产出：以附件 8 的原始浓度回归逐列裁定标度（log10 列 37、原始浓度列 33、自然对数斜率恒为 2.302585 "
     "故排除）；以逐元素重算比对裁定三张矩阵均为斯皮尔曼相关系数（最大绝对偏差为 0）；"
     "查明 3.xlsx 的 12 张评分卡内无任何品评员打分记录、汇总表数值单元格为指向空表的公式，"
     "据此推翻「用原始品评卡做绝对标定」的初始设想，改为阈值锚定。"),
    ("提示词（要义）",
     "“问题一/二/三分别并行推进，另设一个只读 Agent 复核；子 Agent 必须能用 Bash 做数值验证。”",
     "产出：三个执行 Agent 分工产出、复核 Agent 独立审查；先前一批只读 Agent 因无执行权限只能翻历史日志、"
     "产出不可用，遂在派发简报中写死解释器路径与「必须执行数值验证」的硬性要求。"),
    ("提示词（要义）",
     "“核查问题三生成谱是否真的满足缺陷阈值约束，并判断缺陷列边际是否被系统性压低。”",
     "产出：独立复算发现 Tripel 的二甲基硫醚真实中位 −1.4127、约束生成后为 −1.9196，"
     "双乙酰真实中位 −0.5382、生成后为 −2.7881；判定这是拒绝重抽把分布条件化到缺陷项低尾的统计后果"
     "而非代码缺陷，遂要求补一份无约束参照生成以区分「生成器还原度」与「约束对边际的压平作用」。"),
    ("提示词（要义）",
     "“TabPFN 全量五折复现中途报错，且首轮生成结果页数与中文字符校验未过，如何修正？”",
     "产出：定位 TabPFN 报错为并行进程数过多导致的系统资源不足，改为分块续跑并压低并行度；"
     "论文侧修正页数校验口径为「正文不超过 30 页、附录页数不限」，并移除在转换后被渲染为常数的总页数字段。"),
    ("提示词（要义）",
     "“本题附件的原始出处是哪篇论文？请核实其作者、期刊、卷期页码、DOI，并列出论文中需要引用的具体数字。”",
     "产出：定位到 Nature Communications 2024 年第 15 卷文章号 2368（DOI 10.1038/s41467-024-46346-0，"
     "CC BY 4.0）；逐条核对该文可引用的数字与措辞，并确认其 Fig.1 即赛题附件 6.csv、"
     "图题注明使用斯皮尔曼秩相关，从而独立佐证本项目对相关类型的裁定；"
     "同时确认附件 3.xlsx 是该文使用的空白品评卡模板。"),
    ("提示词（要义）",
     "“消费者数据怎么用才算合理？”",
     "产出：查清 7.csv 有 3 个 palate 描述符整列为空（有效 183 维），且第一主成分仅解释 10.4% 方差，"
     "据此判定不存在可直接使用的消费者喜爱度目标，改为建立并行的消费者感知路径并做交叉一致性检验，"
     "辅以多零售商定价的外部价值信号。"),
    ("提示词（要义）",
     "“原文说低浓度的异味物反而提升香气，我们能不能独立检验？”",
     "产出：以浓度四分位分组复算浓度与感官维度的关系，得到双乙酰与 trans-2-壬烯醛呈反向、"
     "二甲基硫醚几乎不相关、乙酸乙酯在最高浓度分位内相关最强的结论；"
     "同时指明原文结论中「提升消费者喜爱度」一环在本附件条件下不可检验。"),
    ("提示词（要义）",
     "“三处绘图脚本风格各不相同，怎么统一？标题也太多冗长。”",
     "产出：抽取统一绘图风格契约模块，改造三处脚本；把图内标题压缩到 12 字以内，"
     "方法、单位、样本量与阈值全部移入论文题注；每张图同时输出位图与矢量 PDF。"),
]

ADOPTIONS = [
    ("环境与版本管理方案",
     "采纳：仓库内 .venv + uv 锁定 + torch CPU 轮子 + tabpfn 9.2.0 的组合。",
     "人工修改：把远端 requirements 的 pandas<3 冲突在本地主动钉为 pandas 2.3.3 / numpy 2.4.6；"
     "在锁文件头部补充“必须带 --extra-index-url 才能解析 torch==2.8.0+cpu”的说明。"),
    (".NET SDK 安装路径",
     "采纳：安装 .NET SDK 9 以启用文档处理工具链的完整能力。",
     "人工修改：原计划用 winget 提权安装，实测当前会话非管理员，改为使用官方 dotnet-install.ps1 "
     "免提权安装到仓库同级的用户级 .dotnet 目录（9.0.320），并写入用户级 PATH。"),
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
     "初始采纳：沿用远端 RESULTS.md 的基线数值（TabPFN 留出 R²=0.2363）；该结果来自另一环境，"
     "不作为本地已验证的最优结果。",
     "人工修改：将其明确标注为“队友在另一环境产出、待本地复现”，并为本地复现设定 ±0.03 的容差，"
     "避免把不可比的结果当作已验证结论。"),
    ("相关系数类型的结论方向",
     "初稿采纳赛题 PDF 的表述，称附件 2 为皮尔逊相关系数。",
     "人工修改：以逐元素重算比对推翻该表述——三张矩阵在斯皮尔曼口径下最大绝对偏差为 0、"
     "逐元素吻合比例 1.0000，而在皮尔逊口径下中位绝对偏差约 0.04；遂统一改为斯皮尔曼，"
     "并在论文中显式声明该勘误及其证据表，而非默认官方文本正确。"),
    ("对数底数的处理",
     "初稿按赛题提示把 1.csv 整体视为 log10 标度。",
     "人工修改：回归发现其中 33 列存的是原始浓度本身，若统一取 10 的幂将使双乙酰等列的量级严重失真；"
     "遂改为逐列审计并统一到 log10(mg/L) 矩阵，同时在测试中固化该裁定。"),
    ("以原始品评卡作绝对标点的设想",
     "初稿设想用 3.xlsx 的 12 张品评卡把标准化值映射回真实 0–6 分。",
     "人工修改：逐格核查发现 12 张卡内无任何评分记录、汇总表为公式且缓存值为 0，"
     "该设想不成立；改为以感官阈值做锚定的一阶近似，并在论文中声明其不等价于实测原分。"),
    ("名称桥的归一化规则",
     "初稿按「剥离全部非字母数字」归一化化合物名。",
     "人工修改：发现希腊字母被剥离后 E-α-紫罗兰酮与 E-β-紫罗兰酮塌缩为同一键，造成静默错配，"
     "遂改为先音译希腊字母再归一化，并对歧义键一律丢弃；该缺陷已写入回归测试。"),
    ("缺陷约束对生成边际的影响",
     "初稿以「零违规」作为问题三生成质量的唯一指标。",
     "人工修改：复算发现拒绝重抽把缺陷列分布条件化到低尾，双乙酰中位由 −0.5382 压至 −2.7881；"
     "遂要求补出无约束参照生成，并在论文中同时呈现「风格代表性」与「合规性」两种用途的生成谱。"),
    ("论文页数口径",
     "初稿以 PDF 总页数对照 30 页上限。",
     "人工修改：按官方规范「正文控制在 30 页以内，附录页数不限」改为以带页脚页码的页识别正文页，"
     "并把附录页从正文限额中剔除；同时移除在文档转换后恒被渲染为常数的总页数字段。"),
    ("数据出处的披露方式",
     "初稿仅把原始论文当作参考文献之一列出。",
     "人工修改：确认附件即该顶刊论文的公开数据与补充材料后，改为在正文明确声明「本文基于该文的公开数据"
     "做独立复现与再分析」，并把三处勘误与两项再分析列为本文贡献；不披露反而不符合学术诚实要求。"),
    ("消费者数据的构造方式",
     "初稿设想从消费者描述符构造一个「总体喜爱度」指标作为建模目标。",
     "人工修改：实测第一主成分仅解释 10.4% 方差、组均值与品评组维度相关很弱，判定该指标不成立；"
     "改为以 183 维消费者描述符空间作为并行感知路径做交叉一致性检验，并以定价作外部价值信号，"
     "同时在论文中声明未声称预测消费者喜爱度。"),
    ("缺陷化合物剂量的结论方向",
     "初稿预设「低浓度异味物提升香气」应在数据中重现。",
     "人工修改：实测双乙酰与 trans-2-壬烯醛呈反向、二甲基硫醚近乎不相关，"
     "遂按实测方向改写结论，并补做风格分层稳健性检查以排除辛普森式混合效应，"
     "同时声明原文结论的「消费者喜爱度」一环不可检验。"),
    ("图表风格的处理",
     "初稿沿用三个绘图脚本各自的样式设置。",
     "人工修改：抽出统一的风格契约模块，三处脚本一律取用；图内标题压缩为极简标题，"
     "方法与口径说明移入论文题注；每张图同时输出位图与矢量 PDF。"),
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
