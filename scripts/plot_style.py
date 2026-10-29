# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：配色取自 Okabe-Ito 色盲友好色板并逐个核对了明度与灰度可分辨性；字号与
# 图宽按正文 5.0 英寸排版宽度反推；PNG(300dpi) 与 PDF(矢量) 双输出已在三处绘图脚本实跑核验。
"""全文统一绘图风格契约（问题一/二/三/消费者层共用）。

为什么需要这个模块
------------------
此前 `q1_figures.py` / `q2_figures.py` / `q3_figures.py` 由三个不同执行者各自实现，
rcParams 各不相同（dpi 220 / 170 / 130，底色变量名各异），图内标题冗长且带方法说明，
导致全文图表风格不统一。本模块把「字体、字号、配色、线条、底色、输出规格、中文标签」
收敛为唯一权威来源，三个绘图脚本与消费者层绘图一律经由本模块取用。

**契约（下游脚本不得改这些名字与语义）**::

    COLORS          dict  分类色板与语义色
    INK/INK2/MUTED/GRID/BASELINE/ACCENT/DEFECT   颜色常量
    FS_TITLE/FS_LABEL/FS_TICK/FS_NOTE              字号常量
    apply_style()                    -> None    写入 matplotlib rcParams
    tidy(ax, grid_axis="y")          -> None    去边框、统一刻度与网格
    simple_title(ax, text)          -> None    简约标题（不含方法说明）
    save(fig, path, dpi=300)         -> None    同名 PNG(300dpi) + PDF(矢量) 双输出
    dim_cn(name)                     -> str     感官维度/消费者描述符的中文名
    STYLE_CONTRACT_VERSION           str      契约版本号（供测试守护）

图内标题原则：**图内只放极简标题（≤12 字）**；方法、单位、样本量、阈值等细节一律
移入论文题注，由 `scripts/build_paper.py` 统一编号（图 1…图 N、表 1…表 N）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE_CONTRACT_VERSION = "plot_style_v1"

# --------------------------------------------------------------------------- 颜色

#: Okabe-Ito 色盲友好分类色板（8 色）
OKABE_ITO: List[str] = [
    "#0072B2",   # 蓝
    "#E69F00",   # 橙
    "#009E73",   # 蓝绿
    "#CC79A7",   # 紫红
    "#D55E00",   # 朱红
    "#56B4E9",   # 天蓝
    "#F0E442",   # 黄
    "#000000",   # 黑
]

#: 语义色
INK = "#1A1A1A"        # 主文字
INK2 = "#4D4D4D"       # 次级文字（轴标签）
MUTED = "#8C8C8C"     # 弱化文字
GRID = "#E3E3E0"       # 网格线
BASELINE = "#B9B9B3"   # 参考线 / 基线
ACCENT = "#0072B2"     # 强调色（蓝）
DEFECT = "#D55E00"     # 缺陷化合物强调色（朱红）
POSITIVE = "#009E73"   # 正向关联
NEGATIVE = "#D55E00"   # 负向关联

#: 发散色标（相关性）与顺序色标（频率/强度）的两端
DIVERGING = ("#2166AC", "#F7F7F7", "#B2182B")
SEQUENTIAL = ("#F7F7F7", "#4393C3", "#2166AC")

COLORS: Dict[str, object] = {
    "categorical": OKABE_ITO,
    "ink": INK, "ink2": INK2, "muted": MUTED, "grid": GRID, "baseline": BASELINE,
    "accent": ACCENT, "defect": DEFECT, "positive": POSITIVE, "negative": NEGATIVE,
    "diverging": DIVERGING, "sequential": SEQUENTIAL,
}

# --------------------------------------------------------------------------- 字号

FS_TITLE = 10.5   # 图内标题
FS_LABEL = 9.0    # 轴标签
FS_TICK = 8.0     # 刻度
FS_NOTE = 9.0     # 论文题注（不在图内使用，留作常量对齐）
FS_LEGEND = 8.5   # 图例

#: 图在正文中的排版宽度（英寸）——figsize 宽度按此反推，避免超宽或过窄
FIG_WIDTH_IN = 5.0
LINE_WIDTH = 0.8
MARKER_SIZE = 3.2


# --------------------------------------------------------------------------- 风格


def apply_style() -> None:
    """把统一的 rcParams 写入 matplotlib 全局状态（每次绘图前调用一次即可）。"""
    plt.rcParams.update({
        # 字体：中文用 Microsoft YaHei，西文用 Times New Roman
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "font.size": FS_TICK,
        "axes.unicode_minus": False,
        # 字号
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        # 底色：纯白，学术印刷友好
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        # 线条
        "axes.linewidth": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "grid.linewidth": 0.6,
        "grid.color": GRID,
        # 边框：只留左/下
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # 刻度
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # 嵌入 TrueType，保证 PDF 中文字形正确
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })


def tidy(ax, grid_axis: str = "y") -> None:
    """统一坐标轴外观：去上右边框、只保留主网格、刻度朝外。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.set_axisbelow(True)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    if grid_axis == "y":
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.yaxis.grid(False)
    ax.tick_params(which="minor", length=0)


def simple_title(ax, text: str, size: float = FS_TITLE, pad: float = 6.0) -> None:
    """极简图内标题：不换行、不带方法说明；方法细节一律交给论文题注。"""
    ax.set_title(text, fontsize=size, color=INK, pad=pad, loc="left")


def save(fig, path: Path, dpi: int = 300) -> List[Path]:
    """把图同时存为 PNG（位图，供 Word 嵌入）与 PDF（矢量，供支撑材料）。

    返回写出的文件路径列表（PNG 在前）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    return [png, pdf]


def figsize(width_in: float = FIG_WIDTH_IN, height_in: float = 3.2,
            ncols: int = 1, nrows: int = 1) -> "tuple":
    """按排版宽度反推 figsize（单图默认 5.0 x 3.2 英寸）。"""
    return (width_in, height_in)


# --------------------------------------------------------------------------- 中文标签

#: 50 个品评组感官维度的中文名
DIM_CN: Dict[str, str] = {
    "A_malt_all": "香气·麦芽总体", "A_malt_grain": "香气·麦芽谷物", "A_malt_bread": "香气·麦芽面包",
    "A_malt_cara": "香气·麦芽焦糖", "A_malt_burn": "香气·麦芽烟熏",
    "A_hops_all": "香气·酒花总体", "A_hops_citrus": "香气·柑橘", "A_hops_tropical": "香气·热带果",
    "A_hops_noble": "香气·醇正酒花",
    "A_hops_woody": "香气·木质",
    "A_esters_all": "香气·酯类总体", "A_esters_ethac": "香气·乙酸乙酯", "A_esters_isoaa": "香气·香蕉",
    "A_esters_flower": "香气·花香", "A_esters_fruity": "香气·果香",
    "F_malt_all": "味道·麦芽总体", "F_malt_grain": "味道·麦芽谷物", "F_malt_bread": "味道·麦芽面包",
    "F_malt_cara": "味道·麦芽焦糖", "F_malt_burn": "味道·麦芽烟熏",
    "F_hops_all": "味道·酒花总体", "F_hops_citrus": "味道·柑橘", "F_hops_tropical": "味道·热带果",
    "F_hops_noble": "味道·醇正酒花", "F_hops_woody": "味道·木质",
    "F_esters_all": "味道·酯类总体", "F_esters_ethac": "味道·乙酸乙酯", "F_esters_isoaa": "味道·香蕉",
    "F_esters_flower": "味道·花香", "F_esters_fruity": "味道·果香",
    "body": "酒体", "bitternes": "苦味", "co2": "碳酸", "aftertaste": "回味",
    "alcohol": "酒精感", "sweetness": "甜度", "acidity": "酸度",
    "dms": "二甲基硫醚", "metallic": "金属感", "X4vg": "4-乙烯基愈创木酚",
    "diacetyl": "双乙酰", "stale_hops": "陈化酒花味", "t2n": "trans-2-壬烯醛",
    "orange": "橙香", "coriander": "香菜", "clove": "丁香", "lactic": "乳酸",
    "acetic": "乙酸", "barnyard": "马厩味", "overall": "总体评价",
}

#: 消费者描述符词根 → 中文（用于 7.csv 的 aroma_/taste_/palate_ 前缀列）
_DESCRIPTOR_CN: Dict[str, str] = {
    "acetaldehyde": "乙醛", "acetate": "乙酸酯", "acetic": "乙酸", "acidic": "酸味",
    "aftertaste": "回味", "alcohol": "酒精感", "anise": "茴香", "apple": "苹果",
    "banana": "香蕉", "barnyard": "马厩味", "bitter": "苦味", "bread": "面包",
    "burnt": "焦糊", "butterscotch": "奶油糖", "buttery": "黄油", "cabbage": "卷心菜",
    "candy": "糖果", "caramel": "焦糖", "carbonation": "碳酸感", "cardboard": "纸板味",
    "cereal": "谷物", "cheesy": "奶酪", "citrus": "柑橘", "clove": "丁香",
    "coriander": "香菜", "corn": "玉米", "dentist": "薄荷感", "diacetyl": "双乙酰",
    "estery": "酯香", "floral": "花香", "fruity": "果香", "grassy": "青草",
    "herbal": "草本", "hoppy": "酒花香", "honey": "蜂蜜", "malty": "麦芽香",
    "metallic": "金属感", "orange": "橙香", "papery": "纸感", "phenolic": "酚感",
    "pine": "松香", "pineapple": "菠萝", "resinous": "树脂", "roasted": "烘烤",
    "smoky": "烟熏", "solvent": "溶剂味", "sour": "酸感", "spicy": "辛香",
    "stale": "陈化味", "sweet": "甜", "toast": "烤面包", "tropical": "热带果",
    "vanilla": "香草", "vegetable": "蔬菜", "woody": "木质",
}

#: 消费者描述符前缀 → 中文（消费者感知与品评组感官的分面）
ASPECT_CN: Dict[str, str] = {"aroma": "香气", "taste": "味道", "palate": "口感"}


def dim_cn(name: str) -> str:
    """感官维度 / 消费者描述符 → 中文名；未知则原样返回。"""
    key = str(name)
    if key in DIM_CN:
        return DIM_CN[key]
    # 消费者描述符：aroma_citrus → 香气·柑橘
    if "_" in key:
        prefix, rest = key.split("_", 1)
        if prefix in ASPECT_CN:
            cn = _DESCRIPTOR_CN.get(rest)
            if cn:
                return f"{ASPECT_CN[prefix]}·{cn}"
    return key


def class_cn(code: str) -> str:
    """化学类别代号 → 中文名（复用 q2_chem_kb 的词典，缺失则原样返回）。"""
    try:
        from q2_chem_kb import CLASS_CN  # type: ignore
        return CLASS_CN.get(code, code)
    except Exception:
        return code


__all__ = [
    "STYLE_CONTRACT_VERSION", "COLORS", "OKABE_ITO", "INK", "INK2", "MUTED", "GRID",
    "BASELINE", "ACCENT", "DEFECT", "POSITIVE", "NEGATIVE", "DIVERGING", "SEQUENTIAL",
    "FS_TITLE", "FS_LABEL", "FS_TICK", "FS_NOTE", "FS_LEGEND", "FIG_WIDTH_IN",
    "LINE_WIDTH", "MARKER_SIZE", "DIM_CN", "ASPECT_CN",
    "apply_style", "tidy", "simple_title", "save", "figsize", "dim_cn", "class_cn",
]
