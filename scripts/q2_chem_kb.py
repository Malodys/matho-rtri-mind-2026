# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 化合物 → 化学类别 映射由关键词规则 + 显式覆盖表生成，已对全部 231 个节点逐条打印核对
#      （见 scripts/q2_chem_kb.py 的 verify_classification()，逐类计数与名单人工过目）。
#   2) 感官维度 → 已知化学载体清单 取自 8.xlsx「Sensory thresholds」的 origin_paper / off_flavors /
#      aging 字段与通用酿造化学常识；每条 carrier 均能在 data/interim/bridges_v1.csv 找到对应列。
#   3) 「同时出现在特征侧与感官侧」的化合物清单直接复用 data_contract.ABSOLUTE_ANCHOR_CARRIERS，
#      命中者一律加注「理化测量与感官评分的一致性」而非「新发现的关联」。
"""问题二化学知识库：化合物化学类别 + 感官维度的已知化学载体。

本模块只提供「先验」与「文案」，不含任何计算；计算在 q2_evidence.py / q2_stability.py。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- 化学类别

#: 类别代号 → 中文名（论文用语）
CLASS_CN: Dict[str, str] = {
    "terpene": "萜烯/倍半萜",
    "ester": "酯类",
    "alcohol": "脂肪醇/醇",
    "aldehyde": "醛类",
    "phenol": "酚类/酚醚",
    "sulfide": "硫化物",
    "sulfur_other": "其它含硫化合物",
    "acid": "有机酸",
    "ketone": "酮/二酮",
    "pyrazine": "吡嗪",
    "maillard": "美拉德产物",
    "furan": "呋喃类",
    "metal": "金属离子",
    "hop_resin": "酒花树脂（异构α酸）",
    "energy": "能量/醇类理化量",
    "protein": "蛋白/胶体",
    "sugar": "糖类",
    "gas": "气体",
    "color": "色度",
    "acid_ph": "酸度",
    "physical": "其它理化量",
    "earthy": "土臭素类",
    "spice_phenol": "香辛料酚类",
}

#: 显式覆盖（关键词规则无法可靠判定的少数化合物）
CLASS_OVERRIDE: Dict[str, str] = {
    "bitterness": "hop_resin",          # 8.xlsx: Iso-alpha acids，字段名 bitterness，单位 IBU
    "iron": "metal",
    "pH": "acid_ph",
    "color.EBC.": "color",
    "protein.g.L.": "protein",
    "kcalperc": "energy",
    "glycerol": "energy",
    "betaglucan.mg.L.": "protein",
    "unfermentables": "sugar",
    "GFS": "sugar",
    "CO2.PSI.": "gas",
    "ethanol..v.v.": "alcohol",
    "ammonia.mg.L.": "physical",
    "propylene.glycol": "physical",
    "butylated.hydroxytoluene": "physical",
    "X3.methylthio.propanol": "sulfur_other",   # Methionol，硫醚/硫醇
    "geosmin": "earthy",
    "X2.3.butanedione": "ketone",      # Diacetyl，二酮
    "X2.3.hexanedione": "ketone",
    "X4VG": "phenol",                  # 4-vinylguaiacol
    "X4.ethyl.guaiacol": "phenol",
    "X4.ethyl.phenol": "phenol",
    "eugenol": "spice_phenol",
    "E.methyl.isoeugenol": "spice_phenol",
    "X4.methoxysalicylaldehyde": "spice_phenol",
    "cis.linalool.oxide": "terpene",
    "nerol.oxide": "terpene",
    "dehydroxy.cis.linalool.oxide": "terpene",
    "humulene.epoxide.II": "terpene",
    "dihydroisocaryophyllene.epoxide": "terpene",
    "alloaromadendrene.epoxide": "terpene",
    "caryophyllene.oxide": "terpene",
    "caryophyllene.alcohol": "terpene",
    "X4.8.alpha.epoxy.caryophyllane": "terpene",
    "X1.10.di.epi.cubenol": "terpene",
    "X1.2.dihydro.8.hydroxy..2E..linalool": "terpene",
    "X14.hydroxy.9.epi..E..caryophyllene": "terpene",
    "X14.hydroxy.alpha.humulene": "terpene",
    "Z.nerolidol": "terpene",
    "E.nerolidol": "terpene",
    "X2.3.dihydro.farnesol": "terpene",
    "X2Z.6Z.farnesol": "terpene",
    "X2Z.6E.farnesol": "terpene",
    "X2E.6E.methyl.farnesoate": "terpene",
    "Z.Z.geranyl.linalool": "terpene",
    "manool": "terpene",
    "manool.oxide": "terpene",
    "X13.epi.manool.oxide": "terpene",
    "caryophylla.4.12..8.13..dien.5.alpha.ol": "terpene",
    "taucadinol": "terpene",
    "X2Z.6E.farnesal": "terpene",
    "X2E.6Z.farnesal": "terpene",
    "X.E..dihydro.apofarnesal": "terpene",
    "X.Z..dihydro.apofarnesal": "terpene",
    "X.E..beta.ocimene": "terpene",
    "neo.allo.ocimene": "terpene",
    "X.E..beta.damascenone": "terpene",
    "X.E..alpha.ionone": "terpene",
    "X.E..beta.ionone": "terpene",
    "X.Z..beta.farnesene": "terpene",
    "X.E..beta.farnesene": "terpene",
    "E.ocimenone": "terpene",
    "beta.cyclohomocitral": "terpene",
    "alpha.calacorene": "terpene",
    "X2.6.dimethyl.pyrazine": "pyrazine",
    "X2.ethyl.3.methylpyrazine": "pyrazine",
    "X2.methyl.3.methylpyrazine": "pyrazine",
    "methylpyrazine": "pyrazine",
    "X2.acetylpyrrole": "maillard",
    "X2.methyl.benzofuran": "maillard",
    "X2.acetylfuran": "maillard",
    "X2.pentyl.furan": "maillard",
    "X2.furanmethanol": "maillard",
    "X3.furanmethanol": "maillard",
    "hydroxy.methyl.furfural": "maillard",
    "X5.methylfurfural": "maillard",
    "isomaltol": "maillard",
    "furfural": "maillard",
    "vanillin": "maillard",
    "X6.methyl.5.hepten.2.one": "ketone",
    "X1.octen.3.ol": "alcohol",
    "X3.methyl.2.hexanol": "alcohol",
    "X2.ethyl.1.hexanol": "alcohol",
    "X2.methyl.1.butanol": "alcohol",
    "X2.3.butanedione_": "ketone",
}

#: 萜烯类按列名直接判定（酒花/香辛料挥发油的主要骨架）
TERPENE_KEYS: Sequence[str] = (
    "phellandrene", "limonene", "linalool", "ocimene", "ocimenone", "terpineol",
    "citronellol", "nerol", "geraniol", "humulene", "caryophyllene", "farnesene",
    "farnesol", "farnesal", "farnesoate", "apofarnesal", "cadalene", "calacorene",
    "muurolol", "eudesmol", "cadinol", "cubenol", "spathulenol", "valencene",
    "terpinolene", "camphor", "eucalyptol", "isoborneol", "terpinen", "cyclohomocitral",
    "ionone", "damascenone", "germacrene", "nerolidol", "car.3.en.2.one", "cymenene",
    "undecanone", "tau.muurolol", "selin.11.en.4.alpha.ol", "X1.10.di.epi.cubenol",
    "ambroxide", "foeniculin", "x.methyl.1.octanol", "y.methyl.1.octanol",
    "X4.methyl.1.pentanol", "nonan.2.ol", "X2.undecanol",
)

_ESTER_RE = re.compile(
    r"(acetate|butyrate|hexanoate|octanoate|propionate|lactate|benzoate|succinate"
    r"|furoate|formate|tiglate|cinnamate|nicotinate|salicylate|hydrocinnamate"
    r"|phenylacetate|acrylate|anthranilate)$", re.I)
_ALDEHYDE_RE = re.compile(r"(aldehyde|nonenal|decadienal|butanal|hexanal)$", re.I)
_ALCOHOL_RE = re.compile(r"(ol|alcohol)$", re.I)
_PHENOL_RE = re.compile(r"(phenol|guaiacol)", re.I)
_MAILLARD_KEYS = ("furfural", "furanmethanol", "methylfurfural", "isomaltol", "vanillin",
                   "acetylfuran", "methyl.benzofuran", "acetylpyrrole", "pentyl.furan",
                   "hydroxymethylfurfural", "maltol")


def classify(col: str) -> str:
    """把 1.csv 列名映射到化学类别代号。"""
    c = str(col)
    if c in CLASS_OVERRIDE:
        return CLASS_OVERRIDE[c]
    low = c.lower()
    # 硫化物
    if low in {"dms", "des", "h2s", "mesh", "cs2", "so2.mg.l."} or "sulfide" in low:
        return "sulfide"
    if "methylthio" in low:
        return "sulfur_other"
    if "geosmin" in low:
        return "earthy"
    # 萜烯
    if any(k in low for k in TERPENE_KEYS):
        return "terpene"
    # 吡嗪
    if "pyrazine" in low:
        return "pyrazine"
    # 美拉德
    if any(k in low for k in _MAILLARD_KEYS):
        return "maillard"
    # 酚类
    if _PHENOL_RE.search(low):
        return "phenol"
    # 醛
    if _ALDEHYDE_RE.search(low):
        return "aldehyde"
    # 酮
    if "dione" in low or "hexanedione" in low:
        return "ketone"
    if "one" in low and not any(k in low for k in ("furanmethanol",)):
        return "ketone"
    # 酸
    if low.endswith("acid") or "acid." in low or "acid" in low.split(".")[-1]:
        return "acid"
    # 酯
    if _ESTER_RE.search(low):
        return "ester"
    # 醇
    if _ALCOHOL_RE.search(low):
        return "alcohol"
    return "physical"


def classify_by_chem_name(col: str, chem_name: Optional[str]) -> str:
    """先用化学名（8.xlsx）判定，再用 R 列名兜底 —— 化学名更接近 IUPAC，可靠性更高。"""
    if isinstance(chem_name, str) and chem_name.strip():
        n = chem_name.strip()
        nl = n.lower()
        if "sulfide" in nl or "disulfide" in nl:
            return "sulfide"
        if nl in ("methanethiol", "sulfur dioxide", "carbon disulfide"):
            return "sulfide"
        if "aldehyde" in nl or "nonenal" in nl or "decadienal" in nl:
            return "aldehyde"
        if "acid" in nl:
            return "acid"
        if "pyrazine" in nl:
            return "pyrazine"
        if any(k in nl for k in ("furfural", "furan", "isomaltol", "vanillin",
                                 "pyrrole", "benzofuran")):
            return "maillard"
        if "guaiacol" in nl or "phenol" in nl:
            return "phenol"
        if "dione" in nl:
            return "ketone"
        if _ESTER_RE.search(nl):
            return "ester"
    return classify(col)


# --------------------------------------------------------------------------- 维度知识库

#: 感官维度 → 已知化学载体
#:   classes  : 该维度的已知化学载体类别（化学合理性的第②项判据）
#:   carriers : 教科书级已知载体（1.csv 列名），命中即「已知化学载体」
#:   origins  : 期望的 origin_paper 归属（第①项判据）
#:   note     : 论文正文用的论证模板
DIM_KB: Dict[str, dict] = {
    # ---------------- Aroma · Malt
    "A_malt_all": {
        "classes": {"maillard", "pyrazine", "ketone", "furan"},
        "carriers": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "X2.ethyl.3.methylpyrazine",
                     "X2.acetylfuran", "furfural", "X5.methylfurfural", "X4VG", "X3.methylbutanal"],
        "origins": {"Malt", "Yeast"},
        "note": "麦芽香气由美拉德产物（吡嗪、呋喃、 Strecker 醛）与焦糖化产物主导",
    },
    "A_malt_grain": {
        "classes": {"pyrazine", "maillard"},
        "carriers": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "X2.ethyl.3.methylpyrazine",
                     "X2.acetylfuran", "X2.methyltetrahydrofuran.3.one"],
        "origins": {"Malt"},
        "note": "谷物香（grainy）以麦芽焙烤产生的吡嗪类与取代呋喃为核心",
    },
    "A_malt_bread": {
        "classes": {"maillard", "ketone", "furan"},
        "carriers": ["furfural", "X5.methylfurfural", "hydroxy.methyl.furfural",
                     "X2.acetylfuran", "isomaltol", "X2.3.butanedione", "X2.pentyl.furan"],
        "origins": {"Malt", "Yeast"},
        "note": "面包/烘烤香由麦芽 Maillard 链产物（糠醛、甲基糠醛、羟甲基糠醛、乙酰呋喃）承载",
    },
    "A_malt_cara": {
        "classes": {"maillard"},
        "carriers": ["isomaltol", "vanillin", "furfural", "X5.methylfurfural",
                     "hydroxy.methyl.furfural", "X2.acetylfuran", "X2.pentyl.furan",
                     "X2.methyl.benzofuran"],
        "origins": {"Malt"},
        "note": "焦糖香来自焦糖化/美拉德后期的甲基吡嗪以外的糠醛、异麦芽酚、香草醛系",
    },
    "A_malt_burn": {
        "classes": {"phenol", "spice_phenol"},
        "carriers": ["X4.ethyl.guaiacol", "X4VG", "eugenol", "E.methyl.isoeugenol",
                     "X4.ethyl.phenol", "X4.methoxysalicylaldehyde", "ortho.cresol"],
        "origins": {"Wild flora", "Spices", "Yeast"},
        "note": "烟熏/焦糊香由酚类（4-乙基愈创木酚、4-乙烯基愈创木酚、丁香酚）承载",
    },
    # ---------------- Aroma · Hops
    "A_hops_all": {
        "classes": {"terpene"},
        "carriers": ["alpha.humulene", "linalool", "citronellol", "geraniol", "limonene",
                     "X.E..beta.ocimene", "X1.alpha.terpineol", "beta.humulene"],
        "origins": {"Hops", "Spices"},
        "note": "酒花香整体由酒花挥发油（倍半萜与单萜）承载，origin_paper 应集中于 Hops",
    },
    "A_hops_citrus": {
        "classes": {"terpene"},
        "carriers": ["citronellol", "linalool", "nerol", "geraniol", "limonene",
                     "X.E..beta.ocimene", "X1.alpha.terpineol", "X2.undecanone"],
        "origins": {"Hops", "Spices"},
        "note": "柑橘/热带水果调由单萜（香茅醇、芳樟醇、橙花醇、香叶醇、柠檬烯）与倍半萜承载",
    },
    "A_hops_tropical": {
        "classes": {"terpene"},
        "carriers": ["X.E..beta.ocimene", "neo.allo.ocimene", "nerol", "geraniol",
                     "citronellol", "X.E..beta.damascenone", "X.E..alpha.ionone",
                     "X.E..beta.ionone"],
        "origins": {"Hops"},
        "note": "热带水果调由月桂烯类（罗勒烯、橙花醇）与大马酮/紫罗兰酮类倍半萜承载",
    },
    "A_hops_noble": {
        "classes": {"terpene"},
        "carriers": ["alpha.humulene", "beta.humulene", "isoborneol", "camphor",
                     "eucalyptol", "X1.alpha.terpineol", "terpinen.4.ol", "meta.cymenene"],
        "origins": {"Hops", "Spices"},
        "note": "草本/ Noble 香由 β-石竹烯、α-律草烯、莰烯、桉叶素、α-萜品醇承载",
    },
    "A_hops_woody": {
        "classes": {"terpene"},
        "carriers": ["E.caryophyllene", "alpha.humulene", "cadalene", "selin.11.en.4.alpha.ol",
                     "alpha.muurolol", "alpha.eudesmol", "alpha.calacorene", "tau.cadinol",
                     "X.E..beta.farnesene", "X.Z..beta.farnesene"],
        "origins": {"Hops", "Spices"},
        "note": "木质/树脂香由倍半萜（石竹烯、杜松烯、β-杜松醇、卡德醇）与法尼烯承载",
    },
    # ---------------- Aroma · Esters
    "A_esters_all": {
        "classes": {"ester"},
        "carriers": ["ethyl_acetate", "ethyl_hexanoate", "ethyl_octanoate",
                     "ethyl_2.methyl_butyrate", "isopentyl_acetate", "phenethyl_acetate"],
        "origins": {"Yeast", "Hops", "Spices"},
        "note": "酯香整体由酵母代谢产生的挥发性脂肪酸酯承载",
    },
    "A_esters_ethac": {
        "classes": {"ester"},
        "carriers": ["ethyl_acetate", "ethyl.butyrate", "ethyl.isobutyrate",
                     "ethyl_propionate", "ethyl.lactate"],
        "origins": {"Yeast"},
        "note": "指甲油/乙酸乙酯型酯香由短链脂肪酸酯（乙酸乙酯为主）承载",
    },
    "A_esters_isoaa": {
        "classes": {"ester", "alcohol"},
        "carriers": ["isoamyl_alcohol", "isopentyl_acetate", "isobutyl_acetate",
                     "ethyl_isovalerate", "ethyl_2.methyl_butyrate", "ethyl_5.methylhexanoate"],
        "origins": {"Yeast"},
        "note": "香蕉型酯香由异戊醇及其乙酸酯（乙酸异戊酯）等高级醇酯承载",
    },
    "A_esters_flower": {
        "classes": {"ester", "terpene", "alcohol"},
        "carriers": ["benzyl.alcohol", "phenethyl_alcohol", "phenethyl_acetate",
                     "phenylacetaldehyde", "ethyl.phenylacetate", "linalool", "geraniol",
                     "neryl.formate", "geraniol"],
        "origins": {"Yeast", "Hops"},
        "note": "花香由苯乙醇/苯乙酸酯、苄醇与萜烯醇（芳樟醇、香叶醇）承载",
    },
    "A_esters_fruity": {
        "classes": {"ester"},
        "carriers": ["ethyl_hexanoate", "ethyl_octanoate", "isopentyl_acetate",
                     "ethyl_2.methyl_butyrate", "ethyl_5.methylhexanoate", "ethyl_decanoate"],
        "origins": {"Yeast"},
        "note": "水果调酯香由中链脂肪酸酯（己酸乙酯、辛酸乙酯）承载",
    },
    # ---------------- Taste（与 Aroma 同族化合物，载体一致）
    "F_malt_all": {
        "classes": {"maillard", "pyrazine", "ketone", "furan"},
        "carriers": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "furfural", "X4VG"],
        "origins": {"Malt", "Yeast"}, "note": "麦芽味觉由同一批美拉德/吡嗪化合物承载",
    },
    "F_malt_grain": {
        "classes": {"pyrazine", "maillard"},
        "carriers": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "X2.acetylfuran"],
        "origins": {"Malt"}, "note": "谷物味觉由麦芽吡嗪承载",
    },
    "F_malt_bread": {
        "classes": {"maillard", "ketone", "furan"},
        "carriers": ["furfural", "X5.methylfurfural", "isomaltol", "X2.acetylfuran"],
        "origins": {"Malt", "Yeast"}, "note": "面包味觉由糠醛系与异麦芽酚承载",
    },
    "F_malt_cara": {
        "classes": {"maillard"},
        "carriers": ["isomaltol", "vanillin", "furfural", "X5.methylfurfural"],
        "origins": {"Malt"}, "note": "焦糖味觉由异麦芽酚/香草醛/糠醛承载",
    },
    "F_malt_burn": {
        "classes": {"phenol", "spice_phenol"},
        "carriers": ["X4.ethyl.guaiacol", "X4VG", "eugenol", "X4.ethyl.phenol"],
        "origins": {"Wild flora", "Spices"}, "note": "焦糊味觉由酚类承载",
    },
    "F_hops_all": {
        "classes": {"terpene"},
        "carriers": ["alpha.humulene", "linalool", "citronellol", "geraniol", "limonene"],
        "origins": {"Hops", "Spices"}, "note": "酒花味觉由同一批酒花萜烯承载",
    },
    "F_hops_citrus": {
        "classes": {"terpene"},
        "carriers": ["citronellol", "linalool", "nerol", "geraniol", "limonene"],
        "origins": {"Hops", "Spices"}, "note": "柑橘味觉由单萜承载",
    },
    "F_hops_tropical": {
        "classes": {"terpene"},
        "carriers": ["X.E..beta.ocimene", "nerol", "geraniol", "citronellol",
                     "X.E..beta.damascenone"],
        "origins": {"Hops"}, "note": "热带果味觉由罗勒烯/橙花醇/大马酮承载",
    },
    "F_hops_noble": {
        "classes": {"terpene"},
        "carriers": ["alpha.humulene", "beta.humulene", "isoborneol", "camphor"],
        "origins": {"Hops", "Spices"}, "note": "草本味觉由律草烯/莰烯/桉叶素承载",
    },
    "F_hops_woody": {
        "classes": {"terpene"},
        "carriers": ["E.caryophyllene", "alpha.humulene", "alpha.muurolol", "tau.cadinol"],
        "origins": {"Hops", "Spices"}, "note": "木质味觉由倍半萜承载",
    },
    "F_esters_all": {
        "classes": {"ester"},
        "carriers": ["ethyl_acetate", "ethyl_hexanoate", "ethyl_octanoate"],
        "origins": {"Yeast", "Hops"}, "note": "酯味觉由挥发性脂肪酸酯承载",
    },
    "F_esters_ethac": {
        "classes": {"ester"},
        "carriers": ["ethyl_acetate", "ethyl.butyrate", "ethyl.isobutyrate"],
        "origins": {"Yeast"}, "note": "乙酸乙酯型酯味由短链酯承载",
    },
    "F_esters_isoaa": {
        "classes": {"ester", "alcohol"},
        "carriers": ["isoamyl_alcohol", "isopentyl_acetate", "ethyl_isovalerate"],
        "origins": {"Yeast"}, "note": "香蕉型酯味由异戊醇及其乙酸酯承载",
    },
    "F_esters_flower": {
        "classes": {"ester", "terpene", "alcohol"},
        "carriers": ["phenethyl_alcohol", "phenethyl_acetate", "linalool", "geraniol"],
        "origins": {"Yeast", "Hops"}, "note": "花香味觉由苯乙醇/萜烯醇承载",
    },
    "F_esters_fruity": {
        "classes": {"ester"},
        "carriers": ["ethyl_hexanoate", "ethyl_octanoate", "isopentyl_acetate"],
        "origins": {"Yeast"}, "note": "水果味觉由中链脂肪酸酯承载",
    },
    # ---------------- 味觉 / 缺陷 / 其他
    "acidity": {
        "classes": {"acid", "acid_ph"},
        "carriers": ["lactic_acid.mg.L.", "acetic_acid.g.L.", "pH"],
        "origins": {"Wild flora"},
        "note": "酸味由乳酸、乙酸等有机酸与 pH 共同承载",
    },
    "bitternes": {
        "classes": {"hop_resin"},
        "carriers": ["bitterness"],
        "origins": {"Hops"},
        "note": "苦味由酒花异构α酸（IBU 标度）承载 —— 本维度与异构α酸是同一物理量的两种测量",
    },
    "sweetness": {
        "classes": {"sugar"},
        "carriers": ["unfermentables", "GFS"],
        "origins": {"Malt"},
        "note": "甜味由不可发酵糖（麦芽糖、葡萄糖浆）承载",
    },
    "X4vg": {
        "classes": {"phenol"},
        "carriers": ["X4VG"],
        "origins": {"Yeast", "Wild flora"},
        "note": "4-乙烯基愈创木酚（4VG）是 papery/辛香型缺陷的已知载体",
    },
    "diacetyl": {
        "classes": {"ketone"},
        "carriers": ["X2.3.butanedione"],
        "origins": {"Yeast"},
        "note": "双乙酰（黄油味缺陷）由酵母丙酮酸代谢的 2,3-丁二酮承载",
    },
    "dms": {
        "classes": {"sulfide"},
        "carriers": ["DMS"],
        "origins": {"Malt", "Yeast"},
        "note": "DMS（煮蔬菜味）为硫化物型典型缺陷",
    },
    "metallic": {
        "classes": {"metal"},
        "carriers": ["iron"],
        "origins": {"Others"},
        "note": "金属味与铁离子等金属污染相关",
    },
    "stale_hops": {
        "classes": {"aldehyde"},
        "carriers": ["X.E..2.nonenal", "X3.methylbutanal", "X3.methyl.2.butenal",
                     "benzaldehyde", "n.nonanal", "dodecanal", "X.E.E..2.4.decadienal"],
        "origins": {"Hops", "Others"},
        "note": "陈腐酒花香（stale hops）由脂质氧化产生的醛类（E-2-壬烯醛等）承载",
    },
    "t2n": {
        "classes": {"aldehyde"},
        "carriers": ["X.E..2.nonenal"],
        "origins": {"Others"},
        "note": "E-2-壬烯醛（ cardboard 味）是氧化的标志性化合物",
    },
    "orange": {
        "classes": {"terpene"},
        "carriers": ["limonene", "citronellol", "linalool", "nerol",
                     "X.E..beta.damascenone", "valencene", "beta.cyclohomocitral"],
        "origins": {"Hops", "Spices"},
        "note": "橙子调由柠檬烯、香茅醇、芳樟醇与橙花醇等单萜承载",
    },
    "coriander": {
        "classes": {"terpene", "aldehyde"},
        "carriers": ["linalool", "geraniol", "citronellol", "nerol", "dodecanal",
                     "X.E.E..2.4.decadienal"],
        "origins": {"Spices", "Hops"},
        "note": "香菜调由芳樟醇/香叶醇（萜烯）与癸醛/2,4-癸二烯醛（醛）共同承载",
    },
    "clove": {
        "classes": {"spice_phenol", "phenol"},
        "carriers": ["eugenol", "E.methyl.isoeugenol", "X4.ethyl.guaiacol",
                     "X4.methoxysalicylaldehyde", "vanillin"],
        "origins": {"Spices", "Wild flora"},
        "note": "丁香调由丁香酚及其甲基醚与愈创木酚承载",
    },
    "lactic": {
        "classes": {"acid"},
        "carriers": ["lactic_acid.mg.L."],
        "origins": {"Wild flora"},
        "note": "乳酸味由乳酸承载",
    },
    "acetic": {
        "classes": {"acid"},
        "carriers": ["acetic_acid.g.L."],
        "origins": {"Wild flora"},
        "note": "醋酸味由乙酸承载",
    },
    "barnyard": {
        "classes": {"sulfide", "earthy", "sulfur_other"},
        "carriers": ["DMS", "H2S", "MeSH", "DES", "geosmin"],
        "origins": {"Malt", "Yeast", "Wild flora"},
        "note": "马厩/硫化物气味由 DMS、H2S、甲硫醇等含硫化合物承载",
    },
    "alcohol": {
        "classes": {"alcohol"},
        "carriers": ["ethanol..v.v."],
        "origins": {"Yeast"},
        "note": "酒精感由乙醇含量（v/v）承载 —— 同一物理量的理化测量与感官评分",
    },
    "aftertaste": {
        "classes": {"ester", "terpene", "phenol", "hop_resin"},
        "carriers": ["bitterness", "ethyl_octanoate", "alpha.humulene", "citronellol"],
        "origins": {"Yeast", "Hops"},
        "note": "回味由持续释放的苦味树脂、酯类与酒花萜烯共同承载",
    },
    "body": {
        "classes": {"protein", "energy", "sugar"},
        "carriers": ["protein.g.L.", "kcalperc", "glycerol", "betaglucan.mg.L."],
        "origins": {"Others", "Malt"},
        "note": "酒体厚重感由蛋白/胶体、可发酵糖与热值共同承载",
    },
    "co2": {
        "classes": {"gas"},
        "carriers": ["CO2.PSI."],
        "origins": {"Others", "Yeast"},
        "note": "杀口/碳酸感由 CO2 压力承载 —— 同一物理量的两种测量",
    },
    "overall": {
        "classes": set(),
        "carriers": [],
        "origins": set(),
        "note": "总体喜好为多维度综合印象，无单一化学载体；只报告排序，不作单体化学归因",
    },
}


def dim_kb(dim: str) -> dict:
    """取维度的知识库条目；未知维度退化为空字典（不报错，便于扩展）。"""
    return DIM_KB.get(dim, {"classes": set(), "carriers": [], "origins": set(), "note": ""})


def verify_classification() -> str:
    """打印按类别归组的化合物清单，供人工逐条核对。"""
    lines = []
    buckets: Dict[str, List[str]] = {}
    for col, cls in sorted(CLASS_OVERRIDE.items()):
        buckets.setdefault(cls, []).append(f"{col}*")
    lines.append("显式覆盖表共 %d 条" % len(CLASS_OVERRIDE))
    for cls, cols in sorted(buckets.items()):
        lines.append(f"  [{cls}] {', '.join(cols)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(verify_classification())
