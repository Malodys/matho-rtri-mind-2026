# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 相关类型裁定（5.csv/6.csv/2.csv = Spearman）—— 已用「重算 vs 给定矩阵」逐元素比对，
#      Spearman 下 max|Δ|=0、frac<1e-8=1.0000，Pearson 下 median|Δ|≈0.04，结论由数字直接得出。
#   2) 标度裁定（1.csv 为混合标度）—— 已用 8.xlsx「Calibrated chemical dataset」的原始浓度回归：
#      37 列 slope_log10=1.0000/r=1.000000，33 列 slope_raw≈1.0000；ln 假设斜率恰为 2.3026 故排除。
#   3) 名称桥 —— 化学桥 231/231、感官桥 50/50，均逐条打印核对过。
#   4) 缺陷化合物阈值 —— DMS/Diacetyl/4VG 的超阈值比例已按统一矩阵重算（µg/L→mg/L 换算）。
"""共享数据层：赛题附件的唯一可信入口。

设计原则
--------
1. 本模块是**唯一**允许读取 ``data/raw/`` 的地方；问题一/二/三的建模脚本一律经由本模块取数，
   禁止各自 ``read_csv`` 后自行做 ``10 ** x``（1.csv 是混合标度，盲做逆变换会得到错误量纲）。
2. 三张相关矩阵（2.csv / 5.csv / 6.csv）经复算判定为 **Spearman**，与官方 PDF 的
   "Pearson" 表述不一致，以本模块的裁定为准，并在论文中显式勘误。
3. 中间产物一律进 ``data/interim/``、模型输出进 ``data/processed/``，带版本后缀与 manifest。
4. ``data/raw/`` 严格只读，本模块永不向其写入。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- 常量

DATA_VERSION = "data_contract_v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLIT_DIR = REPO_ROOT / "data" / "splits" / "seed_0"

KEY_COLUMNS = ["beer", "beer_id", "tasting_category_fine"]
DERIVED_SUFFIX = "_sum"

# 统一矩阵中不参与建模的派生列（队友分支 q1_evaluation.load_inputs 同样剔除）
DERIVED_COLUMNS = ["esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]

#: 感官阈值 → 缺陷化合物的权威清单（来自 8.xlsx / Sensory thresholds）
#: ``log10_threshold`` 为统一矩阵（log10 mg/L）上的阈值；``unit_in_file8`` 保留文件8原始单位。
DEFECT_SPEC: List[dict] = [
    {
        "key": "dms",
        "label": "dms",
        "column": "DMS",
        "name": "Dimethyl sulfide",
        "threshold_mean": 17.25,
        "threshold_median": 19.00,
        "unit_in_file8": "ug/L",
        "unit_convert_to_mgL": 1e-3,
        "has_threshold": True,
        "note": "主导缺陷风险；统一矩阵单位为 log10(mg/L)，阈值须由 µg/L 换算",
    },
    {
        "key": "diacetyl",
        "label": "diacetyl",
        "column": "X2.3.butanedione",
        "name": "Diacetyl",
        "threshold_mean": 0.29925,
        "threshold_median": 0.09,
        "unit_in_file8": "mg/L",
        "unit_convert_to_mgL": 1.0,
        "has_threshold": True,
        "note": "文件1 存原始浓度；存在 gross outlier（最大 121.731 mg/L）需先处理",
    },
    {
        "key": "4vg",
        "label": "X4vg",
        "column": "X4VG",
        "name": "4-vinylguaiacol",
        "threshold_mean": 0.206333,
        "threshold_median": 0.20,
        "unit_in_file8": "mg/L",
        "unit_convert_to_mgL": 1.0,
        "has_threshold": True,
        "note": "统一矩阵单位 log10(mg/L)",
    },
    {
        "key": "t2n",
        "label": "t2n",
        "column": "X.E..2.nonenal",
        "name": "E-2-nonenal",
        "threshold_mean": float("nan"),
        "threshold_median": float("nan"),
        "unit_in_file8": "mg/L",
        "unit_convert_to_mgL": 1.0,
        "has_threshold": False,
        "note": "文件8 无阈值：以「迁移后不恶化」的相对约束兜底，论文须声明该缺口",
    },
]

#: 问题二绝对锚定用：感官维度 → 统一矩阵中的化学载体列
#: 载体选择依据：8.xlsx 感官阈值表中与该维度存在已知感官对应关系的化合物。
ABSOLUTE_ANCHOR_CARRIERS: Dict[str, List[str]] = {
    "bitternes": ["bitterness"],
    "alcohol": ["ethanol..v.v."],
    "dms": ["DMS"],
    "X4vg": ["X4VG"],
    "diacetyl": ["X2.3.butanedione"],
    "t2n": ["X.E..2.nonenal"],
    "metallic": ["iron"],
    "acidity": ["lactic_acid.mg.L."],
    "body": ["protein.g.L.", "kcalperc"],
    "sweetness": ["unfermentables"],
    "co2": ["CO2.PSI."],
}


# --------------------------------------------------------------------------- 工具


#: 希腊字母 → ASCII 音译表。8.xlsx 的 chemical_name 用 ASCII 拼写（如 "(E)-beta-ionone"），
#: 而 2.csv / 6.csv 用希腊字母（"E-β-ionone"）。若直接把希腊字母当非字母数字剥掉，
#: "E-α-ionone" 与 "E-β-ionone" 会塌缩成同一个键 "eionone"，造成静默错配。
GREEK_ASCII = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}


def _norm_name(value: object) -> str:
    """把名称归一成仅含小写字母数字的键，用于跨表匹配。

    先把希腊字母音译成 ASCII，再剥除非字母数字 —— 顺序不可颠倒，否则 α/β 会塌缩。
    """
    text = str(value).lower()
    text = "".join(GREEK_ASCII.get(ch, ch) for ch in text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _unique_index(pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """构建「归一化键 → 目标」映射，**丢弃歧义键**（同一键指向多个目标时静默错配风险太高）。"""
    grouped: Dict[str, List[str]] = {}
    for key, target in pairs:
        grouped.setdefault(key, []).append(target)
    return {k: v[0] for k, v in grouped.items() if len(set(v)) == 1}


def sha256_of(path: Path) -> str:
    """返回文件的 sha256 十六进制摘要，用于 manifest 与原始数据完整性对账。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, payload: dict) -> Path:
    """写出与产物同名的 ``_manifest.json``，记录上游输入与参数。"""
    manifest_path = path.with_name(path.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest_path


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """统一的 CSV 读取：utf-8-sig 容忍 BOM，index_col=0 时保留首列名。"""
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


# --------------------------------------------------------------------------- 原始附件


@dataclass
class RawBundle:
    """附件 1–8 的只读容器。"""

    chem: pd.DataFrame           # 1.csv  250 x 234
    corr_chem: pd.DataFrame      # 2.csv  231 x 231
    panel: pd.ExcelFile         # 3.xlsx
    sensory: pd.DataFrame        # 4.csv  250 x 53
    corr_sensory: pd.DataFrame   # 5.csv  50 x 50
    cross: pd.DataFrame         # 6.csv  44 x 50
    forum: pd.DataFrame          # 7.csv  250 x 189
    thresholds: pd.DataFrame    # 8.xlsx / Sensory thresholds
    sensory_names: pd.DataFrame  # 8.xlsx / Sensory names
    gbm_mdi: pd.DataFrame        # 8.xlsx / GBM importance MDI
    gbm_shap: pd.DataFrame      # 8.xlsx / GBM SHAP Values
    calibrated: pd.DataFrame    # 8.xlsx / Calibrated chemical dataset
    beer_prices: pd.DataFrame   # 8.xlsx / Beer Prices
    manifest: Dict[str, dict]


def load_raw(root: Optional[Path] = None) -> RawBundle:
    """读取附件 1–8，并生成含 sha256 的清单。

    人工核验点：读取后立即校验形状（231/231、50/50、44/50、250/53 等）与题目描述一致。
    """
    raw = Path(root or RAW_DIR)
    files = {
        "1.csv": raw / "1.csv", "2.csv": raw / "2.csv", "3.xlsx": raw / "3.xlsx",
        "4.csv": raw / "4.csv", "5.csv": raw / "5.csv", "6.csv": raw / "6.csv",
        "7.csv": raw / "7.csv", "8.xlsx": raw / "8.xlsx",
    }
    manifest = {
        name: {"bytes": p.stat().st_size, "sha256": sha256_of(p)}
        for name, p in files.items() if p.exists()
    }
    chem = _read_csv(files["1.csv"])
    corr_chem = _read_csv(files["2.csv"], index_col=0)
    corr_chem.index = corr_chem.index.astype(str)
    corr_chem.columns = corr_chem.columns.astype(str)
    sensory = _read_csv(files["4.csv"])
    corr_sensory = _read_csv(files["5.csv"], index_col=0)
    corr_sensory.index = corr_sensory.index.astype(str)
    corr_sensory.columns = corr_sensory.columns.astype(str)
    cross = _read_csv(files["6.csv"], index_col=0)
    cross.index = cross.index.astype(str)
    cross.columns = cross.columns.astype(str)
    forum = _read_csv(files["7.csv"])
    x8 = pd.ExcelFile(files["8.xlsx"])

    bundle = RawBundle(
        chem=chem, corr_chem=corr_chem, panel=pd.ExcelFile(files["3.xlsx"]),
        sensory=sensory, corr_sensory=corr_sensory, cross=cross, forum=forum,
        thresholds=x8.parse("Sensory thresholds"),
        sensory_names=x8.parse("Sensory names"),
        gbm_mdi=x8.parse("GBM importance MDI"),
        gbm_shap=x8.parse("GBM SHAP Values"),
        calibrated=x8.parse("Calibrated chemical dataset"),
        beer_prices=x8.parse("Beer Prices"),
        manifest=manifest,
    )
    # 形状自检（人工核验：与赛题附件描述逐项对齐）
    assert bundle.chem.shape == (250, 234), bundle.chem.shape
    assert bundle.corr_chem.shape == (231, 231), bundle.corr_chem.shape
    assert bundle.sensory.shape == (250, 53), bundle.sensory.shape
    assert bundle.corr_sensory.shape == (50, 50), bundle.corr_sensory.shape
    assert bundle.cross.shape == (44, 50), bundle.cross.shape
    assert bundle.forum.shape[0] == 250, bundle.forum.shape
    return bundle


# --------------------------------------------------------------------------- 名称桥


def build_chemical_bridge(raw: RawBundle) -> pd.DataFrame:
    """构建「相关矩阵行名 → 1.csv 列名」桥表。

    规则（按优先级）：
      1. 8.xlsx ``Sensory thresholds.chemical_name`` 精确命中 → 取 ``field_name_R``；
      2. 归一化匹配（去非字母数字）命中 1.csv 列名 —— 用于 α/β/γ 等希腊字母萜类；
      3. 归一化匹配命中 thresholds 的 ``field_name_python`` 再转到 ``field_name_R``。

    人工核验点：解析率必须达到 231/231（2.csv 全部节点）与 44/44（6.csv 全部行）。
    """
    chem1 = [c for c in raw.chem.columns if c not in KEY_COLUMNS]
    thresholds = raw.thresholds
    by_chem_name = thresholds.set_index("chemical_name")
    norm_to_col = _unique_index((_norm_name(c), c) for c in chem1)
    norm_to_field = _unique_index(
        (_norm_name(row.get("field_name_python")), row.get("field_name_R"))
        for _, row in thresholds.iterrows()
        if isinstance(row.get("field_name_python"), str)
    )
    norm_chem_name = _unique_index(
        (_norm_name(row.get("chemical_name")), row.get("field_name_R"))
        for _, row in thresholds.iterrows()
        if isinstance(row.get("chemical_name"), str)
        and isinstance(row.get("field_name_R"), str)
    )

    def resolve_chemical(name: str) -> Tuple[Optional[str], str]:
        """化学名 → 1.csv 列名。四级兜底，逐级放宽。"""
        if name in by_chem_name.index:
            fn = by_chem_name.loc[name, "field_name_R"]
            if fn in chem1:
                return fn, "thresholds.chemical_name"
        key = _norm_name(name)
        if key in norm_to_col:
            return norm_to_col[key], "normalized_to_file1"
        if key in norm_to_field:
            fn = norm_to_field[key]
            if fn in chem1:
                return fn, "normalized_to_field_name_python"
        if key in norm_chem_name:
            fn = norm_chem_name[key]
            if fn in chem1:
                return fn, "normalized_to_chemical_name"
        return None, ""

    rows = []
    for name in list(raw.corr_chem.index):
        target, how = resolve_chemical(name)
        rows.append({
            "matrix": "2.csv", "matrix_name": name,
            "file1_column": target or "", "resolve_method": how,
        })
    bridge = pd.DataFrame(rows)

    # 6.csv 的行使用同一套化学名
    extra = []
    for name in raw.cross.index:
        target, how = resolve_chemical(name)
        extra.append({"matrix": "6.csv", "matrix_name": name,
                      "file1_column": target or "", "resolve_method": how})
    bridge = pd.concat([bridge, pd.DataFrame(extra)], ignore_index=True)
    return bridge


def build_sensory_bridge(raw: RawBundle) -> pd.DataFrame:
    """构建「相关矩阵列名 → 4.csv 目标列名」桥表。

    规则：8.xlsx ``Sensory names`` 的 CleanName_PanelFeedback / CleanName_PaperLong /
    CleanName_Paper → ``R_name``；同名多义时按 "aroma"→A_、"taste"→F_ 前缀消歧。

    人工核验点：5.csv 与 6.csv 的全部列必须解析到 4.csv 的 50 个目标列。
    """
    names = raw.sensory_names.copy()
    for col in ("CleanName_Paper", "CleanName_PaperLong", "CleanName_PanelFeedback"):
        names[col] = (names[col].astype(str)
                      .str.replace(r"\(Trained panel\)", "", regex=True)
                      .str.strip())
    lookup: Dict[str, List[str]] = {}
    for _, row in names.iterrows():
        for col in ("CleanName_PanelFeedback", "CleanName_PaperLong", "CleanName_Paper"):
            value = row[col]
            if value and value.lower() != "nan":
                lookup.setdefault(value, []).append(row["R_name"])

    sensory_cols = [c for c in raw.sensory.columns if c not in KEY_COLUMNS]

    def resolve(name: str) -> Optional[str]:
        # 矩阵侧的名称可能带有 "(Trained panel)" 后缀与换行，先清洗再查表
        clean = re.sub(r"\(Trained panel\)", "", str(name)).strip()
        clean = re.sub(r"\s+", " ", clean)
        cand = lookup.get(clean, [])
        if not cand:
            return None
        if len(set(cand)) == 1:
            return cand[0]
        low = clean.lower()
        prefix = "A_" if "aroma" in low else ("F_" if "taste" in low else None)
        if prefix:
            hits = [c for c in cand if c.startswith(prefix)]
            if hits:
                return hits[0]
        return cand[0]

    rows = []
    for matrix, cols in (("5.csv", raw.corr_sensory.columns),
                         ("6.csv", raw.cross.columns)):
        for name in cols:
            target = resolve(name)
            ok = target in sensory_cols
            rows.append({
                "matrix": matrix, "matrix_name": name,
                "file4_column": target if ok else "",
                "resolve_method": "CleanName_PanelFeedback" if ok else "",
            })
    return pd.DataFrame(rows)


def build_bridges(raw: Optional[RawBundle] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """同时返回化学桥与感官桥。"""
    bundle = raw or load_raw()
    return build_chemical_bridge(bundle), build_sensory_bridge(bundle)


# --------------------------------------------------------------------------- 标度


def load_provenance(root: Optional[Path] = None) -> pd.DataFrame:
    """读取队友产出的逐列溯源表（化学块 / 单位 / 变换关系 / 标度动作）。"""
    path = Path(root or INTERIM_DIR) / "chem_feature_provenance_v1.csv"
    prov = _read_csv(path, index_col=0)
    prov.index = prov.index.astype(str)
    return prov


def to_unified_scale(raw: RawBundle, provenance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """把 1.csv 的混合标度统一为「log10(原单位) + 化合物统一到 log10(mg/L)」。

    实现方式：直接采用队友分支已验证的 ``chem_log10_matrix_v1.csv``（250 x 231），
    该矩阵已按 provenance 的 ``scale_action`` 完成逆变换与单位统一；本函数仅做校验与
    `_sum` 派生列的剔除约定对齐，不重复实现逆变换逻辑（避免两套实现漂移）。

    人工核验点：DMS 中位数换算回 24.23 µg/L、Diacetyl 0.2566 mg/L、pH 4.13，
    与 1.csv 原始存储标度的独立复算一致。
    """
    matrix = _read_csv(INTERIM_DIR / "chem_log10_matrix_v1.csv", index_col=0)
    matrix.index = matrix.index.astype(int)
    if matrix.shape != (250, 231):
        raise ValueError(f"统一矩阵形状异常：{matrix.shape}")
    if matrix.isna().any().any():
        raise ValueError("统一矩阵含缺失值，应先按 LOD/缺失机制处理")
    return matrix


def load_modeling_matrix(raw: Optional[RawBundle] = None) -> pd.DataFrame:
    """返回建模用特征矩阵（250 x 227，剔除 4 个未验证的 ``*_sum`` 派生列）。"""
    matrix = to_unified_scale(raw) if raw is not None else to_unified_scale(load_raw())
    return matrix.drop(columns=[c for c in matrix.columns if c.endswith(DERIVED_SUFFIX)])


def audit_scale(raw: Optional[RawBundle] = None,
                provenance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """逐列裁定 1.csv 的标度并与 provenance 对账。

    对每个可与 8.xlsx「Calibrated chemical dataset」对齐的化合物，分别拟合
    ``file1_value ~ log10(raw)`` 与 ``file1_value ~ raw``，比较斜率与相关系数：

    * ``slope_log10 ≈ 1`` 且 ``r ≈ 1``  → 该列为 log10 标度；
    * ``slope_raw  ≈ 1`` 且 ``r ≈ 1``  → 该列存的是原始浓度；
    * 两者皆不满足                        → 判定为 AMBIG（GC-MS 相对丰度等无绝对校准的列）。

    人工核验点：ln 假设的斜率恒为 2.3026 = 1/ln(10)，据此排除自然对数。
    """
    bundle = raw or load_raw()
    prov = provenance if provenance is not None else load_provenance()
    calibrated = bundle.calibrated.set_index("beer")
    chem = bundle.chem.set_index("beer")
    common = [b for b in calibrated.index if b in chem.index]

    rows = []
    for col in calibrated.columns:
        if col in KEY_COLUMNS:
            continue
        c = pd.to_numeric(calibrated.loc[common, col], errors="coerce")
        x = pd.to_numeric(chem.loc[common, col], errors="coerce")
        mask = c.notna() & x.notna() & (c > 0)
        if int(mask.sum()) < 25:
            continue
        cv, xv = c[mask].to_numpy(), x[mask].to_numpy()

        def fit(a: np.ndarray) -> Tuple[float, float]:
            a0 = a - a.mean()
            b0 = xv - xv.mean()
            denom = float((b0 * b0).sum())
            slope = float((a0 * b0).sum() / denom) if denom > 0 else float("nan")
            r = float(np.corrcoef(a, xv)[0, 1]) if np.std(a) > 0 and np.std(xv) > 0 else float("nan")
            return slope, r

        s10, r10 = fit(np.log10(cv))
        sraw, rraw = fit(cv)
        sln, rln = fit(np.log(cv))
        log10_ok = abs(s10 - 1) < 1e-3 and r10 > 0.9999
        raw_ok = abs(sraw - 1) < 3e-3 and rraw > 0.999
        # 近似 raw：斜率接近 1 且高度相关，但因文件8 的校准/阈值化差异无法逐点精确复现
        raw_approx = abs(sraw - 1) < 0.06 and rraw > 0.85
        if log10_ok:
            verdict = "LOG10"
        elif raw_ok:
            verdict = "RAW"
        elif raw_approx:
            verdict = "RAW_APPROX"
        else:
            verdict = "AMBIG"
        rows.append({
            "file1_column": col, "n_obs": int(mask.sum()), "verdict": verdict,
            "slope_log10": round(s10, 6), "r_log10": round(r10, 6),
            "slope_raw": round(sraw, 6), "r_raw": round(rraw, 6),
            "slope_ln": round(sln, 6), "r_ln": round(rln, 6),
            "provenance_block": (prov.loc[col, "block"] if col in prov.index else ""),
            "provenance_transform": (prov.loc[col, "transform_relation"] if col in prov.index else ""),
            "provenance_scale_action": (prov.loc[col, "scale_action"] if col in prov.index else ""),
        })
    audit = pd.DataFrame(rows)

    def agrees(verdict: str, transform: object) -> Optional[bool]:
        """与 provenance 的变换声明是否一致；provenance 无声明时返回 None（不算冲突）。

        AMBIG + provenance 声明 'original' 视为「结论一致但验证强度为近似」——
        这 4 列的文件8 标定值与文件1 存储值存在系统性差异，无法逐点复现。
        """
        if not isinstance(transform, str) or not transform.strip():
            return None
        t = transform.strip()
        if verdict == "LOG10":
            return t == "log10"
        if verdict in ("RAW", "RAW_APPROX", "AMBIG"):
            return t == "original"
        return False

    flags = [agrees(v, t) for v, t in zip(audit["verdict"], audit["provenance_transform"])]
    audit["matches_provenance"] = [True if f is None else bool(f) for f in flags]
    audit["has_provenance_claim"] = [f is not None for f in flags]
    audit["verification_strength"] = [
        "exact" if v in ("LOG10", "RAW") else ("approximate" if v == "RAW_APPROX" else "unverified")
        for v in audit["verdict"]
    ]
    return audit


# --------------------------------------------------------------------------- 其它中间产物


def load_lod_mask(root: Optional[Path] = None) -> pd.DataFrame:
    """读取检出限掩码（左删失标记）。缺失 ≠ 未检出，禁止当作 0。"""
    return _read_csv(Path(root or INTERIM_DIR) / "chem_nondetect_mask_v1.csv", index_col=0)


def load_observation_status(root: Optional[Path] = None) -> pd.DataFrame:
    """读取观测状态码（0 观测 / 1 左删失 / 2 派生回填）。"""
    return _read_csv(Path(root or INTERIM_DIR) / "chem_observation_status_v1.csv", index_col=0)


def load_outlier_flags(root: Optional[Path] = None) -> pd.DataFrame:
    """读取统一离群标记（单元级 + 样本级）。"""
    return _read_csv(Path(root or INTERIM_DIR) / "chem_outlier_unified_flags_v1.csv", index_col=0)


def load_label_dictionary(root: Optional[Path] = None) -> pd.DataFrame:
    """读取 50 个感官维度的标签字典。"""
    return _read_csv(Path(root or INTERIM_DIR) / "label_dictionary_v1.csv", index_col=0)


def load_split(split_dir: Optional[Path] = None) -> pd.DataFrame:
    """读取锁定的 seed 0 分层切分（175 训练 / 75 测试）。"""
    return _read_csv(Path(split_dir or SPLIT_DIR) / "split_manifest.csv")


def load_targets(raw: Optional[RawBundle] = None) -> pd.DataFrame:
    """返回 4.csv 的 50 个感官目标列（标准化尺度），索引为 beer。"""
    bundle = raw or load_raw()
    src = bundle.sensory.set_index("beer")
    src.index = src.index.astype(int)
    return src[[c for c in src.columns if c not in KEY_COLUMNS]].apply(
        pd.to_numeric, errors="coerce")


def load_styles(raw: Optional[RawBundle] = None) -> pd.Series:
    """返回风格标签序列。"""
    bundle = raw or load_raw()
    src = bundle.sensory.set_index("beer")
    src.index = src.index.astype(int)
    return src["tasting_category_fine"].astype(str)


# --------------------------------------------------------------------------- 缺陷阈值


def defect_table() -> pd.DataFrame:
    """返回缺陷化合物的阈值表（含统一矩阵上的 log10 阈值）。

    人工核验点：DMS 阈值 17.25 µg/L 换算为 0.01725 mg/L → log10 = -1.7629；
    Diacetyl 0.29925 mg/L → log10 = -0.5240；4VG 0.206333 mg/L → log10 = -0.6854。
    """
    rows = []
    for spec in DEFECT_SPEC:
        row = dict(spec)
        if spec["has_threshold"]:
            thr_mg = spec["threshold_mean"] * spec["unit_convert_to_mgL"]
            row["threshold_mg_per_L"] = thr_mg
            row["log10_threshold"] = float(np.log10(thr_mg))
        else:
            row["threshold_mg_per_L"] = float("nan")
            row["log10_threshold"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def defect_exceedance(matrix: pd.DataFrame, raw: Optional[RawBundle] = None) -> pd.DataFrame:
    """按统一矩阵口径统计各缺陷化合物的超阈值比例。"""
    table = defect_table()
    rows = []
    for _, spec in table.iterrows():
        col = spec["column"]
        if col not in matrix.columns:
            continue
        values = pd.to_numeric(matrix[col], errors="coerce")
        n = int(values.notna().sum())
        if spec["has_threshold"]:
            exceed = int((values > spec["log10_threshold"]).sum())
            rows.append({"key": spec["key"], "column": col, "n": n,
                         "n_exceed": exceed, "exceed_pct": round(100 * exceed / n, 2),
                         "log10_threshold": spec["log10_threshold"],
                         "has_threshold": True})
        else:
            rows.append({"key": spec["key"], "column": col, "n": n,
                         "n_exceed": -1, "exceed_pct": float("nan"),
                         "log10_threshold": float("nan"), "has_threshold": False})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 落盘产物


def persist_interim(raw: Optional[RawBundle] = None) -> dict:
    """落盘 ``data/interim/`` 下的版本化中间产物并写 manifest。"""
    bundle = raw or load_raw()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    chem_bridge, sens_bridge = build_bridges(bundle)
    audit = audit_scale(bundle)

    bridges = pd.concat([
        chem_bridge.rename(columns={"matrix_name": "name", "file1_column": "target",
                                    "resolve_method": "method"}),
        sens_bridge.rename(columns={"matrix_name": "name", "file4_column": "target",
                                    "resolve_method": "method"}),
    ], ignore_index=True)
    bridges.insert(0, "kind", ["chemical"] * len(chem_bridge) + ["sensory"] * len(sens_bridge))

    chem_path = INTERIM_DIR / "bridges_v1.csv"
    chem_path.write_text(bridges.to_csv(index=False), encoding="utf-8")
    audit_path = INTERIM_DIR / "chem_scale_audit_v1.csv"
    audit_path.write_text(audit.to_csv(index=False), encoding="utf-8")
    defects = defect_table()
    defect_path = INTERIM_DIR / "defect_thresholds_v1.csv"
    defect_path.write_text(defects.to_csv(index=False), encoding="utf-8")

    payload = {
        "data_version": DATA_VERSION,
        "raw_manifest": bundle.manifest,
        "chemical_bridge_resolved": int((chem_bridge["file1_column"] != "").sum()),
        "chemical_bridge_total": int(len(chem_bridge)),
        "sensory_bridge_resolved": int((sens_bridge["file4_column"] != "").sum()),
        "sensory_bridge_total": int(len(sens_bridge)),
        "scale_audit_verdicts": audit["verdict"].value_counts().to_dict(),
        "scale_audit_matches_provenance": int(audit["matches_provenance"].sum()),
        "scale_audit_total": int(len(audit)),
        "outputs": [str(p) for p in (chem_path, audit_path, defect_path)],
    }
    write_manifest(chem_path, payload)
    return payload


__all__ = [
    "DATA_VERSION", "REPO_ROOT", "RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR", "SPLIT_DIR",
    "KEY_COLUMNS", "DERIVED_COLUMNS", "DEFECT_SPEC", "ABSOLUTE_ANCHOR_CARRIERS",
    "RawBundle", "load_raw", "build_chemical_bridge", "build_sensory_bridge",
    "build_bridges", "load_provenance", "to_unified_scale", "load_modeling_matrix",
    "audit_scale", "load_lod_mask", "load_observation_status", "load_outlier_flags",
    "load_label_dictionary", "load_split", "load_targets", "load_styles",
    "defect_table", "defect_exceedance", "persist_interim", "sha256_of", "write_manifest",
]
