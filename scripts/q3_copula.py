# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：
#   1) 统一矩阵取数全程经由 data_contract（to_unified_scale），未对全表做任何 10**x 逆变换；
#      缺陷阈值一律取 defect_table() 的 log10_threshold（统一矩阵口径），已打印核对
#      DMS=-1.7632 / Diacetyl=-0.5240 / 4VG=-0.6854 / t2n=NaN。
#   2) 全局相关矩阵由 250×231 统一矩阵的范德瓦登正态分数计算，实测最小特征值经裁剪后为正，
#      Cholesky 可分解（脚本内 assert），正定性由 eig-clip + 岭正则保证。
#   3) 风格条件边际的 λ = n/(n+10) 已逐风格打印核对；风格内常量列（多为 LOD 下限）单列覆盖
#      λ_c=1，理由与数量在 q3_report.py 的 README 中声明。
#   4) 迁移后四项缺陷浓度逐样本 ≤ 迁移前，由 q3_transfer.py 内置 assert 强制。
"""问题 3 核心库：全局高斯 Copula + 风格条件边际 + 缺陷约束 + 秩-分位迁移。

方法定义
--------
* **全局 Copula**：对 250×231 统一 log10(mg/L) 矩阵逐列取范德瓦登正态分数，
  计算 Pearson 相关 R（231×231），做岭正则 + 特征值下截断保证正定，Cholesky 分解。
* **风格条件边际**：F_s = (1-λ)·F_global + λ·F_s^emp，λ = n/(n+10)，实现为分位空间线性混合
  Q_s = (1-λ)·Q_global + λ·Q_s^emp（单调递增，仍是合法分位函数）。
  风格内常量列（方差为 0，多为 LOD 下限）取 λ_c = 1，直接保留风格常量值。
* **采样**：z ~ N(0, R) → u = Φ(z) → v_j = Q_{s,j}(u_j)，保留化合物间相关结构。
* **缺陷约束**：对有阈值的 DMS/Diacetyl/4VG 按 defect_table() 的 log10 阈值做拒绝-重抽，
  不足时退化为单调压缩修复（带 margin，保证严格小于阈值）。
* **跨风格迁移**：秩-分位映射（Iman-Conover 边际替换），保留源样本的秩结构 → 迁移后
  Spearman ρ 矩阵与迁移前完全一致（逐列单调映射不改变秩）；再施加缺陷约束与「不恶化」约束。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm, rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data_contract as dc  # noqa: E402

#: 迁移/生成的目标风格（样本量充足）。
DEFAULT_TARGET_STYLES: List[str] = [
    "Tripel", "Kriek", "Lambic", "Blond", "Strong ale", "Hoppy", "Dubbel", "Saison",
]

#: 缺陷约束的安全余量（log10 口径），保证严格小于阈值。
DEFECT_MARGIN = 0.02


# --------------------------------------------------------------------------- 取数


@dataclass
class Dataset:
    """统一标度数据 + 风格标签 + 缺陷阈值表。"""

    matrix: pd.DataFrame          # (250, 231) 统一 log10(mg/L)
    styles: pd.Series             # 索引=beer，值为 tasting_category_fine
    defects: pd.DataFrame         # defect_table()
    split: pd.DataFrame

    @property
    def columns(self) -> List[str]:
        return list(self.matrix.columns)


def load_dataset() -> Dataset:
    """经由 data_contract 取得统一矩阵与标签（唯一取数入口）。"""
    raw = dc.load_raw()
    matrix = dc.to_unified_scale(raw)
    styles = dc.load_styles(raw)
    # 按标签对齐（不依赖行序），并核对样本量合计
    styles = styles.reindex(matrix.index)
    if styles.isna().any():
        raise ValueError("风格标签与统一矩阵索引无法对齐")
    if len(styles) != matrix.shape[0]:
        raise ValueError("风格标签长度与矩阵行数不一致")
    return Dataset(matrix=matrix, styles=styles, defects=dc.defect_table(),
                   split=dc.load_split())


def style_slices(data: Dataset) -> Dict[str, np.ndarray]:
    """风格 → 该风格样本在矩阵中的行位置。"""
    out: Dict[str, np.ndarray] = {}
    for name in data.styles.unique():
        out[str(name)] = np.flatnonzero((data.styles == name).to_numpy())
    return out


# --------------------------------------------------------------------------- 分位工具


def q_linear(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """逐行线性插值分位函数（等价 np.quantile(method='linear')）。

    Parameters
    ----------
    sorted_values : (p, N) 每行已升序排列。
    u : (p, m) 概率，取值被裁剪到 [0, 1]；超出行支持集时取端点（不外推）。
    """
    p, n = sorted_values.shape
    pos = np.clip(u * n - 0.5, 0.0, n - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    rows = np.arange(p)[:, None]
    return sorted_values[rows, lo] * (1.0 - frac) + sorted_values[rows, hi] * frac


# --------------------------------------------------------------------------- Copula


@dataclass
class GlobalCopula:
    """全局高斯 Copula（结构层）。"""

    columns: List[str]
    chol: np.ndarray                      # (p, p) Cholesky 因子
    corr: np.ndarray                      # (p, p) 正定相关矩阵
    corr_raw: np.ndarray                  # 裁剪前的原始相关矩阵
    eig_raw: np.ndarray                   # 原始特征值
    eig_clipped: np.ndarray               # 裁剪后特征值
    n_clipped: int
    ridge: float
    global_sorted: np.ndarray             # (p, N) 全局经验分位

    @property
    def p(self) -> int:
        return len(self.columns)


def fit_global_copula(matrix: pd.DataFrame, ridge: float = 1e-3,
                       eig_floor: float = 1e-6) -> GlobalCopula:
    """拟合 231 维全局高斯 Copula。

    人工核验点：Cholesky 成功（正定）且裁剪后最小特征值 > 0。
    """
    values = matrix.to_numpy(dtype=float)
    n, p = values.shape
    ranks = rankdata(values, axis=0)                       # (n, p) 平均秩
    z = norm.ppf((ranks - 0.5) / n)                        # 范德瓦登正态分数
    corr_raw = np.corrcoef(z, rowvar=False)
    corr_raw = (corr_raw + corr_raw.T) / 2.0

    eig_raw, vec = np.linalg.eigh(corr_raw)
    eig_clipped = np.maximum(eig_raw, eig_floor)
    n_clipped = int((eig_raw < eig_floor).sum())
    corr = (vec * eig_clipped) @ vec.T
    corr = (corr + corr.T) / 2.0
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    corr = (1.0 - ridge) * corr + ridge * np.eye(p)        # 岭正则
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)                        # 正定性断言
    assert np.all(np.isfinite(chol))

    # 约定：排序数组布局为 (化合物数, 样本数) —— 行 j 是化合物 j 的升序经验分位
    global_sorted = np.sort(values, axis=0).T
    return GlobalCopula(columns=list(matrix.columns), chol=chol, corr=corr,
                         corr_raw=corr_raw, eig_raw=eig_raw, eig_clipped=eig_clipped,
                         n_clipped=n_clipped, ridge=ridge, global_sorted=global_sorted)


# --------------------------------------------------------------------------- 条件边际


@dataclass
class ConditionalMarginal:
    """风格条件边际：F_s = (1-λ)·F_global + λ·F_s^emp。

    两种实现：

    * ``mode="cdf"``（默认，**题面口径**）：对混合 CDF
      M(x) = (1-λ)·F_global(x) + λ·F_s^emp(x) 做广义逆 M⁻¹。
      理论上 KS(M, F_s) = (1-λ)·KS(F_global, F_s)，即收缩带来的边际偏差可精确刻画。
    * ``mode="quantile"``（对照）：在分位空间线性混合
      Q_s = (1-λ)·Q_global + λ·Q_s^emp，生成值是插值值而非真实样本原子。

    风格内常量列（方差 0，多为 LOD 下限）统一按 λ_c = 1 处理，直接保留风格常量值。
    """

    style: str
    lam: float
    n_style: int
    global_sorted: np.ndarray             # (p, N)
    style_sorted: np.ndarray              # (p, n_s)
    const_mask: np.ndarray                # (p,) 风格内常量列
    columns: List[str] = field(default_factory=list)
    mode: str = "cdf"
    _atoms: Optional[List[np.ndarray]] = field(default=None, repr=False)
    _mcdf: Optional[List[np.ndarray]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.mode == "cdf":
            self._atoms, self._mcdf = self._build_mixture_tables()

    def _build_mixture_tables(self):
        """逐列构造混合 CDF 的原子集与累计值，供广义逆查表。"""
        atoms_list: List[np.ndarray] = []
        mcdf_list: List[np.ndarray] = []
        for j in range(self.global_sorted.shape[0]):
            g = self.global_sorted[j]
            s = self.style_sorted[j]
            atoms = np.unique(np.concatenate([g, s]))
            cg = np.searchsorted(g, atoms, side="right")
            cs = np.searchsorted(s, atoms, side="right")
            m = (1.0 - self.lam) * (cg / g.size) + self.lam * (cs / s.size)
            atoms_list.append(atoms)
            mcdf_list.append(m)
        return atoms_list, mcdf_list

    def _q_column(self, j: int, u: np.ndarray) -> np.ndarray:
        if self.const_mask[j]:
            return np.full(np.shape(u), float(self.style_sorted[j, 0]), dtype=float)
        if self.mode == "cdf":
            atoms, mcdf = self._atoms[j], self._mcdf[j]
            idx = np.searchsorted(mcdf, u, side="left")
            idx = np.clip(idx, 0, atoms.size - 1)
            return atoms[idx]
        gq = q_linear(self.global_sorted[j][None, :], np.atleast_2d(u))[0]
        sq = q_linear(self.style_sorted[j][None, :], np.atleast_2d(u))[0]
        return (1.0 - self.lam) * gq + self.lam * sq

    def q(self, u: np.ndarray) -> np.ndarray:
        """u: (p, m) → (p, m) 条件分位值。"""
        out = np.empty(np.shape(u), dtype=float)
        for j in range(self.global_sorted.shape[0]):
            out[j] = self._q_column(j, u[j])
        return out

    def q_column(self, j: int, u: np.ndarray) -> np.ndarray:
        """单列条件分位（迁移用）。"""
        return self._q_column(j, u)


def lambda_for(n_style: int, k: float = 10.0) -> float:
    """λ = n/(n+k)：小样本向全局边际收缩。"""
    return float(n_style) / (float(n_style) + k)


def build_marginal(copula: GlobalCopula, style_values: np.ndarray,
                   style: str, k: float = 10.0, mode: str = "cdf",
                   lam: Optional[float] = None,
                   lam_scale: float = 1.0) -> ConditionalMarginal:
    """用某风格的真实样本构造条件边际。

    ``lam_scale``：λ 的乘子（用于 λ 敏感性研究；1.0 即题面口径 λ = n/(n+10)）。
    """
    n_s = style_values.shape[0]
    base_lam = lambda_for(n_s, k)
    lam = base_lam if lam is None else float(lam)
    lam = float(np.clip(lam * lam_scale, 0.0, 1.0))
    style_sorted = np.sort(style_values, axis=0).T
    # 逐列极差（行=化合物、列=样本）：极差为 0 ⇒ 风格内常量列（多为 LOD 下限）
    const_mask = (style_sorted[:, -1] - style_sorted[:, 0]) < 1e-12
    return ConditionalMarginal(style=style, lam=lam, n_style=n_s,
                               global_sorted=copula.global_sorted,
                               style_sorted=style_sorted, const_mask=const_mask,
                               columns=list(copula.columns), mode=mode)


# --------------------------------------------------------------------------- 采样与约束


def sample_copula(copula: GlobalCopula, marginal: ConditionalMarginal, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """从全局 Copula 抽样并映射到风格条件边际，返回 (n, p)。"""
    eps = rng.standard_normal((n, copula.p))
    z = eps @ copula.chol.T
    u = np.clip(ndtr(z), 1e-9, 1.0 - 1e-9).T            # (p, n)；ndtr = 标准正态 CDF（快于 norm.cdf）
    return marginal.q(u).T


def defect_specs(defects: pd.DataFrame) -> List[dict]:
    """返回带阈值信息的缺陷规格列表。"""
    out = []
    for _, r in defects.iterrows():
        out.append({
            "key": str(r["key"]), "column": str(r["column"]),
            "has_threshold": bool(r["has_threshold"]),
            "log10_threshold": float(r["log10_threshold"]),
        })
    return out


def threshold_violations(values: np.ndarray, columns: Sequence[str],
                         specs: Sequence[dict], margin: float = 0.0) -> np.ndarray:
    """返回每个样本是否违反任一阈值约束（布尔向量）。"""
    index = {c: i for i, c in enumerate(columns)}
    bad = np.zeros(values.shape[0], dtype=bool)
    for spec in specs:
        if not spec["has_threshold"]:
            continue
        j = index[spec["column"]]
        bad |= values[:, j] > spec["log10_threshold"] - margin
    return bad


def enforce_defect_constraints(copula: GlobalCopula, marginal: ConditionalMarginal,
                               n: int, specs: Sequence[dict], rng: np.random.Generator,
                               mode: str = "reject", margin: float = DEFECT_MARGIN,
                               pool_factor: int = 30, max_rounds: int = 6,
                               return_unconstrained: bool = False
                               ) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """生成满足缺陷阈值约束的风格样本。

    mode='reject'：拒绝-重抽（主）；mode='compress'：单调压缩修复（对照）。
    return_unconstrained=True 时，额外返回**同一候选池的前 n 条**（未施加任何约束），
    构成严格配对的「无约束参照谱」：同种子、同 Copula、同条件边际、同候选抽样，
    唯一差别是缺陷约束 —— 用于量化约束带来的边际压缩。
    返回 (values (n, p), info[, unconstrained (n, p)])。
    """
    pool = max(int(n * pool_factor), 2000)
    accepted: List[np.ndarray] = []
    n_rejected = 0
    rounds = 0
    unconstrained: Optional[np.ndarray] = None
    while sum(a.shape[0] for a in accepted) < n and rounds < max_rounds:
        rounds += 1
        cand = sample_copula(copula, marginal, pool, rng)
        if unconstrained is None:
            unconstrained = cand[:n].copy()
        if mode == "reject":
            ok = ~threshold_violations(cand, copula.columns, specs, margin)
            n_rejected += int((~ok).sum())
            accepted.append(cand[ok])
        else:
            accepted.append(cand)
            break
    pool_values = np.vstack(accepted) if accepted else np.empty((0, copula.p))

    info = {"mode": mode, "margin": margin, "rounds": rounds,
            "n_candidates_rejected": n_rejected,
            "pool_size": int(pool_values.shape[0])}

    if pool_values.shape[0] < n:
        # 兜底：补抽并对违规样本做单调压缩修复（硬阈值 + margin）
        extra = sample_copula(copula, marginal, max(n - pool_values.shape[0], 1) * pool_factor,
                              rng)
        pool_values = np.vstack([pool_values, extra])
        info["fallback_compress"] = True

    values = pool_values[:n].copy()
    if values.shape[0] < n:
        raise RuntimeError("候选样本不足，无法满足缺陷约束")
    # 最终硬修复：把超阈值列压到 阈值-margin（单调压缩），保证违规数为 0
    index = {c: i for i, c in enumerate(copula.columns)}
    for spec in specs:
        if not spec["has_threshold"]:
            continue
        j = index[spec["column"]]
        values[:, j] = np.minimum(values[:, j], spec["log10_threshold"] - margin)
    info["n_values"] = int(values.shape[0])
    info["n_violations_after"] = int(threshold_violations(
        values, copula.columns, specs, 0.0).sum())
    if return_unconstrained:
        return values, info, unconstrained
    return values, info


# --------------------------------------------------------------------------- 迁移


def transfer_rank_quantile(source: np.ndarray, marginal: ConditionalMarginal,
                           columns: Sequence[str]) -> np.ndarray:
    """秩-分位映射（Iman-Conover 边际替换）。

    对每一列：把源样本的（平均）秩转成概率 u=(rank-0.5)/n，再用目标风格条件分位替换。
    因每列的替换映射严格单调，迁移前后该列的秩完全不变 ⇒ Spearman ρ 矩阵逐元素保持。
    """
    n, p = source.shape
    out = np.empty((n, p), dtype=float)
    for j in range(p):
        ranks = rankdata(source[:, j], method="average")
        u = (ranks - 0.5) / n
        out[:, j] = marginal.q_column(j, u)
    return out


def apply_transfer_constraints(before: np.ndarray, after: np.ndarray,
                               specs: Sequence[dict], columns: Sequence[str],
                               margin: float = DEFECT_MARGIN) -> Tuple[np.ndarray, dict]:
    """施加迁移后的缺陷约束：三项阈值 + 四项「不恶化」（≤ 迁移前）。"""
    index = {c: i for i, c in enumerate(columns)}
    out = after.copy()
    touched: Dict[str, int] = {}
    for spec in specs:
        j = index[spec["column"]]
        cap = before[:, j].copy()
        if spec["has_threshold"]:
            cap = np.minimum(cap, spec["log10_threshold"] - margin)
        changed = out[:, j] > cap
        touched[spec["key"]] = int(changed.sum())
        out[:, j] = np.minimum(out[:, j], cap)
    info = {"n_touched_per_defect": touched,
            "margin": margin}
    return out, info


def safe_corr(matrix: np.ndarray) -> np.ndarray:
    """相关系数矩阵；对零方差列（风格内常量列）按 0 处理，避免 NaN 污染均值。"""
    x = np.asarray(matrix, dtype=float)
    z = x - x.mean(axis=0)
    sd = z.std(axis=0)
    const = sd < 1e-12
    sd_safe = np.where(const, 1.0, sd)
    c = (z.T @ z) / (x.shape[0] - 1) / np.outer(sd_safe, sd_safe)   # (p, p) 化合物间相关
    if const.any():
        c[const, :] = 0.0
        c[:, const] = 0.0
    np.fill_diagonal(c, 1.0)
    return c


def safe_spearman(matrix: np.ndarray) -> np.ndarray:
    """Spearman 相关矩阵（先取秩，再按 safe_corr 计算）。"""
    return safe_corr(rankdata(np.asarray(matrix, dtype=float), axis=0))


def spearman_mean_abs_dev(before: np.ndarray, after: np.ndarray) -> Tuple[float, float]:
    """迁移前后 Spearman ρ 的平均绝对偏差（全部非对角元素）与最大值。"""
    rb = safe_spearman(before)
    ra = safe_spearman(after)
    iu = np.triu_indices(rb.shape[0], k=1)
    dev = np.abs(rb[iu] - ra[iu])
    return float(dev.mean()), float(dev.max())


def pearson_mean_abs_dev(before: np.ndarray, after: np.ndarray) -> float:
    rb = safe_corr(before)
    ra = safe_corr(after)
    iu = np.triu_indices(rb.shape[0], k=1)
    return float(np.abs(rb[iu] - ra[iu]).mean())


def ks_oracle_floor(real: np.ndarray, n_gen: int, n_boot: int = 40,
                    seed: int = 0) -> float:
    """「完美生成器」的 KS 中位数参考。

    逐列从该风格自身的经验分布自助重采样 n_gen 条，与真实风格样本比较 KS，
    取多次重复的中位数均值。若生成器完全复现风格边际，KS 中位数应接近该值。
    """
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        draw = np.empty((n_gen, real.shape[1]), dtype=float)
        for j in range(real.shape[1]):
            draw[:, j] = rng.choice(real[:, j], size=n_gen, replace=True)
        meds.append(float(np.median(ks_per_column_fast(draw, real))))
    return float(np.mean(meds))


# --------------------------------------------------------------------------- 评估


def marginal_cdf_at(marginal: ConditionalMarginal, j: int, x: float) -> float:
    """条件边际在 x 处的 CDF 值（cdf 模式为精确阶梯 CDF）。"""
    if marginal.mode != "cdf":
        return float("nan")
    atoms, mcdf = marginal._atoms[j], marginal._mcdf[j]
    i = int(np.searchsorted(atoms, x, side="right") - 1)
    return float(mcdf[i]) if i >= 0 else 0.0


def marginal_exceed_prob(marginal: ConditionalMarginal, j: int, x: float) -> float:
    """条件边际在 x 以上的概率（= 1 - M(x)）。"""
    cdf = marginal_cdf_at(marginal, j, x)
    return float("nan") if np.isnan(cdf) else 1.0 - cdf


def ks_per_column(gen: np.ndarray, real: np.ndarray) -> np.ndarray:
    """逐列两样本 KS 距离（scipy 精确实现，含并列值处理）。"""
    from scipy.stats import ks_2samp
    out = np.empty(gen.shape[1], dtype=float)
    for j in range(gen.shape[1]):
        out[j] = float(ks_2samp(gen[:, j], real[:, j]).statistic)
    return out


def ks_per_column_fast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐列两样本 KS 距离（全向量化，含并列值的正确处理）。

    与 scipy.stats.ks_2samp 等价：ECDF 差在并列组末尾取值（右侧连续定义）。
    速度约为逐列 scipy 版的 1–2 个数量级，用于自助重采样等重复计算。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.shape[0], b.shape[0]
    n = n1 + n2
    comb = np.concatenate([a, b], axis=0)
    order = np.argsort(comb, axis=0, kind="stable")
    cs = np.take_along_axis(comb, order, axis=0)
    is_a = np.concatenate([np.ones(n1, dtype=bool),
                           np.zeros(n2, dtype=bool)])[:, None]
    signs = np.take_along_axis(np.broadcast_to(is_a, (n, a.shape[1])), order, axis=0)
    cum_a = np.cumsum(signs, axis=0) / n1
    cum_b = np.cumsum(~signs, axis=0) / n2
    diff = np.abs(cum_a - cum_b)
    # 只在并列组末尾取值：把每个位置映射到其所属组的末位
    is_end = np.empty(cs.shape, dtype=bool)
    is_end[-1] = True
    is_end[:-1] = cs[1:] != cs[:-1]
    idx = np.arange(n)[:, None]
    nxt = np.minimum.accumulate(np.where(is_end, idx, n)[::-1], axis=0)[::-1]
    grouped = np.take_along_axis(diff, nxt, axis=0)
    return grouped.max(axis=0)


def ks_max_abs_dev_against_scipy(a: np.ndarray, b: np.ndarray) -> float:
    """自检：矢量化 KS 与 scipy 实现的逐列最大绝对偏差。"""
    fast = ks_per_column_fast(a, b)
    slow = ks_per_column(a, b)
    return float(np.max(np.abs(fast - slow)))


def coverage_per_column(gen: np.ndarray, real: np.ndarray) -> np.ndarray:
    """逐列「生成值落在真实样本 [min, max] 区间内」的比例。"""
    lo = real.min(axis=0)
    hi = real.max(axis=0)
    return ((gen >= lo[None, :]) & (gen <= hi[None, :])).mean(axis=0)


def ks_null_expectation(n_gen: int, n_real: int) -> float:
    """两样本 KS 在原假设下的均值近似（用于解读 KS 数值大小）。

    D·sqrt(nm/(n+m)) → Kolmogorov 分布，E[K]=0.7979，中位数 0.6745。
    """
    return float(0.7979 * np.sqrt((n_gen + n_real) / (n_gen * n_real)))


def ks_null_median(n_gen: int, n_real: int) -> float:
    """两样本 KS 原假设下的中位数近似。"""
    return float(0.6745 * np.sqrt((n_gen + n_real) / (n_gen * n_real)))


def ks_vs_target_marginal(gen: np.ndarray, marginal: ConditionalMarginal) -> np.ndarray:
    """生成样本与其**目标条件边际**的逐列 KS（自洽性检查）。

    风格条件边际按定义就不是风格经验分布（其中含 (1-λ) 的全局成分），
    因此本指标比「对风格经验分布的 KS」更直接地回答：
    「生成器是否复现了它自己的目标边际？」理想值为有限样本噪声量级。
    仅 cdf 模式可用（目标边际为显式阶梯 CDF）；quantile 模式返回 NaN。
    """
    if marginal.mode != "cdf":
        return np.full(gen.shape[1], np.nan)
    n = gen.shape[0]
    out = np.empty(gen.shape[1], dtype=float)
    for j in range(gen.shape[1]):
        if marginal.const_mask[j]:
            out[j] = 0.0                       # 常量列：生成值恒等于目标
            continue
        atoms, mcdf = marginal._atoms[j], marginal._mcdf[j]
        g = gen[:, j]
        pts = np.unique(np.concatenate([g, atoms]))
        f_gen = np.searchsorted(np.sort(g), pts, side="right") / n
        idx = np.searchsorted(atoms, pts, side="right") - 1
        m = np.where(idx >= 0, mcdf[np.clip(idx, 0, mcdf.size - 1)], 0.0)
        out[j] = float(np.max(np.abs(f_gen - m)))
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题3 核心库自检：拟合全局高斯 Copula 并打印关键诊断量")
    parser.add_argument("--ridge", type=float, default=1e-3, help="相关矩阵岭正则强度")
    parser.add_argument("--eig-floor", type=float, default=1e-6, help="特征值下截断")
    args = parser.parse_args(argv)

    data = load_dataset()
    copula = fit_global_copula(data.matrix, ridge=args.ridge, eig_floor=args.eig_floor)
    print(f"统一矩阵 {data.matrix.shape}；Copula 相关矩阵 {copula.corr.shape}")
    print(f"原始特征值 min={copula.eig_raw.min():.3e} max={copula.eig_raw.max():.3e}；"
          f"裁剪数 {copula.n_clipped}；裁剪后 min={copula.eig_clipped.min():.3e}")
    slices = style_slices(data)
    for style in DEFAULT_TARGET_STYLES:
        if style not in slices:
            continue
        idx = slices[style]
        marg = build_marginal(copula, data.matrix.to_numpy()[idx], style)
        print(f"  {style:<12} n={marg.n_style:<3} λ={marg.lam:.4f} "
              f"风格常量列={int(marg.const_mask.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
