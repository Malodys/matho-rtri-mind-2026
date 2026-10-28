# 问题二 · 化合物–风味分析：共表达图与各感官维度 Top-3 关键化合物

> **AI 辅助声明**：本目录全部脚本由 MiniMax Code（MiniMax-M3.1-Flash，MiniMax，2026-10-28）辅助生成，
> 脚本首部均带 AI 辅助标注块；关键结论（相关类型裁定、图规模、Top-3 数值、稳定性统计）均由
> `scripts/q2_verify.py` 实测输出核对，未做形式化声明。

---

## 0. 结论速览

| 指标 | 实测值 |
|---|---|
| 共表达图规模（\|ρ\| ≥ 0.6） | **231 节点 / 266 边 / 114 连通分量 / 120 个 Leiden 社区**（非平凡社区 19 个，最大社区 21 节点） |
| 模块度 Q（0.6 档） | **0.7455**；化学纯度（社区内来源众数占比）**0.6615** |
| 孤立节点（0.6 档） | **101 / 231 = 43.72%**（0.5 档降至 48 / 231 = 20.78%） |
| 四路证据覆盖 | E1=44 化合物 / E2=**227** / E3=**227** / E4=231 |
| Top-3 表 | **50 维 × 3 = 150 行，64 个不重复化合物，NaN = 0，全部可映射回 1.csv 列** |
| bootstrap 稳定性 | **B = 200**，频率步长 0.005，取值域 [0, 1]，全部有限 |
| 判为「稳健关键因素」 | **32 / 150 条目，22 / 50 个维度**（入选频率 ≥ 0.6 且跨方法平均 Jaccard ≥ 0.3） |
| 跨方法一致性 | E1–E2 **0.546**，E1–E3 **0.198**，E2–E3 **0.224**，三路平均 **0.3227** |
| 化学合理性 | 强证据 **15 条 / 14 个维度**；强+中 **36 条 / 23 个维度**（要求 ≥ 11 个维度） |
| 与 E4（原论文 MDI 先验）重合 | 平均 **0.30** 个/维，最大 2 个，12 个维度有 ≥ 1 重合 |

---

## 1. 方法定义式

### 1.1 数据入口
全部取数经 `scripts/data_contract.py`（唯一允许读 `data/raw/` 的模块）：

- `load_raw()` → 1.csv / 2.csv / 5.csv / 6.csv / 8.xlsx 各表
- `build_bridges()` → 化学桥 275/275、感官桥 100/100 全部解析
- `load_modeling_matrix()` → **250 × 227** 统一 `log10(mg/L)` 矩阵（已剔除 4 个未验证的 `*_sum` 派生列）
- `load_targets()` → **250 × 50** 标准化感官目标
- `load_provenance()` → 逐列溯源（block / unit / scale_action）
- `defect_table()` / `defect_exceedance()` → 缺陷阈值与超标率

### 1.2 相关类型勘误（论文必须显式声明）

**2.csv / 5.csv / 6.csv 全部是 Spearman，不是 Pearson。** 复算证据：

- 5.csv、6.csv 相对重算 Spearman：`max|Δ| = 0`、`frac(<1e-8) = 1.0000`；Pearson 下 `median|Δ| ≈ 0.04`。
- 2.csv 相对**1.csv 原始存储标度**重算 Spearman：`max|Δ| = 0`（2.csv 未做单位统一，直接在文件 1 的存储标度上计算）。

官方 PDF 表述为 "Pearson"，与《数据说明.txt》及上述复算矛盾，**以 Spearman 为准并在论文中勘误**。

### 1.3 共表达图

- 边：`abs(rho) >= 0.6`（默认档），敏感性扫描 0.5 / 0.6 / 0.7 / 0.8。
- 节点属性：`origin_paper`（8.xlsx Sensory thresholds → `field_name_R`）、`block`（provenance）、
  `unit`、`is_defect` 与缺陷阈值、`degree`、`community`。
- 社区发现：`igraph` `community_leiden(objective_function="modularity", n_iterations=5)`，
  种子 20261028。**231 节点 / 266 边 / 114 连通分量 / 社区规模 [21,17,16,15,15]，与
  `data/processed/smoke/stack_smoke.txt` 的 V8 记录逐项一致。**

### 1.4 四路证据

| 证据路 | 方法 | 覆盖 | 角色 |
|---|---|---|---|
| E1 | 6.csv 给出的 Spearman 交叉相关 | 44 | **交叉验证**，不进主排序 |
| E2 | 自算全量 Spearman：`load_modeling_matrix()` × `load_targets()` | 227 | **主排序** |
| E3 | GBDT 归因 + SHAP（LightGBM 逐维度拟合，`mean|SHAP|`） | 227 | **主排序** |
| E4 | 8.xlsx `GBM importance MDI`（原论文模型先验） | 231 | **仅外部印证**，不进主排序 |

> E4 只作外部印证的理由：MDI 是原论文模型的**全局**重要性先验，不区分感官维度；
> 把它并入主排序等于把别人的结论当作自己的证据。

**E1 ↔ E2 一致性核查**：在 1.csv 原始存储标度上重算 6.csv 的 2200 个交叉相关，
`max|Δ| = 8.88e-16`，**2199 / 2200** 个单元 < 1e-8。唯一例外是 **Diacetyl**
（`X2.3.butanedione`，1.csv 有 **9 个缺失值**，统一矩阵做了插补），其 ρ 偏移 ≤ 0.047。
E2 用统一矩阵，故仅对 Diacetyl 存在可解释的偏差。

### 1.5 融合规则

E2 与 E3 各自**按维度做秩归一化**（`(rank-1)/(n-1)`，值域 [0,1]）后加权求和：

```
fused_score(d, c) = 0.5 * normrank_E2(d, c) + 0.5 * normrank_E3(d, c)
```

权重取等权的理由：(a) E2 是题面点名的关联性度量，可解释为「与感官评分的单调关联强度」；
(b) E3 捕捉非线性与交互，但 SHAP 重要度无量纲、随模型设定漂移；(c) 等权避免任一路因量纲或超参主导。
排序按关联强度（E2 取 |ρ|、E3 取 mean|SHAP|），**方向单列**（正 / 负 / 无方向），方向取 E2 的 ρ 符号。

### 1.6 稳定性

- **bootstrap**：对 seed_0 的 **175 瓶训练集**有放回重采样 B = **200** 次，每次重算 E2（全量 Spearman）
  与 E3（GBDT + SHAP）并按同一融合规则重排，记录每个化合物进入 Top-3 的频率。
  每次迭代的 RNG 由 `[seed=20261028, b]` 派生，**分块运行与一次性运行结果逐元素一致**
  （chunk=1 与 chunk=4 的计数 `max|Δ| = 0`）。
- **跨方法 Jaccard**：E1 / E2 / E3 三路各自的 Top-3 集合两两求 Jaccard，取平均作「方法间一致性」。
- **入选判据**：`bootstrap 频率 ≥ 0.6` **且** `跨方法平均 Jaccard ≥ 0.3` → **稳健关键因素**；
  否则 → **候选（需谨慎）**。

### 1.7 化学合理性论证（四条判据，逐条落到 `top3_by_sensory.csv`）

1. `origin_paper` 归属是否与该维度的已知化学来源一致；
2. 是否属于该感官维度的**已知化学载体类别**（命中教科书级载体记「强」，仅类别一致记「中」，
   仅来源/表征一致记「弱」）；
3. 与 8.xlsx `off_flavors` / `aging` / 缺陷阈值（统一矩阵 log10(mg/L) 口径与超标率）的一致性；
4. 与 E4 先验的一致性（MDI 分位 + 原论文 SHAP 值的符号方向）。

> **重要约束**：同时出现在特征侧与感官侧的量（`bitterness`/异构α酸、`X4VG`、`DMS`、
> `X2.3.butanedione`、`ethanol..v.v.`、`lactic_acid.mg.L.`、`acetic_acid.g.L.`、`iron`、`pH`、
> `CO2.PSI.`、`protein.g.L.`、`kcalperc`、`unfermentables` 等，共 **50 条** Top-3 记录命中）
> 一律加注「理化测量与感官评分的一致性」，**不得表述为新发现的关联**。

---

## 2. 共表达图阈值择优（量化依据）

| \|ρ\| 阈值 | 边数 | 连通分量 | Leiden 社区 | 非平凡社区 | 最大社区 | 孤立节点 | 孤立占比 | 模块度 Q | 化学纯度 | 平均边 \|ρ\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 645 | 52 | 59 | 11 | 33 | 48 | 20.78% | 0.6513 | 0.5628 | 0.6077 |
| **0.6（默认档）** | **266** | **114** | **120** | **19** | **21** | **101** | **43.72%** | **0.7455** | **0.6615** | 0.7016 |
| 0.7 | 109 | 169 | 171 | 21 | 11 | 150 | 64.94% | 0.7507 | 0.7778 | 0.7884 |
| 0.8 | 39 | 207 | 208 | 14 | 7 | 194 | 83.98% | 0.6824 | 0.8108 | 0.8619 |

跨阈值社区一致性（ARI，标签置换不变）：

| | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|
| **0.5** | 1.000 | **0.4381** | 0.1677 | 0.0385 |
| **0.6** | 0.4381 | 1.000 | **0.3909** | 0.0997 |
| **0.7** | 0.1677 | 0.3909 | 1.000 | 0.3262 |
| **0.8** | 0.0385 | 0.0997 | 0.3262 | 1.000 |

### 推荐与理由（规则可复算，见 `graph/threshold_recommendation.json`）

- **作业默认档 = |ρ| ≥ 0.6**。规则：「孤立节点占比 ≤ 50% 且 非平凡社区数 ≥ 15，在可行档中取模块度 Q 最高者」。
  0.6 档是唯一同时满足两约束的档位：模块度 Q = 0.7455（社区划分质量高）、化学纯度 0.6615、
  非平凡社区 19 个（最多的可用社区结构），且与作业预置的 V8 冒烟记录（266 边 / 114 分量）一致。
  **全部社区 × `block` / `origin_paper` 交叉表与化学合理性论证均基于此档。**
- **连通性档 = |ρ| ≥ 0.5**。规则：「孤立节点占比最低者」。边数 266 → 645、孤立节点占比 43.72% → 20.78%，
  网络拓扑更完整；但模块度（0.6513）与化学纯度（0.5628）均低于 0.6 档，说明放宽阈值会把化学性质不同的
  化合物并入同一社区。与 0.6 档的社区划分 **ARI = 0.4381**，主体一致但非等价。
- **正文并列报告**：拓扑与连通性结论用 0.5 档，社区化学构成与合理性论证用 0.6 档。

### 社区化学构成的实证（非平凡社区，均基于 0.6 档）

| 社区 | 节点数 | 主导 origin_paper | 占比 | 典型成员 |
|---|---|---|---|---|
| C28 | 21 | Yeast | 57.1% | 麦芽吡嗪、乙酰呋喃、异麦芽酚、高级醇 |
| C39 | 17 | **Hops** | **88.2%** | α/β-律草烯、卡德醇、杜松醇、石竹烯氧化物 |
| C5 | 16 | Yeast | 75.0% | 长链脂肪酸酯 + 法尼烯（酵母酯 + 酒花倍半萜） |
| C0 | 15 | Yeast | 86.7% | 乙酸乙酯、己酸乙酯、辛酸乙酯、乙酸异戊酯、苯乙酸乙酯 |
| C4 | 15 | Yeast | 33.3% | 乙基愈创木酚、甲基糠醛、香豆酸乙酯、β-杜松醇 |
| C47 | 12 | Hops | 58.3% | **柠檬烯、芳樟醇、α-萜品醇、莰烯、桉叶素、四氢月桂烯** |
| C3 | 3 | Malt | 33.3% | **DMS + H₂S + 硫总量**（含缺陷化合物） |
| C61 | 3 | Yeast | 33.3% | **E-2-壬烯醛 + β-萜品醇**（含缺陷化合物） |

**化学合理性成立**：萜烯社区落 Hops（C39、C47）、酯类落 Yeast（C0、C5）、硫化物落 Malt（C3）、
氧化醛类自成一社区（C61）——与题面预期一致，是化学合理性论证的主证据。

---

## 3. 各感官维度 Top-3（代表性维度）

> `ρ` 为 E2 自算全量 Spearman；`E1` 为 6.csv 给出的值（`NA` = 6.csv 未覆盖该化合物）；
> `freq` 为 bootstrap 入选频率（B=200）；`J` 为该维度三路 Top-3 的平均两两 Jaccard；
> `cls` 为化学合理性等级（强 / 中 / 弱）。

| 感官维度 | Top-3（化合物，方向，ρ，E1，freq，J，cls） | 判定 |
|---|---|---|
| `bitternes` | `bitterness`(正, +0.682, 0.682, 0.990, 0.667, **强**) ; `ethyl_octanoate`(正, +0.593, 0.593, 0.970, 0.667, 弱) ; `X.E..2.nonenal`(正, +0.503, NA, 0.060, 0.667, 弱) | 稳健 |
| `alcohol` | `ethanol..v.v.`(正, +0.817, 0.817, **1.000**, 0.667, **强**) ; `kcalperc`(正, +0.730, 0.730, 0.980, 0.667, 弱) ; `ethyl_octanoate`(正, +0.619, 0.619, 0.550, 0.667, 弱) | 稳健 |
| `X4vg` | `X4VG`(正, +0.473, 0.473, **0.970**, 0.300, **强**) ; `delta.terpinyl.acetate`(正, +0.399, NA, 0.435, 0.300, 弱) ; `X4.ethyl.guaiacol`(负, −0.359, NA, 0.250, 0.300, 中) | 稳健 |
| `A_hops_citrus` | `CO2.PSI.`(正, +0.329, 0.329, 0.400, 0.300, 弱) ; `citronellol`(正, +0.370, 0.370, 0.255, 0.300, **强**) ; `nerol.oxide`(正, +0.271, NA, 0.445, 0.300, 中) | 候选 |
| `A_esters_isoaa` | `n.hexanol`(正, +0.340, NA, 0.805, 0.233, 中) ; `isopentyl_acetate`(正, +0.283, 0.283, 0.360, 0.233, **强**) ; `tau.cadinol`(负, −0.209, NA, 0.115, 0.233, 弱) | 候选 |
| `A_malt_bread` | `color.EBC.`(正, +0.329, 0.329, 0.525, 0.333, 弱) ; `acetaldehyde`(正, +0.232, 0.232, 0.255, 0.333, 弱) ; `ethyl.tiglate`(正, +0.238, NA, 0.250, 0.333, 弱) | 候选 |
| `dms` | `protein.g.L.`(负, −0.322, −0.322, 0.710, 0.300, 弱) ; `E.nerolidol`(正, +0.194, NA, 0.325, 0.300, 弱) ; `color.EBC.`(负, −0.248, −0.248, 0.275, 0.300, 弱) | 稳健（但化学论证弱，见局限） |
| `stale_hops` | `alloaromadendrene.epoxide`(正, +0.278, NA, 0.465, 0.067, 弱) ; `X2.undecanol`(正, +0.300, NA, 0.285, 0.067, 弱) ; `gamma.eudesmol`(正, +0.247, NA, 0.270, 0.067, 弱) | 候选 |
| `overall` | `ethyl_octanoate`(正, +0.286, 0.286, 0.265, 0.467, 弱) ; `unfermentables`(正, +0.226, NA, 0.300, 0.467, 弱) ; `DMS`(正, +0.227, 0.227, 0.225, 0.467, 弱) | 候选 |

**「强」证据且稳健的 9 条**（同时满足频率 ≥ 0.6、Jaccard ≥ 0.3、命中教科书级载体）：

| 维度 | 化合物 | ρ | freq | 化学载体说明 |
|---|---|---|---|---|
| `alcohol` | `ethanol..v.v.` | +0.817 | 1.000 | 酵母发酵产物，酒精感直接载体（理化测量与感官评分一致性） |
| `bitternes` | `bitterness`（异构α酸, IBU） | +0.682 | 0.990 | 酒花异构α酸，苦味唯一主载体（一致性） |
| `body` | `kcalperc` | +0.687 | 0.995 | 热值/酒体理化量（一致性） |
| `F_esters_ethac` | `ethyl_acetate` | +0.503 | 0.965 | 短链脂肪酸酯，指甲油型酯香载体 |
| `X4vg` | `X4VG` | +0.473 | 0.970 | 4-乙烯基愈创木酚，papery/辛香缺陷载体（一致性） |
| `sweetness` | `GFS`（糖） | +0.373 | 0.935 | 糖类，甜味直接载体 |
| `co2` | `CO2.PSI.` | +0.461 | 0.755 | CO₂ 压力，杀口感直接载体（一致性） |
| `lactic` | `lactic_acid.mg.L.` | +0.448 | 0.760 | 乳酸，乳酸味直接载体（一致性） |
| `acetic` | `acetic_acid.g.L.` | +0.354 | 0.620 | 乙酸，醋酸味直接载体（一致性） |

> 注意：`bitternes`/`alcohol`/`X4vg`/`co2`/`lactic`/`acetic`/`body` 等的「强」证据本质上是
> **理化测量与感官评分的一致性**（同一物理量的两种读数），`top3_by_sensory.csv` 中已逐条加注，
> 论文中不得表述为「新发现的关联」。

---

## 4. 稳定性结果

### 4.1 B = 200（正式结果，`summary.json` → `stability.bootstrap_B = 200`）

- bootstrap 入选频率取值域 **[0, 1]**，最小非零步长 **0.005 = 1/200**，全部有限。
- **稳健关键因素：32 / 150 条目，22 / 50 个维度**（同时满足 freq ≥ 0.6 与平均 Jaccard ≥ 0.3）。
- 其中「稳健 + 强化学证据」共 **9 条**。
- 跨方法平均 Jaccard：E1–E2 **0.5460**、E1–E3 **0.1980**、E2–E3 **0.2240**，三路平均 **0.3227**。

22 个稳健维度：`A_malt_all`、`A_malt_burn`、`A_malt_cara`、`A_hops_all`、`A_hops_noble`、
`A_esters_all`、`F_malt_all`、`F_malt_burn`、`F_malt_cara`、`F_hops_all`、`F_hops_noble`、
`F_esters_all`、`F_esters_ethac`、`bitternes`、`alcohol`、`body`、`co2`、`sweetness`、`X4vg`、
`lactic`、`acetic`、`dms`。

### 4.2 B 敏感性（稳健性信息）

| B | 稳健条目 / 150 | 稳健维度 / 50 | 平均 Jaccard | E1–E3 | E2–E3 |
|---|---|---|---|---|---|
| 8（试跑） | 35 | 21 | 0.3227 | 0.198 | 0.224 |
| **200（正式）** | **32** | **22** | **0.3227** | **0.198** | **0.224** |

结论：判定结果对 B **不敏感**——稳健维度数 21 → 22（+1），稳健条目数 35 → 32（−3），
而 Jaccard 完全不变（它由全样本点估计的 E1/E2/E3 排名算出，与 B 无关）。
B=8 时「频率 ≥ 0.6」等价于「至少 5/8」，统计强度不足，故正式结果一律以 B=200 为准。

### 4.3 与 E4（原论文先验）的重合

8.xlsx `GBM importance MDI` 的全局 Top-3 为 `ethyl_acetate`、`ethanol..v.v.`、`ethyl_octanoate`。
与本方案各维度 Top-3 的平均重合 **0.30 个/维**，最大 2 个，**12 个维度**有 ≥ 1 重合。
低重合是预期的：MDI 是原论文模型的全局重要性，不区分感官维度，因此只能作外部印证。

---

## 5. 产出清单

### 脚本（`scripts/`，均带 `--help`）

| 文件 | 作用 |
|---|---|
| `q2_common.py` | 共享配置（阈值、权重、超参）与工具（Spearman、秩归一化、Jaccard） |
| `q2_chem_kb.py` | 化学知识库：化合物→化学类别、维度→已知化学载体 |
| `q2_graph.py` | Step 1：共表达图、Leiden 社区、阈值择优、四档敏感性 |
| `q2_evidence.py` | Step 2：E1/E2/E3/E4 四路证据与融合主排序 |
| `q2_stability.py` | Step 3：bootstrap、Jaccard、化学合理性论证、汇总 |
| `q2_figures.py` | 出图（6 张） |
| `q2_verify.py` | 验收自检（逐条核对题面验收标准） |
| `q2_explore*.py`、`q2_report_peek.py` | 只读探查与摘录脚本（不产出交付物） |

### 数据（`data/processed/q2_runs/`）

```
top3_by_sensory.csv        150 行 × 31 列：50 维 × Top-3 × 四路证据 + 方向 + 稳定性 + 合理性标注
stability.csv              bootstrap 频率明细 + 每维度三路两两 Jaccard + 入选状态
bootstrap_top3_freq.csv    (维度 × 化合物) 长表，仅含频率 > 0 的条目
jaccard_summary.csv        每维度 Top-3 集合与三路两两 Jaccard
jaccard_matrix.csv         跨方法 Jaccard 汇总（与 jaccard_summary 同源，便于直接引用）
summary.json               图规模 / 阈值敏感性 / 稳健维度计数 / E4 重合 / 全部超参
summary_B8_sensitivity.json  B=8 试跑结果副本（B 敏感性对照用）
acceptance_q2.txt          验收自检报告
README.md                  本文件
graph/
  coexpression_graph.graphml / .json   231 节点 + 266 边 + 全属性
  communities.csv          节点级属性（来源/block/单位/缺陷/度数/社区）
  community_composition.csv  社区 × block / × origin_paper 交叉表（0.6 档）
  community_composition_0p5.csv  同上（0.5 连通性档）
  edge_sensitivity.csv     四档敏感性表
  threshold_community_ari.csv  跨阈值社区一致性（ARI）
  threshold_recommendation.json 阈值择优规则与结论
evidence/
  e1_cross_rho_6csv.csv / e2_spearman_full.csv / e3_shap_importance.csv / e4_gbm_mdi.csv
  evidence_scores.csv      (维度 × 化合物) 长表，含四路证据分值、排名与融合得分
  evidence_coverage.json   四路证据覆盖数
  bootstrap_counts_part{0..3}_of4.npz  4×50 次 bootstrap 的计数（可续跑/可复算）
```

### 图（`data/processed/q2_figures/`）

| 文件 | 内容 |
|---|---|
| `fig1_coexpression_network_by_origin.png` | 共表达图，按 `origin_paper` 着色（外圈为孤立节点） |
| `fig2_coexpression_network_by_community.png` | 共表达图，按 Leiden 社区着色 |
| `fig3_community_composition_heatmap.png` | 社区 × block 热图 + 社区 × origin_paper 热图 |
| `fig4_top3_bootstrap_frequency.png` | 50 维 Top-3 的 bootstrap 入选频率条形图（B=200，标出 0.6 阈值线） |
| `fig5_method_consistency_jaccard.png` | E1–E3 平均 Jaccard 热图 + 逐维度一致性 |
| `fig6_threshold_selection.png` | 阈值择优：模块度/化学纯度/孤立占比 + 跨阈值 ARI |

---

## 6. 复现命令

```powershell
# 0) 环境（已就绪，勿改 .venv 与 requirements.lock.txt）
.\.venv\Scripts\python.exe -V          # Python 3.12.10

# 1) Step 1：共表达图 + Leiden 社区 + 阈值择优
.\.venv\Scripts\python.exe scripts\q2_graph.py

# 2) Step 2：四路证据 + 融合主排序
.\.venv\Scripts\python.exe scripts\q2_evidence.py

# 3) Step 3：bootstrap（B=200，分 4 块；分块与一次性结果逐元素一致）
.\.venv\Scripts\python.exe scripts\q2_stability.py --chunk 0 --nchunks 4 --bootstrap-b 200
.\.venv\Scripts\python.exe scripts\q2_stability.py --chunk 1 --nchunks 4 --bootstrap-b 200
.\.venv\Scripts\python.exe scripts\q2_stability.py --chunk 2 --nchunks 4 --bootstrap-b 200
.\.venv\Scripts\python.exe scripts\q2_stability.py --chunk 3 --nchunks 4 --bootstrap-b 200
.\.venv\Scripts\python.exe scripts\q2_stability.py --aggregate --bootstrap-b 200 --nchunks 4

# 4) 出图
.\.venv\Scripts\python.exe scripts\q2_figures.py

# 5) 验收自检（退出码 0 = 全部通过）
.\.venv\Scripts\python.exe scripts\q2_verify.py
```

单块 bootstrap 约 2.7–3.8 分钟（8 线程）；全流程约 15 分钟。
所有脚本支持 `--help`；`data/raw/` 全程只读。

---

## 7. 已知局限

1. **相关类型勘误**：官方 PDF 称 2.csv / 5.csv / 6.csv 为 Pearson，实测均为 Spearman；
   且 2.csv 未做单位统一（直接在 1.csv 原始存储标度上计算）。论文须显式勘误。
2. **E1 覆盖不足**：6.csv 只覆盖 44 / 231 个化合物，只能作交叉验证，不能支撑主排序。
3. **E1 与 E2 的唯一系统性偏差在 Diacetyl**：1.csv 有 9 个缺失值，统一矩阵插补后其 ρ 偏移 ≤ 0.047
   （其余 2199 / 2200 单元 `max|Δ| = 8.88e-16`）。
4. **方法间一致性本身偏弱**：E1–E3 平均 Jaccard 仅 0.198、E2–E3 仅 0.224，均低于 0.3 阈值。
   SHAP 归因排序与相关排序的一致性有限，因此 50 个维度中只有 22 个达到「稳健关键因素」——
   这是真实结果而非失败，应在论文中如实报告。
5. **麦芽类子维度缺少可定量的香气载体**：`A_malt_grain` / `A_malt_bread` / `A_malt_cara` 的 Top-3
   常落在色度（`color.EBC.`）、醛类等表征变量上，|ρ| ≤ 0.33。根因是麦芽吡嗪/呋喃类在 1.csv 中
   多为 **GC-MS 相对丰度（无绝对校准）**，无法建立定量因果链。`A_malt_grain` 的 Top-3 甚至落到
   酒花/香料萜烯，属噪声，须标注为候选。
6. **共表达图偏碎**：0.6 档 101 个节点（43.72%）无任何边，这些化合物在该阈值下不携带共表达信息；
   0.5 档降至 48 个（20.78%）。两档社区划分 ARI = 0.4381，主体一致但非等价。
   正文并列报告 0.6（社区化学构成）与 0.5（网络拓扑）。
7. **E4 不可分维度**：MDI 是原论文模型的全局重要性先验；8.xlsx `GBM SHAP Values` 的模型
   也无法唯一对应 50 个维度中的某一个（与 `aftertaste` 的 |r| 最高，仅 0.6513），
   故只用作 E4 的**符号方向**参照，不参与任何排序。
8. **覆盖 227 而非 231**：4 个 `*_sum` 派生列按共享数据契约被剔除，不参与 E2/E3。
9. **`stale_hops` 是 4.csv 的感官维度**，不是化合物；其 Top-3 无已知缺陷载体（真实缺陷载体应为
   E-2-壬烯醛等醛类，但本数据中该维度与醛类的关联未进入 Top-3）。
10. **样本量**：n = 250（训练 175），bootstrap 频率分辨率 1/200 = 0.005；ρ 的抽样误差本身未做校正。
11. **化学合理性是规则化先验判定**，不是因果验证：等级由「化学名/列名关键词分类 + 人工载体清单」得出，
    载体清单本身取自 8.xlsx 的 `origin_paper`/`off_flavors`/`aging` 与通用酿造化学常识；
    「中」与「弱」两级不构成机制性结论。
12. **未做的事**：未做留出集上的 Top-3 泛化验证（Top-3 属描述性归纳，非预测任务）；
    未做 E2/E3 权重的敏感性扫描（等权假设未做扰动检验），论文中应说明这一点。
