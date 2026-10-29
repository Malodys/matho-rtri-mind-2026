# B2 · 消费者层 + 缺陷化合物剂量-响应再检验

生成脚本：`scripts/q4_consumer.py`　｜　随机种子：0　｜　运行耗时：172.8 s

## 0. 边界声明（先读）

- **附件未提供任何消费者评分 / 喜爱度标签**（7.csv 只有 186 个描述符强度，没有 overall liking、
  评分或打分列）。因此本层**不声称**预测「消费者喜爱度」，只做两件事：
  1. **两条感知路径的一致性**：化学 → 品评组感官（50 维）与化学 → 消费者描述符（183 维）
     是否指向同一化学驱动因子；
  2. **外部市场价值的可解释比例**：化学预测感知能解释多少 log 几何均价的差异。
- 183 维消费者描述符上 PCA 第一主成分仅解释 **10.4%** 方差，不存在单一「总体喜爱度」轴；
  本层**不构造**任何「消费者喜爱度」目标变量。
- 价格为**单一零售源附件**（8.xlsx / Beer Prices）的零售商报价，跨币种（USD/EUR/GBP）
  未做汇率换算，**只有 ≥2 家报价的 189 款**进入主回归；价格同时受 ABV 与风格混杂，
  且缺少消费者评分标签，市场价值只能解释为「可解释比例」而非因果效应。

## 1. 数据来源与清洗口径

| 环节 | 来源 | 说明 |
| --- | --- | --- |
| 化学特征 | `data_contract.to_unified_scale()` | 统一 log10(mg/L) 矩阵，231 列 |
| 品评组感官 | `data_contract.load_targets()` | 4.csv 的 50 维标准化感官分数 |
| 消费者描述符 | `data/raw/7.csv` | 250 × 186 描述符 |
| 风格 | `data_contract.load_styles()` | 22 个风格 |
| 切分 | `data/splits/seed_0/split_manifest.csv` | **锁定 175/75（seed 0）**，未重新划分 |
| 价格 | 8.xlsx / Beer Prices | 250 行 × 11 列（ABV + 7 个零售商报价） |
| 感官阈值 | 8.xlsx / Sensory thresholds | `threshold_median` 统一换算为 log10(mg/L) |

### A1 消费者矩阵清洗

- 原始描述符 186 列 → 剔除 **3 个整列为空的列**：`palate_acetaldehyde`、`palate_acetate`、`palate_dentist`（各 250 个 NA）
- 保留 **183 维**；零星 NA 按列中位数填补，实际填补 **0 个单元格**（0 列）
- 索引与 1.csv 的 `beer` 逐行对齐（250 行，顺序一致）

### 化学特征口径：保留还是剔除 `*_sum` 派生列？

`to_unified_scale()` 返回 231 列，其中 4 列是 `esters_sum / aroma_hops_sum / acids_sum /
sulfur_sum`（组内浓度和取 log10 的非线性派生量，与成分列高度共线）。**主口径剔除这 4 列**
（227 列，与 Q1 建模矩阵 `load_modeling_matrix()` 完全一致），231 列作为对照口径：

| 模型 | 227 列（剔除派生）Macro-RMSE | 231 列（保留派生）Macro-RMSE | Δ | Spearman 中位（剔除 / 保留） |
| --- | --- | --- | --- | --- |
| global_mean | 0.8904 | 0.8904 | +0.0000 | nan / nan |
| ridge | 0.8171 | 0.8166 | -0.0006 | 0.3095 / 0.3157 |
| pls2 | 0.8245 | 0.8233 | -0.0012 | 0.3400 / 0.3467 |
| global_extra_trees | 0.7737 | 0.7717 | -0.0020 | 0.3891 / 0.4080 |
| independent_gbr | 0.8027 | 0.7992 | -0.0034 | 0.3121 / 0.3165 |

## 2. 模型与切分

- 切分：**锁定的 175/75**（`split_manifest.csv`，seed 0），基线为**训练折目标均值**。
- 模型族（与 Q1 同名同族）：global_mean, ridge, pls2, global_extra_trees, independent_gbr；`ridge` / `pls2` 用训练折内层 5 折
  GridSearchCV 选参；`global_extra_trees` / `independent_gbr` 直接复用 Q1 在 50 维品评组上
  已选定的超参（只读引用 `data/processed/q1_runs/*/model_best_params.csv`），避免本层二次
  调参、保证两层同口径，也把运行时间控制在 CPU/内存预算内。
- OOF 预测（5 折，`KFold(shuffle=True, random_state=0)`）用于交叉一致性与价格回归，
  避免自证（每个样本的预测都来自未见过它的模型）。
- 用于 A3/A4 的模型族：**global_extra_trees**（消费者留出集 Macro-RMSE 最小者）。

### A2 消费者层模型（锁定 175/75 留出集）

| 模型 | 目标数 | Macro-RMSE | Macro-MAE | 逐维 R² 中位 | Spearman 中位 |
| --- | --- | --- | --- | --- | --- |
| global_mean | 183 | 0.8904 | 0.5416 | -0.0087 | nan |
| ridge | 183 | 0.8171 | 0.4811 | 0.0856 | 0.3095 |
| pls2 | 183 | 0.8245 | 0.4953 | 0.0789 | 0.3400 |
| global_extra_trees | 183 | 0.7737 | 0.4341 | 0.1742 | 0.3891 |
| independent_gbr | 183 | 0.8027 | 0.4614 | 0.0953 | 0.3121 |

- 基线 `global_mean` Macro-RMSE = **0.8904**；
  最优模型 **global_extra_trees** Macro-RMSE = **0.7737**（相对基线 -0.1167），Spearman 中位 0.3891。
- 结论：消费者层模型 Macro-RMSE **优于**全局均值基线。

### A2 同口径对比：品评组 50 维 vs 消费者 183 维

`consumer_vs_panel_models.csv` 同时给出：同一留出集、同一超参下两层各自的结果，
以及 Q1 已发布的 5 折 OOF 结果（只读引用）。

| 模型 | 目标集 | 评估口径 | Macro-RMSE | Macro-MAE | R² 中位 | Spearman 中位 |
| --- | --- | --- | --- | --- | --- | --- |
| global_mean | consumer_183 | locked_holdout_175_75 | 0.8904 | 0.5416 | -0.0087 | nan |
| global_mean | panel_50 | locked_holdout_175_75 | 0.5762 | 0.4205 | -0.0066 | nan |
| ridge | consumer_183 | locked_holdout_175_75 | 0.8171 | 0.4811 | 0.0856 | 0.3095 |
| ridge | panel_50 | locked_holdout_175_75 | 0.4976 | 0.3593 | 0.1766 | 0.3669 |
| pls2 | consumer_183 | locked_holdout_175_75 | 0.8245 | 0.4953 | 0.0789 | 0.3400 |
| pls2 | panel_50 | locked_holdout_175_75 | 0.4915 | 0.3576 | 0.1766 | 0.4033 |
| global_extra_trees | consumer_183 | locked_holdout_175_75 | 0.7737 | 0.4341 | 0.1742 | 0.3891 |
| global_extra_trees | panel_50 | locked_holdout_175_75 | 0.4900 | 0.3492 | 0.1643 | 0.4016 |
| independent_gbr | consumer_183 | locked_holdout_175_75 | 0.8027 | 0.4614 | 0.0953 | 0.3121 |
| independent_gbr | panel_50 | locked_holdout_175_75 | 0.4951 | 0.3543 | 0.1619 | 0.3768 |
| independent_gbr | panel_50 | q1_published_5fold_oof | 0.4880 | 0.3505 | 0.1833 | 0.3819 |
| grouped_extra_trees | panel_50 | q1_published_5fold_oof | 0.4881 | 0.3475 | 0.1883 | 0.4164 |
| global_extra_trees | panel_50 | q1_published_5fold_oof | 0.4893 | 0.3486 | 0.1652 | 0.4112 |
| pls2 | panel_50 | q1_published_5fold_oof | 0.4915 | 0.3576 | 0.1766 | 0.4033 |
| multitask_elastic_net | panel_50 | q1_published_5fold_oof | 0.4929 | 0.3542 | 0.1422 | 0.4143 |
| ridge | panel_50 | q1_published_5fold_oof | 0.4976 | 0.3593 | 0.1766 | 0.3669 |
| global_mean | panel_50 | q1_published_5fold_oof | 0.5762 | 0.4205 | -0.0066 | nan |

**对账**：本脚本在品评组 50 维上的 Macro-RMSE 与 Q1 已发布 `model_summary.csv` 的最大差异 = **0.0071**；其中 global_mean, ridge, pls2 **完全一致**（说明两层确为同一锁定切分、同一评价口径）；`global_extra_trees` / `independent_gbr` 的微小差异来自本层**复用 Q1 已选定的固定超参**而未重跑其内层 GridSearchCV。

**解读**（以最优模型 global_extra_trees 为例）：消费者描述符的**绝对误差更大**（Macro-RMSE 0.7737 vs 品评组 0.4900），但**逐维 R² 中位**（0.1742 vs 0.1643）与 **Spearman 中位**（0.3891 vs 0.4016）相当甚至更高 → 消费者描述符更难预测主要来自**个体化噪声更大**（186 个分散的消费者描述符打分 vs 训练有素的品评组 50 维打分），而不是化学可解释性更差。

## 3. 三张核心表怎么读

### `cross_consistency.csv`（A3）

- `rho_truth` = 品评组感官维度**真值**与消费者描述符**真值**之间的 Spearman（n=250）；
  `rho_pred` = 两者**模型预测**之间的 Spearman（OOF）；`delta = rho_pred − rho_truth`。
- `delta` 接近 0 → 两条感知路径（品评组 vs 消费者）对同一化学驱动因子的指向一致；
  `delta` 明显为负 → 化学模型在消费者路径上的还原度弱于品评组路径。
- **注意 `delta > 0` 的结构性来源**：两条预测列由**同一套 227 维化学特征**、**同一个模型族**
  投影得到，预测值天然共享化学主轴方向，因此 `rho_pred` 往往高于 `rho_truth`；
  `delta > 0` 反映的是「化学结构被还原得比感知更整齐」，**不能**直接读作「两条感知路径
  更一致」。判读应以 `rho_truth` 的符号/量级为主，`delta` 只用于判断模型是否**削弱**了
  原有关系（`delta` 大幅为负时才值得警惕）。
- **词汇表循环性风险（必须声明）**：品评组感官维度与消费者描述符共用同一套啤酒感官
  词汇（citrus / fruity / bitter / sour / stale / cardboard …），二者不是相互独立的测量；
  共享词汇会把「品评组与消费者说同一件事」直接写进 `rho_truth`，因此 `rho_truth` 偏高
  **部分是词汇重叠的产物**，不能当作两条独立证据。`delta`（预测 vs 真值的落差）比
  `rho_truth` 的绝对值更可靠。

| 品评组维度 | 消费者描述符 | 配对依据 | ρ 真值 | ρ 预测 | Δ |
| --- | --- | --- | --- | --- | --- |
| A_hops_citrus | aroma_citrus | 词根一致：柑橘 | +0.395 | +0.708 | +0.313 |
| A_hops_tropical | aroma_tropical | 词根一致：热带果 | +0.263 | +0.740 | +0.476 |
| A_hops_woody | aroma_woody | 词根一致：木质 | -0.001 | +0.198 | +0.199 |
| A_esters_fruity | aroma_fruity | 词根一致：果香 | -0.034 | -0.165 | -0.131 |
| A_esters_ethac | aroma_acetate | 同指乙酸乙酯 | +0.006 | -0.104 | -0.109 |
| A_esters_isoaa | aroma_banana | 异戊乙酸酯=香蕉 | +0.196 | +0.307 | +0.111 |
| A_esters_flower | aroma_flower | 词根一致：花香 | +0.108 | -0.146 | -0.255 |
| A_esters_all | aroma_estery | 酯类总体 ↔ 酯香 | +0.121 | +0.093 | -0.028 |
| A_malt_bread | aroma_bread | 词根一致：面包 | +0.140 | +0.230 | +0.090 |
| A_malt_cara | aroma_caramel | 词根一致：焦糖 | +0.599 | +0.879 | +0.279 |
| F_hops_citrus | taste_citrus | 词根一致：柑橘 | +0.366 | +0.813 | +0.448 |
| F_esters_fruity | taste_fruity | 词根一致：果香 | -0.027 | -0.356 | -0.329 |
| F_esters_ethac | taste_acetate | 同指乙酸乙酯 | +0.081 | +0.126 | +0.045 |
| F_esters_all | taste_estery | 酯类总体 ↔ 酯香 | +0.062 | -0.208 | -0.270 |
| bitternes | taste_bitter | 词根一致：苦味 | +0.645 | +0.679 | +0.034 |
| acidity | taste_sour | 酸度 ↔ 酸感 | +0.461 | +0.730 | +0.269 |
| sweetness | taste_sweet | 词根一致：甜 | +0.524 | +0.585 | +0.061 |
| alcohol | palate_alcohol | 词根一致：酒精感 | +0.561 | +0.780 | +0.220 |
| co2 | palate_carbonation | 碳酸 ↔ 碳酸感 | +0.237 | +0.233 | -0.004 |
| body | palate_buttery | 酒体 ↔ 奶油质地（最接近的口感体描述符） | +0.119 | +0.516 | +0.396 |
| dms | taste_stale | 硫化物/氧化 → 陈化味（缺陷感） | +0.051 | +0.528 | +0.478 |
| stale_hops | aroma_stale | 陈化酒花味 → 陈化味 | +0.217 | +0.311 | +0.094 |
| diacetyl | taste_diacetyl | 词根一致：双乙酰 | +0.074 | +0.034 | -0.040 |
| X4vg | taste_smoky | 4-乙烯基愈创木酚 → 烟熏/酚感 | +0.057 | +0.076 | +0.019 |
| t2n | taste_cardboard | 2-壬烯醛 → 纸板味 | +0.198 | +0.364 | +0.166 |
| metallic | taste_metallic | 词根一致：金属感 | +0.364 | +0.478 | +0.114 |
| orange | aroma_orange | 词根一致：橙香 | +0.167 | +0.046 | -0.121 |
| clove | taste_clove | 词根一致：丁香 | +0.215 | +0.313 | +0.099 |
| coriander | aroma_coriander | 词根一致：香菜 | +0.124 | +0.537 | +0.413 |
| lactic | taste_lactic | 词根一致：乳酸 | +0.483 | +0.749 | +0.266 |
| acetic | taste_acetic | 词根一致：乙酸 | +0.397 | +0.695 | +0.298 |
| barnyard | taste_barnyard | 词根一致：马厩味 | +0.199 | +0.467 | +0.269 |
| aftertaste | taste_aftertaste | 词根一致：回味 | -0.078 | -0.132 | -0.054 |

### `price_regression.csv`（A4）

- 被解释变量：**log(几何均价)**；样本 = **189 款**（≥2 家零售商报价）；
  对照口径（≥1 家）n = **229** 款，R² = 0.3398（ols_with_abv），≥2 家口径 R² = 0.3664。
- `ols` = 仅化学预测感知；`ols_with_abv` = 再加 ABV；
  `fixed_effects` = 再加风格固定效应（样本量 ≥ 5 的风格设虚拟变量，其余并入 Other；基准组 = `Blond`）并用**按风格聚类的稳健标准误**（聚类数远小于参数量，标准误偏保守，只作方向性参考）。
- log 价格与 ABV 的相关系数 = **+0.266**（价格受 ABV 混杂明显）。
- 三套规格的关键系数：

| 模型 | n | R² | 调整 R² | 稳健标准误 |
| --- | --- | --- | --- | --- |
| ols | 189 | 0.3402 | 0.3109 | classical |
| ols_with_abv | 189 | 0.3664 | 0.3346 | classical |
| fixed_effects | 189 | 0.4723 | 0.3876 | cluster-robust (by style) |

- 关键系数（完整表见 `price_regression.csv`，含全部风格虚拟变量）：

| 项 | ols 系数 (SE) | ols_with_abv 系数 (SE) | fixed_effects 系数 (SE) |
| --- | --- | --- | --- |
| aroma_fruity | -0.2340 (0.3941) | -0.4184 (0.3931) | -0.1064 (0.3650) |
| aroma_hoppy | -0.6314 (0.1930) | -0.5282 (0.1934) | -0.7760 (0.2451) |
| aroma_citrus | -0.1187 (0.0887) | -0.0766 (0.0886) | -0.0153 (0.1591) |
| taste_fruity | +0.4516 (0.4179) | +0.5859 (0.4136) | +0.1835 (0.3066) |
| taste_sweet | -0.6079 (0.1193) | -0.5774 (0.1178) | -0.5894 (0.1908) |
| taste_bitter | +0.3226 (0.1407) | +0.1730 (0.1488) | +0.6193 (0.2794) |
| taste_stale | -0.2396 (0.1892) | -0.1425 (0.1893) | -0.2038 (0.1713) |
| palate_carbonation | +0.2393 (0.1908) | +0.0571 (0.1991) | +0.0840 (0.3235) |
| ABV | — | +0.0533 (0.0196) | +0.0671 (0.0375) |

- 读法：`taste_sweet`（预测甜度）与 `aroma_hoppy`（预测酒花香）的系数为负、`taste_bitter`
  为正，方向与直觉相反 —— 因为自变量是**模型预测的感知值**而非实测消费者评分，
  且价格跨币种未换算、并与 ABV/品牌混杂；这些系数只应解读为「化学预测感知对 log 价格的
  条件相关」，**不能**解释为「消费者更偏爱甜/啤酒花」。

### `defect_dose_response.csv` / `defect_dose_verdict.csv` / `style_robustness.csv`（B）

- `defect_dose_response.csv`：6 组配对 × 4 个浓度四分位，共 24 行。`conc_median` 是
  分位内浓度中位（统一 log10(mg/L)），`rho_within` 是分位内 Spearman，`sensory_median` 是
  分位内感官维度中位，`exceed_frac` 是分位内浓度超过**感官阈值（中位阈值）**的样品比例
  （阈值统一换算为 log10(mg/L)；无阈值的化合物留空）。
- `defect_dose_verdict.csv`：`pattern` 取值仅 `monotonic_positive / monotonic_negative / none / non_monotonic`；
  判定依据 = **组内 ρ 的符号一致性**（主）+ **感官中位单调性**（辅）；
  `style_robust_fraction` = n≥8 的风格中保持同一模式的比例。
- `style_robustness.csv`：n≥8 的全部风格逐个重算（风格内用该风格自己的浓度四分位；
  四分位格 <3 样本的风格标记 `insufficient_quarters`，不计入稳健比例的分母）。

## 4. 剂量-响应再检验结果

**基准复现核对**：6/6 行与主控实测基准在 ±0.005 内一致（`dose_benchmark_check.csv`）。

| 化合物 | 品评组维度 | 全体 ρ | ρ(Q1..Q4) | 感官中位 Q1→Q4 | 模式 | 风格稳健比例 |
| --- | --- | --- | --- | --- | --- | --- |
| ethyl_acetate | A_esters_all | +0.321 | +0.171 / +0.014 / -0.038 / +0.407 | -0.246 → +0.262 | non_monotonic | 0.75 |
| ethyl_acetate | F_esters_all | +0.380 | +0.159 / +0.042 / -0.052 / +0.498 | -0.328 → +0.343 | non_monotonic | 0.88 |
| linalool | A_hops_citrus | +0.327 | +0.114 / +0.033 / +0.112 / +0.128 | -0.268 → +0.124 | monotonic_positive | 0.17 |
| DMS | dms | +0.088 | +0.031 / -0.173 / -0.044 / -0.004 | -0.167 → -0.029 | non_monotonic | 1.00 |
| X2.3.butanedione | diacetyl | -0.093 | +0.083 / -0.041 / +0.171 / +0.030 | -0.143 → -0.233 | non_monotonic | 1.00 |
| X.E..2.nonenal | t2n | -0.151 | -0.073 / -0.167 / -0.033 / -0.219 | +0.085 → -0.105 | monotonic_negative | 0.00 |

### 风格分层稳健性与辛普森式混合效应（B3）

- **四分位口径**：风格内用该风格自己的浓度四分位；四分位格 <3 样本记 `insufficient_quarters`，不计入分母（n≥8 的 16 个风格中仅 6–8 个可判，
  因为每个四分位格平均只有 2–8 个样本，组内 ρ 本身方差极大）。
- **半分割口径**（更稳健的伴随指标）：风格内按浓度中位数分低/高两半，n≥8 的全部风格都可判。

| 化合物 | 品评组维度 | 整体模式 | 四分位口径保持比例 | 半分割口径保持比例 | 方向翻转的风格 |
| --- | --- | --- | --- | --- | --- |
| ethyl_acetate | A_esters_all | non_monotonic | 0.75 | 7/16 = 0.44 | Stout/Porter(ρ=+0.27, monotonic_negative)、Amber(ρ=+0.53, monotonic_positive) |
| ethyl_acetate | F_esters_all | non_monotonic | 0.88 | 9/16 = 0.56 | Amber(ρ=+0.76, monotonic_positive) |
| linalool | A_hops_citrus | monotonic_positive | 0.17 | 7/16 = 0.44 | — |
| DMS | dms | non_monotonic | 1.00 | 8/16 = 0.50 | — |
| X2.3.butanedione | diacetyl | non_monotonic | 1.00 | 6/16 = 0.38 | — |
| X.E..2.nonenal | t2n | monotonic_negative | 0.00 | 3/16 = 0.19 | Strong ale(ρ=+0.18, monotonic_positive)、Stout/Porter(ρ=+0.38, monotonic_positive) |

（「—」表示该配对在可判风格中**没有出现方向相反**的模式；部分配对只是从单调**退化**为
`non_monotonic`，这种退化同样计入「保持同一模式比例」的下降。全部风格明细见
`style_robustness.csv` 与 `style_robustness_detail.csv`。）

判读要点（数字由实跑结果填入）：

- **trans-2-壬烯醛 → t2n**：整体判定 `monotonic_negative`，但风格分层后**没有一个风格**
  保持该模式（四分位口径 0/7，半分割口径 3/16），且 Strong ale 与 Stout/Porter
  两个风格内的模式**翻转为正**（风格级 ρ 为正）。这是典型的**辛普森式混合效应**：整体负向
  剂量-响应由风格间差异驱动，在风格内部并不成立，**必须如实报告为不稳健**。
- **芳樟醇 → 香气·柑橘**：整体 `monotonic_positive`，但四分位口径只有 1/6 的风格保持
  （Tripel），半分割口径 7/16；风格级 ρ 的符号在 6/6 可判风格中均为正，
  说明**方向稳健而「单调」不稳健**。
- **乙酸乙酯 → 香气·酯类总体**：整体 `non_monotonic`（非单调），四分位口径 6/8、
  半分割口径 7/16；Stout/Porter 翻转为 `monotonic_negative`、Amber 翻转为
  `monotonic_positive`，进一步支持「整体正相关来自组间差异而非组内剂量-响应」的判断。
- **DMS → 二甲基硫醚、双乙酰 → 双乙酰**：均为 `non_monotonic`，且风格级 ρ 的符号只有 5/8 与
  整体一致 → 缺陷物与对应感官维度的关系**既非单调、也不在风格间稳定**。

### 与原文结论的对照

原文主张：「通常被视为异味的风味物（乙酸乙酯、二甲基硫醚、双乙酰、某些醛）在低浓度且与
果香酯类共存时反而提升香气」，其后半段通常被引申为「提升消费者喜爱度」。

1. **可检验的部分**：「浓度—品评组感知」关系。本附件可检验，数据结论见上表。
2. **不可检验的部分（必须明写）**：原文结论的后半段「提升消费者喜爱度」在本附件条件下
   **不可检验**——附件**没有消费者评分/喜爱度标签**，且 183 维消费者描述符的 PCA 第一主成分
   仅解释 10.4% 方差，不存在可当作喜爱度的低维轴。本层因此**不声称**任何关于消费者
   喜爱度或偏好的结论。
3. **本数据能说的是**：乙酸乙酯与酯类香气在**全体样本上**同向（ρ=+0.321 / +0.380），
   感官中位随浓度四分位**单调上升**，但**分位内 ρ 的符号不一致**（Q3 转负、Q4 转强正），
   因此判定为 `non_monotonic`：整体正相关主要来自「低浓度组与高浓度组之间的差异」，
   组内并不存在稳定的剂量-响应斜率。这是典型的**分组均值差掩盖了组内关系**的情形。
4. **辛普森式混合效应**：凡风格分层后方向翻转或消失的配对，均在 `note` 与
   `style_robustness.csv` 中如实标注，不做掩盖。


## 5. 复现命令

```bash
python scripts/q4_consumer.py                 # 全流程（消费者层 + 剂量再检验 + 图）
python scripts/q4_consumer.py --stages consumer,dose
python scripts/q4_consumer.py --help
```

随机种子固定为 0，重复运行结果一致（幂等）。上游输入 sha256、参数与耗时见 `manifest.json`。

## 6. 产物清单

| 文件 | 内容 |
| --- | --- |
| `consumer_model_summary.csv` | 消费者 183 维的模型汇总（Macro-RMSE/MAE、R² 中位、ρ 中位） |
| `consumer_vs_panel_models.csv` | 同口径的品评组 vs 消费者模型对比 + Q1 已发布 OOF 结果 |
| `cross_consistency.csv` | A3 交叉一致性（ρ 真值 / ρ 预测 / Δ / n） |
| `price_regression.csv` | A4 价格回归系数（ols / ols_with_abv / fixed_effects） |
| `defect_dose_response.csv` | B1 分位级统计（24 行） |
| `defect_dose_verdict.csv` | B2 模式判定与风格稳健比例 |
| `style_robustness.csv` | B3 风格分层稳健性（n≥8 的全部风格） |
| `style_robustness_detail.csv` | 风格内四分位 ρ 与半分割明细（补充） |
| `dose_benchmark_check.csv` | 与主控实测基准的逐行核对（补充） |
| `consumer_per_dimension_metrics.csv` | 消费者层逐维 RMSE/MAE/R²/ρ（补充） |
| `panel_per_dimension_metrics.csv` | 品评组 50 维逐维指标（补充） |
| `oof_predictions_consumer.csv.gz` / `oof_predictions_panel.csv.gz` | 5 折 OOF 预测（250 行，供论文取数与复核） |
| `manifest.json` | 上游 sha256、参数、种子、耗时 |
| `../q4_figures/fig1…fig4` | 4 张图，每张 .png(300dpi) + .pdf |
| `_run_stdout.log` / `_run_stderr.log` | 本次实跑的控制台日志（运行证据） |

