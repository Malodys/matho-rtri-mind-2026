# overnight-q123-20261030 · 02 Q2 稳健关联优化审查

## 结论

**Share with caveats（审查范围内完成）**。本轮未覆盖任何 Q2 主结果，保留 `abs_rho >= 0.6` 的描述性共表达网络主口径：**231 节点、266 边、114 个连通分量、101 个孤立节点**。当前主产物可继续作为候选关联与网络描述使用，但图表题注/标签有 3 项实质问题，且“稳健关键因素”不能直接等同于化学合理或因果关系。

三个指定不变量均由当前 CSV/JSON 独立复算通过：

- **22/50** 个感官维度至少有一条稳健项；
- **28/50** 个维度没有稳健项；
- **9** 行同时满足“稳健”与“化学合理”。

历史 `docs/review_round2.md` 同时写过“32 条”和“33 条”稳健行。当前 `summary.json`、`README.md` 与 `top3_by_sensory.csv` 一致为 **32 条**；本轮仅记录这一历史不一致，不修改主结果。

## 补充稳健性结果

### 有符号 Spearman

当前 150 条 Top-3 中，正相关 **129** 条、负相关 **21** 条；32 条稳健项中正相关 **31** 条、负相关 **1** 条。唯一的稳健负相关是 `dms -> protein.g.L.`（全样本 rho=-0.322），其化学合理性为“弱”，留出集 rho=-0.250，方向一致但 BH q=0.055，未通过 q<=0.05。

有符号明细见 [signed_spearman_main_top3.csv](q2/signed_spearman_main_top3.csv)。

### 训练选择 + 留出置换/FDR

为避免主 E2 使用全 250 样本而造成留出泄漏，本轮在 175 个训练样本上重新计算 E2，并与既有、同样基于训练集的 E3 SHAP 等权融合后重选 50×3 个候选；随后只在固定 75 样本留出集上做 5,000 次双侧置换检验，并对 150 项统一做 Benjamini-Hochberg FDR。

- 150 项训练选择关联中，**83/150** 通过留出 BH q<=0.05，覆盖 **34/50** 个维度；
- 训练/留出 Spearman 符号一致 **130/150（86.7%）**；
- 训练重选与已发布主 Top-3 的逐维平均 Jaccard 为 **0.628**，**19/50** 个维度三项完全相同；
- 当前 **32/32** 条稳健项全部被训练重选保留且留出符号一致，**30/32** 通过留出 FDR；
- **9/9** 条“稳健且化学合理”全部通过留出 FDR。

未过 FDR 的两条稳健项均为化学合理性“弱”：

| 感官维度 | 化合物 | 留出 rho | BH q | 处置 |
|---|---|---:|---:|---|
| `A_esters_all` | `n.hexanol` | +0.202 | 0.125 | 保留为候选关联，不升级 |
| `dms` | `protein.g.L.` | -0.250 | 0.055 | 保留方向性提示，不升级 |

完整结果见 [train_selected_holdout_permutation_fdr.csv](q2/train_selected_holdout_permutation_fdr.csv) 与 [published_robust_holdout_review.csv](q2/published_robust_holdout_review.csv)。该检验验证的是关联复现，不验证化学机理或因果关系。

### 相关特征组与 conditional SHAP

当前 Top-3 中有 **23/50** 个维度的三个候选至少两项落在同一个 `abs_rho >= 0.6` Leiden 社区，说明单特征重要性可能在相关代理变量间分摊。后续低风险优先方案应为：按既有 0.6 社区联合置换整组特征，以留出 Spearman 与 RMSE 的下降量报告组重要性；再用 0.5 社区作敏感性对照，孤立节点保持单元素组。

本轮未运行 conditional SHAP。原因是训练样本仅 175、特征 227，仓库没有已验证的条件分布采样器或交叉拟合 conditional SHAP 实现；临时引入会增加模型假设和泄漏风险。分组置换与 conditional SHAP 都只解释相关条件下的预测归因，不提供因果识别。方案记录见 [method_review.json](q2/method_review.json)，组冗余明细见 [correlated_group_review.csv](q2/correlated_group_review.csv)。

## 数据与脚本口径审查

1. 共表达图读取官方 `2.csv` 的 231×231 Spearman 矩阵，阈值固定为 `abs_rho >= 0.6`；E2/E3 排序使用剔除 4 个未验证派生求和列后的 **227** 个特征。两者分母不同但用途明确，不应互换。
2. 主 E2 是全 250 样本描述性 Spearman，E3 和 bootstrap 只用 175 个训练样本。主 Top-3 因此是描述性融合，不是纯留出验证结果；本轮训练选择/留出置换是补充验证，不能倒写主结果。
3. `summary_B8_sensitivity.json` 的 35 条/21 维/8 条化学合理只是低 B 敏感性快照；正式口径必须使用 B=200 的 32 条/22 维/9 条。
4. `q2_stability.py` 的稳健判据确为 bootstrap 频率 >=0.60 且逐维平均 Jaccard >=0.30；当前 CSV 复算与 `summary.json` 一致。
5. `q2_runs/README.md` 与 `q2_common.py` 仍把 `2.csv` 的逐元素 Spearman 复现解释成“未做单位统一”。Spearman 对逐列严格单调变换不变，因此只能裁定相关类型，不能由该复现裁定标度。该文字错误不改变 0.6 网络的边，但应在后续正式修订中撤回。
6. `q2_verify.py`、`q2_stability.py` 与 `q2_figures.py` 都把输出目录固定为主 `q2_runs`/`q2_figures`，无法安全重定向到 nightly。本轮为遵守“不覆盖主结果”，没有运行这些写主目录的入口，只做只读复算与夜间独立验证。

## 图表审查

自动检查通过：6 张图均有同名 PNG/PDF，PDF 头有效，manifest 6 条且文件存在；题注中的 0.6 网络数值与 `q2_runs` 一致。PNG 均为高分辨率，图 3、图 4 未观察到裁切或缺字。

需要修订但本轮未覆盖主图：

1. **P1：图 1/2 节点标签重叠。** Top-6 标签在网络中心与节点及彼此重叠，多项被截断成难以区分的前缀，与脚本头“无重叠”的声明不一致。
2. **P1：图 1/2 题注对孤立节点来源的指引不成立。** 题注称 101 个未绘出的孤立节点来源构成见图 3，但图 3 只保留 `n>=2` 的非平凡社区，未展示单节点社区/孤立节点构成。
3. **P1：图 6 色标与题注不一致。** 题注称 ARI 发散色标“0 为白色”，实际 0 映射为蓝色，约 0.5 才接近白色；ARI 在本图中为非负量，顺序色标更直接。
4. **P2：图 5 覆盖口径提示不足。** E1 只在 44 个化合物内排名，E2/E3 为 227 个；题注未提示这一覆盖不对称，Jaccard 不宜被读成同一候选空间下的完全可比一致性。

逐图记录见 [figure_audit.json](q2/figure_audit.json)。

## 候选关联、化学合理性与因果关系

- **候选关联**：150 条 Top-3 全部属于候选；其中 32 条通过现有 bootstrap + Jaccard 稳健判据。留出 FDR 为补充证据，不改写主标签。
- **化学合理性**：由知识库的载体/类别/来源/阈值规则独立判定。32 条稳健项中仅 9 条同时成立；统计复现不能把“弱”化学证据自动升级为“强”。
- **因果关系**：本项目没有干预、时间顺序或可识别的混杂控制。Spearman、SHAP、bootstrap、置换与分组置换都不能证明某化合物导致某感官变化。

## 深度审查覆盖表

表中分母是本轮建立的范围清单，不是“准确率”；`0/N` 仅表示未观察到缺陷。

### 结果与图表质量

| Category | Observed defects | Assessment |
|---|---:|---|
| Usefulness and completeness | 1 / 3 | 网络、Top-3/稳定性、图包均齐；缺少相关特征组层面的正式归因结果，本轮给出方案。 |
| Analytical clarity | 4 / 8 | README/Top-3 表可追溯；图 1/2 标签、图 5 覆盖说明、图 6 色标说明需修订。 |
| Visual and interaction consistency | 3 / 6 | PNG/PDF 配对完整；图 1/2/6 有可复现的视觉或题注缺陷。 |

### 分析正确性与稳健性

| Category | Observed defects | Assessment |
|---|---:|---|
| Source authority and confidence | 1 / 3 | 输入与知识库来源可追溯；`2.csv` 标度推断文字超出 Spearman 能支持的范围。 |
| Value accuracy | 0 / 5 | 网络规模、150 行 Top-3、32 条、22/28 维和 9 条联合计数均独立复算一致。 |
| Within-chart agreement | 3 / 6 | 图 1/2 对孤立节点的题注指引错误；图 6 的“0 为白色”与实际色标不符。 |
| Complete source details | 1 / 6 | 图 5 未在题注披露 E1=44 与 E2/E3=227 的覆盖不对称。 |
| Cross-artifact consistency | 1 / 4 | 当前 Q2 主产物一致为 32 条；round2 历史评审中的 33 条为旧文字冲突。 |
| Data-quality controls | 0 / 4 | 250×227、250×50、175/75 切分、有限性与频率域检查均通过。 |
| Conclusion support | 2 / 5 | 30/32 稳健项通过留出 FDR；两条弱化学证据未过 FDR，且所有结果仍只支持关联而非因果。 |

## 复现与失败记录

复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File "F:\cs\mothercup\worktrees\overnight-q123-20261030\data\processed\nightly\overnight-q123-20261030\q2\reproduce.ps1"
```

脚本固定 seed `20261030`，默认 5,000 次置换，只写本运行的 `nightly/.../q2/`。首次运行的可恢复 NumPy 错误、修复方式，以及未运行 `q2_verify.py` 的守护原因见 [failures.log](q2/failures.log)。机器可读汇总见 [robustness_summary.json](q2/robustness_summary.json)。
