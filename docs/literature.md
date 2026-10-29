# 文献锚定与扩充（第二轮升级 · B1 交付物）

本文件是 `docs/refs_verified.json` 的人读版，两份文件内容一一对应。参考文献编号顺序与
`scripts/build_paper.py` 的第二章「相关工作与文献综述」主题顺序严格对齐，可直接供
`build_paper.py` 自动生成参考文献表使用。

- 全部网络核验均在 **2026-10-29** 完成；核验来源包括 Crossref works API、DataCite API、
  OpenAlex API、Europe PMC 全文 XML（PMC10966102）、Zenodo DataCite 记录，以及各出版方
  官方页面。
- 文献总数 **35 条**（主锚点 1 条 + 方法学 13 条 + 领域综述 21 条），全部
  `verified: true`，即每条的 DOI/URL 均由上述来源回显确认。
- 本文件与 `docs/refs_verified.json` 均不包含任何本地绝对路径。

---

## 一、主锚点原文的详细核实结果

主锚点为：

> Schreurs M, Piampongsant S, Roncoroni M, Cool L, Herrera-Malaver B, Vanderaa C, Theßeling FA,
> Kreft Ł, Botzki A, Malcorps P, Daenen L, Wenseleers T, Verstrepen KJ. **Predicting and
> improving complex beer flavor through machine learning.** Nature Communications, 2024,
> 15(1): 2368. DOI: 10.1038/s41467-024-46346-0（CC BY 4.0，2024-03-26 在线）

来源：Nature 文章页 <https://www.nature.com/articles/s41467-024-46346-0>（访问日期 2026-10-29）；
Europe PMC 全文 XML：<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10966102/fullTextXML>；
Crossref works API <https://api.crossref.org/works/10.1038/s41467-024-46346-0>。

### 1.1 受训品评组人数：16 人（原文明确）

原文 Methods「Trained panel」段落逐字为：

> "30 volunteers were screened through a series of triangle tests. **The sixteen most sensitive
> and consistent tasters were retained as taste panel members.**"

引言亦写 "a **16-person** in-house trained tasting panel"。故**品评组人数为 16**，题面所述
「15–16 人」中的 15 一说在原文中**未在原文明示**（未核实）。品评组构成：年龄 22–42（均值 29）、
性别 56% 男性、国籍 7 国；评分 50 项属性、采用 **7 点强度标度**；评价在 10–12 点进行，
黑色品酒杯，每场 5–12 款同风格啤酒、12–16 ℃ 供样，每场含 2 款参照啤酒用于校准。
一致性以 12 个样品跨场重复 + ANOVA 检验，**95% 情形下跨场无显著差异（p > 0.05）**。

### 1.2 RateBeer 评论条数：采集 232,288 条，清洗后保留 181,025 条（原文明确）

原文 Methods 明确区分两个数字，务必不可混用：

- 抓取总量：**232,288** 条（mean = 922, min = 6, max = 5343）；
- 清洗后保留：**181,025** 条，来自 **>6000** 名评论者、**>40** 个国家。清洗规则为：语言须被
  `langdetect` 与 `langid` **同时**判为英语；评论者总条目数 **<100** 者剔除。

摘要与引言处的表述是「over 180,000 consumer reviews」「map data from over 180,000 consumer
reviews」，引言正文另有一处写 "we collected 180,000 reviews of our 250 beers"。因此
**181,025 是确切保留数，232,288 是抓取数，"18 万余条"是摘要口径**。

### 1.3 训练的 10 个模型（原文 Methods 与 Results 双处列出，表述一致）

原文按归纳偏置分为四类，共 10 个：

| 类别 | 模型（原文缩写与全称） |
| --- | --- |
| 线性回归类（3） | **LR**（linear regression with first-order interaction terms）、**Lasso**（lasso regression with first-order interaction terms）、**PLSR**（partial least squares regressor） |
| 决策树类（5） | **ABR**（AdaBoost regressor）、**ET**（extra trees）、**GBR**（gradient boosting regressor）、**RF**（random forest）、**XGBR**（XGBoost regressor） |
| 支持向量机（1） | **SVR**（support vector regression） |
| 人工神经网络（1） | **ANN**（scikit-learn 的 MLPRegressor） |

实现环境：scikit-learn v1.2.2、xgboost v1.7.3、Python v3.9.16；ANN 用 Optuna v3.2.0 的
贝叶斯树结构 Parzen 估计器调参；其余模型用**五折交叉验证网格搜索**、以 R² 为评价指标。
数据划分：250 款按风格分层随机分为训练集 70%（175 款）与测试集 30%（75 款）。

### 1.4 最优模型及其对消费者评分的表现（原文明确）

- 最优算法为 **GBR（梯度提升回归器）**：Methods「Model dissection」写 "GBR was found to
  **outperform other methods**, resulting in models with the highest average R² values in both
  trained panel and public review data sets"；Table 1 中 Gradient boosting 的 RateBeer R² = **0.69**、
  平均秩 **1.50**，均为全表最佳；摘要称 "The best-performing algorithm, Gradient Boosting"。
- Table 1（多输出模型在测试集上的 R²）：AdaBoost 0.21/0.61；ANN 0.14/0.46；Extra Trees
  0.22/0.61；**Gradient boosting 0.21/0.69**；Lasso 0.05/0.64；Linear regression −4.13/−11.02；
  PLSR −0.25/0.57；Random Forest 0.22/0.62；SVR 0.18/0.59；XGBoost 0.22/0.62
  （每组依次为 trained panel R² / RateBeer R²）。
- 分目标的表现：GBR 对**消费者欣赏度（RateBeer）的预测 R² = 0.67**，而对**品评组欣赏度仅
  R² = 0.09**（Supplementary Table S3/S4）；整体上 R² 随被预测感官属性不同可达 **0.75**。
- 原文明确指出**所有模型预测 RateBeer 的效果都优于预测品评组**（"all models perform better at
  predicting RateBeer results than results from our trained tasting panel"），推测原因是品评数据
  本身变异性大而被大量公开评论平均掉。

> **「是否优于人类专家」——原文未作此断言（未核实）。**
> 原文比较的对象是**传统统计方法**（相关系数与线性模型），摘要与讨论均写
> "significantly outperform predictions based on conventional statistics"、"modern machine
> learning models outperform conventional statistical tools, such as correlations and linear
> models"。原文**没有**声称模型表现优于受训品评组这一「人类专家」，也**没有**任何
> "outperform human experts" 之类的表述。相反，原文报告品评组总体欣赏度与 RateBeer 总体
> 欣赏度的相关仅 **rho = 0.29**，且明确指出模型无法区分因果化合物、必须靠验证实验补足。
> 因此本文**不得**把「模型优于人类专家」写成本文对原文的复现结论。

### 1.5 化学测量手段与被测物类别（原文 Methods 明确）

| 手段 | 设备 | 被测物 |
| --- | --- | --- |
| HS-GC-FID/FPD | Shimadzu GC 2010 Plus（FID 柱 DB-WAXetr，FPD 柱 HP-5） | 高级醇、**乙醛**、酯类、4-乙烯基愈创木酚、**硫化合物** |
| HS-SPME-GC-MS | Shimadzu GCMS-QP-2010 Ultra + Thermo Trace 1300 / ISQ（50/30 μm 三相 DVB/Carboxen/PDMS 纤维） | 额外挥发物，**主要是萜类与酯类**；目标清单共 **284** 个化合物，经色谱峰人工判读后 **167** 个被认为洗脱曲线可靠并进入最终分析 |
| 离散光度法与酶法 | Thermo Scientific Gallery Plus Beermaster 离散分析仪 | 乙酸、氨、β-葡聚糖、异α-酸、色度、糖、**甘油**、铁、pH、蛋白、亚硫酸盐 |
| NIR | Anton Paar Alcolyzer Beer ME | 乙醇 |
| 瓶压（手工压力计） | Steinfurth Mess-Systeme | 由瓶压换算 CO₂ 浓度 |

样品均以生物重复制备后取均值。全部性质合计 **226 项**（原文正文两处均作 226："we measured
226 different chemical properties" 与 Discussion 的 "226 chemical parameters"）；Methods 建模段
另写作 "Chemical measurements (p = 231)"，为建模矩阵中的变量数，与描述性统计口径 226 略有差异，
属原文自身的口径差，本文引用时以正文的 226 为准。

### 1.6 关键化合物结论的原文措辞（逐条摘录原文）

**乙酸乙酯（ethyl acetate）——两路归因均排第一：**

> "Both approaches identified **ethyl acetate as the most predictive parameter** for beer
> appreciation. Ethyl acetate is the most abundant ester in beer with a typical 'fruity',
> 'solvent' and 'alcoholic' flavor, but is often considered less important than other esters like
> isoamyl acetate."

按 SHAP 排序：第 1 位乙酸乙酯、第 2 位**乙醇**（"the most abundant beer compound after water"）；
**蛋白水平**在两种方法下均居高位（"Despite not often being considered a driver of beer
appreciation, protein level also ranks highly in both approaches"）；**乳酸**为 SHAP 第 4 位
（"the fourth most important parameter identified by SHAP"），原因推测为数据集里酸型啤酒普遍
获得较高欣赏度。

**「低浓度 + 与果香酯类共存」——原文并无此合并表述（未核实）。**
原文中与「共存 / 协同」有关的只有两处，且都不是针对乙酸乙酯：

1. 关于甲硫醇（methanethiol）与乙酸苯乙酯（ethyl phenyl acetate）这一对「通常被视为负面的」
   化合物，原文写的是：

   > "our models identify methanethiol and ethyl phenyl acetate, an ester commonly linked to beer
   > staling, as a key factor contributing to beer appreciation. Although there is no doubt that
   > **high concentrations** of these compounds are considered unpleasant, **the positive effects of
   > modest concentrations are not yet known**."

   即原文的措辞是「**高浓度令人不适，但低（适中）浓度的正向效应尚不清楚**」，**并未**说
   「低浓度且与果香酯类共存」。此外原文还特别指出模型的一个局限：部分依赖曲线在浓度持续升高
   时预测值保持不变，而实际上乙酸乙酯（"nail polish"）与甲硫醇（"sulfury"、"rotten cabbage"）
   在高浓度下会变得难闻，说明模型无法把握化合物的最优浓度。

2. 关键词 "coexist" 在原文中**只出现在参考文献题名里**——即 Takoi K 等 2010 年关于香叶醇与
   β-柠檬醇（geraniol / β-citronellol）在过量芳樟醇下共存（"under coexistence with excess
   linalool"）的论文[21]。该文与乙酸乙酯无直接关系。

**双乙酰（diacetyl）——原文未列为关键化合物结论（未核实）。**
双乙酰在原文中仅以三种方式出现：(a) Fig. 3 的 RateBeer 文本挖掘相关性讨论中，作为「在网络评论
中体现不足的具体香气之一」与 ester、coriander 并列（其与品评组的相关 p = 0.0134）；
(b) 2,3-己二酮（2,3-hexanedione）作为 GC-MS 内标；(c) 其浓度与品评组分值的关系只出现在
Supplementary Data 6 的相关值表中。原文**没有**把双乙酰写进「驱动喜爱度的关键化合物」名单。

**二甲基硫醚（dimethyl sulfide）——原文全文未出现该化合物名（未核实）。**
对全文 XML 检索 "dimethyl"、"sulfide"、"thiol" 的结果是：**"dimethyl" 0 次、"sulfide" 0 次**；
"thiol" 仅 3 次，且全部与**甲硫醇（methanethiol）**或**酒花来源硫醇（hop-derived thiols）**相关
（后者出现在局限部分："hop-derived thiols, that influence flavor at very low concentrations, are
notoriously difficult to measure"）。原文只笼统地提到 HS-GC-FID/FPD 测得「硫化合物」。
因此**「二甲基硫醚」是赛题附件/阈值表中的化合物名，不是原文点名分析的化合物**；
本文在问题三把它作为缺陷化合物做剂量-响应再检验，属于本文的独立处理，须在正文中明确区分。

**醛类（aldehydes）——原文只测了乙醛，并把陈化醛类作为「反向」证据（部分核实）。**
乙醛由 HS-GC-FID/FPD 测定；「陈化醛类（staling aldehydes）」出现在参考文献中[22]，以及局限段
关于「模型无法识别最优浓度」的讨论中（引用了氧化/纸板味的嗅觉线索文献[23]）。原文并未把醛类
整体列为风味驱动因子。

**乳酸（lactic acid）与甘油（glycerol）——原文明确（已核实）。**
乳酸：SHAP 第 4 位；并与异α-酸呈强负相关（反映产酸菌抑制或酸型风格少投酒花）；
验证实验的加标混合物含乳酸，且 Discussion 明说「许多酿酒商已用乳酸酸化糖化用水，
本文结果提示加乳酸也能提高欣赏度」。甘油：属酵母代谢物，与乙醇、高级醇、酯类同源共变
（Spearman 系数 > 0.5），与品评组的 alcohol / body 感知相关（rho = 0.72 / 0.57），
且低醇/无醇啤酒中浓度偏低；验证实验的加标混合物同样含甘油。

**验证实验（Model validation）原文明确：** 选取 Blond 风格 Beer 64（Supplementary Data 1）
与无醇 Beer 223，把「最重要驱动物」（乙酸乙酯、乙醇、乳酸、乙酸苯乙酯）连同其相关化合物
（乙酸己酸酯、乙酸异戊酯、甘油）提到**风格内第 95 百分位的乙醇归一化浓度**；加标组相对对照
组的总体欣赏度**显著提高**（二项检验，α = 0.05，n = 20 或 13），品评员报告酯香、甜度、酒精感
与酒体增强；去掉乙醇的第二组实验结论相同；无醇啤酒亦显著提升。

### 1.7 原文自述的局限（原文 Discussion「Our study is limited to…」段，明确）

原文逐条自述的局限如下：

1. **只覆盖比利时酒厂**：尽管覆盖了全球相当部分的啤酒风格，仍可能遗漏某些风格与消费模式，
   或使某些特征被过度代表（如许多比利时艾尔呈现酵母主导风味）。
2. **GBR 架构的固有偏倚**：共线变量中「最大主效应者」总是被优先保留，故共线变量的重要性
   （不论 impurity 还是 SHAP）会被人为压低；GBR 认定的关键化合物**未必是真正的因果化合物**，
   例如乙酸乙酯的重要性可能部分来自总酯、乙醇或乙酸己酸酯（rho = 0.77 / 0.72 / 0.68），
   乙酸苯乙酯可能掩盖香叶基异丁酸酯与苯甲酸乙酯（rho = 0.77 / 0.76）。
3. **风格维度未带来额外信息**（R² = 0.66 对 0.67），部分归因于维度灾难使模型主要吸收单参数效应，
   也因为风格本身并非严格科学分类。
4. **可测风味化合物仍是全体的一部分**：复杂食物中的风味化合物总数多于本文可测的子集，
   如在极低浓度即影响风味、且难以高通量测定的酒花来源硫醇。
5. **消费者感知主观且带偏差**，难以避免。
6. **模型尚不成熟**，需要更大的数据集才能形成更完整的模型。
7. **数据集不含品评者人口学信息**：原文明说 "our dataset does not include any demographic
   information about the tasters"，并指出补充年龄、文化等人口学信息可能带来更好的模型。
8. **样本集缺少不适合销售的啤酒**，限制了模型在「被极差评价」产品上的预测能力。
9. **模型无法辨别因果关系**，在感官科学与产品开发中的应用因此受限，必须靠验证实验识别真正
   的因果化合物。

另有两项与「人口学因素」相关的原文事实需注意：原文**未**将品评者性别纳入感官数据库构建
（"The sex of the tasting subject was not considered when building our sensory database"），
并给出受训组 56% 男 / 44% 女、RateBeer 全体 70% 男 / 30% 女的对照；此外原文报告
**价格与欣赏度相关 rho = 0.49（网络评论）、盲评品评组 rho = 0.19**，且盲评不受价格影响。

### 1.8 Fig. 1 图注要点与配色分组含义（原文图注逐字）

Fig. 1 标题为 "Correlations between selected chemical parameters (upper right panel) and sensory
descriptors used by the tasting panel (bottom left panel)"。图注明确：

> "**Spearman rank correlations are shown.** Descriptors are grouped according to their
> **origin (malt (blue), hops (green), yeast (red), wild flora (yellow), Others (black))**, and
> **sensory aspect (aroma, taste, palate, and overall appreciation)**. Please note that for the
> chemical compounds, for the sake of clarity, **only a subset of the total number of measured
> compounds is shown**, with an emphasis on the key compounds for each source."

要点归纳：① 相关类型为**斯皮尔曼秩相关**（非皮尔逊）；② 化学来源五色分组为
**麦芽蓝 / 酒花绿 / 酵母红 / 野生菌群黄 / 其他黑**；③ 感官面分组为
**aroma、taste、palate、overall appreciation** 四类；④ 图中化合物为**子集**，完整结果见
Supplementary Fig. S2 与 Supplementary Data 2。Fig. 2 的图注同样以颜色标识
"malt (blue), hops (green), yeast (red), wild flora (yellow), Others (black)"。

### 1.9 补充数据编号与内容对应关系（原文图注与 Methods 明确）

| 编号 | 内容 | 原文依据 |
| --- | --- | --- |
| Supplementary Data 1 | 化学实测数据（各化合物的量纲与数值） | Fig. 1 图注："Chemical data can be found in Supplementary Data 1" |
| Supplementary Data 2 | 全部化合物之间的相关值 | Fig. 1 图注："correlation values can be found in Supplementary Data 2"（图形见 Supplementary Fig. S2） |
| Supplementary Data 3 | 品评打分表（scoring sheet） | Methods："The scoring sheet is included as Supplementary Data 3" |
| Supplementary Data 4 | 品评组感官评估结果 | Fig. 1 图注："See Supplementary Data 4 for sensory panel assessments" |
| Supplementary Data 5 | 全部感官描述符之间的相关值 | Fig. 1 图注："Supplementary Data 5 for correlation values between all sensory descriptors" |
| Supplementary Data 6 | 化学品—感官交叉相关值 | Fig. 2 图注："See Supplementary Data 6 for all correlation values" |
| Supplementary Data 7 | RateBeer 文本挖掘结果 | Fig. 3 图注："RateBeer text mining results can be found in Supplementary Data 7" |

配套的补充表格：Supplementary Table S1（实测化合物清单）、S2（品评组一致性 ANOVA）、
S3（品评组逐描述符结果）、S4（RateBeer 逐描述符结果）、S5/S6（风格附加实验）、
S7（试剂与标准溶液）、S8（软件包版本）、S9（标准溶液）；补充图 S1（风格构成）、S2（全部化合物
相关）、S3（风格—化合物）、S4（风格—感官）、S5（化学—感官相关）、S6（价格相关）、
S7（部分依赖图）、S8（100 次重复建模稳定性）、S9（风格/合并数据实验）。

### 1.10 数据与代码可得性（Zenodo 记录已核实）

原文 Data availability 与 Code availability 均写明：支撑本工作的数据存放于 **Zenodo，
accession code 10653704**（即 DOI 10.5281/zenodo.10653704），且模型训练、模型分析与作图代码
亦在同一 Zenodo 记录中。该记录的 DataCite 元数据（标题 "Data from: Predicting and improving
complex beer flavor through machine learning"、13 位作者、CC BY 4.0、发布日期 2024-01-15）
与文件清单均已核实；**但需注意 RateBeer 评分数据本身属受限访问**（RateBeer 为 ZX Ventures
所有物），不可公开获取。

---

## 二、文献完整著录（共 35 条，按引用编号）

著录格式按赛题要求：期刊为「作者，论文名，杂志名，卷期号：起止页码，出版年」；网络资源为
「作者，资源标题，网址，访问时间（年月日）」。

**[1]** Schreurs M, Piampongsant S, Roncoroni M, Cool L, Herrera-Malaver B, Vanderaa C, Theßeling FA,
Kreft Ł, Botzki A, Malcorps P, Daenen L, Wenseleers T, Verstrepen KJ. Predicting and improving
complex beer flavor through machine learning. Nature Communications, 15(1): 2368, 2024.
DOI: 10.1038/s41467-024-46346-0.

**[2]** Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M,
Prettenhofer P, 等. Scikit-learn: Machine Learning in Python.
https://jmlr.org/papers/v12/pedregosa11a.html, 访问时间（2026年10月29日）.

**[3]** Lundberg S M, Lee S I. A Unified Approach to Interpreting Model Predictions.
https://arxiv.org/abs/1705.07874, 访问时间（2026年10月29日）.
DOI: 10.48550/arXiv.1705.07874.

**[4]** Hollmann N, Müller S, Eggensperger K, Hutter F. TabPFN: A Transformer That Solves Small
Tabular Classification Problems in a Second. https://arxiv.org/abs/2207.01848,
访问时间（2026年10月29日）. DOI: 10.48550/arXiv.2207.01848.

**[5]** Traag V A, Waltman L, van Eck N J. From Louvain to Leiden: guaranteeing well-connected
communities. Scientific Reports, 9(1): 5233, 2019. DOI: 10.1038/s41598-019-41695-z.

**[6]** Nelsen R B. An Introduction to Copulas[M]. New York: Springer (Springer Series in
Statistics), 2006. DOI: 10.1007/0-387-28678-0.

**[7]** Iman R L, Conover W J. A distribution-free approach to inducing rank correlation among
input variables. Communications in Statistics - Simulation and Computation, 11(3): 311-334,
1982. DOI: 10.1080/03610918208812265.

**[8]** Stevens S S. On the Theory of Scales of Measurement. Science, 103(2684): 677-680, 1946.
DOI: 10.1126/science.103.2684.677.

**[9]** Seppä L, Latvala T, Akaichi F, Gil J M, Tuorila H. What are domestic apples worth?
Hedonic responses and sensory information as drivers of willingness to pay.
Food Quality and Preference, 43: 97-105, 2015. DOI: 10.1016/j.foodqual.2015.02.013.

**[10]** Spearman C. The Proof and Measurement of Association between Two Things.
The American Journal of Psychology, 15(1): 72, 1904. DOI: 10.2307/1412159.

**[11]** Hollmann N, Müller S, Purucker L, Krishnakumar A, Körfer M, Hoo S B, Schirrmeister R T,
Hutter F. Accurate predictions on small data with a tabular foundation model.
Nature, 637(8045): 319-326, 2025. DOI: 10.1038/s41586-024-08328-6.

**[12]** Restrepo J P, Rivera J C, Laniado H, Osorio P, Becerra O A. Nonparametric Generation of
Synthetic Data Using Copulas. Electronics, 12(7): 1601, 2023.
DOI: 10.3390/electronics12071601.

**[13]** Moskowitz H R. The dollar value of product quality: the effect of pricing versus overall
liking on consumer choice. Journal of Sensory Studies, 10(3): 239-247, 1995.
DOI: 10.1111/j.1745-459x.1995.tb00016.x.

**[14]** Meilgaard M C. Prediction of flavor differences between beers from their chemical
composition. Journal of Agricultural and Food Chemistry, 30(6): 1009-1017, 1982.
DOI: 10.1021/jf00114a002.

**[15]** Olaniran A O, Hiralal L, Mokoena M P, Pillay B. Flavour-active volatile compounds in beer:
production, regulation and control. Journal of the Institute of Brewing, 123(1): 13-23, 2017.
DOI: 10.1002/jib.389.

**[16]** Verstrepen K J, Derdelinckx G, Dufour J P, Winderickx J, Thevelein J M, Pretorius I S,
Delvaux F R. Flavor-active esters: Adding fruitiness to beer.
Journal of Bioscience and Bioengineering, 96(2): 110-118, 2003.
DOI: 10.1016/S1389-1723(03)90112-5.

**[17]** Dzialo M C, Park R, Steensels J, Lievens B, Verstrepen K J. Physiology, ecology and
industrial applications of aroma formation in yeast. FEMS Microbiology Reviews, 41(Suppl_1):
S95-S128, 2017. DOI: 10.1093/femsre/fux031.

**[18]** Bokulich N A, Bamforth C W. The Microbiology of Malting and Brewing.
Microbiology and Molecular Biology Reviews, 77(2): 157-172, 2013.
DOI: 10.1128/MMBR.00060-12.

**[19]** Spitaels F, Wieme A D, Janssens M, Aerts M, Daniel H M, Van Landschoot A, De Vuyst L,
Vandamme P. The Microbial Diversity of Traditional Spontaneously Fermented Lambic Beer.
PLoS ONE, 9(4): e95384, 2014. DOI: 10.1371/journal.pone.0095384.

**[20]** Snauwaert I, Roels S P, Van Nieuwerburgh F, Van Landschoot A, De Vuyst L, Vandamme P.
Microbial diversity and metabolite composition of Belgian red-brown acidic ales.
International Journal of Food Microbiology, 221: 1-11, 2016.
DOI: 10.1016/j.ijfoodmicro.2015.12.009.

**[21]** Takoi K, Itoga Y, Koie K, Kosugi T, Shimase M, Katayama Y, Nakayama Y, Watari J.
The Contribution of Geraniol Metabolism to the Citrus Flavour of Beer: Synergy of Geraniol and
beta-citronellol under coexistence with excess linalool.
Journal of the Institute of Brewing, 116(3): 251-260, 2010.
DOI: 10.1002/j.2050-0416.2010.tb00428.x.

**[22]** Wauters R, Herrera-Malaver B, Schreurs M, Bircham P, Cautereels C, Cortebeeck J,
Duffin P M, Steensels J, 等. Novel Saccharomyces cerevisiae variants slow down the accumulation
of staling aldehydes and improve beer shelf-life. Food Chemistry, 398: 133863, 2023.
DOI: 10.1016/j.foodchem.2022.133863.

**[23]** Lusk L T, Kay S B, Porubcan A, Ryder D S. Key Olfactory Cues for Beer Oxidation.
Journal of the American Society of Brewing Chemists, 70(4): 257-261, 2012.
DOI: 10.1094/asbcj-2012-0907-01.

**[24]** Blanco C A, Andrés-Iglesias C, Montero O. Low-alcohol Beers: Flavor Compounds, Defects, and
Improvement Strategies. Critical Reviews in Food Science and Nutrition, 56(8): 1379-1388, 2016.
DOI: 10.1080/10408398.2012.733979.

**[25]** Gonzalez Viejo C, Fuentes S, Torrico D D, Godbole A, Dunshea F R. Chemical
characterization of aromas in beer and their effect on consumers liking.
Food Chemistry, 293: 479-485, 2019. DOI: 10.1016/j.foodchem.2019.04.114.

**[26]** Rodrigues H, Parr W V. Contribution of cross-cultural studies to understanding wine
appreciation: A review. Food Research International, 115: 251-258, 2019.
DOI: 10.1016/j.foodres.2018.09.008.

**[27]** Chambers E, Koppel K. Associations of Volatile Compounds with Sensory Aroma and Flavor:
The Complex Nature of Flavor. Molecules, 18(5): 4887-4905, 2013.
DOI: 10.3390/molecules18054887.

**[28]** Korneva E, Blockeel H. Towards Better Evaluation of Multi-target Regression Models.
ECML PKDD 2020 Workshops（Communications in Computer and Information Science）: 353-362, 2020.
DOI: 10.1007/978-3-030-65965-3_23.

**[29]** Friedman J H. Greedy function approximation: A gradient boosting machine.
The Annals of Statistics, 29(5): 1189-1232, 2001. DOI: 10.1214/aos/1013203451.

**[30]** Efron B. Bootstrap Methods: Another Look at the Jackknife.
The Annals of Statistics, 7(1): 1-26, 1979. DOI: 10.1214/aos/1176344552.

**[31]** Arlot S, Celisse A. A survey of cross-validation procedures for model selection.
Statistics Surveys, 4: 40-79, 2010. DOI: 10.1214/09-SS054.

**[32]** Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning: Data Mining,
Inference, and Prediction[M]. New York: Springer (Springer Series in Statistics), 2009.
DOI: 10.1007/978-0-387-84858-7.

**[33]** Lawless H T, Heymann H. Sensory Evaluation of Food: Principles and Practices[M].
New York: Springer (Food Science Text Series), 2010. DOI: 10.1007/978-1-4419-6488-5.

**[34]** Stone H, Sidel J, Oliver S, Woolsey A, Singleton R C. Sensory Evaluation by Quantitative
Descriptive Analysis. Descriptive Sensory Analysis in Practice（Wiley）: 23-34, 2004.
DOI: 10.1002/9780470385036.ch1c.

**[35]** Harris C R, Millman K J, van der Walt S J, 等. Array programming with NumPy.
Nature, 585(7825): 357-362, 2020. DOI: 10.1038/s41586-020-2649-2.

---

## 三、可支撑论断对照表

「支撑本文哪一节」一列采用论文正文的实际节号（与 `scripts/build_paper.py` 中的标题一致）。

| 编号 | 文献（简） | 支撑本文哪一节 | 支撑的具体论断 |
| --- | --- | --- | --- |
| [1] | Schreurs 2024 | 二、2.1；一、1.1；三、3.1–3.6 | 本文附件即该文的公开数据与补充材料；原文的 Fig.1 即附件相关矩阵的来源；原文以 250 款比利时啤酒、22 个风格、226 项化学性质与 16 人受训品评组感官结果为基准 |
| [2] | Pedregosa 2011（scikit-learn） | 二、2.2；四、4.1 | 问题一以 scikit-learn 统一实现线性/树模型族，构成高维小样本回归的通用基线 |
| [3] | Lundberg & Lee 2017（SHAP） | 二、2.2；四、4.2；五、5.3 | SHAP 把模型输出分解到单化合物的加性归因，是问题二把模型排序与秩相关并列为两路证据的方法基础 |
| [4] | Hollmann 2022（TabPFN） | 二、2.2；四、4.1；五、5.1 | 预训练先验模型为小样本表格回归提供第三条归纳偏置，与手工特征工程、树模型并列 |
| [5] | Traag 2019（Leiden） | 二、2.3；四、4.2；五、5.3 | 以 Leiden 修正 Louvain 的分辨率缺陷，使「共表达模块对应可追溯化学来源」不依赖单一阈值 |
| [6] | Nelsen 2006（Copula） | 二、2.4；四、4.3 | Copula 把联合分布分解为边际与依赖结构，是问题三以全局 Copula 保留跨化合物相关结构的理论依据 |
| [7] | Iman & Conover 1982 | 二、2.4；四、4.3；五、5.5 | 秩—分位映射在保持秩相关不变的前提下迁移样本，是跨风格迁移的核心方法 |
| [8] | Stevens 1946 | 二、2.5；五、5.1 | Stevens 量表理论给出 QDA 强度量表的心理学基础，是以强度标度替代类别标签的前提 |
| [9] | Seppä 2015 | 二、2.5；五、5.2 | hedonic 响应与感官信息共同解释支付意愿，说明「化学→感官」链条最终可在市场价值上被外部观察 |
| [10] | Spearman 1904 | 二、2.2 段末；三、3.2 与 3.6 | 本文全部相关分析用斯皮尔曼秩相关系数而非官方赛题文本所述皮尔逊系数，并以逐元素复算比对裁定 |
| [11] | Hollmann 2025（Nature） | 二、2.2；五、5.1 | TabPFN 的同行评议版本，为以预训练先验模型处理小样本表格回归提供正式出处 |
| [12] | Restrepo 2023 | 二、2.4；四、4.3 | 以 Copula 非参数生成保留相关结构的合成样本，是「相关结构 + 风格条件边际」生成化合谱的可比先例 |
| [13] | Moskowitz 1995 | 二、2.5；五、5.2 | 区分定价效应与总体喜好效应，支持把价格与感官喜好分列为两条外部验证信号，并与原文「价格 ρ=0.49、盲评 ρ=0.19」对照 |
| [14] | Meilgaard 1982 | 二、领域综述（啤酒香气与风味） | 以化学组成预测啤酒风味差异的经典工作，为以化学谱预测感官维度提供同题先例 |
| [15] | Olaniran 2017 | 二、领域综述（啤酒香气与风味） | 系统梳理风味活性挥发物的产生、调控与控制，是按化学来源给化合物分组的领域依据 |
| [16] | Verstrepen 2003 | 二、领域综述（酵母来源） | 酵母来源酯类是把果香带入啤酒的主要通路，支撑把酯类群组视为共表达模块的预期来源 |
| [17] | Dzialo 2017 | 二、领域综述（酵母来源） | 酵母香气形成的生理与工业应用综述，支撑把乙醇/高级醇/酯类的同源共变解释为发酵参数与酵母遗传背景的产物 |
| [18] | Bokulich & Bamforth 2013 | 二、领域综述（麦芽来源） | 制麦与糖化阶段微生物学决定麦芽来源风味物质形成，支撑把麦芽衍生物视为独立化学来源 |
| [19] | Spitaels 2014 | 二、领域综述（野生菌群） | 拉姆比自发发酵的微生物多样性是野生菌群风味贡献的直接证据 |
| [20] | Snauwaert 2016 | 二、领域综述（野生菌群） | 酸型啤酒的微生物多样性与代谢物组成相互对应，支撑对乳酸/乙酸抗相关簇的来源解释 |
| [21] | Takoi 2010 | 二、领域综述（酒花来源） | 香叶醇与 β-柠檬醇共存条件下的协同作用，是原文引用以说明「共存」效应的原始文献来源 |
| [22] | Wauters 2023 | 二、领域综述（异味与陈化）；五、5.4 | 陈化醛类累积与货架期改善，支撑把 Staling/Others 维度异味化合物单列并做剂量—响应再检验 |
| [23] | Lusk 2012 | 二、领域综述（异味与陈化）；五、5.4 | 氧化型异味的关键嗅觉线索，支撑氧化/纸板味维度的阈值锚定与缺陷判定 |
| [24] | Blanco 2016 | 二、领域综述（异味与陈化）；五、5.4 | 低醇啤酒的风味化合物、缺陷与改进策略，对应无醇/低醇风格的风味稀薄与缺陷讨论 |
| [25] | Gonzalez Viejo 2019 | 二、领域综述（消费者接受度）；五、5.2 | 以香气化学解释消费者喜好，把化学—香气—喜好链条落到可测量的接受度层面 |
| [26] | Rodrigues & Parr 2019 | 二、领域综述（接受度与价格）；五、5.2 | 跨文化欣赏度差异的价格效应，是原文引用以说明价格影响欣赏度的来源，提示本文价格信号的混杂风险 |
| [27] | Chambers & Koppel 2013 | 二、领域综述（挥发物—感官复杂关系）；六、6.1 | 挥发物与感官属性含协同与拮抗效应，支撑不把单相关当作因果、并以剂量—响应再检验复核关键化合物 |
| [28] | Korneva & Blockeel 2020 | 五、5.1 | 多目标回归的逐描述符排名取平均秩口径，是原文与本文在多输出感官预测上对齐的排名方法 |
| [29] | Friedman 2001 | 二、2.2；四、4.1 | 梯度提升的原始算法文献；原文最优模型即梯度提升回归器 |
| [30] | Efron 1979 | 五、5.1、5.3；六、6.2 | 自助法支撑在自助重采样区间下判定头部模型差异是否可区分、以及关键化合物入选频率 |
| [31] | Arlot & Celisse 2010 | 四、4.1、4.3；六、6.2 | 交叉验证与模型选择方法综述，支撑按风格分层的训练/留出划分与「调参只在训练集内完成」 |
| [32] | Hastie 2009 | 四、4.1、4.3 | 偏差—方差权衡、集成学习与正则化，支撑对线性模型过拟合与树模型/先验模型的比较分析 |
| [33] | Lawless & Heymann 2010 | 二、2.5；三、3.4 | QDA 强度标度与品评组训练、重复测量的标准实践，支撑对空白品评卡与标准化 z 分数的处理判断 |
| [34] | Stone 等 2004 | 二、2.5；三、3.4 | QDA 的术语形成、强度标度与品评员一致性标准操作，是「附件不含原始 0–6 打分、0–6 分制须显式构造」的方法学参照 |
| [35] | Harris 2020 | 附录 A | NumPy 版本的正式出处，用于附录软件清单的可核实著录 |

---

## 四、引用编号顺序表（供 `scripts/build_paper.py` 直接使用）

编号规则：**原文为 [1]**；其余按「第二章相关工作与文献综述各主题的首次出现顺序」排列，
与 `build_paper.py` 中 `_pick_entry` 的主题占位符一一对应。下表第 3 列即该主题在正文中的
占位符名，可直接用于核对自动生成结果。

| 编号 | 文献（简） | 正文占位符 / 节位 | 类型 |
| --- | --- | --- | --- |
| 1 | Schreurs 2024 | `{anchor}`（二、2.1） | 主锚点 |
| 2 | Pedregosa 2011（scikit-learn） | `{sklearn}`（二、2.2） | 方法学 |
| 3 | Lundberg & Lee 2017（SHAP） | `{shap}`（二、2.2） | 方法学 |
| 4 | Hollmann 2022（TabPFN） | `{tabpfn}`（二、2.2） | 方法学 |
| 5 | Traag 2019（Leiden） | `{leiden}`（二、2.3） | 方法学 |
| 6 | Nelsen 2006（Copula） | `{copula}`（二、2.4） | 方法学 |
| 7 | Iman & Conover 1982 | `{rank}`（二、2.4） | 方法学 |
| 8 | Stevens 1946 | `{stevens}`（二、2.5） | 方法学 |
| 9 | Seppä 2015 | `{price}`（二、2.5） | 方法学 |
| 10 | Spearman 1904 | `{spearman}`（二、2.2 段末；三、3.2） | 方法学 |
| 11 | Hollmann 2025（Nature, TabPFN） | 二、2.2（补充） | 方法学 |
| 12 | Restrepo 2023（Copula 合成数据） | 二、2.4（补充） | 方法学 |
| 13 | Moskowitz 1995（价格 vs 总体喜好） | 二、2.5（补充）；五、5.2 | 方法学 |
| 14 | Meilgaard 1982 | 二、领域综述 | 领域综述 |
| 15 | Olaniran 2017 | 二、领域综述 | 领域综述 |
| 16 | Verstrepen 2003 | 二、领域综述（酵母） | 领域综述 |
| 17 | Dzialo 2017 | 二、领域综述（酵母） | 领域综述 |
| 18 | Bokulich & Bamforth 2013 | 二、领域综述（麦芽） | 领域综述 |
| 19 | Spitaels 2014 | 二、领域综述（野生菌群） | 领域综述 |
| 20 | Snauwaert 2016 | 二、领域综述（野生菌群） | 领域综述 |
| 21 | Takoi 2010 | 二、领域综述（酒花） | 领域综述 |
| 22 | Wauters 2023 | 二、领域综述（异味与陈化） | 领域综述 |
| 23 | Lusk 2012 | 二、领域综述（异味与陈化） | 领域综述 |
| 24 | Blanco 2016 | 二、领域综述（异味与陈化） | 领域综述 |
| 25 | Gonzalez Viejo 2019 | 二、领域综述（消费者接受度） | 领域综述 |
| 26 | Rodrigues & Parr 2019 | 二、领域综述（接受度与价格） | 领域综述 |
| 27 | Chambers & Koppel 2013 | 二、领域综述（挥发物—感官） | 领域综述 |
| 28 | Korneva & Blockeel 2020 | 五、5.1 | 方法学 |
| 29 | Friedman 2001 | 二、2.2（补充） | 方法学 |
| 30 | Efron 1979 | 五、5.1、5.3 | 方法学 |
| 31 | Arlot & Celisse 2010 | 四、4.1 | 方法学 |
| 32 | Hastie 2009 | 四、4.1、4.3 | 方法学 |
| 33 | Lawless & Heymann 2010 | 二、2.5（补充）；三、3.4 | 方法学 |
| 34 | Stone 等 2004 | 二、2.5（补充）；三、3.4 | 方法学 |
| 35 | Harris 2020 | 附录 A | 方法学 |

**编号一致性约定（请正文引用时遵守）：**

1. `[1]` 固定指主锚点，论文正文 1.1 节「公开数据与补充材料 [1]」即指本条目。
2. `build_paper.py` 的 `_build_related_work` 会用 `_pick_entry` 按关键词自动定位条目并填入
   `{anchor}`、`{sklearn}`、`{shap}`、`{tabpfn}`、`{leiden}`、`{copula}`、`{rank}`、`{stevens}`、
   `{price}`、`{spearman}` 十个占位符。上表第 1–10 条即按此顺序布置，可保证占位符解析到
   语义正确、且互不串位的条目（例如「copula」只会命中 [6] 与 [12]、「price」只会命中 [9] 与 [13]）。
3. 正文若需引用 [11]–[35]，请按本表顺序首次出现处依次引用，不得跳号或倒序。
4. 参考文献表由 `build_paper.py` 依据 `docs/refs_verified.json` 自动生成，
   编号即 `citation_number`，**不得**在论文中另行手工编号。

---

## 五、存疑与未核实项

本节逐条列出**未能由权威来源确认、或原文与题面表述存在出入**的项目，供正文作者据实处理。

### 5.1 著录层面的存疑（已写入清单但需注意事项）

1. **Spearman 1904 的完整页码范围未能核实。**
   本清单 [10] 依 Crossref 与 OpenAlex 的回显著录为
   `The American Journal of Psychology, 15(1): 72`——**Crossref/OpenAlex 未收录该条目的完整页码
   （其记录中只有 first_page=72），著录值以来源为准**。业界常引作「15(1): 72–101」，但本次
   未能从 Crossref、OpenAlex、JSTOR 页面元数据或出版方页面回显确认结束页 101，故未写入。
   同时提请注意：`build_paper.py` 内置的降级清单把该文献写作 `17(1): 72-101`，**卷号 17 有误**
   （正确为 15），本清单已按 Crossref 回显更正为 15(1)。
2. **Anderson & Simester 关于啤酒市场「价格—品质」的经典实证未能核实到可引用著录。**
   该主题原拟引用的条目（Management Science 50(10): 1369–1385 一类）在 Crossref 与 OpenAlex 中
   均**未检索到与之匹配的记录**（该刊该期的 11 条记录无一对应，页码 1369–1385 区间在回显记录中
   对应的是另一篇 Montmarquette 等的合著论文），JSTOR 侧亦未定位到对应稳定 ID。
   **Crossref/OpenAlex 未收录该条目的回溯 DOI，著录值以来源为准**；按主控指令，
   本清单的 `{price}` 主题改用**已核实 DOI 的替代条目**：[9] Seppä 2015（hedonic 响应与感官信息
   共同决定支付意愿，DOI 10.1016/j.foodqual.2015.02.013）与 [13] Moskowitz 1995
   （定价效应与总体喜好效应的分离，DOI 10.1111/j.1745-459x.1995.tb00016.x）。
   **正文 2.5 节现有措辞点名「Anderson 与 Simester 对啤酒市场的实证」，与新的引用编号 [9]/[13]
   不一致，需由正文作者改写该句或另补一条经核实的文献。**
3. **Korneva & Blockeel 2020 的卷号未能核实。** 清单 [28] 著录了来源回显的页码 353–362 与
   会议录名（CCIS），但卷号未在来源回显中给出，故未著录卷号。

### 5.2 原文事实层面的未核实项（正文不得据此断言）

4. **原文是否称模型「优于人类专家」——未在原文明示，且原文有相反方向的证据。**
   原文只与**传统统计方法**（相关系数、线性模型）比较，全文无任何「优于人类专家/受训品评组」
   的表述；原文自己报告品评组与 RateBeer 总体欣赏度的相关仅 rho = 0.29，且强调模型无法辨别
   因果。本文若引用原文，只能表述为「机器学习显著优于传统统计方法」，**不能**写成
   「模型表现优于受训品评组」。
5. **「受训品评组 15 人」未在原文明示。** 原文明确为 **16 人**（30 人三角试验筛选后保留
   16 名最敏感且最一致的品评员），题面所述「15–16 人」中的 15 一说在原文中无对应表述。
6. **「低浓度 + 与果香酯类共存」这一合并表述未在原文明示。** 原文的相关措辞是：甲硫醇与
   乙酸苯乙酯在**高浓度**下令人不适，而**适中浓度**的正向效应「尚不清楚」；原文并未把该效应
   与「果香酯类共存」绑定，也未把乙酸乙酯纳入该论断。关键词 "coexist" 在原文中仅出现于
   其参考文献 [21] 的题名（香叶醇/β-柠檬醇共存），与乙酸乙酯无直接关系。
7. **双乙酰未在原文中被列为关键化合物结论。** 原文仅在网络评论相关性的讨论中把它与 ester、
   coriander 并列为「在公开评论中体现不足的具体香气」，并作为 2,3-己二酮内标出现。
8. **「二甲基硫醚」未在原文出现。** 对全文的检索显示 "dimethyl" 与 "sulfide" 各 0 次，
   "thiol" 仅 3 次且全部指甲硫醇或酒花来源硫醇。原文只笼统地说 HS-GC-FID/FPD 测得「硫化合物」。
   因此二甲基硫醚是**赛题附件/阈值表中的化合物名**，本文对它的剂量—响应再检验属独立处理，
   应与原文结论明确区分。
9. **醛类未在原文中作为一类风味驱动因子出现。** 原文只测了乙醛，并把陈化醛类作为「模型无法
   把握最优浓度」的反向证据引用，未把醛类整体列入关键化合物。
10. **化学性质总数的口径差（原文自身）。** 正文两处为 226 项，Methods 建模段写
    "Chemical measurements (p = 231)"。本文引用应以正文 226 为准，并注意附件 1.csv 与
    Supplementary Data 1 的列数可能落在 226 与 231 之间，撰稿时应以附件实际列数为准。
11. **原文未提供 RateBeer 评分的公开副本**（属 RateBeer/ZX Ventures 所有物，受限访问），
    因此本文无法直接复算原文的 RateBeer 侧指标，只能引用其已发表的数值。
12. **性别未被纳入原文的感官数据库构建**，且原文明说数据集**不含品评者人口学信息**；
    故本文在消费者层与价格回归中讨论人口学混杂时，只能引用原文这一「缺口」陈述，
    不能把原文当作提供人口学变量的证据。

### 5.3 本清单的核验边界

13. 全部 35 条的 DOI/URL 均由 Crossref works API、DataCite API、OpenAlex API 或出版方官方
    页面回显确认，故一律置 `verified: true`；本清单**不含**仅凭书目记忆写入的条目。
14. 出版方页面的题名/卷期/页码以回显值为准；同一文献在不同聚合源之间若存在拼写差异
    （如「Theßeling」的变音符号），以 Crossref 返回的作者串为准。
15. 本文件与 `docs/refs_verified.json` 相互对应，JSON 为机读真值来源；两处著录如有出入，
    以 `docs/refs_verified.json` 为准。
