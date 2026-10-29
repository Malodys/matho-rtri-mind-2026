# 啤酒数据字段概览

本文档是原始字段的简要语义地图。完整的逐列表由 `notebooks/01_data_quality.ipynb` 生成到 `data/interim/chemical_dictionary.csv`、`sensory_dictionary.csv` 和 `forum_descriptor_dictionary.csv`。

## 1.csv：化学测量与派生特征

- 标识列：`beer`、`beer_id`、`tasting_category_fine`。
- 化学/派生特征：231 列。
- 来源分类来自 `8.xlsx` 的 `Sensory thresholds.origin_paper`。它们是物质的通常来源分类，不是对单个样品的因果结论。

### Yeast / 酵母相关（78 列）

代表性物质包括 acetaldehyde、carbon disulfide、diethyl sulfide、ethyl acetate、ethyl hexanoate、ethyl octanoate、hydrogen sulfide、isoamyl alcohol、isobutanol、isoamyl acetate、methanethiol、phenethyl acetate、phenethyl alcohol、4-vinylguaiacol、sulfur dioxide、glycerol、ethanol、diacetyl 以及多种酯、高级醇和醛。

### Hops / 酒花相关（77 列）

包括 iso-alpha acids、linalool、limonene、eucalyptol、citronellol、geraniol、nerol、caryophyllene、humulene、farnesene、nerolidol 及多种单落和倍半落衍生物。

### Spices / 香料相关（23 列）

包括 terpinolene、camphor、terpinen-4-ol、cumin aldehyde、cinnamaldehyde、eugenol、methyl isoeugenol、valencene、spathulenol 和 ambroxide 等。

### Malt / 麦芽相关（21 列）

包括 dimethyl sulfide、ammonia、color、betaglucan、sugars、unfermentables、3-methylbutanal、methylpyrazine、furfural、5-methylfurfural、isomaltol 和多种呋喃/吡嗪类化合物。

### Wild flora / 野生微生物相关（6 列）

acetic acid、lactic acid、isoamyl lactate、4-ethyl-phenol、4-ethyl-guaiacol、geosmin。

### Others / 其他理化或风味指标（21 列）

包括 pH、iron、protein、carbon dioxide、caloric value、ethyl lactate、phenylacetaldehyde、E-2-nonenal、vanillin 等。

### 内标和汇总特征

- 内标相关：3-methyl-2-butenal（1 列）。
- 汇总特征（4 列）：`esters_sum`、`aroma_hops_sum`、`acids_sum`、`sulfur_sum`。

## 4.csv：训练品评小组感官指标

- `A_` 系列（15 列）：Aroma，包括 malt、hops、esters 的 overall 与子类评分。
- `F_` 系列（15 列）：Flavour，采用与 Aroma 相对应的 malt、hops、esters 分类。
- 其他指标（20 列）：acidity、bitterness（源列名 `bitternes`）、sweetness、alcohol、aftertaste、body、carbonation（`co2`）、overall，以及 4VG、diacetyl、DMS、metallic、stale hops、trans-2-nonenal、orange、coriander、clove、lactic、acetic、barnyard 等缺陷或专项风味。

## 7.csv：论坛评论感官描述符

62 个描述符各自出现在三个语境中，共 186 列：

- `aroma_*`：香气描述特征。
- `palate_*`：口腔感受/口感描述特征。
- `taste_*`：味觉描述特征。

描述符包括 acetaldehyde、acetate、acetic、acidic、aftertaste、alcohol、apple、banana、barnyard、bitter、bread、burnt、butterscotch、buttery、cabbage、candy、caramel、carbonation、cardboard、cereal、cheesy、citrus、clove、coriander、corn、diacetyl、estery、flower、fruity、grainy、grass、hay、herb、hoppy、horse、lactic、malt、medicinal、metallic、orange、papery、pear、pine、resinous、smoky、solvent、sour、stale、sweet、toast、tropical、woody 等。

由于数值包含负数，在获得方法说明前称为“描述符特征/得分”，不称为原始词频。

## 命名约定与数值口径

- `beer` 是项目内样品键；`beer_id` 是外部商品标识。部分文件中 `beer_id` 已损失精度，跨表联结使用 `beer`。
- `A_` 指 Aroma，`F_` 指 Flavour。
- 列名中的 `X`和句点多数是 R 语言合法化列名的结果，例如 `X4VG`、`X2.3.butanedione`。
- 文件 1 与文件 8 的 `Calibrated chemical dataset` 按 `beer` 对齐后，70 个共有指标中 37 个与 `log10(标定值)` 一致，33 个与原标定值一致，没有列支持 `ln(标定值)`。另有 157 个非派生指标没有标定表对照，4 个为派生汇总指标。
- 对已确认的 log10 列，文件 8 中的 0 与文件 1 中的 `log10(该列最小正值 / 2)` 一一对应，表明这些重复底限是左删失/检出限填补，不是普通缺失。
- 文件 8 的单位是变换前单位。文件 1 的 log10 列应解读为 `log10(原单位数值)`，不应再直接标注为 mg/L、µg/L 等物理单位。逐点匹配不支持在变换前还发生了额外的 mg/L↔µg/L 换算。
- 论坛特征的原始计算方法仍需要数据提供方或原论文确认。
