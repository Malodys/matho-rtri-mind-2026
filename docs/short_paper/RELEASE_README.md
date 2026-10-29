# 化学指纹啤酒建模：当前版本交付包

本包对应 `docs/short_paper/main.pdf` 的五页工作稿，包含论文源码与图、主线模型代码、主要结果和复算所需输入。根目录的相对路径与项目一致。完整候选池与已拟合的二进制模型保留在工作区内，压缩包通过所附代码、参数与数据可重新生成它们。

## 从哪里看

- `docs/short_paper/main.pdf`：当前论文；同目录的 `main.tex` 和 `figures/`：可编辑原稿及文中六幅图。
- `docs/short_paper/figure_*.py`：论文图的生成脚本。
- `reports/modeling/q1_mainline_retrain_v1/`：问题一的 175/75 留出评估，论文中的泛化误差以此为准。
- `reports/modeling/q1_full_delivery_v1/`：用全部 250 款重新拟合后的 50 维画像、代理等级；这些全量拟合输出不用于估计泛化误差。
- `reports/modeling/consumer_vocabulary_v1/` 与 `forum_bridge_v1/`：消费者词汇预测及专家—论坛对应。
- `reports/modeling/q2_mainline_retrain_v1/`：211 个化合物的网络、社区、Top-3、500 次重采样及社区—感官映射。
- `reports/modeling/q3_source_guided_v2/`：两部分生成、源引导对照、选中候选谱；`q3_copula_v1/` 和 `q3_analog_v1/` 为相关参照。

## 模型与复算

主要入口是 `scripts/q1_evaluation.py`、`scripts/q1_full_delivery.py`、`scripts/consumer_vocabulary_model.py`、`scripts/forum_bridge_audit.py`、`scripts/run_q2_submission.py`、`scripts/q3_source_guided_design_v2.py`。`scripts/README.md` 说明数据、模型、实验与图表脚本的分工。`data/raw/`、`data/interim/`、`data/splits/seed_0/` 与 `configs/` 是相应输入和配置。问题二的本次配置见 `reports/modeling/q2_mainline_retrain_v1/submission_config.json`。

Python 运行环境见 `requirements.txt`。生成论文 PDF 使用 XeLaTeX：在 `docs/short_paper/` 内执行 `xelatex -interaction=nonstopmode -halt-on-error main.tex` 两次。

## 解释边界

论坛列是文本描述符富集得分，不是消费者喜好评分。问题一的 0–6 等级是代理等级，不能还原原始品评员打分。问题二的社区是化合物共变图的结构，社区—感官热力图是描述性投影。问题三的候选是化学谱方案，尚未经过酿造和感官品评。
