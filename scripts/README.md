# 代码分工

主线代码统一放在 `scripts/`，按问题编号命名；图表脚本放在 `docs/short_paper/`，与论文和图片同处。数据在 `data/raw/`、`data/interim/`、`data/splits/`；结果在 `reports/modeling/`。

| 工作 | 入口与分工 |
| --- | --- |
| 数据准备 | `build_data_contracts.py`、`build_label_dictionary.py` 生成统一输入；`split_dataset.py` 固定 175/75 划分。 |
| 问题一 | `q1_models.py` 定义模型；`q1_evaluation.py` 统一数据与指标；`q1_submit.py` 执行留出评估；`q1_full_delivery.py` 用全部样本生成交付画像。 |
| 消费者描述符 | `consumer_vocabulary_model.py` 预测评论词；`forum_bridge_audit.py` 对齐专家与论坛感官。 |
| 问题二 | `q2_baseline.py` 建共表达网络和 Top-3；`run_q2_submission.py` 串起结果表与图；`q2_submission_report.py`、`q2_submission_figures.py` 负责呈现。 |
| 问题三 | `q3_copula.py` 实现两部分分布；`q3_source_guided_design_v2.py` 生成、筛选和评估候选；`q3_analog_retrieval.py` 检查真实目标参照；`export_q3_fitted_models.py` 可另行保存拟合对象。 |

其余 `q1_*`、`q2_*`、`q3_*` 文件是建模探索或消融实验，保留在工作区，不进入当前交付包。结果目录中的 `fitted_models/` 和 `pool_*.csv` 也留在工作区，需要时可用主线脚本重建。
