# Q2脚本说明

## 提交版唯一入口

```bash
python src/run_q2_submission.py
```

提交版的所有参数集中在`configs/q2_submission_v1.json`，不依赖脚本中的默认值。已有模型结果、只需重新生成化学表和图时运行：

```bash
python src/run_q2_submission.py --reuse-results
```

| 文件 | 作用 |
| --- | --- |
| `configs/q2_submission_v1.json` | 提交版参数的唯一来源 |
| `src/run_q2_submission.py` | 依次运行建图、Top-3、化学核查和绘图 |
| `src/q2_baseline.py` | Spearman共表达图、冗余惩罚Top-3和Bootstrap；提交调用不启用额外方法对照 |
| `src/q2_submission_report.py` | 生成提交版Top-3化学核查表 |
| `src/q2_submission_figures.py` | 生成提交版图形 |

## 探索脚本

其他`src/q2_*.py`是建模过程中的独立实验，不参与当前提交版。它们使用了不同的特征范围、阈值、Bootstrap次数或建模目标，不应将不同目录的输出混合为一次实验。

| 类别 | 脚本 | 状态 |
| --- | --- | --- |
| 模块/Hub路线 | `q2_louvain_consensus.py`、`q2_stratified_stability.py`、`q2_sensory_pipeline.py` | 探索，不提交 |
| 条件重要性路线 | `q2_module_conditional_model.py`、`q2_module_shapley.py`、`q2_formal_attribution.py` | 探索，不提交 |
| 多输出路线 | `q2_multitask_rrr.py` | 探索，不提交 |
| 非线性路线 | `q2_nonlinear_experiments.py`、`q2_two_part_experiment.py` | 探索，不提交 |
| 图传播/平滑路线 | `q2_network_propagation.py`、`q2_soft_graph_smoothing.py` | 探索，不提交 |
| 其他对照 | `q2_hierarchical_crl.py`、`q2_node_consensus_redundancy.py`、`q2_core_validation.py` | 探索，不提交 |

## 提交版固定设置

| 设置 | 值 |
| --- | --- |
| 特征 | 211个单项化合物 |
| 共表达相关来源 | 2.csv |
| 网络阈值 | `|rho|≥0.5` |
| Top-3冗余惩罚 | `0.35` |
| Bootstrap | `500` |
| 方法对照 | 无；稳定性直接用Bootstrap评估 |
| 随机种子 | `20261027` |
