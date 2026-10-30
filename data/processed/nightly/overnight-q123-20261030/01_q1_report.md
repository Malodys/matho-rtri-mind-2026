# overnight-q123-20261030 · 01 Q1 模型与指标优化审查

## 阶段状态

- 状态：`dependency_missing`
- 分支：`night/overnight-q123-20261030`
- 工作树：`F:\cs\mothercup\worktrees\overnight-q123-20261030`
- HEAD：`f49581ff563f626abc10eb1727e0b5c07ead9d2d`
- 阻塞依赖：任务指定的 `data/processed/nightly/overnight-q123-20261030/state/00_baseline.json` 不存在。
- 旁证：发现 `state/00_baseline.json`，其内容声明 `00_baseline` 已完成，但它不位于任务要求的依赖路径，因此没有替代规范依赖继续执行。
- 执行边界：按约定转为只读审查；未拟合模型、未运行 bounded 重算、未增加重复种子、未修改 `data/processed/q1_final`、未触碰 `data/raw`、未生成提交包、未提交或推送。

## 只读审查结果

### 主协议与产物完整性

- `scripts/q1_pipeline.py` 对输入硬断言为 250×227 特征、250×50 目标、175/75 固定留出；调参入口在 175 个训练样本的内层 5 折中完成。
- `data/interim/chem_log10_matrix_v1.csv` 实测 250 行、232 个 CSV 列（1 个索引列、231 个化学列），其中 4 个 `*_sum` 派生列；排除索引和派生列后为 227 个建模特征。
- `data/splits/seed_0/1_train.csv` 与 `4_train.csv` 各 176 行（含表头，即 175 样本），测试文件各 76 行（含表头，即 75 样本）。
- 7 个既有模型 run 均存在，每个 `test_predictions.csv.gz` 解压后为 3,750 行，即 75×50；未重跑网格搜索。
- 官方交付表 `sensory_scores_0_6.csv` 为 250×153；本轮未覆盖官方结果。
- `git diff --check` 通过。

### 既有留出指标

| 模型 | Macro-RMSE | Macro-MAE | R² mean | Spearman mean |
|---|---:|---:|---:|---:|
| independent_gbr | 0.487951 | 0.350474 | 0.213319 | 0.393793 |
| grouped_extra_trees | 0.488051 | 0.347538 | 0.216956 | 0.423935 |
| global_extra_trees | 0.489329 | 0.348551 | 0.220878 | 0.434550 |
| pls2 | 0.491485 | 0.357635 | 0.202248 | 0.409836 |
| multitask_elastic_net | 0.492875 | 0.354200 | 0.205164 | 0.408339 |
| ridge | 0.497636 | 0.359255 | 0.189221 | 0.396861 |
| global_mean | 0.576241 | 0.420498 | -0.013089 | 不适用 |

`independent_gbr` 仍是既有 Macro-RMSE 最小模型，但与树模型的差距极小，且已有 bootstrap 区间高度重叠；本轮没有产生足以修改主模型选择的新证据。

### 绝对轨汇总检查

- 有效维度为 3 个：`bitternes`、`alcohol`、`co2`；其中阈值锚定 2 个，相对 fallback 1 个；另外 8 个候选维度未通过有效性规则。
- `abs_threshold_*`：MAE 0.686025、exact accuracy 0.453333、within-one accuracy 0.886667、QWK 0.792056。
- `abs_fallback_*`：MAE 1.155384、exact accuracy 0.240000、within-one accuracy 0.693333、QWK 0.489370。
- `abs_valid_*`：MAE 0.842478、exact accuracy 0.382222、within-one accuracy 0.822222、QWK 0.691161。
- 四个兼容字段 `abs_*` 与对应 `abs_valid_*` 在 7 行模型汇总中均完全一致，无覆盖回归。
- 异常/表达风险：上述绝对轨指标只用被选中的 `independent_gbr` 留出预测计算，却被复制到 `model_summary.csv` 的全部模型行。数值计算没有发现覆盖错误，但表结构容易被误读成逐模型绝对轨指标。建议后续增加 `abs_metrics_model=independent_gbr` 来源字段，或仅在选中模型行填值；本轮因依赖缺失未修改。

### TabPFN 公平性

- 现有 `data/processed/q1_repro_tabpfn/run.csv` 明确记录 175 样本、231 特征、5 折 KFold、seed 0；其 CV mean RMSE 为 0.459497，不能与主协议 227 特征、固定 175/75 留出的 0.487951 直接比较。
- `data/processed/q1_fair_tabpfn` 不存在。`scripts/q1_tabpfn_cv.py` 虽已提供 `--protocol fair_main`，但本轮没有执行。
- 结论固定为：**队友协议复现，不可直接比较**。拒绝把现有 TabPFN 数值加入主模型优劣结论。

### Conformal 区间候选设计（未进入主结论）

可在 175 个训练样本内部做交叉 conformal：对 50 个目标分别形成 OOF 残差，按目标校准有限样本分位数，再由固定 75 样本一次性报告边际覆盖率、宏平均覆盖率、平均/中位区间宽度以及按目标的最差覆盖。不得用 75 样本调分位数；风格分组样本太小时不做无收缩的 Mondrian 区间。由于本轮没有完整覆盖率与宽度验证，该设计不写入主结论，也不生成区间交付物。

## 回归测试与命令日志

1. `python -m pytest tests_q1 -q -p no:cacheprovider`：退出 1；系统 `python` 指向 LibreOffice 内置解释器，缺少 `pytest`。
2. `C:\Users\10857\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pytest tests_q1 -q -p no:cacheprovider`：退出 1；缺少 `pytest`。
3. `F:\cs\mothercup\matho-rtri-mind-2026\.venv\Scripts\python.exe -m pytest tests_q1 -q -p no:cacheprovider`：退出 103；该 venv 指向已不存在的 WindowsApps Python。
4. `py -3.12 -m pytest tests_q1 -q -p no:cacheprovider`：退出 101；Python launcher 无法创建目标进程。
5. `git diff --check`：退出 0。
6. 使用 PowerShell/标准库只读核对 CSV、gzip 行数、JSON、SHA-256：通过。

以上失败均发生在 pytest 收集前，属于运行环境不可用，不是 Q1 断言失败。为避免制造新环境或写入缓存，本轮没有安装依赖。`q1_pipeline --skip-fit` 也因规范基线缺失而按约定跳过。

## 推荐与拒绝的修改

推荐：

1. 先由基线阶段把 `00_baseline.json` 写到规范路径，或由调度器统一依赖路径；随后再运行 bounded 命令，并把输出定向到本次 nightly 的 `q1/` 子目录。
2. 修复/重建仓库 Python 3.12 环境后，先运行 `pytest tests_q1 -q -p no:cacheprovider`，再运行 `q1_pipeline.py --skip-fit --n-boot 2000`；不得覆盖 `q1_final`。
3. 为绝对轨汇总增加所用模型的显式 provenance，并在报告层明确 2/1/3 个维度的分母。
4. 如执行重复验证，只在 nightly `q1/` 下保留不超过 5 个种子，并保持特征、切分、指标和训练折内调参一致。

拒绝：

1. 不以缺失的规范基线之外的同名 JSON 静默替代依赖。
2. 不重跑无界网格搜索，不修改或覆盖 `data/processed/q1_final`。
3. 不把 231 特征 KFold TabPFN 结果并入 227 特征固定留出比较。
4. 不在缺少覆盖率/宽度验证时把 conformal 区间写入主结论。

## 本阶段变更文件

- `data/processed/nightly/overnight-q123-20261030/01_q1_report.md`
- `state/01_q1.json`
