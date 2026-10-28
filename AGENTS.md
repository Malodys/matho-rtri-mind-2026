# AGENTS.md

2026 年 MathorCup 数学应用挑战赛 · 大数据挑战赛 **赛道 A（啤酒酿造化合物分析与啤酒风味预测）** 的分析仓库。
三个初赛问题：①缺失条件下的感官评分预测；②化合物共表达图与各感官维度 Top-3 关键化合物；③风格条件生成与跨风格迁移。

## Setup commands

- 建环境：`uv venv --python 3.12 .venv`（`uv` 0.11+；Windows 上 venv 落在仓库内，F 盘）
- 装依赖：`uv pip sync requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`（锁文件为权威版本源，勿直接 `pip install` 改动；`--extra-index-url` 不可省，否则 `torch==2.8.0+cpu` 解析不到）
- 测试：`.\.venv\Scripts\python.exe -m pytest worktrees/prediction-models/tests -q`
- 单个问题入口：`.\.venv\Scripts\python.exe scripts\<脚本名>.py`（脚本自带 `--help`）
- 问题一 TabPFN 五折复现（并行版，口径与远端 `tabpfn_cv.py` 等价，支持 `--resume` 断点续跑）：
  ```powershell
  .\.venv\Scripts\python.exe scripts\q1_tabpfn_cv.py `
    --worktree worktrees\prediction-models `
    --model-path worktrees\pretrained\Prior-Labstabpfn_3_5\tabpfn-v3.5-20260909.safetensors `
    --output-dir data\processed\q1_repro_tabpfn --workers 3 --threads 2 --resume
  ```
  > 本机 13.9 GB 内存下每分块加载一次 876 MB 权重：**workers 不得超过 4**，否则触发 WinError 1450（系统资源不足）。
- 工具链自检：`.\.venv\Scripts\python.exe scripts\stack_smoke.py`（图论/归因/生成栈）、`scripts\acceptance_check.py`（V1–V15）
- 论文 PDF 合规校验：`.\.venv\Scripts\python.exe scripts\check_paper_pdf.py <pdf> --docx <源docx>`
- 锁定新版本：`uv pip freeze --python .venv\Scripts\python.exe > requirements.lock.txt`

> 环境为 **CPU-only**：torch 2.8.0+cpu，TabPFN 走 CPU。本机 GTX 1650 仅 4GB 显存，多任务 TabPFN 易 OOM，故不装 CUDA 轮子。

## Project layout

- `data/raw/` — 赛题原始附件 1–8（`1.csv` 化合物丰度、`2.csv` 化合物相关、`3.xlsx` 品评表、`4.csv` 感官结果、`5.csv`/`6.csv` 相关矩阵、`7.csv` 论坛评述、`8.xlsx` 参考数据）。**只读，永不原地改写**
- `data/splits/` — 分层训练/留出切分（`seed_0/`）
- `data/interim/` — 探查与插补中间产物（缺失机制、检出限掩码、离群标记、log10 矩阵）
- `data/processed/` — 模型输出、指标、图表、论文冒烟产物
- `scripts/` — 数据语义诊断脚本 + 问题一/二/三建模脚本
- `tests/` — 单元测试（远端 `experiments/prediction-models` 分支）
- `worktrees/prediction-models/` — 远端管线分支的并行工作树（勿在主工作区切分支）

## 仓库与分支模型

- `main` — 仅放已合并的结论性产物（切分脚本、切分数据）
- `feat/initial-data-analysis` — `data/interim/**` 探查与插补产物（Malodys）
- `experiments/prediction-models` — 问题一建模管线与 `tests/**`（Thryzen）
- `feat/*` — 新工作分支，从最新 `origin/main` 切出
- 远端：`Malodys/matho-rtri-mind-2026`；`gh` 已登录并具备 push 权限，但**推 main 必须走 PR**

**边界**：`main` 只放已合并结论；各分支写各自命名的子目录（如 `data/processed/q1_runs/<timestamp>/`），避免并行写同一路径。

## 外部模型权重（不入库，需单独获取）

- `experiments/prediction-models` 分支的 `scripts/tabpfn_cv.py` **必须**显式传入 TabPFN 3.5 权重文件，缺文件直接 `FileNotFoundError`：
  `pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors`（相对该分支的项目根）。
- 该分支 `.gitignore` 含 `/pretrained/` 与 `*.safetensors`，**权重永不入库**，换机器需自行拉取：
  来源 Hugging Face 仓库 `Prior-Labs/tabpfn_3_5` 的 `tabpfn-v3.5-20260909.safetensors`（约 836 MB）。
  ```powershell
  # 本仓库布局下（worktree 并行检出时）目标路径为：
  #   F:\cs\mothercup\matho-rtri-mind-2026\worktrees\pretrained\Prior-Labstabpfn_3_5\
  ```
- 队友分支 `feat/initial-data-analysis` 上的 28 个插补/离群中间产物同样不在 `main`，需切到该分支或用 worktree 取用。

## AI 辅助注释规范（竞赛硬性合规）

依据《MathorCup 数学应用挑战赛人工智能工具使用规定（试行）》，任何由 AI 辅助产出的脚本，**文件头必须**写明工具名、版本/型号、开发机构、使用日期：

```python
# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：<写清哪些结论被人工逐项核对过>
```

- 函数级注释需标注该段逻辑是否经人工复核/修改。
- AI 生成的图表、参数、分析结论，要在**图注与正文对应位置**注明 AI 贡献。
- 论文参考文献后须声明所用 AI 工具，格式：`[编号] 工具名称，版本/型号，开发机构/公司，使用日期`。
- 支撑材料须附《AI 工具使用详情说明》PDF：工具与版本、使用目的与环节、关键交互记录、采纳内容与人工修改情况。
- 无法确认来源的模型/算法/数据必须显式标注说明。

## 数据分层与不可变约定

- `data/raw/` 严格只读；任何脚本不得向其写入或原地改写。
- `data/interim/` 放探查/插补中间产物，`data/processed/` 放模型输出与图表。
- 中间产物一律带版本后缀（如 `_v1`），并配套 `_manifest.json` 记录来源与参数。
- 缺失值与检出限必须区分处理，不得把"未检出"直接当 0 或直接丢弃。

## Testing instructions

- 数据层回归测试：`pytest tests -q`（20 项：数据契约 / 标度审计 / 相关类型，锁定本轮裁定，改动 `scripts/data_contract.py` 必须全绿）
- 建模管线单元测试：`pytest worktrees/prediction-models/tests -q`（7 个测试文件，改动建模脚本后必须全绿）
- 指标口径：按感官维度分别计算后取平均；R² 越大越好，MAE/RMSE 越小越好
- 数据划分：按 `tasting_category_fine` 分层，固定 seed，训练/留出比 175:75（`data/splits/seed_0`，只读勿改）
- 新增行为都要补测试；提 PR 前测试必须全绿
- 交付前跑 `python scripts/acceptance_check.py`，V1–V15 必须全 PASS

## 数据语义备忘（避免重复探查；均为本轮实测裁定，附回归测试）

> 权威入口：`scripts/data_contract.py` —— **唯一允许读 `data/raw/` 的模块**。
> 回归测试：`tests/test_data_contract.py`、`tests/test_scale_audit.py`、`tests/test_corr_type.py`（20 项，全绿）。
> 裁定产物：`data/interim/bridges_v1.csv`、`chem_scale_audit_v1.csv`、`defect_thresholds_v1.csv`。

- **对数底数 = log10，但 1.csv 是混合标度**。以 8.xlsx「Calibrated chemical dataset」的原始浓度回归：37 列严格
  `file1 = log10(原浓度)`（斜率 1.0000、r=1.000000）；33 列存的是**原始浓度本身**（如 `X2.3.butanedione`
  中位 0.2847 mg/L）。`ln` 假设斜率恒为 2.302585，据此排除。
  **任何脚本都不得对全表盲做 `10 ** x`** —— 用 `data_contract.to_unified_scale()` 取已统一的 log10(mg/L) 矩阵
  （`data/interim/chem_log10_matrix_v1.csv`，250×231，已剔除 4 个 `*_sum` 派生列后为 227 特征）。
  逐列裁定见 `data/interim/chem_scale_audit_v1.csv`（LOG10=37 / RAW=19 / RAW_APPROX=10 / AMBIG=4，与队友
  provenance 表 70/70 一致；4 列 AMBIG 因文件8 标定值与文件1 存储值存在系统性差异，采信 provenance 的 `original`）。
- **2.csv / 5.csv / 6.csv 全部是 Spearman，不是 Pearson**。逐元素复算比对：Spearman 下 `max|Δ|=0`、
  `frac<1e-8=1.0000`（Pearson 下 `median|Δ|≈0.04`）。2.csv 是在**文件1 原始存储标度**上直接算的（未做单位统一）。
  ⚠️ **官方 PDF 称「表格 2 单元格为皮尔逊相关系数」与此矛盾**，《数据说明.txt》的 Spearman 表述才正确；
  论文必须显式勘误并附证据表。
- **名称桥**（`data/interim/bridges_v1.csv`）：化学桥 2.csv(231)+6.csv(44) **275/275 全解析**；
  感官桥 5.csv(50)+6.csv(50) **100/100 全解析**。
  - 化学：矩阵名 → `Sensory thresholds.chemical_name` → `field_name_R` → 1.csv 列名；兜底四级
    （归一化 → field_name_python → 归一化 chemical_name）。**归一化必须先音译希腊字母再剥除非字母数字**，
    否则 `E-α-ionone` 与 `E-β-ionone` 会塌缩成同一键 `eionone` 造成静默错配。
  - 感官：矩阵名 → `Sensory names.CleanName_PanelFeedback` → `R_name` → 4.csv 列名；
    矩阵侧名称可能带 `(Trained panel)` 后缀与换行，查表前须清洗。
- **3.xlsx 是空白品评卡模板，不含任何评分数据**。12 张 `beer_N` 只有属性名与 0–6 刻度，无品评员打分行；
  `summary` 表 600 个数值格全为 0，数值单元格是公式（`=beer_1!$Q$7` 等）。
  **附件中不存在原始 0–6 分数**，4.csv 的标准化值是唯一标签源（mean≈0、std∈[0.37,0.86]，非 z 分数，
  原始尺度不可反解）。→ 论文的「0–6 分制」必须显式构造并声明（相对七分位分级 + 阈值锚定双轨）。
  3.xlsx 公式本身有价值：它确认了属性→维度映射（`A_malt_grain→Q10`、`A_hops_citrus→Q23` …）。
  语义锚点：0 = `Not present`，6 = `Predominant/overwhelming`。
- **感官维度全集 = 50**：Aroma 15（`A_*`）+ Flavour 15（`F_*`）+ 通用 20；
  `8.xlsx/Sensory names` 的 Trained panel 50 行与 4.csv 的 50 个目标列**集合完全相同**（已验证）。
  `Category_Paper` 分布：Aroma 15 / Taste 30 / Palate 4 / Overall 1。题面「约 40 个」为「约」数，交付全 50 维即可覆盖。
- **缺陷化合物阈值**（统一矩阵 log10(mg/L) 口径，见 `defect_table()`）：
  DMS 阈值 −1.7632（17.25 µg/L，超标 168/250=67.2%）；Diacetyl 阈值 −0.5240（0.29925 mg/L，112/250=44.8%）；
  4VG 阈值 −0.6854（0.206333 mg/L，16/250=6.4%）；**trans-2-nonenal 在文件8 中无阈值**（以「迁移后不恶化」兜底）；
  `stale_hops` 是 4.csv 的**感官维度**而非化合物。
- `7.csv` = 250×187 论坛评述派生描述符（已标准化，仅 3 列含 NA 共 750 格），与 4.csv 互补。
- `8.xlsx` 其余 sheet：`GBM importance MDI`（231 行，原论文模型重要性先验）、`GBM SHAP Values`（500 行，
  `base_prediction` 为常数 −0.3263 即模型截距）、`Calibrated chemical dataset`（250×73，**含 70 个化合物的原始浓度**，
  是裁定对数底数与标度的基准）、`Beer Prices`。
- `tasting_category_fine` 为风格标签（22 类，Blond 31 / Tripel 28 最大，Faro 3 / Brut 4 最小）；
  问题三的 Copula 条件边际对 n<10 的风格需向全局边际收缩（λ = n/(n+10)）。

## PR & commit conventions

- 从最新 `origin/main` 切 `feat/*` 分支，不直接推 `main`。
- 提交信息用 conventional commits（如 `feat:` / `fix:` / `chore:` / `docs:`）。
- 合并走 PR：`gh pr create --repo Malodys/matho-rtri-mind-2026`。
- 不提交大体积中间数据；只提交可由脚本重生成的产物与 manifest。

## 附录素材（论文格式要求）

论文附录须列出**实际使用的软件名称、命令与全部源程序**，材料从仓库直接生成：

- 软件清单：由 `requirements.lock.txt` 汇总（名称 + 版本），逐条对应论文附录的软件列表。
- 命令清单：取本文件 "Setup commands" 与各脚本 `--help` 的实际调用。
- 源程序清单：`scripts/` 与 `tests/` 全部文件，随支撑材料 `#*******附件.zip` 一并打包。

## Security

- 论文与支撑材料**不得出现姓名、学校、学号、班级等个人信息**（赛规要求匿名）；提交前扫描源码与日志中的绝对用户路径。
- 绝不提交密钥：`.env`、`.env.*`、`*.pem`、`*.key` 已在 `.gitignore`。
- `data/raw/` 不外传、不上传。
- 不得调用任何会把仓库内容发往外部服务的脚本或命令，除非用户明确要求。
