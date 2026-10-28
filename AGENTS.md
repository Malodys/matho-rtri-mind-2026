# AGENTS.md

2026 年 MathorCup 数学应用挑战赛 · 大数据挑战赛 **赛道 A（啤酒酿造化合物分析与啤酒风味预测）** 的分析仓库。
三个初赛问题：①缺失条件下的感官评分预测；②化合物共表达图与各感官维度 Top-3 关键化合物；③风格条件生成与跨风格迁移。

## Setup commands

- 建环境：`uv venv --python 3.12 .venv`（`uv` 0.11+；Windows 上 venv 落在仓库内，F 盘）
- 装依赖：`uv pip sync requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`（锁文件为权威版本源，勿直接 `pip install` 改动；`--extra-index-url` 不可省，否则 `torch==2.8.0+cpu` 解析不到）
- 测试：`.\.venv\Scripts\python.exe -m pytest worktrees/prediction-models/tests -q`
- 单个问题入口：`.\.venv\Scripts\python.exe scripts\<脚本名>.py`（脚本自带 `--help`）
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

- 单元测试：`pytest worktrees/prediction-models/tests -q`（7 个测试文件，改动建模脚本后必须全绿）
- 指标口径：按感官维度分别计算后取平均；R² 越大越好，MAE/RMSE 越小越好
- 数据划分：按 `tasting_category_fine` 分层，固定 seed，训练/留出比 175:75
- 新增行为都要补测试；提 PR 前测试必须全绿

## 数据语义备忘（避免重复探查）

- `1.csv` 化合物丰度含**对数变换后的值**（负值对应原始浓度 < 1）；`ethanol / color / bitterness / pH / CO2` 等理化参数保留原量纲。**对数底数（log10 还是 ln）需自行核实**，正负值与原浓度的对应关系必须写清。
- `2.csv` 化合物相关性矩阵（方阵）、`5.csv` 感官属性相关性矩阵（方阵）、`6.csv` 为 46 化学/理化指标 × 50 感官属性的**非方阵**交叉相关矩阵。
- `2.csv / 5.csv / 6.csv` 的相关类型（Pearson 还是 Spearman）与是否在 log 变换量上计算——**尚未最终裁定**，引用前先用 `scripts/resolve_corr_type.py` 与 `scripts/align_and_confirm.py` 复核。
- `3.xlsx` 为品评卡：sheet `beer_1..beer_12` 各 100 行 × 33 列，`summary` sheet 汇总各啤酒的感官属性（Aroma / Flavour 等维度）。
- `4.csv` 是**标准化/中心化后的连续值**，不是原始 0–6 分制打分。
- `7.csv` 为 250 样本 × 187 感官描述符的论坛评述数据，同样经标准化/中心化（存在负值，部分列含 `NA`），与 `4.csv` 的冗余/互补性见 `scripts/explore_7csv.py`、`scripts/probe_7csv_value.py`。
- `8.xlsx` 提供两套名称桥：`Sensory thresholds`（`field_name_R`/`field_name_python` ↔ `chemical_name`）与 `Sensory names`（`R_name` ↔ `CleanName_PanelFeedback`）。
- `tasting_category_fine` 为风格标签；缺陷化合物阈值项为 DMS / Diacetyl / 4VG / trans-2-nonenal / Stale hops。

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
