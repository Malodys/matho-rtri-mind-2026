# overnight-q123-20261030 · 00 基线审计

## 运行状态

- status: `completed`
- started_at: `2026-10-30T00:31:04.2049709+08:00`
- ended_at: `2026-10-30T00:32:09.4961926+08:00`
- branch: `night/overnight-q123-20261030`
- worktree: `F:\cs\mothercup\worktrees\overnight-q123-20261030`
- commit_sha: `f49581ff563f626abc10eb1727e0b5c07ead9d2d`
- environment: Windows PowerShell `5.1.26100.9711`; Git `2.55.0.windows.2`; Python `3.12.13`
- tests: `PASS` — `state/00_baseline.json` 与锁元数据均可解析；`git diff --check` 通过
- failure_reason: `null`

## Git 可观测事实

- 审计开始时 `git status --porcelain=v2 --branch` 仅显示分支 OID 与分支名，无已修改、未跟踪或冲突条目。
- `git ls-files -u` 无输出，即索引中无未合并条目。
- `HEAD` 为 `f49581f`；`origin/main` 为 `10cfca6`；两者 merge-base 为 `10cfca6`。
- `origin/experiments/prediction-models` 为 `15d0c99`，与当前 HEAD 的 merge-base 为 `10cfca6`。该远端引用有 1 个当前 HEAD 未包含的非合并提交：`15d0c99 Add prediction model tuning and summarized results`。
- 该预测分支相对共同祖先包含 18 个差异路径：修改 `.gitignore`；新增 `RESULTS.md`、`requirements-tabpfn.txt`、7 个脚本和 8 个测试文件。
- `origin/feat/initial-data-analysis` 为 `2d72cae`，与当前 HEAD 的 merge-base 为 `10cfca6`。当前 HEAD 未包含其 3 个非合并提交：`0dc7f52`、`738a5d0`、`dfdf59d`。
- 该初始分析分支相对共同祖先包含数据中间产物、数据字典、notebook、Q1 报告产物与建模脚本；差异清单由本次 `git diff --name-status HEAD...origin/feat/initial-data-analysis` 获得，未将这些文件合并或复制进本工作树。
- 本次未执行 `git fetch`，以上 `origin/*` 均为本地可见的远端跟踪引用；未推送 GitHub。
- 未观察或推断任何队友在线状态。

## 输入产物快照

- Q1：`data/processed/q1_final`（14 文件）、`q1_runs`（56）、`q1_repro_tabpfn`（273）、`q1_figures`（9）。
- Q2：`data/processed/q2_runs`（33 文件）、`q2_figures`（13）。
- Q3：`data/processed/q3_runs`（60 文件）、`q3_figures`（15）。
- 论文：`data/processed/paper`（5 文件）。
- 审计依据：`AGENTS.md`、`README.md`、`docs/git_team_status.md`、`docs/review_round2.md`、`docs/review_round2_response.md`。
- `docs/git_team_status.md` 记录的是另一发布分支快照；本文件以本次命令观测到的当前工作树、分支和 SHA 为准。

## 主要审稿风险

1. Q1 特征口径必须统一为 227（231 列中剔除 4 个未经验证的派生求和列），不能在摘要或表格中写成直接使用 231。
2. 相关矩阵可裁定为 Spearman，但单调变换不改变 Spearman，不能据 `max|Δ|=0` 推断其浓度标度。
3. Q2 的稳健关键因素须与化学合理性联合披露；28/50 感官维度无稳健关键化合物，且不得把关联写成因果。
4. Q3 仅约束 4 项缺陷化合物；KS 基准、受约束列计数、8/22 风格选择规则、单对相关最大偏差均须如实披露。
5. 消费者/价格层自变量为 OOF 预测值；价格跨 USD/EUR/GBP 且未换汇，只能作方向性证据，不能构造消费者喜爱度标签。
6. 论文正文页数余量有限；参考文献引用点、附录脚本清单、构建期残留与占位队号仍是提交前核验重点。当前任务禁止生成提交包。

## Q1 / Q2 / Q3 后续执行边界

- 共同边界：`data/raw/` 严格只读；不把外部数据并入主训练集；固定既有拆分和口径；新结果只写本运行命名空间；不接触禁入 worktree、发布分支或 `origin/main`；不生成提交包、不推送。
- Q1：只使用统一标度后的 227 个化学特征预测 50 个感官维度；保持训练折内调参与留出集隔离；TabPFN 权重仅作本地依赖且不得入库。
- Q2：共表达与 Top-3 必须保留 bootstrap 稳定性、化学合理性和负结果；输出限定为候选关联，不作机制或因果推断。
- Q3：保持风格条件、稀有风格收缩和 4 项缺陷硬约束；报告无约束参照、总体与极端相关偏差；“零违规”仅表示约束合规，不表示无约束真实性。

## 阶段锁

- 已建立 `state/locks/00_baseline.lock/`，本次开始时不存在同阶段活动锁。
- 锁在阶段完成后保留并刷新为 `completed`，供后续任务识别本阶段已完成。
