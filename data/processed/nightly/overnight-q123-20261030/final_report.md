# overnight-q123-20261030 最终验收报告

## 结论

- 状态：`completed_with_known_failures`
- 分支：`night/overnight-q123-20261030`
- 验收开始前 HEAD：`f49581ff563f626abc10eb1727e0b5c07ead9d2d`
- 验收时间：`2026-10-30T09:11:17+08:00`
- 处置：允许提交已审计的报告、验证脚本、论文排版和图表硬化改动；不补造缺失的 Q1 模型工件，不新增或升级未经验证的科学结论。

全量测试仍有一个前序已记录的失败：默认 Q1 预测接口缺少 `data/processed/q1_final/model_independent_gbr.joblib`。该文件只能通过一致协议重训和工件审计获得，本阶段没有伪造或临时训练。除此之外，Q1 绝对指标回归、Q2/Q3 验证、图表风格和论文五项合规检查均通过。

## 明确环境

- Python：`C:\Users\10857\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`，版本 3.12.14。
- 项目依赖：通过 `PYTHONPATH=F:\cs\mothercup\matho-rtri-mind-2026\.venv\Lib\site-packages` 显式加载锁定依赖。
- 已核对版本：pytest 9.2.0、NumPy 2.4.6、pandas 2.3.3、SciPy 1.18.1、scikit-learn 1.9.1、python-docx 1.2.0、pypdf 6.21.3、torch 2.8.0+cpu。
- Q3 验证器已新增 `--python` 参数，默认使用当前解释器，消除对不存在的工作树 `.venv` 的硬编码；未改模型或数值结果。

## 验收结果

| 检查 | 结果 | 说明 |
|---|---|---|
| 全量 pytest | FAIL | 57 passed, 1 failed；唯一失败为缺少 `model_independent_gbr.joblib` |
| Q1 绝对指标回归 | PASS | 1 passed |
| Q2 验证 | PASS | 全部节点、边、Top-3、稳定性、化学合理性和产物检查通过 |
| Q3 验证 | PASS | 7/7；三风格生成、阈值约束、迁移和脚本入口通过 |
| 图表风格 | PASS | 10 passed；PNG/PDF 配对与 manifest 另由验收 V17 通过 |
| 论文 PDF | PASS | 5/5；总 36 页，正文 30 页、前置 2 页、附录 4 页 |
| `acceptance_check.py` | FAIL | 13 项通过，V6/V11/V12/V15 四项失败，见下节 |
| `git diff --check` | PASS | 仅有 LF/CRLF 提示，无空白错误 |

## acceptance_check 失败解释

1. `V6_remote_tests`：旧验收项固定查找当前 worktree 内的 `worktrees/prediction-models/tests`，本次夜间 worktree 不嵌套队友 worktree；没有复制或触碰队友目录。
2. `V11_docx_skill`：旧验收项要求历史 `data/processed/paper_smoke/paper_smoke.docx` 冒烟文件，本次最终论文不以该占位冒烟工件作为交付。
3. `V12_paper_pdf`：旧验收项固定查找 `paper_smoke/paper_smoke_check.json`；实际最终论文检查位于 `data/processed/paper/paper_check.json`，本轮已独立 5/5 通过。
4. `V15_git`：该项只允许 `feat/toolchain-env` 或 `release/toolchain-hardening-*`，并要求远端分支和嵌套 prediction-models worktree，与本次指定的 `night/overnight-q123-20261030`、只做本地提交且禁止 push 的任务冲突；其运行时还处于提交前工作树状态，网络也不可达。

这些失败没有被改写为通过。V1–V5、V7–V10、V13、V14、V16、V17 均通过。

## 论文工件核对

| 工件 | 字节 | 页数口径 | 生成/修改时间 | SHA-256 |
|---|---:|---|---|---|
| `paper.pdf` | 2,128,684 | 36；正文 30、附录 4、前置 2 | PDF 元数据 `2026-10-30T07:53:31+08:00` | `540153755F7C8B8765C258FD1C0EF9F96A26001D3D28FA3B8DE5AB8EFCD8670A` |
| `paper.docx` | 1,864,433 | Word 导出的 PDF 为 36；OOXML `Pages=3` 为陈旧模板属性，不作权威页数 | 文件修改 `2026-10-30T07:52:45+08:00` | `1BE0448DE772637DD66B2D865980CFFF89963A8071D13E24FBA9D7E9EE11DD9F` |
| `paper_check.json` | 1,850 | 记录 36/30/4 | `generated_at_utc=2026-10-30T01:03:50Z` | `E61AD4978EC9D2768643A0461D2B6366AB6FE1DB6FA6D8CF0D0B2C02940EB5E6` |
| `paper_check.txt` | 1,145 | 记录 36/30/4 | 与 JSON 同批生成 | `B48795802A3DCE63A8B476EBFB433B005B5A40F03FB1342A4A8A6C63AF15A1D5` |

PDF 使用捆绑 Poppler 重新渲染为 36 张页面图并逐页目检；未见裁切、重叠、缺字、异常空白或页码中断。DOCX 的核心属性已清空最后修改者，但创建/修改时间和 `Pages=3` 仍含模板历史，故页数以 Word 2019 实际导出 PDF 和 PDF 校验器为准。

## 数据与提交边界

- `git diff HEAD -- data/raw` 为空，退出码 0；`data/raw` 未改变。
- 仓库内没有 ZIP/7z/RAR/TAR 或提交目录；仅保留未执行的 `scripts/package_submission.py`，未生成真实或占位提交包。
- `external/source_manifest.csv` 的五项来源均为 `not_downloaded`，用途限定为外部验证、元数据、基准复现或方法讨论；没有外部行进入训练、调参、补缺失或样本扩增。
- 未执行 push、merge、rebase、`reset --hard`，未触碰队友 worktree，未提交秘密或 `data/raw`。
- 首页的 `MCB260XXXX` 是赛规要求保留的匿名队伍编号占位，不是提交包；本轮未替换。

## 可分享范围

可分享：论文排版/题注硬化、Q2 图表修复、Q2/Q3 nightly 验证结果、Q3 验证器的解释器参数修复及本报告。

不可宣称：全量测试全绿、Q1 默认预测接口可用、TabPFN 与主协议公平优胜、Q2 因果关系、Q3 硬约束等同于真实性、消费者喜爱度得到验证。

最终本地 commit SHA 在提交完成后记录于 automation memory 和任务回执；提交对象自身无法可靠保存其自引用 SHA。
