# MathorCup 赛道 A：啤酒化学指纹与风味建模

本项目围绕 250 款啤酒的化学矩阵、感官维度、化合物关联、风格条件生成和外部价格/描述符验证，提供可复现的分析脚本、论文产物和提交前验收。`data/raw/` 是只读赛题附件；主训练集不会并入外部数据，外部来源只按 [external/source_manifest.csv](external/source_manifest.csv) 和 [docs/external_data_policy.md](docs/external_data_policy.md) 做验证或元数据补充。

## 环境与复现

锁定依赖见 [requirements.lock.txt](requirements.lock.txt)。标准环境使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

本机若不能使用 `.venv`，可使用 Codex/CI 提供的 Python 解释器，并显式传给验收脚本：

```powershell
python scripts/acceptance_check.py --python <python-3.12-path> --json data/processed/acceptance_report.json
```

问题一的正式结果优先使用已锁定的 runs：

```powershell
.\.venv\Scripts\python.exe scripts/q1_pipeline.py --skip-fit --n-boot 2000
.\.venv\Scripts\python.exe -m pytest tests_q1 -q
```

`model_summary.csv` 中 `abs_valid_*` 是三维有效绝对轨汇总；`abs_threshold_*` 和 `abs_fallback_*` 只表示各自子集，兼容字段 `abs_*` 等于 `abs_valid_*`。TabPFN 当前产物采用原始 231 特征和 KFold 的既有复现协议，不与主模型的 227 特征、固定 175/75 留出结果直接比较。

## 论文与提交

```powershell
.\.venv\Scripts\python.exe scripts/build_paper.py --team-no <真实队号>
.\.venv\Scripts\python.exe scripts/check_paper_pdf.py data/processed/paper/paper.pdf `
  --docx data/processed/paper/paper.docx `
  --json data/processed/paper/paper_check.json `
  --text data/processed/paper/paper_check.txt
.\.venv\Scripts\python.exe scripts/build_release_manifest.py
.\.venv\Scripts\python.exe scripts/package_submission.py --team-no <真实队号>
```

没有真实队号时禁止生成最终提交包；脚本会拒绝 `MCB260XXXX` 或含 `X` 的占位符。提交包不包含 `data/raw/`，也不包含预训练权重。论文中的 Q2 稳健关联是候选关联而非因果结论；Q3 的零违规是硬约束合规，不是无约束生成真实性；Q4 不构造消费者喜爱度标签，跨币种价格结果只作方向性证据。

## 交付检查

发布前应确认 `git diff --check`、全量测试、论文 JSON/TXT 同批生成、PDF 正文不超过 30 页、无占位队号、无意外未跟踪文件，并运行 [scripts/build_release_manifest.py](scripts/build_release_manifest.py) 记录 Git SHA、环境、数据/论文哈希、测试和外部来源清单。
