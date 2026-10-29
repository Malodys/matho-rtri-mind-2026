# 论文产物说明（data/processed/paper）

- 生成时间口径：论文 AI 辅助声明日期 2026-10-29。
- 论文总页数：36 页（其中正文 30 页，附录 4 页）。
- 参考文献来源：docs/refs_verified.json，条目数 38。
- 消费者层产物：已就绪并并入 5.2 / 5.4。

## 降级说明


## 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\build_paper.py
.\.venv\Scripts\python.exe scripts\check_paper_pdf.py data/processed/paper/paper.pdf --docx data/processed/paper/paper.docx
```
