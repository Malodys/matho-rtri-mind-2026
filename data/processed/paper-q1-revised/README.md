# 论文产物说明（data/processed/paper-q1-revised）

- 生成时间口径：论文 AI 辅助声明日期 2026-10-29。
- 论文总页数：29 页（其中正文 22 页，附录 5 页）。
- 参考文献来源：docs/refs_verified.json，条目数 38。
- 消费者层产物：已并入 5.2 / 5.4。

## 复现命令

```powershell
$env:DOCX_TO_PDF_BACKEND = "libreoffice"  # 无 Word COM 时使用
python scripts\build_paper.py --team-no MCB260XXXX --track A
python scripts\check_paper_pdf.py data/processed/paper-q1-revised/paper.pdf --docx data/processed/paper-q1-revised/paper.docx
```
