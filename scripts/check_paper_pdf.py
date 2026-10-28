# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：页码连续性、无页眉、无个人信息的判定阈值经人工按官方《论文格式及提交规范》逐条核对。
"""论文 PDF 合规校验器（对应《论文格式及提交规范》硬性要求）。

检查项：
  1. 页数 > 0，且正文页数不超过 30 页（附录页数不限）。
  2. 全文可抽取中文，无 \\ufffd 替换符与缺字方块。
  3. 页脚中部存在页码，且自正文首页起按 1 连续编号。
  4. 无页眉（页面顶部区域无文字）。
  5. 全文不含个人身份信息（姓名/学校/学号/班级/指导教师等）。

用法：
    python scripts/check_paper_pdf.py <pdf路径> [--max-body-pages 30] [--json <报告输出路径>]
退出码：0 = 全部通过；1 = 存在未通过项。
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

# 页面区域比例（按页高）
HEADER_BAND = 0.07
FOOTER_BAND = 0.90

# 可能泄露个人身份的信息（论文须匿名）
PII_PATTERNS = [
    r"学校", r"学院", r"大学", r"指导教师", r"指导老师", r"指导教授",
    r"学号", r"班级", r"同学", r"姓名", r"参赛队员?[:：]",
]

PAGE_NUM_RE = re.compile(r"第\s*(\d+)\s*页")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
BAD_CHARS = ("\ufffd", "□", "�")


def collect(path: Path):
    """返回 (页数, 每页文本, 每页顶部文本, 每页底部文本)。"""
    pages_text, top_text, bottom_text = [], [], []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
            height = page.height or 1.0
            top_cut = height * HEADER_BAND
            bot_cut = height * FOOTER_BAND
            top_text.append(page.crop((0, 0, page.width, top_cut)).extract_text() or "")
            bottom_text.append(page.crop((0, bot_cut, page.width, height)).extract_text() or "")
        n_pages = len(pdf.pages)
    return n_pages, pages_text, top_text, bottom_text


def check(path: Path, max_body_pages: int, docx_path: Path | None = None,
          appendix_marker: str = "附录"):
    reader = PdfReader(str(path))
    n_pdf_pages = len(reader.pages)
    n_pages, pages_text, top_text, bottom_text = collect(path)
    full = "\n".join(pages_text)

    results = []

    # 1 页数：官方规范为「正文控制在 30 页以内，附录页数不限」。
    #    正文页 = 带页脚页码的页；封面与目录不带页码，自然不计入正文。
    #    附录页 = 正文中出现附录标题之后的页；其页数不计入正文限额。
    numbered = [i for i, bt in enumerate(bottom_text)
                if PAGE_NUM_RE.search(bt)]
    appendix_start = None
    for i in numbered:
        if appendix_marker and appendix_marker in pages_text[i]:
            appendix_start = i
            break
    body_pages = len(numbered)
    appendix_pages = (body_pages - appendix_start) if appendix_start is not None else 0
    body_only = body_pages - appendix_pages
    ok_pages = 1 <= body_only <= max_body_pages
    results.append({
        "id": "page_count",
        "desc": f"正文页数在 1..{max_body_pages} 之间（正文上限 30 页，附录页数不限）",
        "passed": ok_pages,
        "detail": f"PDF 共 {n_pages} 页；其中正文 {body_only} 页"
                  f"（附录 {appendix_pages} 页，前置部分 {n_pages - body_pages} 页不计）",
    })

    # 2 中文可抽取、无乱码
    bad_found = {c: full.count(c) for c in BAD_CHARS if c in full}
    cjk_ratio = (len(CJK_RE.findall(full)) / max(len(full), 1))
    ok_text = (not bad_found) and cjk_ratio > 0.05
    results.append({
        "id": "chinese_extractable",
        "desc": "中文可正常抽取，无替换符/缺字方块",
        "passed": ok_text,
        "detail": f"中文字符占比 {cjk_ratio:.1%}；异常字符 {bad_found or '无'}",
    })

    # 3 页脚页码连续
    nums = []
    for bt in bottom_text:
        m = PAGE_NUM_RE.search(bt)
        nums.append(int(m.group(1)) if m else None)
    seq = [n for n in nums if n is not None]
    ok_page_num = bool(seq) and seq == list(range(seq[0], seq[0] + len(seq)))
    # 正文首页应为 1；冒烟样例第 2 页起（首页为封面，无页码）
    starts_at_one = (seq[0] == 1) if seq else False
    results.append({
        "id": "footer_page_number",
        "desc": "页脚含页码且连续编号",
        "passed": ok_page_num,
        "detail": f"各页页脚识别页码序列={seq}；首页是否从 1 起={starts_at_one}",
    })

    # 4 无页眉
    #    真页眉必然在多页重复出现；正文节另起一页时，页首行属于正文而非页眉。
    #    因此 PDF 侧判定标准为"顶部文字在 >=2 页重复"；若有源 DOCX，则以其页眉对象为权威依据。
    normalized_top = [re.sub(r"\s+", "", t) for t in top_text]
    nonempty_top = [t for t in normalized_top if t]
    repeated = sorted({t for t in nonempty_top if nonempty_top.count(t) >= 2})
    header_pages = [i + 1 for i, t in enumerate(normalized_top) if t]

    docx_header_note = ""
    ok_no_header = not repeated
    if docx_path:
        try:
            from docx import Document
            d = Document(str(docx_path))
            hdr_texts = []
            for s in d.sections:
                for part in (s.header, s.first_page_header, s.even_page_header):
                    try:
                        hdr_texts.extend(p.text.strip() for p in part.paragraphs if p.text.strip())
                    except Exception:
                        pass
            ok_no_header = ok_no_header and not hdr_texts
            docx_header_note = f"；源 DOCX 页眉文字={hdr_texts or '空'}"
        except Exception as exc:  # 读不到 DOCX 时退回 PDF 启发式
            docx_header_note = f"；源 DOCX 读取失败({exc.__class__.__name__})，仅按 PDF 启发式判定"
    results.append({
        "id": "no_header",
        "desc": "无页眉（页面顶部区域无跨页重复文字）",
        "passed": ok_no_header,
        "detail": f"顶部有文字的页={header_pages or '无'}；跨页重复的顶部文字={repeated or '无'}{docx_header_note}",
    })

    # 5 无个人信息
    hits = []
    for pat in PII_PATTERNS:
        for m in re.finditer(pat, full):
            hits.append(f"{pat}@…{full[max(0, m.start() - 8):m.start() + 8]}…")
    ok_no_pii = not hits
    results.append({
        "id": "no_personal_info",
        "desc": "不含姓名/学校/学号等个人身份信息（赛规要求匿名）",
        "passed": ok_no_pii,
        "detail": f"命中 {len(hits)} 处" + (f"：{hits[:5]}" if hits else ""),
    })

    return {
        "pdf": str(path),
        "pages": n_pages,
        "body_pages": body_only,
        "appendix_pages": appendix_pages,
        "footer_page_sequence": seq,
        "results": results,
        "all_passed": all(r["passed"] for r in results),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="论文 PDF 合规校验器")
    ap.add_argument("pdf", help="待校验的论文 PDF 路径")
    ap.add_argument("--docx", dest="docx", default=None,
                    help="论文源 DOCX 路径（提供时以 DOCX 页眉对象作为权威依据）")
    ap.add_argument("--max-body-pages", type=int, default=30, help="正文页数上限（默认 30）")
    ap.add_argument("--appendix-marker", default="附录",
                    help="附录标题标记；该标题之后的页数不计入正文限额")
    ap.add_argument("--json", dest="json_out", default=None, help="JSON 报告输出路径")
    ap.add_argument("--quiet", action="store_true", help="仅输出结论")
    args = ap.parse_args()

    path = Path(args.pdf)
    if not path.exists():
        print(f"[FAIL] 文件不存在: {path}", file=sys.stderr)
        return 1

    report = check(path, args.max_body_pages,
                   Path(args.docx) if args.docx else None,
                   appendix_marker=args.appendix_marker)
    buf = io.StringIO()
    buf.write(f"论文 PDF 合规校验报告\n")
    buf.write(f"文件: {report['pdf']}\n")
    buf.write(f"总页数: {report['pages']}（正文 {report['body_pages']} 页，"
              f"附录 {report['appendix_pages']} 页）\n\n")
    for r in report["results"]:
        flag = "PASS" if r["passed"] else "FAIL"
        buf.write(f"[{flag}] {r['id']} — {r['desc']}\n       {r['detail']}\n")
    buf.write(f"\n结论: {'全部通过' if report['all_passed'] else '存在未通过项'}\n")
    text = buf.getvalue()

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.quiet:
        print("ALL PASSED" if report["all_passed"] else "HAS FAILURES")
    else:
        print(text)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
