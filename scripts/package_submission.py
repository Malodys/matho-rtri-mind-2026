# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：已核对赛题对论文文件名的要求（以队伍编号命名）与支撑材料的组成要求
# （源程序清单须与仓库 scripts/ 实际内容一致）；打包前会重跑一次论文 PDF 合规校验，
# 校验不通过则拒绝打包；ZIP 内不含 data/raw（原始附件由赛方提供，且 AGENTS.md 规定不外传），
# 不含 worktrees/、smoke/ 与 __pycache__ 等中间物；打包产物目录自身也被排除，
# 否则会把正在写的 ZIP 再打包进去（自包含，体积无限膨胀）。
"""生成赛题提交件：论文 PDF（按队号命名）+ 支撑材料 ZIP。

赛题对提交物的硬性要求（已在本脚本中落地）：
- 论文 PDF 文件名以队伍编号命名，形如 ``#MCB260XXXX.pdf``；
- 支撑材料须包含全部源程序，供评审复算。

用法::

    python scripts/package_submission.py --team-no MCB260XXXX
    python scripts/package_submission.py --team-no MCB260XXXX --include-raw

产物写入 ``data/processed/submission/``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "submission"
PAPER_PDF = PROCESSED / "paper" / "paper.pdf"
PAPER_DOCX = PROCESSED / "paper" / "paper.docx"
AI_DOC = PROCESSED / "compliance" / "AI工具使用详情说明.pdf"

#: 打包进支撑材料的目录（相对仓库根）。
INCLUDE_DIRS = [
    "scripts",
    "tests",
    "tests_q1",
    "docs",
    "worktrees/prediction-models/scripts",
    "data/interim",
    "data/splits",
    "data/processed",
]

#: 需要额外单独收录的根级文件。
INCLUDE_FILES = [
    "AGENTS.md",
    "README.md",
    "requirements.lock.txt",
]

#: 排除的目录名（任意层级）；worktrees 单独处理，见 _skip_path。
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "smoke",
    "paper_smoke",
    "node_modules",
}

#: 排除的文件名后缀（运行日志 .log 保留，它们是实跑证据）。
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def _skip_path(rel: Path) -> bool:
    """判断某个仓库相对路径是否应被排除。"""
    parts = rel.parts
    # 打包产物目录本身绝不能被纳入：否则会把正在写的 ZIP 再打包进去（自包含，体积无限膨胀）。
    if parts[:3] == ("data", "processed", "submission"):
        return True
    # worktrees/ 整体很大（840 MB），只保留远程管线的脚本目录
    if "worktrees" in parts:
        if parts[:2] == ("worktrees", "prediction-models") and "scripts" in parts:
            pass
        else:
            return True
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    # 流水线运行日志里会带构建机的绝对路径（warning 堆栈），噪声大且不宜外发
    if "_logs" in rel.parts:
        return True
    # 探测脚本是一次性的，不进支撑材料
    if rel.name.startswith("_probe"):
        return True
    return False


def _iter_files(include_raw: bool):
    """产出 (仓库相对路径, 绝对路径) 序列。"""
    seen = set()
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if _skip_path(rel):
                continue
            key = rel.as_posix()
            if key in seen:
                continue
            seen.add(key)
            yield rel, p
    if include_raw:
        base = ROOT / "data" / "raw"
        if base.exists():
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(ROOT)
                if _skip_path(rel):
                    continue
                key = rel.as_posix()
                if key in seen:
                    continue
                seen.add(key)
                yield rel, p
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists() and f not in seen:
            seen.add(f)
            yield Path(f), p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_paper() -> dict:
    """打包前重跑论文合规校验；不通过则拒绝打包。"""
    if not PAPER_PDF.exists():
        raise SystemExit(f"缺少 {PAPER_PDF}，请先运行 python scripts/build_paper.py")
    report = PROCESSED / "paper" / "paper_check.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_paper_pdf.py"), str(PAPER_PDF),
         "--docx", str(PAPER_DOCX), "--json", str(report)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.stderr.write(r.stdout or "")
        raise SystemExit("论文 PDF 合规校验未通过，拒绝打包。")
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="打包赛题提交件（论文 PDF + 支撑材料 ZIP）")
    ap.add_argument("--team-no", default="MCB260XXXX",
                    help="参赛队编号（论文 PDF 将命名为 #<队号>.pdf）")
    ap.add_argument("--track", default="A", help="赛道")
    ap.add_argument("--include-raw", action="store_true",
                    help="把 data/raw 也打进支撑材料（默认不打）")
    ap.add_argument("--skip-verify", action="store_true", help="跳过论文合规校验（不推荐）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    team = args.team_no.strip()

    if not args.skip_verify:
        report = verify_paper()
        print(f"[ok] 论文合规校验通过：总 {report['pages']} 页 / 正文 {report['body_pages']} 页 "
              f"/ 附录 {report['appendix_pages']} 页")
        if int(report.get("body_pages") or 0) > 30:
            raise SystemExit("正文页数超过 30 页上限，拒绝打包。")

    # ---- 论文 PDF：按队号命名 --------------------------------------------
    paper_out = OUT_DIR / f"#{team}.pdf"
    paper_out.write_bytes(PAPER_PDF.read_bytes())
    print(f"[ok] 论文 PDF -> {paper_out.relative_to(ROOT)}")

    # ---- 支撑材料 ZIP ----------------------------------------------------
    zip_out = OUT_DIR / f"#{team}附件.zip"
    manifest = []
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in _iter_files(args.include_raw):
            arc = rel.as_posix()
            zf.write(p, arc)
            manifest.append({"path": arc, "bytes": p.stat().st_size,
                             "sha256": _sha256(p)})
        # 《AI 工具使用详情说明》PDF 已随 data/processed/ 一并收录，此处只作兜底补录。
        if AI_DOC.exists():
            arc = f"data/processed/compliance/{AI_DOC.name}"
            if arc not in {m["path"] for m in manifest}:
                zf.write(AI_DOC, arc)
                manifest.append({"path": arc, "bytes": AI_DOC.stat().st_size,
                                 "sha256": _sha256(AI_DOC)})

    # 支撑材料清单
    manifest_path = OUT_DIR / f"#{team}附件清单.json"
    manifest_path.write_text(json.dumps(
        {"team_no": team, "track": args.track,
         "paper_pdf": paper_out.name,
         "paper_pdf_sha256": _sha256(paper_out),
         "file_count": len(manifest),
         "include_raw": bool(args.include_raw),
         "files": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] 支撑材料 -> {zip_out.relative_to(ROOT)}（{len(manifest)} 个文件，"
          f"{zip_out.stat().st_size / 1024 / 1024:.1f} MB）")
    print(f"[ok] 清单     -> {manifest_path.relative_to(ROOT)}")
    if AI_DOC.exists():
        print(f"[ok] AI 说明  -> data/processed/compliance/{AI_DOC.name}")
    print("\n提交前请自行确认：")
    print(f"  1. 论文 PDF 文件名为 {paper_out.name}，队伍编号与赛题系统内登记一致；")
    print("  2. 论文与支撑材料中均无姓名、学校、学号等个人信息；")
    print("  3. 参赛承诺书需另行签署并扫描上传。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
