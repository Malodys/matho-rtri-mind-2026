# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：验收项判定口径与《论文格式及提交规范》《AI 工具使用规定》逐条对应后写定。
"""工具链交付验收：逐条核验 V1–V15，输出可复查的 PASS/FAIL 报告。

用法：
    python scripts/acceptance_check.py [--repo-root <仓库根>] [--json <报告路径>]
退出码：0 = 全部通过。
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

PII_PATTERNS = [r"学校", r"学院", r"大学", r"学号", r"指导教师", r"指导老师", r"班级"]
# AGENTS.md 属内部契约文件，会正当地写出"不得出现姓名/学校/学号"这类禁止性表述，
# 因此对其只查"具体身份值"（真实校名、长数字串、邮箱、姓名取值），不查词语本身。
PII_VALUE_PATTERNS = [
    r"[\u4e00-\u9fff]{2,10}(?:大学|学院)",   # 真实校名
    # 学号/长数字标识：**独立出现**的 6 位以上数字串才算命中。
    # 不能紧跟在字母数字词或小数点之后，否则会误伤合法的文件名与数值常量
    # （如 "tabpfn-v3.5-20260909.safetensors"、ln(10)=2.302585、阈值 0.206333）。
    r"(?<![\w.])\d{6,}(?![\w.])",
    r"[\w.+-]+@[\w-]+\.[\w.]+",               # 邮箱
    r"(?:姓名|学号|班级|指导教师|指导老师)\s*[:：]\s*[^\s，。；]{2,}",  # 字段取值
]
AGENTS_SECTIONS = [
    "Setup commands", "Project layout", "仓库与分支模型", "AI 辅助注释规范",
    "数据分层与不可变约定", "Testing instructions", "数据语义备忘",
    "PR & commit conventions", "附录素材", "Security",
]
TARGET_MODULES = [
    "pandas", "numpy", "sklearn", "scipy", "statsmodels", "matplotlib", "seaborn",
    "openpyxl", "torch", "tabpfn", "networkx", "igraph", "xgboost", "lightgbm",
    "shap", "optuna", "copulas", "einops", "tqdm", "pytest", "docx", "pypdf", "pdfplumber",
]


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=60)
        return out.stdout.strip()
    except Exception:
        return ""


def _py(root: Path, code: str) -> tuple[int, str]:
    py = root / ".venv" / "Scripts" / "python.exe"
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=300)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return 1, f"{exc.__class__.__name__}: {exc}"


def run_checks(root: Path) -> list[dict]:
    res: list[dict] = []

    def add(vid, desc, passed, detail):
        res.append({"id": vid, "desc": desc, "passed": bool(passed), "detail": str(detail)})

    # V1 venv 就位
    py_exe = root / ".venv" / "Scripts" / "python.exe"
    v1 = py_exe.is_file()
    v1_detail = "缺失"
    if v1:
        rc, out = _py(root, "import sys;print(sys.version.split()[0])")
        v1_detail = out.strip() or "无法执行"
        v1 = v1 and rc == 0 and v1_detail.startswith("3.12")
    add("V1_venv", ".venv 存在且为 Python 3.12.x", v1, v1_detail)

    # V2 全量导入
    code = ("import importlib,sys\n"
            "mods=" + repr(TARGET_MODULES) + "\n"
            "bad=[]\n"
            "for m in mods:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception as e: bad.append(m+':'+type(e).__name__)\n"
            "print('BAD='+','.join(bad) if bad else 'BAD=')\n")
    rc, out = _py(root, code)
    bad = out.strip().splitlines()[-1] if out.strip() else "no output"
    add("V2_imports", "全部目标包可导入", rc == 0 and "BAD=" in bad and bad.endswith("BAD="), bad)

    # V3 锁文件可复现
    lock = root / "requirements.lock.txt"
    v3 = lock.is_file() and "torch==2.8.0+cpu" in lock.read_text(encoding="utf-8")
    add("V3_lock", "requirements.lock.txt 存在且含 torch CPU 钉版", v3,
        f"{lock.name} 存在={lock.is_file()}")

    # V4 torch 变体
    rc, out = _py(root, "import torch;print(torch.__version__, torch.cuda.is_available())")
    v4 = rc == 0 and out.strip().endswith("False")
    add("V4_torch_cpu", "torch 为 CPU 轮子且 cuda 不可用", v4, out.strip())

    # V5 中文出图
    png = root / "data" / "processed" / "cjk_probe.png"
    v5 = png.is_file() and png.stat().st_size > 5000
    add("V5_cjk_plot", "中文出图产物存在且非空", v5, f"{png.name} {png.stat().st_size if png.is_file() else 0} bytes")

    # V6 远端测试
    wt = root / "worktrees" / "prediction-models"
    tests = wt / "tests"
    v6 = tests.is_dir() and len(list(tests.glob("test_*.py"))) >= 7
    add("V6_remote_tests", "远端管线测试文件已就位（≥7 个）", v6,
        f"{len(list(tests.glob('test_*.py'))) if tests.is_dir() else 0} 个测试文件")

    # V7 TabPFN 复现结果（优先取本地并行复现产物，其次取远端管线 results 目录）
    cands = []
    local_repro = root / "data" / "processed" / "q1_repro_tabpfn" / "cv_summary.csv"
    if local_repro.is_file():
        cands.append(local_repro)
    wt_results = root / "worktrees" / "prediction-models" / "results"
    if wt_results.is_dir():
        cands.extend(sorted(wt_results.glob("tabpfn_3_5_cv5_seed_*/cv_summary.csv")))

    v7 = False
    v7_detail = "未产出结果"
    for cand in cands:
        try:
            import pandas as pd
            df = pd.read_csv(cand, index_col=0)
            row = df[df.index.astype(str).str.contains("cv_mean", na=False)]
            if not len(row):
                continue
            val = float(row["r2"].iloc[0])
            n_targets = len(pd.read_csv(cand.parent / "fold_target_metrics.csv")["target"].unique())
            ok = abs(val - 0.2055) <= 0.03
            v7 = v7 or ok
            v7_detail = (f"{cand.parent.name}: 五折平均 R2={val:.4f}"
                         f"（参考 0.2055，容差 ±0.03 → 判定{'通过' if ok else '超差'}）；"
                         f"目标数={n_targets}")
        except Exception as exc:
            v7_detail = f"{cand}: 读取失败 {exc}"
    add("V7_tabpfn_repro", "TabPFN 五折平均 R² 复现且落在容差内", v7, v7_detail)

    # V8/V9/V10 能力栈
    smoke = root / "data" / "processed" / "smoke" / "stack_smoke.json"
    if smoke.is_file():
        items = json.loads(smoke.read_text(encoding="utf-8"))
        for it in items:
            vid = it.get("id", "")
            if vid.startswith("V8"):
                add("V8_graph_stack", it.get("desc", ""), it.get("passed"), it)
            elif vid.startswith("V9"):
                add("V9_attribution_stack", it.get("desc", ""), it.get("passed"),
                    f"R2={it.get('holdout_r2')}, shap Top3={it.get('shap_top3')}")
            elif vid.startswith("V10"):
                add("V10_generation_stack", it.get("desc", ""), it.get("passed"),
                    f"cols={it.get('n_numeric_cols')}, styles={it.get('styles_seen')}")
    else:
        add("V8_V9_V10_stack", "能力栈冒烟报告存在", False, "stack_smoke.json 缺失")

    # V11 docx 技能
    cli = Path(r"C:\Users\10857\.minimax\.builtin-skills\docx\scripts\dotnet\MiniMaxAIDocx.Cli")
    smoke_docx = root / "data" / "processed" / "paper_smoke" / "paper_smoke.docx"
    v11 = cli.is_dir() and smoke_docx.is_file()
    add("V11_docx_skill", "docx 技能后端可用且模板冒烟产物存在", v11,
        f"CLI 存在={cli.is_dir()}；冒烟 docx={smoke_docx.is_file()}")

    # V12 论文 PDF 五项校验
    chk = root / "data" / "processed" / "paper_smoke" / "paper_smoke_check.json"
    v12 = False
    v12_detail = "校验报告缺失"
    if chk.is_file():
        rep = json.loads(chk.read_text(encoding="utf-8"))
        v12 = bool(rep.get("all_passed"))
        v12_detail = "; ".join(f"{r['id']}={'PASS' if r['passed'] else 'FAIL'}"
                               for r in rep.get("results", []))
    add("V12_paper_pdf", "论文 PDF 五项校验全部通过", v12, v12_detail)

    # V13 AI 说明 PDF
    comp = root / "data" / "processed" / "compliance"
    ai_pdf = comp / "AI工具使用详情说明.pdf"
    ai_chk = comp / "AI工具使用详情说明_check.json"
    v13 = ai_pdf.is_file() and ai_chk.is_file()
    v13_detail = f"pdf={ai_pdf.is_file()}"
    if ai_chk.is_file():
        rep = json.loads(ai_chk.read_text(encoding="utf-8"))
        v13 = v13 and bool(rep.get("all_passed"))
        v13_detail += "; " + "; ".join(f"{r['id']}={'PASS' if r['passed'] else 'FAIL'}"
                                       for r in rep.get("results", []))
    add("V13_ai_usage_doc", "《AI 工具使用详情说明》PDF 生成且校验通过", v13, v13_detail)

    # V14 AGENTS.md
    agents = root / "AGENTS.md"
    v14 = False
    v14_detail = "AGENTS.md 缺失"
    if agents.is_file():
        text = agents.read_text(encoding="utf-8")
        missing = [s for s in AGENTS_SECTIONS if s not in text]
        hits = []
        for pat in PII_VALUE_PATTERNS:
            for m in re.finditer(pat, text):
                hits.append(f"{m.group(0)!r}")
        has_clause = ("个人信息" in text) and ("不得" in text or "禁止" in text)
        v14 = not missing and not hits and has_clause
        v14_detail = (f"缺失章节={missing or '无'}；身份值命中={hits or '无'}；"
                      f"含匿名化约束条款={has_clause}")
    add("V14_agents_md", "AGENTS.md 章节齐全、含匿名化约束且无具体身份信息", v14, v14_detail)

    # V15 Git 归位
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    ahead = _git(root, "rev-list", "--count", "origin/main..HEAD")
    wt_list = _git(root, "worktree", "list")
    dirty_tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    untracked = [ln[3:].strip() for ln in _git(root, "status", "--porcelain").splitlines()
                 if ln.startswith("??")]
    unexpected_untracked = [u for u in untracked if not u.startswith("data/processed/")]
    commits = _git(root, "log", "--oneline", "origin/main..HEAD")
    commit_lines = [ln for ln in commits.splitlines() if ln.strip()]
    remote_branches = _git(root, "ls-remote", "--heads", "origin")
    on_remote = f"refs/heads/{branch}" in remote_branches
    v15 = (branch == "feat/toolchain-env" and ahead.isdigit() and int(ahead) >= 2
           and "prediction-models" in wt_list and not dirty_tracked
           and not unexpected_untracked and bool(commit_lines) and on_remote)
    add("V15_git", "分支/提交/worktree 就位，工作树干净，分支已在远端留存", v15,
        f"branch={branch}, upstream={upstream}, 相对 origin/main 提交数={ahead}, "
        f"跟踪文件改动={dirty_tracked or '无'}, 非预期未跟踪={unexpected_untracked or '无'}, "
        f"远端存在该分支={on_remote}")

    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="工具链交付验收 V1–V15")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    root = Path(args.repo_root)

    results = run_checks(root)
    all_ok = all(r["passed"] for r in results)

    buf = io.StringIO()
    buf.write("MathorCup 赛道A · 工具链交付验收报告\n")
    buf.write(f"仓库: {root}\n\n")
    for r in results:
        buf.write(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']} — {r['desc']}\n")
        buf.write(f"       {r['detail']}\n\n")
    buf.write(f"结论: {'全部通过' if all_ok else '存在未通过项'}\n")

    out = root / "data" / "processed" / "acceptance_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue(), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(buf.getvalue())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
