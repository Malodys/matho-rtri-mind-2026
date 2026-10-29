# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-28
# 人工核验点：本目录只新增问题一的测试，不触碰队友的 worktrees/prediction-models/tests；
#   断言内容与 scripts/q1_leakage_check.py 同源（同一批数值口径），两处互相印证。
"""问题一测试的公共配置：把仓库 scripts/ 目录加入 sys.path。"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
