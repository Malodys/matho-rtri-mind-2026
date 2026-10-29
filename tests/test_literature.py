# -*- coding: utf-8 -*-
# AI 辅助：MiniMax Code (MiniMax-M3.1-Flash), MiniMax, 2026-10-29
# 人工核验点：断言的字段集与主控下发给 B1 的著录契约逐项对齐；主锚点 DOI 与
# Nature Communications 原文页面的 JSON-LD 元数据一致。
"""文献著录完整性测试：守住「真实存在、著录可核实、编号有序」三条底线。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REFS_JSON = DOCS / "refs_verified.json"
LITERATURE_MD = DOCS / "literature.md"

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
REQUIRED_FIELDS = ["key", "citation_number", "type", "authors", "title", "year",
                   "supports_section", "supports_claim", "verified"]

ANCHOR_DOI = "10.1038/s41467-024-46346-0"
ANCHOR_TITLE = "Predicting and improving complex beer flavor through machine learning"


@pytest.fixture(scope="module")
def refs():
    assert REFS_JSON.exists(), f"缺少 {REFS_JSON}"
    data = json.loads(REFS_JSON.read_text(encoding="utf-8"))
    entries = data["refs"] if isinstance(data, dict) and "refs" in data else data
    assert isinstance(entries, list) and entries, "refs_verified.json 结构异常"
    return entries


def test_at_least_14_references(refs):
    assert len(refs) >= 14, f"文献仅 {len(refs)} 条，要求 ≥14 条"


def test_required_fields_present(refs):
    for entry in refs:
        for field in REQUIRED_FIELDS:
            assert field in entry, f"文献 {entry.get('key')} 缺字段 {field}"
        assert entry["type"] in ("journal", "book", "web", "proceedings"), entry["type"]
        assert isinstance(entry["citation_number"], int)
        assert entry["year"] and 1900 <= int(entry["year"]) <= 2030
        assert entry["authors"].strip(), f"文献 {entry['key']} 无作者"


def test_citation_numbers_unique_and_sequential(refs):
    numbers = sorted(e["citation_number"] for e in refs)
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"引用编号应从 1 连续：{numbers}")


def test_anchor_paper_is_reference_one(refs):
    anchor = next((e for e in refs if e["citation_number"] == 1), None)
    assert anchor is not None, "参考文献 [1] 缺失"
    assert anchor["title"] == ANCHOR_TITLE, f"[1] 标题不符：{anchor['title']}"
    assert anchor["doi"] == ANCHOR_DOI, f"[1] DOI 不符：{anchor.get('doi')}"
    assert "Nature Communications" in anchor["journal"]
    assert "Verstrepen" in anchor["authors"]
    assert int(anchor["year"]) == 2024


def test_journal_entries_have_volume_pages_and_doi(refs):
    for entry in refs:
        if entry["type"] != "journal":
            continue
        assert entry.get("volume_issue"), f"期刊文献缺卷期：{entry['key']}"
        assert entry.get("pages"), f"期刊文献缺页码：{entry['key']}"
        doi = entry.get("doi", "")
        assert DOI_RE.match(doi), f"DOI 格式非法：{entry['key']} -> {doi}"


def test_web_entries_have_url_and_access_date(refs):
    for entry in refs:
        if entry["type"] != "web":
            continue
        assert entry.get("url", "").startswith("http"), f"网络资源缺网址：{entry['key']}"
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", entry.get("accessed", "")), (
            f"网络资源缺访问日期：{entry['key']}")


def test_verified_flag_is_boolean(refs):
    for entry in refs:
        assert isinstance(entry.get("verified"), bool), (
            f"verified 必须是布尔：{entry['key']}")
    unverified = [e["key"] for e in refs if not e["verified"]]
    # 未核实条目允许存在，但必须列入 literature.md 的存疑清单
    if unverified:
        md = LITERATURE_MD.read_text(encoding="utf-8")
        assert "未核实" in md or "存疑" in md, (
            f"存在未核实文献 {unverified} 但 literature.md 未声明存疑清单")


def test_literature_md_has_required_sections():
    assert LITERATURE_MD.exists(), f"缺少 {LITERATURE_MD}"
    text = LITERATURE_MD.read_text(encoding="utf-8")
    for token in ["Nature Communications", "10.1038/s41467-024-46346-0", "DOI", "访问"]:
        assert token in text, f"literature.md 缺少关键内容：{token}"


def test_no_local_path_leakage_in_docs():
    for path in [REFS_JSON, LITERATURE_MD]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for bad in ["F:\\cs\\mothercup", "C:\\Users\\", "F:/cs/mothercup"]:
            assert bad not in text, f"{path.name} 泄漏本地路径：{bad}"
