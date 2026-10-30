# -*- coding: utf-8 -*-
"""Regression tests for the final integrated paper and Q1 protocol."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
FINAL_PAPER = ROOT / "data" / "processed" / "paper" / "paper.docx"
ARCHIVE_Q1 = ROOT / "data" / "processed" / "q1_tabpfn_final"


class Q1PaperRevisionTests(unittest.TestCase):
    def test_archived_protocol_remains_traceable(self):
        row = pd.read_csv(ARCHIVE_Q1 / "model_summary.csv").set_index("model").loc["tabpfn"]
        np.testing.assert_allclose(
            [row.R2_mean, row.Macro_MAE, row.Macro_RMSE],
            [0.2407293912437345, 0.3362367581706291, 0.47470748219473685],
            atol=1e-12,
        )

    def test_final_paper_uses_fixed_primary_metrics(self):
        text = "\n".join(p.text for p in Document(FINAL_PAPER).paragraphs)
        self.assertIn("227", text)
        self.assertIn("175/75", text)
        self.assertIn("0.2133", text)
        self.assertIn("0.8425", text)
        self.assertNotIn("0.240729", text)

    def test_final_paper_does_not_present_tabpfn_as_primary_comparison(self):
        doc = Document(FINAL_PAPER)
        table_first_cells = [table.rows[0].cells[0].text.strip() for table in doc.tables if table.rows]
        self.assertFalse(any("TabPFN" in cell for cell in table_first_cells))

    def test_final_paper_contains_validated_q2_q3_q4_evidence(self):
        text = "\n".join(p.text for p in Document(FINAL_PAPER).paragraphs).replace(" ", "")
        for value in ("83/150", "30/32", "42/48", "13.5%", "12.7%", "22/50", "28/50"):
            self.assertIn(value, text)
        self.assertIn("消费者喜爱度", text)

    def test_final_paper_has_no_engineering_residue(self):
        text = "\n".join(p.text for p in Document(FINAL_PAPER).paragraphs)
        for phrase in ("队友协议", "待完成", "当前提交", "AI辅助"):
            self.assertNotIn(phrase, text)

    def test_integration_manifest_and_descriptive_contract(self):
        stats = json.loads(
            (ROOT / "data" / "processed" / "final_integration" / "descriptive_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stats["samples"], 250)
        self.assertEqual(stats["raw_numeric_features"], 231)
        self.assertEqual(stats["modeling_features"], 227)
        self.assertEqual(stats["sensory_targets"], 50)
        self.assertEqual(stats["split"]["train"], 175)
        self.assertEqual(stats["split"]["holdout"], 75)

    def test_q3_manifest_has_one_slope_figure(self):
        manifest = json.loads(
            (ROOT / "data" / "processed" / "q3_figures" / "figures_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        names = [item.get("file") for item in manifest["figures"]]
        slope_count = sum("transfer_slope" in str(name) for name in names)
        self.assertEqual(slope_count, 1)

    def test_code_zip_hash_is_recorded(self):
        code_zip = ROOT / "code.zip"
        self.assertTrue(code_zip.is_file())
        digest = hashlib.sha256(code_zip.read_bytes()).hexdigest()
        manifest = json.loads(
            (ROOT / "data" / "processed" / "final_integration" / "team_integration_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        code_source = next(item for item in manifest["sources"] if item.get("path") == "code.zip")
        self.assertEqual(code_source["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
