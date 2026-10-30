# -*- coding: utf-8 -*-
# AI 辅助：Codex (GPT-6), OpenAI, 2026-10-30
# 人工复核状态：待团队核验；验证最终上下文与交叉验证输出。
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
from contextlib import nullcontext
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_tabpfn", ROOT / "scripts/q1_final_tabpfn.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FinalTabPFNTest(unittest.TestCase):
    def test_context_keeps_rows_order_and_missing_values(self):
        x = pd.DataFrame({"x": [np.nan, 2., 5.]})
        y = pd.DataFrame({"y": [3., 4., 7.]})
        a, b = runner.doubled_context(x, y)
        pd.testing.assert_frame_equal(a, pd.concat([x, x]))
        pd.testing.assert_frame_equal(b, pd.concat([y, y]))

    def test_fivefold_and_holdout_export(self):
        project = ROOT / "prediction-models"
        if not (project / "data/raw/1.csv").exists():
            project = ROOT.parent / "repo"
        if not (project / "data/raw/1.csv").exists():
            self.skipTest("Prepare the competition data in prediction-models first.")
        calls = []
        def predict(params, x, y, evaluation, **kwargs):
            n = len(x) // 2
            pd.testing.assert_frame_equal(x.iloc[:n], x.iloc[n:])
            pd.testing.assert_frame_equal(y.iloc[:n], y.iloc[n:])
            training = set(pd.util.hash_pandas_object(x, index=False))
            validation = set(pd.util.hash_pandas_object(evaluation, index=False))
            self.assertFalse(training & validation)
            calls.append((n, len(evaluation)))
            return np.broadcast_to(y.mean().to_numpy(), (len(evaluation), y.shape[1])).copy()
        (ROOT / "tmp").mkdir(exist_ok=True)
        with nullcontext(tempfile.mkdtemp(dir=ROOT / "tmp")) as folder:
            checkpoint = Path(folder) / "checkpoint"
            checkpoint.write_bytes(b"test")
            output = Path(folder) / "out"
            args = SimpleNamespace(experiment_root=project, model_path=checkpoint,
                data_dir=Path("data/splits/seed_0"), seed=0, output_dir=output, device="cpu")
            runner.run(args, predictor=predict)
            self.assertEqual(calls, [(140, 35)] * 5 + [(175, 75)])
            oof = pd.read_csv(output / "oof_predictions.csv")
            holdout = pd.read_csv(output / "test/test_predictions.csv")
            self.assertEqual(len(oof), 175 * 50)
            self.assertEqual(len(holdout), 75 * 50)
            self.assertFalse(set(oof.beer) & set(holdout.beer))
            self.assertFalse(oof.duplicated(["beer", "target"]).any())


if __name__ == "__main__":
    unittest.main()
