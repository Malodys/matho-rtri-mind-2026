import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tune_tabpfn import candidate_grid, store_batch
from tuning_protocol import make_store


class TabPFNSearchTests(unittest.TestCase):
    def test_short_prediction_batch_does_not_mark_missing_targets_complete(self):
        output = Path(tempfile.mkdtemp())
        predictions, done = make_store(output, [{'n_estimators': 4}], (1, 10, 2))
        with self.assertRaises(ValueError):
            store_batch(predictions, done, 0, 0, np.array([0, 1]), [0, 1], [np.array([1., 2.])])
        self.assertFalse(done.any())
        self.assertTrue(np.isnan(predictions).all())

    def test_grid_covers_each_configuration_once_and_includes_default(self):
        candidates = candidate_grid()
        self.assertEqual(len(candidates), 18)
        self.assertEqual(len({tuple(sorted(p.items())) for p in candidates}), 18)
        for estimators in [4, 8, 16]:
            for temperature in [0.7, 1., 1.3]:
                for average in [False, True]:
                    self.assertIn({
                        'n_estimators': estimators, 'softmax_temperature': temperature,
                        'average_before_softmax': average,
                    }, candidates)


if __name__ == '__main__':
    unittest.main()
