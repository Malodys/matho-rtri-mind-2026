import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tuning_protocol import make_store, save_selection


class TuningProtocolTests(unittest.TestCase):
    def test_complete_five_fold_search_selects_predictions_against_real_targets(self):
        output = Path(tempfile.mkdtemp())
        params = [{'width': 8}, {'width': 16}]
        targets = pd.DataFrame({'taste': np.arange(10.), 'aroma': np.arange(10.) * 2. + 1.})
        metadata = pd.DataFrame({'beer': np.arange(1, 11), 'tasting_category_fine': 'A'})
        folds = list(KFold(5).split(targets))
        predictions, done = make_store(output, params, (2, 10, 2))
        predictions[0] = targets.to_numpy()
        predictions[1] = 0.
        done[:] = True
        predictions.flush()
        done.flush()
        selected = save_selection(output, params, predictions, targets, metadata, folds)
        self.assertEqual(selected, 0)
        results = pd.read_csv(output / 'cv_results.csv')
        self.assertAlmostEqual(results.iloc[0]['mean_r2'], 1.)
        self.assertLess(results.iloc[1]['mean_r2'], 0.)
        oof = pd.read_csv(output / 'oof_predictions.csv')
        self.assertEqual(len(oof), 20)
        np.testing.assert_allclose(oof['actual'], oof['prediction'])
        resumed, flags = make_store(output, params, (2, 10, 2))
        self.assertTrue(flags.all())
        np.testing.assert_allclose(resumed[0], targets)
        with self.assertRaises(ValueError):
            make_store(output, [{'width': 4}, {'width': 16}], (2, 10, 2))

    def test_incomplete_search_cannot_select_a_winner(self):
        output = Path(tempfile.mkdtemp())
        values, _ = make_store(output, [{'width': 8}], (1, 10, 1))
        targets = pd.DataFrame({'taste': np.arange(10.)})
        metadata = pd.DataFrame({'beer': np.arange(10), 'tasting_category_fine': 'A'})
        with self.assertRaises(ValueError):
            save_selection(output, [{'width': 8}], values, targets, metadata, list(KFold(5).split(targets)))


if __name__ == '__main__':
    unittest.main()
