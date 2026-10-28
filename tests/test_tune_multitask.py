import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tune_multitask as tuning
from multitask_cv import fit_member as original_fit_member


class MultiTaskSearchTests(unittest.TestCase):
    def test_candidates_are_distinct_reproducible_and_include_original(self):
        candidates = tuning.search_candidates()
        self.assertEqual(len(candidates), 48)
        self.assertEqual(candidates, tuning.search_candidates())
        self.assertNotEqual(candidates, tuning.search_candidates(1))
        self.assertEqual(len({tuple(sorted(p.items())) for p in candidates}), 48)
        self.assertIn(dict(hidden=(64, 16), dropout=.15, learning_rate=.001,
                           weight_decay=.01, input_clip=5.), candidates)
        for name in ('hidden', 'dropout', 'learning_rate', 'weight_decay', 'input_clip'):
            self.assertGreater(len({p[name] for p in candidates}), 1)

    def test_fit_predict_keeps_evaluation_out_of_training_and_uses_member_seeds(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 3))
        x[2, 1] = np.nan
        y = np.column_stack((x[:, 0] + 5, x[:, 0] * 2))
        evaluation = np.array([[999., np.nan, 999.], [-999., 999., 0.]])
        training_calls = []

        def observed_fit(features, targets, **kwargs):
            np.testing.assert_equal(features, x)
            np.testing.assert_equal(targets, y)
            training_calls.append(kwargs['seed'])
            return original_fit_member(features, targets, **kwargs)

        params = tuning.search_candidates()[0]
        with patch.object(tuning, 'fit_member', side_effect=observed_fit):
            predicted, records = tuning.fit_predict(
                params, x, y, evaluation, seed=1000, device='cpu', members=2,
                max_epochs=3, patience=2, return_info=True,
            )
        self.assertEqual(training_calls, [1000, 1001])
        self.assertEqual(predicted.shape, (2, 2))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertEqual([row['seed'] for row in records], [1000, 1001])
        self.assertTrue(all(1 <= row['selected_epochs'] <= 3 for row in records))

    def test_empty_ensemble_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            tuning.fit_predict(tuning.search_candidates()[0], np.ones((10, 2)),
                               np.ones((10, 2)), np.ones((2, 2)), members=0)


if __name__ == '__main__':
    unittest.main()
