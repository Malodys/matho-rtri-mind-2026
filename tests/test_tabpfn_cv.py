import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tabpfn_cv import cross_validate, evaluate_test, load_split


class CrossValidationTests(unittest.TestCase):
    def test_holdout_predictions_use_only_training_scores(self):
        training = (
            pd.DataFrame({'chemical': [1., 2., 3., 4.]}),
            pd.DataFrame({'taste': [1., 2., 3., 4.]}),
            pd.DataFrame({'beer': [1, 2, 3, 4], 'tasting_category_fine': 'A'}),
        )
        testing = (
            pd.DataFrame({'chemical': [5., 6.]}),
            pd.DataFrame({'taste': [5., 6.]}),
            pd.DataFrame({'beer': [5, 6], 'tasting_category_fine': 'A'}),
        )
        path = Path(tempfile.mkdtemp())
        evaluate_test(training, testing, path, DummyRegressor)
        predictions = pd.read_csv(path / 'test_predictions.csv')
        self.assertEqual(predictions['beer'].tolist(), [5, 6])
        np.testing.assert_allclose(predictions['prediction'], [2.5, 2.5])
        metric = pd.read_csv(path / 'target_metrics.csv').iloc[0]
        self.assertAlmostEqual(metric['mae'], 3.0)
        self.assertAlmostEqual(metric['rmse'], np.sqrt(9.25))
        self.assertAlmostEqual(metric['r2'], -36.0)
        testing[2].loc[0, 'beer'] = 1
        with self.assertRaisesRegex(ValueError, 'overlap'):
            evaluate_test(training, testing, path, DummyRegressor)

    def test_loading_aligns_beers_and_preserves_missing_values(self):
        path = Path(tempfile.mkdtemp())
        pd.DataFrame({
            'beer': [1, 2], 'beer_id': [101, 102],
            'tasting_category_fine': ['A', 'B'], 'chemical': [np.nan, 3.0],
        }).to_csv(path / '1_train.csv', index=False)
        pd.DataFrame({
            'beer': [2, 1], 'beer_id': [102, 101],
            'tasting_category_fine': ['B', 'A'], 'sensory': [20.0, 10.0],
        }).to_csv(path / '4_train.csv', index=False)
        features, targets, metadata = load_split(path)
        self.assertEqual(list(features.columns), ['chemical'])
        self.assertEqual(list(targets.columns), ['sensory'])
        self.assertEqual(targets['sensory'].tolist(), [10.0, 20.0])
        self.assertEqual(metadata['beer'].tolist(), [1, 2])
        self.assertTrue(pd.isna(features.iloc[0, 0]))

    def test_each_beer_is_predicted_once_using_only_other_fold_samples(self):
        beers = np.arange(1, 11)
        features = pd.DataFrame({'chemical': beers.astype(float)})
        targets = pd.DataFrame({'taste': beers.astype(float), 'aroma': beers * 2.0})
        metadata = pd.DataFrame({'beer': beers, 'tasting_category_fine': 'A'})
        path = Path(tempfile.mkdtemp())
        cross_validate(features, targets, metadata, path, DummyRegressor)
        predictions = pd.read_csv(path / 'oof_predictions.csv')
        self.assertEqual(len(predictions), 20)
        self.assertFalse(predictions.duplicated(['beer', 'target']).any())
        taste = predictions[predictions['target'] == 'taste']
        self.assertEqual(set(taste['beer']), set(range(1, 11)))
        self.assertEqual(sorted(taste.groupby('fold').size()), [2] * 5)
        for _, group in taste.groupby('fold'):
            expected = (55 - group['beer'].sum()) / 8
            np.testing.assert_allclose(group['prediction'], expected)
            np.testing.assert_allclose(group['baseline_prediction'], expected)
        metrics = pd.read_csv(path / 'target_metrics.csv')
        self.assertEqual(set(metrics['target']), {'taste', 'aroma'})
        np.testing.assert_allclose(metrics['rmse'], metrics['baseline_rmse'])


if __name__ == '__main__':
    unittest.main()
