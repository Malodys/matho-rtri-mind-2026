import sys
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import paper_baselines as b

class BaselineTests(unittest.TestCase):
    def test_grid_counts(self):
        self.assertEqual(len(b.paper_grid('gbr')), 648)
        self.assertEqual(len(b.paper_grid('rf')), 2304)

    def test_preprocessing_training_only(self):
        a, z, transform = b.preprocess(np.array([[1., np.nan], [3., 4.], [5., 6.]]), np.array([[1000., np.nan]]))
        assert_allclose(transform.named_steps['imputer'].statistics_, [3., 5.])
        assert_allclose(z[0, 1], 0.)
        assert_allclose(a.mean(axis=0), 0., atol=1e-12)

    def test_gbr_prefix_equivalence(self):
        rng = np.random.RandomState(4)
        x = rng.normal(size=(35, 5)); y = rng.normal(size=35)
        for loss in ['squared_error', 'absolute_error']:
            params = dict(subsample=.6, learning_rate=.08, max_depth=6, loss=loss)
            shared = b.prefix_predictions('gbr', params, x, y, x[:7], (3, 8, 15))
            for n, pred in shared.items():
                model = GradientBoostingRegressor(random_state=0, n_estimators=n, **params).fit(x, y)
                assert_allclose(pred, model.predict(x[:7]), rtol=0, atol=0)

    def test_rf_prefix_equivalence(self):
        rng = np.random.RandomState(5)
        x = rng.normal(size=(40, 5)); y = rng.normal(size=(40, 3))
        params = dict(max_features=2, max_depth=4, min_impurity_decrease=0.)
        shared = b.prefix_predictions('rf', params, x, y, x[:7], (30, 40, 50))
        for n, pred in shared.items():
            model = RandomForestRegressor(random_state=0, n_estimators=n, oob_score=True, n_jobs=1, **params).fit(x, y)
            assert_allclose(pred, model.predict(x[:7]), rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
