import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tune_rf import forest_prefixes, candidate_grid


class ForestSearchTests(unittest.TestCase):
    def test_tree_prefix_predictions_equal_independent_forest_fits(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
        config = {'max_features': 2, 'max_depth': 4, 'min_impurity_decrease': 0.}
        predictions, _ = forest_prefixes(x[:30], y[:30], x[30:], config, (10, 20))
        for position, count in enumerate([10, 20]):
            independent = RandomForestRegressor(n_estimators=count, random_state=0, n_jobs=1, **config)
            independent.fit(x[:30], y[:30])
            np.testing.assert_allclose(predictions[position], independent.predict(x[30:]), rtol=0, atol=1e-14)

    def test_stump_predictions_do_not_depend_on_depth_limit(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
        config = {'max_features': 2, 'max_depth': 4, 'min_impurity_decrease': 100.}
        predictions, stumps = forest_prefixes(x[:30], y[:30], x[30:], config, (10, 20))
        self.assertTrue(stumps)
        config['max_depth'] = 18
        different_depth = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1, **config).fit(x[:30], y[:30])
        np.testing.assert_allclose(predictions[1], different_depth.predict(x[30:]), rtol=0, atol=1e-14)
        self.assertEqual(len(candidate_grid()), 2304)


if __name__ == '__main__':
    unittest.main()
