import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multitask_cv import MaskedMultiTaskNet, ObservedScaler, fit_member


class MultiTaskTests(unittest.TestCase):
    def test_input_clipping_bounds_extrapolation_and_keeps_missing_values(self):
        scaler = ObservedScaler(clip=5.).fit(np.array([[1., 2.], [3., 4.]]))
        result = scaler.transform(np.array([[100., np.nan], [-100., 3.]]))
        np.testing.assert_allclose(result[:, 0], [5., -5.])
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertEqual(result[1, 1], 0.)
        np.testing.assert_allclose(scaler.mean, [2., 3.])

    def test_scaling_uses_only_observed_training_values(self):
        training = np.array([[1., np.nan, 5.], [3., 4., 5.]])
        scaler = ObservedScaler().fit(training)
        np.testing.assert_allclose(scaler.mean, [2., 4., 5.])
        np.testing.assert_allclose(scaler.scale, [1., 1., 1.])
        transformed = scaler.transform(np.array([[101., np.nan, 5.]]))
        self.assertEqual(transformed[0, 0], 99.)
        self.assertTrue(np.isnan(transformed[0, 1]))
        self.assertEqual(transformed[0, 2], 0.)
        self.assertTrue(np.isnan(training[0, 1]))

    def test_missing_values_allow_joint_outputs_and_finite_gradients(self):
        torch.manual_seed(0)
        model = MaskedMultiTaskNet(3, 2, hidden=(8, 4), dropout=0.)
        prediction = model(torch.tensor([[1., float('nan'), 3.], [2., 4., 5.]]))
        self.assertEqual(tuple(prediction.shape), (2, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.missing_values.grad[1].abs().item(), 0.)

    def test_short_training_predicts_all_targets_and_preserves_inputs(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(30, 3)).astype(np.float32)
        features[0, 1] = np.nan
        original = features.copy()
        targets = np.column_stack([features[:, 0] + 10., 2. * features[:, 0] - 5.])
        model, x_scaler, y_scaler, info = fit_member(
            features, targets, seed=0, device='cpu', hidden=(8, 4), dropout=0.,
            max_epochs=8, patience=3, learning_rate=0.01, weight_decay=0.01,
        )
        model.eval()
        with torch.no_grad():
            predicted = model(torch.from_numpy(x_scaler.transform(features[:4]))).numpy()
        predicted = y_scaler.inverse_transform(predicted)
        self.assertEqual(predicted.shape, (4, 2))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue(1 <= info['selected_epochs'] <= 8)
        np.testing.assert_allclose(x_scaler.mean, np.nanmean(features, axis=0), atol=1e-6)
        np.testing.assert_allclose(y_scaler.mean, targets.mean(axis=0), atol=1e-6)
        np.testing.assert_equal(features, original)


if __name__ == '__main__':
    unittest.main()
