import sys
import hashlib
import platform
from importlib.metadata import version
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finalize_tuning import freeze_selection, validate_checkpoint, read_parameters, nn_refit_options, validate_refit_settings, validate_folds


class FinalSelectionTests(unittest.TestCase):
    def test_changed_runtime_is_rejected_and_frozen_nn_settings_returned(self):
        import torch
        root = Path(tempfile.mkdtemp())
        checkpoint = root / 'weights.bin'
        checkpoint.write_bytes(b'weights')
        settings = {
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': version('numpy'), 'pandas': version('pandas'),
            'scikit-learn': version('scikit-learn'), 'sklearn': version('scikit-learn'),
            'tabpfn': version('tabpfn'), 'device': 'cuda',
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'members': 5, 'max_epochs': 200, 'patience': 10,
        }
        for model in ['gbr', 'rf', 'tabpfn', 'multitask']:
            (root / model).mkdir()
            pd.Series(settings, name='value').rename_axis('setting').to_csv(root / model / 'run.csv')
        frozen = validate_refit_settings(root, checkpoint, 'cuda')
        self.assertEqual(nn_refit_options(frozen['multitask']), {'members': 5, 'max_epochs': 200, 'patience': 10})
        settings['torch'] = 'different-version'
        pd.Series(settings, name='value').rename_axis('setting').to_csv(root / 'multitask/run.csv')
        with self.assertRaises(ValueError):
            validate_refit_settings(root, checkpoint, 'cuda')

    def test_mismatched_fold_assignments_are_rejected(self):
        root = Path(tempfile.mkdtemp())
        for model in ['first', 'second']:
            (root / model).mkdir()
            pd.DataFrame({'beer': [1, 2], 'fold': [1, 2]}).to_csv(root / model / 'fold_assignments.csv', index=False)
        validate_folds(root, ['first', 'second'])
        pd.DataFrame({'beer': [1, 2], 'fold': [2, 1]}).to_csv(root / 'second/fold_assignments.csv', index=False)
        with self.assertRaises(ValueError):
            validate_folds(root, ['first', 'second'])

    def test_duplicate_trial_ids_cannot_masquerade_as_complete_grid(self):
        root = Path(tempfile.mkdtemp())
        output = root / 'tabpfn'
        output.mkdir()
        pd.Series({'status': 'completed', 'dataset_sha256': 'training-only'},
                  name='value').rename_axis('setting').to_csv(output / 'run.csv')
        pd.Series({'trial': 0, 'n_estimators': 4},
                  name='value').rename_axis('parameter').to_csv(output / 'best_parameters.csv')
        pd.DataFrame({'trial': [0] * 18, 'mean_r2': [0.25] * 18, 'std_r2': [0.1] * 18}).to_csv(output / 'cv_results.csv', index=False)
        with self.assertRaises(ValueError):
            freeze_selection(root, ['tabpfn'], 'training-only')

    def test_changed_pretrained_checkpoint_is_rejected(self):
        root = Path(tempfile.mkdtemp())
        checkpoint = root / 'weights.bin'
        checkpoint.write_bytes(b'first checkpoint')
        settings = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        self.assertEqual(validate_checkpoint(settings, checkpoint), settings['checkpoint_sha256'])
        checkpoint.write_bytes(b'different checkpoint')
        with self.assertRaises(ValueError):
            validate_checkpoint(settings, checkpoint)

    def test_frozen_neural_ensemble_settings_are_preserved(self):
        self.assertEqual(nn_refit_options({'members': '5', 'max_epochs': '200', 'patience': '10'}),
                         {'members': 5, 'max_epochs': 200, 'patience': 10})

    def test_integer_tree_parameters_survive_csv_numeric_coercion(self):
        path = Path(tempfile.mkdtemp()) / 'best_parameters.csv'
        pd.Series({'trial': 1, 'n_estimators': 500, 'max_depth': 12,
                   'max_features': 122, 'min_impurity_decrease': 0.05},
                  name='value').rename_axis('parameter').to_csv(path)
        params = read_parameters(path)
        self.assertIsInstance(params['n_estimators'], int)
        self.assertIsInstance(params['max_depth'], int)
        self.assertIsInstance(params['max_features'], int)
        self.assertEqual(params['min_impurity_decrease'], 0.05)

    def test_paper_baseline_requires_complete_candidate_grid(self):
        root = Path(tempfile.mkdtemp())
        output = root / 'rf'
        output.mkdir()
        pd.Series({'status': 'completed', 'dataset_sha256': 'training-only'},
                  name='value').rename_axis('setting').to_csv(output / 'run.csv')
        pd.Series({'trial': 0, 'n_estimators': 200},
                  name='value').rename_axis('parameter').to_csv(output / 'best_parameters.csv')
        pd.DataFrame({'trial': [0], 'mean_r2': [0.25], 'std_r2': [0.1]}).to_csv(output / 'cv_results.csv', index=False)
        with self.assertRaises(ValueError):
            freeze_selection(root, ['rf'], 'training-only')

    def test_selection_requires_completed_searches_and_cannot_change_after_freezing(self):
        root = Path(tempfile.mkdtemp())
        for model in ['first', 'second']:
            output = root / model
            output.mkdir()
            pd.Series({'status': 'running', 'dataset_sha256': 'training-only'}, name='value').rename_axis('setting').to_csv(output / 'run.csv')
            pd.Series({'trial': 0, 'width': 8}, name='value').rename_axis('parameter').to_csv(output / 'best_parameters.csv')
            pd.DataFrame({'trial': [0], 'mean_r2': [0.25], 'std_r2': [0.1]}).to_csv(output / 'cv_results.csv', index=False)
        with self.assertRaises(ValueError):
            freeze_selection(root, ['first', 'second'], 'training-only')
        for model in ['first', 'second']:
            pd.Series({'status': 'completed', 'dataset_sha256': 'training-only'}, name='value').rename_axis('setting').to_csv(root / model / 'run.csv')
        frozen = freeze_selection(root, ['first', 'second'], 'training-only')
        self.assertEqual(set(frozen['model']), {'first', 'second'})
        self.assertEqual(frozen['mean_cv_r2'].tolist(), [0.25, 0.25])
        pd.Series({'trial': 0, 'width': 16}, name='value').rename_axis('parameter').to_csv(root / 'first/best_parameters.csv')
        with self.assertRaises(ValueError):
            freeze_selection(root, ['first', 'second'], 'training-only')


if __name__ == '__main__':
    unittest.main()
