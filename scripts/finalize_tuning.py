from __future__ import annotations

import argparse
import ast
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd

from tabpfn_cv import load_split, project_path
from tuning_protocol import data_signature, export_predictions, load_protocol, macro_scores


MODELS = ['gbr', 'rf', 'tabpfn', 'multitask']
EXPECTED_CANDIDATES = {'gbr': 648, 'rf': 2304, 'tabpfn': 18, 'multitask': 48}


def read_parameters(path):
    values = pd.read_csv(path).set_index('parameter')['value']
    result = {}
    for name, value in values.items():
        if name == 'trial':
            continue
        try:
            result[name] = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            result[name] = value
        if name.split('__')[-1] in {'n_estimators', 'max_depth', 'max_features'}:
            if isinstance(result[name], float) and result[name].is_integer():
                result[name] = int(result[name])
    return result


def freeze_selection(root, models, signature):
    records = []
    for model in models:
        output = root / model
        settings = pd.read_csv(output / 'run.csv').set_index('setting')['value']
        if settings.get('status') != 'completed' or settings.get('dataset_sha256') != signature:
            raise ValueError(f'{model}: search incomplete or training data changed.')
        best_path = output / 'best_parameters.csv'
        best = pd.read_csv(best_path).set_index('parameter')['value']
        results = pd.read_csv(output / 'cv_results.csv').set_index('trial')
        if model in EXPECTED_CANDIDATES:
            expected = EXPECTED_CANDIDATES[model]
            if not results.index.is_unique or set(results.index) != set(range(expected)):
                raise ValueError(f'{model}: candidate coverage is incomplete.')
            if not np.isfinite(results[['mean_r2', 'std_r2']].to_numpy()).all():
                raise ValueError(f'{model}: non-finite selection scores.')
        trial = int(best['trial'])
        if trial != int(results['mean_r2'].idxmax()):
            raise ValueError(f'{model}: selected trial does not maximize five-fold R2.')
        candidates_path = output / 'candidates.csv'
        if model in EXPECTED_CANDIDATES:
            candidates = pd.read_csv(candidates_path, dtype=str, keep_default_na=False).set_index('trial')
            if not candidates.index.is_unique or set(candidates.index) != set(map(str, range(EXPECTED_CANDIDATES[model]))):
                raise ValueError(f'{model}: candidate manifest coverage is incomplete.')
            if candidates.duplicated().any():
                raise ValueError(f'{model}: duplicate candidate configurations.')
            chosen = read_parameters(best_path)
            expected_parameters = {column.removeprefix('param_') for column in candidates.columns}
            if set(chosen) != expected_parameters:
                raise ValueError(f'{model}: selected parameter keys differ from the scored candidate.')
            for parameter, value in chosen.items():
                candidate_value = candidates.at[str(trial), 'param_' + parameter]
                try:
                    candidate_value = ast.literal_eval(candidate_value)
                except (ValueError, SyntaxError):
                    pass
                if value != candidate_value:
                    raise ValueError(f'{model}: selected parameters differ from the scored candidate.')
        records.append({
            'model': model, 'best_trial': trial,
            'mean_cv_r2': results.at[trial, 'mean_r2'],
            'std_cv_r2': results.at[trial, 'std_r2'], 'candidates': len(results),
            'parameters_sha256': hashlib.sha256(best_path.read_bytes()).hexdigest(),
            'dataset_sha256': signature,
            'checkpoint_sha256': settings.get('checkpoint_sha256', ''),
        })
    manifest = pd.DataFrame(records)
    path = root / 'frozen_selection.csv'
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False)
        try:
            pd.testing.assert_frame_equal(previous, manifest, check_dtype=False)
        except AssertionError as error:
            raise ValueError('Frozen selection changed; test evaluation cannot proceed.') from error
    else:
        manifest.to_csv(path, index=False, encoding='utf-8-sig')
    return manifest


def validate_checkpoint(settings, path):
    with path.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != settings.get('checkpoint_sha256'):
        raise ValueError('Final TabPFN checkpoint differs from the selected CV checkpoint.')
    return actual


def validate_refit_settings(root, model_path, device):
    import torch
    runtime = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': version('numpy'), 'pandas': version('pandas'),
        'scikit-learn': version('scikit-learn'),
    }
    settings = {model: pd.read_csv(root / model / 'run.csv').set_index('setting')['value'].to_dict() for model in MODELS}
    validate_checkpoint(settings['tabpfn'], model_path)
    for model in ['tabpfn', 'multitask']:
        for key in ['python', 'torch', 'device']:
            expected = device if key == 'device' else runtime[key]
            if str(settings[model].get(key)) != str(expected):
                raise ValueError(f'{model}: final runtime differs from selection: {key}')
    if str(settings['tabpfn'].get('tabpfn')) != version('tabpfn'):
        raise ValueError('TabPFN runtime version changed.')
    for key in ['numpy', 'pandas', 'scikit-learn']:
        if str(settings['multitask'].get(key)) != runtime[key]:
            raise ValueError(f'NN runtime changed: {key}')
    for model in ['gbr', 'rf']:
        if str(settings[model].get('sklearn')) != runtime['scikit-learn']:
            raise ValueError(f'{model}: sklearn runtime changed.')
    return settings


def validate_folds(root, models):
    reference = pd.read_csv(root / models[0] / 'fold_assignments.csv')
    for model in models[1:]:
        try:
            pd.testing.assert_frame_equal(reference, pd.read_csv(root / model / 'fold_assignments.csv'))
        except AssertionError as error:
            raise ValueError('Models used different CV folds.') from error


def nn_refit_options(settings):
    return {key: int(settings[key]) for key in ['members', 'max_epochs', 'patience']}


def wait_for_searches(root, models, interval=30):
    while True:
        statuses = {}
        for model in models:
            path = root / model / 'run.csv'
            try:
                settings = pd.read_csv(path).set_index('setting')['value']
                statuses[model] = settings.get('status', 'starting')
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
                statuses[model] = 'starting'
        if all(value == 'completed' for value in statuses.values()):
            return
        if any('interrupted' in str(value) or 'failed' in str(value) for value in statuses.values()):
            raise ValueError(f'A search stopped before completion: {statuses}')
        print(f'Waiting for all searches: {statuses}', flush=True)
        sleep(interval)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--search-dir', type=Path, default=Path('results/tuned_seed_0'))
    parser.add_argument('--model-path', type=Path, default=Path('../pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data_dir, root = project_path(args.data_dir), project_path(args.search_dir)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.wait:
        wait_for_searches(root, MODELS)
    features, targets, metadata, _ = load_protocol(data_dir, args.seed)
    manifest = freeze_selection(root, MODELS, data_signature(features, targets, metadata))
    frozen_settings = validate_refit_settings(root, project_path(args.model_path), args.device)
    validate_folds(root, MODELS)
    test_features, test_targets, test_metadata = load_split(data_dir, 'test')
    if set(metadata['beer']) & set(test_metadata['beer']):
        raise ValueError('Training and test beers overlap.')
    if set(test_features.columns) != set(features.columns) or set(test_targets.columns) != set(targets.columns):
        raise ValueError('Train and test columns differ.')
    test_features = test_features.loc[:, features.columns]
    test_targets = test_targets.loc[:, targets.columns]
    baseline = np.broadcast_to(targets.mean().to_numpy(), test_targets.shape)
    summaries = []
    for model in MODELS:
        output = root / model / 'test'
        output.mkdir(parents=True, exist_ok=True)
        params = read_parameters(root / model / 'best_parameters.csv')
        print(f'Final full training: {model}, train={len(features)}, test={len(test_features)}, {params}', flush=True)
        started = perf_counter()
        if model in ['gbr', 'rf']:
            from paper_baselines import fit_predict
            predicted = fit_predict(model, params, features, targets, test_features, jobs=args.jobs)
        elif model == 'tabpfn':
            from tune_tabpfn import fit_predict
            predicted = fit_predict(
                params, features, targets, test_features, model_path=project_path(args.model_path),
                device=args.device, seed=args.seed,
            )
        else:
            from tune_multitask import fit_predict
            predicted = fit_predict(
                params, features, targets, test_features, seed=args.seed,
                device=args.device, **nn_refit_options(frozen_settings[model]),
            )
        predicted = np.asarray(predicted)
        if predicted.shape != test_targets.shape or not np.isfinite(predicted).all():
            raise ValueError(f'{model}: invalid final predictions.')
        export_predictions(output, test_targets, predicted, test_metadata, baseline)
        metric = macro_scores(test_targets.to_numpy(), predicted)
        summaries.append({
            'model': model, 'mean_cv_r2': manifest.set_index('model').at[model, 'mean_cv_r2'],
            'n_train': len(features), 'n_test': len(test_features), **metric,
            'seconds': perf_counter() - started,
        })
        pd.Series({
            'status': 'completed', 'model': model, 'n_train': len(features), 'n_test': len(test_features),
            'dataset_sha256': data_signature(features, targets, metadata),
            'parameters_sha256': hashlib.sha256((root / model / 'best_parameters.csv').read_bytes()).hexdigest(),
            'test_previously_inspected': True,
            'checkpoint_sha256': frozen_settings[model].get('checkpoint_sha256', ''),
        }, name='value').rename_axis('setting').to_csv(output / 'run.csv', encoding='utf-8-sig')
        print(f'{model}: {metric}', flush=True)
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(root / 'comparison.csv', index=False, encoding='utf-8-sig')
    print(comparison.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
