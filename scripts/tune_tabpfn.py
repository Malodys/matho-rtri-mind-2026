from __future__ import annotations

import argparse
import hashlib
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from tabpfn_cv import project_path
from tuning_protocol import data_signature, load_protocol, macro_scores, make_store, save_selection


def candidate_grid():
    return list(ParameterGrid({
        'n_estimators': [4, 8, 16], 'softmax_temperature': [0.7, 1., 1.3],
        'average_before_softmax': [False, True],
    }))


def store_batch(predictions, done, trial, fold, valid, columns, results):
    if len(results) != len(columns):
        raise ValueError('TabPFN returned an incomplete target batch.')
    results = [np.asarray(result) for result in results]
    if any(result.shape != (len(valid),) or not np.isfinite(result).all() for result in results):
        raise ValueError('Invalid TabPFN predictions.')
    for column, result in zip(columns, results):
        predictions[trial, valid, column] = result
    predictions.flush()
    done[trial, fold, columns] = True
    done.flush()


def fit_predict(params, features, targets, evaluation, *, model_path, device='cuda', batch_size=4, seed=0):
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        model_path=model_path, device=device, random_state=seed,
        show_progress_bar=False, **params,
    )
    predictions = []
    for start in range(0, targets.shape[1], batch_size):
        columns = range(start, min(start + batch_size, targets.shape[1]))
        results = model.predict_batched(
            [features] * len(columns), [targets.iloc[:, c] for c in columns],
            [evaluation] * len(columns),
        )
        predictions.extend(np.asarray(result) for result in results)
    return np.column_stack(predictions)


def run_search(data_dir, output, model_path, *, seed=0, device='cuda', batch_size=4):
    import torch
    from tabpfn import TabPFNRegressor

    os.environ.setdefault('TABPFN_MODEL_CACHE_SIZE', '1')
    torch.set_num_threads(4)
    features, targets, metadata, folds = load_protocol(data_dir, seed)
    signature = data_signature(features, targets, metadata)
    with model_path.open('rb') as handle:
        checkpoint_signature = hashlib.file_digest(handle, 'sha256').hexdigest()
    runtime = {'python': platform.python_version(), 'tabpfn': version('tabpfn'), 'torch': torch.__version__, 'device': device, 'seed': seed, 'fold_order': 'author_original_train_order_no_shuffle'}
    run_path = output / 'run.csv'
    if run_path.exists():
        previous = pd.read_csv(run_path).set_index('setting')['value']
        if previous['dataset_sha256'] != signature or previous['checkpoint_sha256'] != checkpoint_signature:
            raise ValueError('Dataset or pretrained checkpoint changed; use a new output directory.')
        for name, value in runtime.items():
            if str(previous.get(name)) != str(value):
                raise ValueError(f'Runtime version changed: {name}; use a new output directory.')
    params = candidate_grid()
    predictions, done = make_store(output, params, (len(params), len(features), targets.shape[1]))
    settings = {
        'model': 'TabPFN-3.5', 'status': 'running', 'seed': seed, 'folds': 5,
        'fold_order': 'author_original_train_order_no_shuffle',
        'candidates': len(params), 'samples': len(features), 'features': features.shape[1],
        'targets': targets.shape[1], 'device': device, 'python': platform.python_version(),
        'tabpfn': version('tabpfn'), 'torch': torch.__version__,
        'dataset_sha256': signature, 'checkpoint_sha256': checkpoint_signature,
        'weights_updated': False,
    }
    pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
    started = perf_counter()
    for trial, config in enumerate(params):
        model = TabPFNRegressor(
            model_path=model_path, device=device, random_state=seed,
            show_progress_bar=False, **config,
        )
        print(f'Trial {trial + 1}/{len(params)}: {config}', flush=True)
        for fold, (train, valid) in enumerate(folds):
            invalid = done[trial, fold] & ~np.isfinite(predictions[trial, valid]).all(axis=0)
            if invalid.any():
                done[trial, fold, invalid] = False
                done.flush()
            pending = np.flatnonzero(~done[trial, fold])
            tick = perf_counter()
            for start in range(0, len(pending), batch_size):
                columns = pending[start:start + batch_size]
                results = model.predict_batched(
                    [features.iloc[train]] * len(columns),
                    [targets.iloc[train, column] for column in columns],
                    [features.iloc[valid]] * len(columns),
                )
                store_batch(predictions, done, trial, fold, valid, columns, results)
            score = macro_scores(targets.iloc[valid].to_numpy(), predictions[trial, valid])['r2']
            print(f'  fold {fold + 1}/5 R2={score:.5f}, {perf_counter() - tick:.1f}s', flush=True)
            pd.Series({
                'completed_outputs': int(done.sum()), 'total_outputs': int(done.size),
                'elapsed_seconds': perf_counter() - started,
            }, name='value').rename_axis('setting').to_csv(output / 'progress.csv', encoding='utf-8-sig')
    selected = save_selection(output, params, predictions, targets, metadata, folds)
    print(f'Elapsed: {perf_counter() - started:.1f}s', flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/tuned_seed_0/tabpfn'))
    parser.add_argument('--model-path', type=Path, default=Path('../pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive.')
    run_search(
        project_path(args.data_dir), project_path(args.output_dir), project_path(args.model_path),
        seed=args.seed, device=args.device, batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
