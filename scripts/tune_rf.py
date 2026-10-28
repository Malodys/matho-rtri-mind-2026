from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from tabpfn_cv import PROJECT, project_path
from tuning_protocol import data_signature, load_protocol, make_store, save_selection


def candidate_grid():
    return list(ParameterGrid({
        'max_features': np.arange(2, 231, 20), 'max_depth': np.arange(4, 20, 2),
        'min_impurity_decrease': np.arange(0, 0.4, 0.05),
        'n_estimators': [200, 500, 1000],
    }))


def forest_prefixes(x_train, y_train, x_valid, params, counts=(200, 500, 1000)):
    model = RandomForestRegressor(
        n_estimators=max(counts), oob_score=True, random_state=0, n_jobs=1, **params,
    ).fit(x_train, y_train)
    cumulative = np.zeros((len(x_valid), y_train.shape[1]))
    results = []
    for number, tree in enumerate(model.estimators_, 1):
        cumulative += tree.predict(x_valid)
        if number in counts:
            results.append(cumulative.copy() / number)
    stumps = all(tree.tree_.max_depth == 0 for tree in model.estimators_)
    return np.stack(results), stumps


def job(base, fold, prepared):
    x_train, y_train, x_valid = prepared[fold]
    predicted, stumps = forest_prefixes(x_train, y_train, x_valid, base)
    return base, fold, predicted, stumps


def run_search(data_dir, output, *, seed=0, jobs=4):
    features, targets, metadata, folds = load_protocol(data_dir, seed)
    signature = data_signature(features, targets, metadata)
    run_path = output / 'run.csv'
    if run_path.exists():
        previous = pd.read_csv(run_path).set_index('setting')['value']
        if previous['dataset_sha256'] != signature:
            raise ValueError('Training data changed; choose a new output directory.')
        for name, value in {'sklearn': sklearn_version, 'fold_order': 'author_original_train_order_no_shuffle'}.items():
            if str(previous.get(name)) != str(value):
                raise ValueError(f'Resume setting changed: {name}; choose a new output directory.')
    params = candidate_grid()
    predictions, done = make_store(output, params, (len(params), len(features), targets.shape[1]))
    settings = {
        'model': 'paper_RF', 'status': 'running', 'seed': seed, 'folds': 5,
        'candidates': len(params), 'search': 'exhaustive_original_grid',
        'dataset_sha256': signature, 'sklearn': sklearn_version, 'source_sklearn': '1.2.2',
        'preprocessing': 'fold_local_mean_imputation_then_standard_scaling',
        'author_difference': 'author_imputes_before_split_and_scales_before_cv; current_sklearn_version',
        'fold_order': 'author_original_train_order_no_shuffle',
        'source_doi': '10.1038/s41467-024-46346-0', 'jobs': jobs,
    }
    pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
    prepared = []
    for train, valid in folds:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(features.iloc[train]))
        x_valid = scaler.transform(imputer.transform(features.iloc[valid]))
        prepared.append((x_train, targets.iloc[train].to_numpy(), x_valid))
    lookup = {
        (p['max_features'], p['max_depth'], p['min_impurity_decrease'], p['n_estimators']): i
        for i, p in enumerate(params)
    }
    bases = list(ParameterGrid({
        'max_features': np.arange(2, 231, 20), 'max_depth': np.arange(4, 20, 2),
        'min_impurity_decrease': np.arange(0, 0.4, 0.05),
    }))

    def pending():
        for base in bases:
            indices = [lookup[(base['max_features'], base['max_depth'], base['min_impurity_decrease'], n)] for n in [200, 500, 1000]]
            for fold, (_, valid) in enumerate(folds):
                if done[indices, fold].all() and np.isfinite(predictions[indices][:, valid]).all():
                    continue
                yield delayed(job)(base, fold, prepared)

    temp = Path(tempfile.gettempdir()) / 'mathorcup_joblib_rf'
    started = last_report = perf_counter()
    completed_jobs = 0
    with parallel_config(backend='loky', inner_max_num_threads=1, temp_folder=str(temp)):
        results = Parallel(n_jobs=jobs, return_as='generator_unordered', pre_dispatch=jobs * 2, max_nbytes=None)(pending())
        for base, fold, result, stumps in results:
            valid = folds[fold][1]
            depths = np.arange(4, 20, 2) if stumps else [base['max_depth']]
            indices = []
            for depth in depths:
                for position, count in enumerate([200, 500, 1000]):
                    trial = lookup[(base['max_features'], depth, base['min_impurity_decrease'], count)]
                    predictions[trial, valid] = result[position]
                    indices.append(trial)
            predictions.flush()
            done[indices, fold] = True
            done.flush()
            completed_jobs += 1
            if perf_counter() - last_report >= 30:
                progress = {
                    'completed_configuration_folds': int(done.all(axis=2).sum()),
                    'total_configuration_folds': len(params) * 5,
                    'fitted_forests': completed_jobs, 'elapsed_seconds': perf_counter() - started,
                }
                pd.Series(progress, name='value').rename_axis('setting').to_csv(output / 'progress.csv', encoding='utf-8-sig')
                print(progress, flush=True)
                last_report = perf_counter()
    selected = save_selection(output, params, predictions, targets, metadata, folds)
    print(f'RF exhaustive search complete in {perf_counter() - started:.1f}s', flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/tuned_seed_0/rf'))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_search(project_path(args.data_dir), project_path(args.output_dir), seed=args.seed, jobs=args.jobs)


if __name__ == '__main__':
    main()
