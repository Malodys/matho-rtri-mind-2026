"""Exhaustive paper tree baseline grids with exact tree-prefix reuse."""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COUNTS = (200, 500, 1000)


def paper_grid(model, n_features=231):
    if model == 'gbr':
        grid = {'estimator__subsample': np.arange(.2, 1., .2),
                'estimator__learning_rate': np.arange(.02, .2, .02),
                'estimator__max_depth': np.arange(6, 15, 3),
                'estimator__n_estimators': list(COUNTS),
                'estimator__loss': ['squared_error', 'absolute_error']}
    elif model == 'rf':
        grid = {'max_features': np.arange(2, n_features, 20),
                'max_depth': np.arange(4, 20, 2),
                'min_impurity_decrease': np.arange(0, .4, .05),
                'n_estimators': list(COUNTS)}
    else:
        raise ValueError(f'Unknown model: {model}')
    return list(ParameterGrid(grid))


def preprocess(x_train, x_eval):
    transform = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    return transform.fit_transform(x_train), transform.transform(x_eval), transform


def plain_params(params):
    return {k.removeprefix('estimator__'): v for k, v in params.items()}


def prefix_predictions(model, params, x_train, y_train, x_eval, counts=COUNTS):
    params = plain_params(params)
    params.pop('n_estimators', None)
    if model == 'gbr':
        fitted = GradientBoostingRegressor(random_state=0, n_estimators=max(counts), **params).fit(x_train, y_train)
        return {n: p.copy() for n, p in enumerate(fitted.staged_predict(x_eval), 1) if n in counts}
    if model != 'rf':
        raise ValueError(f'Unknown model: {model}')
    fitted = RandomForestRegressor(random_state=0, oob_score=True, n_jobs=1, n_estimators=max(counts), **params).fit(x_train, y_train)
    cumulative = np.zeros((len(x_eval), y_train.shape[1]))
    result = {}
    for n, tree in enumerate(fitted.estimators_, 1):
        cumulative += tree.predict(x_eval)
        if n in counts:
            result[n] = cumulative.copy() / n
    return result


def fit_predict(model, params, X_train, Y_train, X_eval, jobs=12):
    train, evaluation, _ = preprocess(X_train, X_eval)
    params = plain_params(params)
    if model == 'gbr':
        estimator = MultiOutputRegressor(GradientBoostingRegressor(random_state=0, **params), n_jobs=jobs)
    elif model == 'rf':
        estimator = RandomForestRegressor(oob_score=True, random_state=0, n_jobs=jobs, **params)
    else:
        raise ValueError(f'Unknown model: {model}')
    with parallel_config(backend='loky', inner_max_num_threads=1, temp_folder=str(Path(tempfile.gettempdir()) / 'mathorcup_joblib_gbr')):
        return estimator.fit(train, Y_train).predict(evaluation)


def _gbr_job(base, fold, target, prepared):
    train, y, evaluation = prepared
    return base, fold, target, prefix_predictions('gbr', base, train, y[:, target], evaluation)


def run_search(data_dir, output, *, seed=0, jobs=8):
    from tuning_protocol import load_protocol, data_signature, make_store, save_selection
    features, targets, metadata, folds = load_protocol(data_dir, seed)
    signature = data_signature(features, targets, metadata)
    run_path = output / 'run.csv'
    fixed = {'model': 'paper_GBR', 'seed': seed, 'folds': 5, 'candidates': 648,
             'search': 'exhaustive_original_grid', 'dataset_sha256': signature,
             'sklearn': sklearn_version, 'source_sklearn': '1.2.2',
             'preprocessing': 'fold_local_mean_imputation_then_standard_scaling',
             'author_difference': 'author_imputes_whole250_before_split_and_scales175_before_cv; sklearn_version',
             'fold_order': 'author_original_train_order_no_shuffle', 'additional_log_transform': False,
             'source_doi': '10.1038/s41467-024-46346-0'}
    previous = {}
    if run_path.exists():
        previous = pd.read_csv(run_path).set_index('setting')['value'].to_dict()
        for key, value in fixed.items():
            if str(previous.get(key)) != str(value):
                raise ValueError(f'Run setting changed: {key}; choose a new output directory.')
    params = paper_grid('gbr')
    predictions, done = make_store(output, params, (len(params), len(features), targets.shape[1]))
    prepared = []
    for train, valid in folds:
        x_train, x_valid, _ = preprocess(features.iloc[train], features.iloc[valid])
        prepared.append((x_train, targets.iloc[train].to_numpy(), x_valid))
    bases = list(ParameterGrid({'subsample': np.arange(.2, 1., .2), 'learning_rate': np.arange(.02, .2, .02),
                               'max_depth': np.arange(6, 15, 3), 'loss': ['squared_error', 'absolute_error']}))
    def key(p):
        return tuple(sorted(plain_params(p).items()))
    lookup = {key(p): i for i, p in enumerate(params)}
    def indices(base):
        return [lookup[key(dict(base, n_estimators=n))] for n in COUNTS]
    def pending():
        for base in bases:
            trials = indices(base)
            for fold, (_, valid) in enumerate(folds):
                for target in range(targets.shape[1]):
                    if done[trials, fold, target].all() and np.isfinite(predictions[trials][:, valid, target]).all():
                        continue
                    yield delayed(_gbr_job)(base, fold, target, prepared[fold])
    started = last_report = perf_counter()
    prior_seconds = float(previous.get('elapsed_seconds', 0))
    settings = dict(fixed, status='running', jobs=jobs)
    def checkpoint(status='running'):
        predictions.flush()
        done.flush()
        progress = dict(completed_configuration_fold_targets=int(done.sum()), total_configuration_fold_targets=int(done.size),
                        elapsed_seconds=prior_seconds + perf_counter() - started)
        settings.update(progress, status=status)
        pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
        pd.Series(progress, name='value').rename_axis('setting').to_csv(output / 'progress.csv', encoding='utf-8-sig')
        print(progress, flush=True)
    checkpoint()
    try:
        with parallel_config(backend='loky', inner_max_num_threads=1, temp_folder=str(Path(tempfile.gettempdir()) / 'mathorcup_joblib_gbr')):
            stream = Parallel(n_jobs=jobs, return_as='generator_unordered', pre_dispatch=jobs * 2, max_nbytes=None)(pending())
            for base, fold, target, result in stream:
                trials, valid = indices(base), folds[fold][1]
                for trial, count in zip(trials, COUNTS):
                    if not np.isfinite(result[count]).all():
                        raise ValueError('Non-finite GBR predictions.')
                    predictions[trial, valid, target] = result[count]
                # Persist prediction bytes before completion flags for crash-safe resumption.
                predictions.flush()
                done[trials, fold, target] = True
                done.flush()
                if perf_counter() - last_report >= 30:
                    checkpoint()
                    last_report = perf_counter()
        checkpoint()
        selected = save_selection(output, params, predictions, targets, metadata, folds)
        return selected
    except BaseException:
        checkpoint('interrupted')
        raise


def main():
    from tabpfn_cv import project_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gbr'], default='gbr')
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/tuned_seed_0/gbr'))
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_search(project_path(args.data_dir), project_path(args.output_dir), seed=args.seed, jobs=args.jobs)

if __name__ == '__main__':
    main()
