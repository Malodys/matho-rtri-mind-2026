from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from tabpfn_cv import PROJECT, load_split, scores


def load_protocol(data_dir: Path, seed=0):
    features, targets, metadata = load_split(data_dir)
    original = pd.read_csv(PROJECT / 'data/raw/1.csv', usecols=['beer', 'tasting_category_fine'])
    train, _ = train_test_split(
        np.arange(len(original)), test_size=0.3, random_state=seed,
        stratify=original['tasting_category_fine'], shuffle=True,
    )
    author_beers = original.iloc[train]['beer'].tolist()
    if set(author_beers) != set(metadata['beer']):
        raise ValueError('Saved training beers differ from the author split.')
    lookup = pd.Series(np.arange(len(metadata)), index=metadata['beer'])
    order = lookup.loc[author_beers].to_numpy()
    features = features.iloc[order].reset_index(drop=True)
    targets = targets.iloc[order].reset_index(drop=True)
    metadata = metadata.iloc[order].reset_index(drop=True)
    return features, targets, metadata, list(KFold(5, shuffle=False).split(features))


def data_signature(features, targets, metadata):
    digest = hashlib.sha256()
    for frame in [features, targets, metadata]:
        digest.update(frame.to_csv(index=False).encode('utf-8'))
    return digest.hexdigest()


def make_store(output: Path, params, shape):
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(params).add_prefix('param_')
    manifest.insert(0, 'trial', np.arange(len(params)))
    manifest = manifest.map(str)
    path = output / 'candidates.csv'
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not existing.equals(manifest):
            raise ValueError('Search candidates changed; choose a new output directory.')
    else:
        manifest.to_csv(path, index=False, encoding='utf-8-sig')
    prediction_path, done_path = output / 'cv_predictions.npy', output / 'cv_done.npy'
    if prediction_path.exists() != done_path.exists():
        raise ValueError('Incomplete checkpoint files; choose a new output directory.')
    if prediction_path.exists():
        predictions = np.load(prediction_path, mmap_mode='r+')
        done = np.load(done_path, mmap_mode='r+')
        if predictions.shape != tuple(shape) or done.shape != (shape[0], 5, shape[2]):
            raise ValueError('Checkpoint dimensions differ.')
    else:
        predictions = np.lib.format.open_memmap(prediction_path, mode='w+', dtype='float64', shape=shape)
        done = np.lib.format.open_memmap(done_path, mode='w+', dtype='bool', shape=(shape[0], 5, shape[2]))
        predictions[:] = np.nan
        done[:] = False
        predictions.flush()
        done.flush()
    return predictions, done


def macro_scores(actual, predicted):
    residual = np.square(np.asarray(actual) - np.asarray(predicted))
    denominator = np.square(actual - np.mean(actual, axis=0)).sum(axis=0)
    numerator = residual.sum(axis=0)
    r2 = np.ones_like(numerator, dtype=float)
    nonconstant = denominator != 0
    r2[nonconstant] = 1 - numerator[nonconstant] / denominator[nonconstant]
    r2[~nonconstant & (numerator != 0)] = 0.
    return {
        'mae': np.abs(actual - predicted).mean(axis=0).mean(),
        'rmse': np.sqrt(residual.mean(axis=0)).mean(), 'r2': r2.mean(),
    }


def export_predictions(output, actual, predicted, metadata, baseline, folds=None):
    output.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], []
    for column, target in enumerate(actual.columns):
        block = metadata.copy()
        block['target'] = target
        block['actual'] = actual.iloc[:, column].to_numpy()
        block['prediction'] = predicted[:, column]
        block['baseline_prediction'] = baseline[:, column]
        rows.append(block)
        metrics.append({
            'target': target, 'n_samples': len(actual),
            **scores(block['actual'], block['prediction'], block['baseline_prediction']),
        })
    name = 'oof_predictions.csv' if folds is not None else 'test_predictions.csv'
    pd.concat(rows, ignore_index=True).to_csv(output / name, index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(output / 'target_metrics.csv', index=False, encoding='utf-8-sig')
    if folds is None:
        pd.DataFrame(metrics).drop(columns=['target', 'n_samples']).mean().to_frame().T.to_csv(
            output / 'test_summary.csv', index=False, encoding='utf-8-sig',
        )


def save_selection(output, params, predictions, targets, metadata, folds):
    done = np.load(output / 'cv_done.npy', mmap_mode='r')
    if not done.all() or not np.isfinite(predictions).all():
        raise ValueError('Cannot select from an incomplete search.')
    actual = targets.to_numpy()
    results = []
    for trial, config in enumerate(params):
        fold_scores = [macro_scores(actual[valid], predictions[trial, valid]) for _, valid in folds]
        record = {
            'trial': trial, **{'param_' + key: str(value) for key, value in config.items()},
            'mean_r2': np.mean([item['r2'] for item in fold_scores]),
            'std_r2': np.std([item['r2'] for item in fold_scores], ddof=1),
            'mean_mae': np.mean([item['mae'] for item in fold_scores]),
            'mean_rmse': np.mean([item['rmse'] for item in fold_scores]),
        }
        record.update({f'fold_{i + 1}_r2': item['r2'] for i, item in enumerate(fold_scores)})
        results.append(record)
    table = pd.DataFrame(results)
    table.to_csv(output / 'cv_results.csv', index=False, encoding='utf-8-sig')
    selected = int(table['mean_r2'].idxmax())
    pd.Series({'trial': selected, **params[selected]}, name='value', dtype=object).rename_axis('parameter').to_csv(
        output / 'best_parameters.csv', encoding='utf-8-sig',
    )
    assignment = metadata.copy()
    assignment['fold'] = 0
    baseline = np.empty_like(actual)
    target_metrics = []
    for fold, (train, valid) in enumerate(folds, 1):
        assignment.loc[valid, 'fold'] = fold
        baseline[valid] = actual[train].mean(axis=0)
        for column, target in enumerate(targets.columns):
            target_metrics.append({
                'fold': fold, 'target': target,
                **scores(actual[valid, column], predictions[selected, valid, column], baseline[valid, column]),
            })
    assignment.to_csv(output / 'fold_assignments.csv', index=False, encoding='utf-8-sig')
    per_fold = pd.DataFrame(target_metrics)
    per_fold.to_csv(output / 'fold_target_metrics.csv', index=False, encoding='utf-8-sig')
    by_fold = per_fold.drop(columns='target').groupby('fold').mean()
    summary = by_fold.copy()
    summary.index = [f'fold_{i}' for i in summary.index]
    summary.loc['cv_mean'], summary.loc['cv_std'] = by_fold.mean(), by_fold.std(ddof=1)
    summary.index.name = 'evaluation'
    summary.to_csv(output / 'cv_summary.csv', encoding='utf-8-sig')
    export_predictions(output, targets, predictions[selected], assignment, baseline, folds)
    run_path = output / 'run.csv'
    settings = pd.read_csv(run_path).set_index('setting')['value'].to_dict() if run_path.exists() else {}
    settings.update(status='completed', best_trial=selected, candidates=len(params), target_metadata_sha256=data_signature(pd.DataFrame(), targets, metadata))
    pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
    print(f'Best trial={selected}, five-fold R2={table.loc[selected, "mean_r2"]:.6f}', flush=True)
    return selected
