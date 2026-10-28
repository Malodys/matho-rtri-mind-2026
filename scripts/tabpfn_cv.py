from __future__ import annotations

import argparse
import gc
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


PROJECT = Path(__file__).resolve().parents[1]
METADATA = ['beer', 'beer_id', 'tasting_category_fine']


def load_split(directory: Path, subset: str = 'train'):
    chemical = pd.read_csv(directory / f'1_{subset}.csv')
    sensory = pd.read_csv(directory / f'4_{subset}.csv')
    if not chemical['beer'].is_unique or not sensory['beer'].is_unique:
        raise ValueError('Duplicate beer identifiers.')
    if set(chemical['beer']) != set(sensory['beer']):
        raise ValueError('Chemical and sensory beer identifiers differ.')
    sensory = sensory.set_index('beer').loc[chemical['beer']].reset_index()
    if not chemical['tasting_category_fine'].equals(sensory['tasting_category_fine']):
        raise ValueError('Chemical and sensory styles differ.')
    features = chemical.drop(columns=METADATA).apply(pd.to_numeric, errors='raise')
    targets = sensory.drop(columns=METADATA).apply(pd.to_numeric, errors='raise')
    if np.isinf(features.to_numpy()).any():
        raise ValueError('Infinite chemical values.')
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError('Non-finite sensory scores.')
    return features, targets, chemical[['beer', 'tasting_category_fine']].copy()


def scores(actual, predicted, baseline):
    result = {}
    for prefix, values in [('', predicted), ('baseline_', baseline)]:
        result[prefix + 'mae'] = mean_absolute_error(actual, values)
        result[prefix + 'rmse'] = np.sqrt(mean_squared_error(actual, values))
        result[prefix + 'r2'] = r2_score(actual, values)
    return result


def cross_validate(features, targets, metadata, output, model_factory, seed=0):
    output.mkdir(parents=True, exist_ok=True)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(features))
    assignment = metadata.copy()
    assignment['fold'] = 0
    for fold, (_, valid) in enumerate(splits, 1):
        assignment.loc[valid, 'fold'] = fold
    assignment.to_csv(output / 'fold_assignments.csv', index=False, encoding='utf-8-sig')

    predictions, fold_metrics = [], []
    started = perf_counter()
    for fold, (train, valid) in enumerate(splits, 1):
        print(f'Fold {fold}/5: train={len(train)}, validation={len(valid)}', flush=True)
        for number, target in enumerate(targets.columns, 1):
            tick = perf_counter()
            model = model_factory()
            model.fit(features.iloc[train], targets[target].iloc[train])
            predicted = np.asarray(model.predict(features.iloc[valid])).reshape(-1)
            if len(predicted) != len(valid) or not np.isfinite(predicted).all():
                raise ValueError(f'Invalid predictions: fold={fold}, target={target}')
            actual = targets[target].iloc[valid].to_numpy()
            baseline = np.full(len(valid), targets[target].iloc[train].mean())
            metric = {
                'fold': fold, 'target': target,
                'n_train': len(train), 'n_valid': len(valid),
                **scores(actual, predicted, baseline),
                'seconds': perf_counter() - tick,
            }
            fold_metrics.append(metric)
            rows = metadata.iloc[valid].copy()
            rows['fold'], rows['target'] = fold, target
            rows['actual'], rows['prediction'] = actual, predicted
            rows['baseline_prediction'] = baseline
            predictions.append(rows)
            print(
                f'  {number:02d}/{len(targets.columns)} {target}: '
                f'R2={metric["r2"]:.4f}, RMSE={metric["rmse"]:.4f}, '
                f'{metric["seconds"]:.1f}s', flush=True,
            )
            del model
            gc.collect()
        pd.concat(predictions, ignore_index=True).to_csv(
            output / 'oof_predictions.csv', index=False, encoding='utf-8-sig',
        )
        pd.DataFrame(fold_metrics).to_csv(
            output / 'fold_target_metrics.csv', index=False, encoding='utf-8-sig',
        )

    oof = pd.concat(predictions, ignore_index=True)
    per_target = []
    for target, group in oof.groupby('target', sort=False):
        per_target.append({
            'target': target, 'n_samples': len(group),
            **scores(group['actual'], group['prediction'], group['baseline_prediction']),
        })
    target_metrics = pd.DataFrame(per_target)
    target_metrics.to_csv(output / 'target_metrics.csv', index=False, encoding='utf-8-sig')

    metric_columns = ['mae', 'rmse', 'r2', 'baseline_mae', 'baseline_rmse', 'baseline_r2']
    by_fold = pd.DataFrame(fold_metrics).groupby('fold')[metric_columns].mean()
    summary = by_fold.copy()
    summary.index = [f'fold_{fold}' for fold in summary.index]
    summary.loc['cv_mean'] = by_fold.mean()
    summary.loc['cv_std'] = by_fold.std(ddof=1)
    summary.loc['pooled_oof'] = target_metrics[metric_columns].mean()
    summary.index.name = 'evaluation'
    summary.to_csv(output / 'cv_summary.csv', encoding='utf-8-sig')
    print(summary.round(5).to_string(), flush=True)
    print(f'Elapsed: {perf_counter() - started:.1f}s', flush=True)
    return summary


def evaluate_test(training, testing, output, model_factory):
    features, targets, train_metadata = training
    test_features, test_targets, test_metadata = testing
    if set(train_metadata['beer']) & set(test_metadata['beer']):
        raise ValueError('Training and test beers overlap.')
    if set(features.columns) != set(test_features.columns):
        raise ValueError('Training and test chemical columns differ.')
    test_features = test_features.loc[:, features.columns]
    test_targets = test_targets.loc[:, targets.columns]
    output.mkdir(parents=True, exist_ok=True)
    predictions, metrics = [], []
    started = perf_counter()
    for number, target in enumerate(targets.columns, 1):
        tick = perf_counter()
        model = model_factory()
        model.fit(features, targets[target])
        predicted = np.asarray(model.predict(test_features)).reshape(-1)
        if len(predicted) != len(test_features) or not np.isfinite(predicted).all():
            raise ValueError(f'Invalid test predictions: target={target}')
        baseline = np.full(len(test_features), targets[target].mean())
        actual = test_targets[target].to_numpy()
        metric = {
            'target': target, 'n_train': len(features), 'n_test': len(test_features),
            **scores(actual, predicted, baseline), 'seconds': perf_counter() - tick,
        }
        metrics.append(metric)
        rows = test_metadata.copy()
        rows['target'] = target
        rows['actual'], rows['prediction'] = actual, predicted
        rows['baseline_prediction'] = baseline
        predictions.append(rows)
        print(
            f'{number:02d}/{len(targets.columns)} {target}: '
            f'R2={metric["r2"]:.4f}, RMSE={metric["rmse"]:.4f}, '
            f'{metric["seconds"]:.1f}s', flush=True,
        )
        del model
        gc.collect()
    pd.concat(predictions, ignore_index=True).to_csv(
        output / 'test_predictions.csv', index=False, encoding='utf-8-sig',
    )
    pd.DataFrame(metrics).to_csv(output / 'target_metrics.csv', index=False, encoding='utf-8-sig')
    metric_columns = ['mae', 'rmse', 'r2', 'baseline_mae', 'baseline_rmse', 'baseline_r2']
    summary = pd.DataFrame(metrics)[metric_columns].mean().to_frame().T
    summary.index = pd.Index(['holdout_test'], name='evaluation')
    summary.to_csv(output / 'test_summary.csv', encoding='utf-8-sig')
    print(summary.round(5).to_string(), flush=True)
    print(f'Elapsed: {perf_counter() - started:.1f}s', flush=True)
    return summary


def project_path(value: Path):
    return value if value.is_absolute() else PROJECT / value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument(
        '--model-path', type=Path,
        default=Path('../pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors'),
    )
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-estimators', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--targets', nargs='+')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()

    import torch
    from tabpfn import TabPFNRegressor

    model_path = project_path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    data_dir = project_path(args.data_dir)
    evaluation = 'test' if args.evaluate_test else 'cv5'
    output = project_path(args.output_dir or Path(f'results/tabpfn_3_5_{evaluation}_seed_{args.seed}'))
    features, targets, metadata = load_split(data_dir)
    if args.targets:
        targets = targets.loc[:, args.targets]
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        'python': platform.python_version(), 'tabpfn': version('tabpfn'),
        'torch': torch.__version__, 'scikit-learn': version('scikit-learn'),
        'numpy': version('numpy'), 'pandas': version('pandas'),
        'device': device, 'seed': args.seed, 'evaluation': evaluation,
        'folds': 0 if args.evaluate_test else 5,
        'n_estimators': args.n_estimators, 'samples': len(features),
        'features': len(features.columns), 'targets': len(targets.columns),
        'data_dir': args.data_dir.as_posix(), 'model_path': args.model_path.as_posix(),
        'missing_features': int(features.isna().sum().sum()), 'status': 'running',
    }
    pd.Series(settings, name='value').rename_axis('setting').to_csv(
        output / 'run.csv', encoding='utf-8-sig',
    )
    print(pd.Series(settings).to_string(), flush=True)
    def model_factory():
        return TabPFNRegressor(
            model_path=model_path, device=device, random_state=args.seed,
            n_estimators=args.n_estimators, show_progress_bar=False,
        )

    if args.evaluate_test:
        evaluate_test(
            (features, targets, metadata), load_split(data_dir, 'test'), output, model_factory,
        )
    else:
        cross_validate(features, targets, metadata, output, model_factory, seed=args.seed)
    settings['status'] = 'completed'
    pd.Series(settings, name='value').rename_axis('setting').to_csv(
        output / 'run.csv', encoding='utf-8-sig',
    )


if __name__ == '__main__':
    main()
