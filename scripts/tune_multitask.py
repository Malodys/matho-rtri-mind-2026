from __future__ import annotations

import argparse
import itertools
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from multitask_cv import fit_member
from tabpfn_cv import project_path
from tuning_protocol import data_signature, load_protocol, make_store, save_selection


def search_candidates(seed=0):
    """Freeze 48 distinct configurations before observing any validation score."""
    keys = ('hidden', 'dropout', 'learning_rate', 'weight_decay', 'input_clip')
    grid = list(itertools.product(
        [(32, 8), (64, 16), (64, 32), (128, 32)],
        [0., .15, .3], [.0003, .001, .003], [.001, .01, .1], [3., 5., 10.],
    ))
    base = ((64, 16), .15, .001, .01, 5.)
    grid.remove(base)
    indices = np.random.default_rng(seed).choice(len(grid), size=47, replace=False)
    return [dict(zip(keys, values)) for values in [base] + [grid[i] for i in indices]]


def fit_predict(params, X_train, Y_train, X_eval, seed=0, device='cuda',
                members=3, max_epochs=600, patience=60, return_info=False):
    """Select epochs on an inner split, refit all supplied training rows, ensemble.

    Evaluation labels are deliberately absent from this API. ``return_info`` adds
    the per-member epoch and seed audit records without changing training.
    """
    if min(members, max_epochs, patience) < 1:
        raise ValueError('Members, max_epochs and patience must be positive.')
    predictions, records = [], []
    X_train, Y_train, X_eval = map(np.asarray, (X_train, Y_train, X_eval))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for member in range(members):
        member_seed = seed + member
        model, x_scaler, y_scaler, info = fit_member(
            X_train, Y_train, seed=member_seed, device=device,
            max_epochs=max_epochs, patience=patience, **params,
        )
        with torch.no_grad():
            values = torch.as_tensor(x_scaler.transform(X_eval), device=device)
            predicted = y_scaler.inverse_transform(model(values).cpu().numpy())
        predictions.append(predicted)
        records.append({'member': member + 1, 'seed': member_seed, **info})
        del model, values
    predicted = np.mean(predictions, axis=0)
    if not np.isfinite(predicted).all():
        raise ValueError('Non-finite ensemble predictions.')
    return (predicted, records) if return_info else predicted


def run_search(features, targets, metadata, folds, output, args):
    params = search_candidates(args.seed)
    signature = data_signature(features, targets, metadata)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / 'run.csv'
    settings = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': version('numpy'), 'pandas': version('pandas'),
        'scikit-learn': version('scikit-learn'), 'device': args.device,
        'seed': args.seed, 'folds': 5, 'samples': len(features),
        'features': features.shape[1], 'targets': targets.shape[1],
        'candidates': len(params), 'members': args.members, 'max_epochs': 600,
        'patience': 60, 'dataset_sha256': signature, 'status': 'running',
        'search': '47_seeded_without_replacement_plus_original_base',
        'selection_metric': 'mean_R2_over_5_folds_and_all_targets',
        'evaluation': 'selection_CV_not_unbiased_generalization_estimate',
        'early_stopping': 'inner_80_20_then_refit_all_outer_training',
    }
    if run_path.exists():
        previous = pd.read_csv(run_path).set_index('setting')['value'].to_dict()
        for key in ('dataset_sha256', 'seed', 'members', 'max_epochs', 'patience', 'python', 'torch', 'numpy', 'pandas', 'scikit-learn', 'device'):
            if str(previous.get(key)) != str(settings[key]):
                raise ValueError(f'Resume setting differs: {key}; choose a new output directory.')
    pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
    predictions, done = make_store(output, params, (len(params), len(features), targets.shape[1]))
    training_path = output / 'training.csv'
    records = pd.read_csv(training_path).to_dict('records') if training_path.exists() else []
    started = perf_counter()
    try:
        for trial, config in enumerate(params):
            print(f'Trial {trial + 1}/{len(params)}: {config}', flush=True)
            for fold, (train, valid) in enumerate(folds, 1):
                if done[trial, fold - 1].all():
                    continue
                tick = perf_counter()
                predicted, info = fit_predict(
                    config, features.iloc[train], targets.iloc[train], features.iloc[valid],
                    seed=args.seed + fold * 1000, device=args.device,
                    members=args.members, return_info=True,
                )
                # Replace interrupted fold records rather than duplicating them on resume.
                records = [row for row in records if not (row['trial'] == trial and row['fold'] == fold)]
                records.extend({'trial': trial, 'fold': fold, **row} for row in info)
                pd.DataFrame(records).to_csv(training_path, index=False, encoding='utf-8-sig')
                predictions[trial, valid] = predicted
                predictions.flush()
                done[trial, fold - 1] = True
                done.flush()
                print(f'  fold {fold}/5 completed in {perf_counter() - tick:.1f}s; '
                      f'epochs={[r["selected_epochs"] for r in info]}', flush=True)
        selected = save_selection(output, params, predictions, targets, metadata, folds)
        print(f'Search completed in {perf_counter() - started:.1f}s', flush=True)
        return selected
    except BaseException:
        settings['status'] = 'interrupted_or_failed'
        pd.Series(settings, name='value').rename_axis('setting').to_csv(run_path, encoding='utf-8-sig')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/tuned_seed_0/multitask'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.members < 1:
        parser.error('Members must be positive.')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    features, targets, metadata, folds = load_protocol(project_path(args.data_dir), seed=args.seed)
    run_search(features, targets, metadata, folds, project_path(args.output_dir), args)


if __name__ == '__main__':
    main()
