from __future__ import annotations

import argparse
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn

from tabpfn_cv import PROJECT, load_split, project_path, scores


class ObservedScaler:
    def __init__(self, clip=None):
        self.clip = clip

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        count = np.isfinite(values).sum(axis=0)
        self.mean = np.nansum(values, axis=0) / np.maximum(count, 1)
        variance = np.nansum((values - self.mean) ** 2, axis=0) / np.maximum(count, 1)
        self.scale = np.sqrt(variance)
        self.scale[self.scale < 1e-8] = 1.
        return self

    def transform(self, values):
        scaled = ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)
        return np.clip(scaled, -self.clip, self.clip) if self.clip is not None else scaled

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale + self.mean


class MaskedMultiTaskNet(nn.Module):
    def __init__(self, n_features, n_targets, hidden=(64, 16), dropout=0.15):
        super().__init__()
        self.missing_values = nn.Parameter(torch.zeros(n_features))
        layers = []
        width = n_features
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), nn.GELU(), nn.Dropout(dropout)])
            width = next_width
        layers.append(nn.Linear(width, n_targets))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        values = torch.where(torch.isnan(values), self.missing_values, values)
        return self.network(values)


def fit_member(
    features, targets, *, seed, device, hidden=(64, 16), dropout=0.15,
    max_epochs=600, patience=60, learning_rate=0.001, weight_decay=0.01,
    input_clip=5., sample_weight_factory=None,
):
    features = np.asarray(features)
    targets = np.asarray(targets)
    fit_rows, stop_rows = train_test_split(
        np.arange(len(features)), test_size=0.2, random_state=seed,
    )
    x_scaler = ObservedScaler(clip=input_clip).fit(features[fit_rows])
    y_scaler = ObservedScaler().fit(targets[fit_rows])

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def new_model():
        torch.manual_seed(seed)
        model = MaskedMultiTaskNet(features.shape[1], targets.shape[1], hidden, dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        return model, optimizer

    def step(model, optimizer, x, y, sample_weights=None):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        if sample_weights is None:
            loss = nn.functional.mse_loss(prediction, y)
        else:
            loss = ((prediction-y).square().mean(dim=1)*sample_weights).sum()/sample_weights.sum()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss.')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        return loss.item()

    x_fit, y_fit = tensor(x_scaler.transform(features[fit_rows])), tensor(y_scaler.transform(targets[fit_rows]))
    x_stop, y_stop = tensor(x_scaler.transform(features[stop_rows])), tensor(y_scaler.transform(targets[stop_rows]))
    def weighting(values):
        if sample_weight_factory is None:
            return None
        weights = np.asarray(sample_weight_factory(values), dtype=float)
        if weights.shape != (len(values),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Invalid training sample weights.')
        return tensor(weights)

    fit_weights = weighting(features[fit_rows])
    model, optimizer = new_model()
    best_loss, best_epoch = float('inf'), 1
    for epoch in range(1, max_epochs + 1):
        step(model, optimizer, x_fit, y_fit, fit_weights)
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.mse_loss(model(x_stop), y_stop).item()
        if not np.isfinite(validation_loss):
            raise ValueError('Non-finite inner validation loss.')
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch = validation_loss, epoch
        if epoch - best_epoch >= patience:
            break

    x_scaler = ObservedScaler(clip=input_clip).fit(features)
    y_scaler = ObservedScaler().fit(targets)
    x_all, y_all = tensor(x_scaler.transform(features)), tensor(y_scaler.transform(targets))
    all_weights = weighting(features)
    del model, optimizer
    model, optimizer = new_model()
    for _ in range(best_epoch):
        final_loss = step(model, optimizer, x_all, y_all, all_weights)
    model.eval()
    info = {
        'selected_epochs': best_epoch, 'selection_epochs_run': epoch,
        'inner_validation_loss': best_loss, 'refit_training_loss': final_loss,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, x_scaler, y_scaler, info


def run_cv(features, targets, metadata, output, args):
    output.mkdir(parents=True, exist_ok=True)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(features))
    assignment = metadata.copy()
    assignment['fold'] = 0
    for fold, (_, valid) in enumerate(folds, 1):
        assignment.loc[valid, 'fold'] = fold
    reference = PROJECT / 'results' / f'tabpfn_3_5_cv5_seed_{args.seed}'
    if (reference / 'fold_assignments.csv').is_file():
        pd.testing.assert_frame_equal(assignment, pd.read_csv(reference / 'fold_assignments.csv'))
    assignment.to_csv(output / 'fold_assignments.csv', index=False, encoding='utf-8-sig')

    predictions, metrics, training_records = [], [], []
    started = perf_counter()
    for fold, (train, valid) in enumerate(folds, 1):
        tick = perf_counter()
        member_predictions = []
        print(f'Fold {fold}/5: train={len(train)}, validation={len(valid)}', flush=True)
        for member in range(args.members):
            member_seed = args.seed + fold * 1000 + member
            model, x_scaler, y_scaler, info = fit_member(
                features.iloc[train].to_numpy(), targets.iloc[train].to_numpy(),
                seed=member_seed, device=args.device, hidden=tuple(args.hidden),
                dropout=args.dropout, max_epochs=args.max_epochs, patience=args.patience,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                input_clip=args.input_clip or None,
            )
            with torch.no_grad():
                x_valid = torch.as_tensor(
                    x_scaler.transform(features.iloc[valid].to_numpy()), device=args.device,
                )
                predicted = model(x_valid).cpu().numpy()
            member_predictions.append(y_scaler.inverse_transform(predicted))
            training_records.append({'fold': fold, 'member': member + 1, 'seed': member_seed, **info})
            print(
                f'  member {member + 1}/{args.members}: '
                f'epochs={info["selected_epochs"]}, '
                f'inner_loss={info["inner_validation_loss"]:.4f}', flush=True,
            )
            del model
        predicted = np.mean(member_predictions, axis=0)
        if not np.isfinite(predicted).all():
            raise ValueError(f'Non-finite fold {fold} predictions.')
        for column, target in enumerate(targets.columns):
            actual = targets[target].iloc[valid].to_numpy()
            baseline = np.full(len(valid), targets[target].iloc[train].mean())
            metrics.append({
                'fold': fold, 'target': target, 'n_train': len(train), 'n_valid': len(valid),
                **scores(actual, predicted[:, column], baseline),
            })
            rows = metadata.iloc[valid].copy()
            rows['fold'], rows['target'] = fold, target
            rows['actual'], rows['prediction'] = actual, predicted[:, column]
            rows['baseline_prediction'] = baseline
            predictions.append(rows)
        fold_score = pd.DataFrame(metrics).query('fold == @fold')['r2'].mean()
        print(f'  mean R2={fold_score:.4f}, {perf_counter() - tick:.1f}s', flush=True)
        pd.DataFrame(training_records).to_csv(output / 'training.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(metrics).to_csv(output / 'fold_target_metrics.csv', index=False, encoding='utf-8-sig')
        pd.concat(predictions, ignore_index=True).to_csv(
            output / 'oof_predictions.csv', index=False, encoding='utf-8-sig',
        )

    oof = pd.concat(predictions, ignore_index=True)
    per_target = pd.DataFrame([
        {'target': target, 'n_samples': len(group),
         **scores(group['actual'], group['prediction'], group['baseline_prediction'])}
        for target, group in oof.groupby('target', sort=False)
    ])
    per_target.to_csv(output / 'target_metrics.csv', index=False, encoding='utf-8-sig')
    columns = ['mae', 'rmse', 'r2', 'baseline_mae', 'baseline_rmse', 'baseline_r2']
    by_fold = pd.DataFrame(metrics).groupby('fold')[columns].mean()
    summary = by_fold.copy()
    summary.index = [f'fold_{fold}' for fold in summary.index]
    summary.loc['cv_mean'], summary.loc['cv_std'] = by_fold.mean(), by_fold.std(ddof=1)
    summary.loc['pooled_oof'] = per_target[columns].mean()
    summary.index.name = 'evaluation'
    summary.to_csv(output / 'cv_summary.csv', encoding='utf-8-sig')
    if (reference / 'cv_summary.csv').is_file():
        previous = pd.read_csv(reference / 'cv_summary.csv').set_index('evaluation')
        comparison = pd.DataFrame({
            'TabPFN': previous.loc['cv_mean', ['mae', 'rmse', 'r2']],
            'MultiTaskMLP': summary.loc['cv_mean', ['mae', 'rmse', 'r2']],
        }).T
        comparison.index.name = 'model'
        comparison.to_csv(output / 'comparison.csv', encoding='utf-8-sig')
    print(summary.round(5).to_string(), flush=True)
    print(f'Elapsed: {perf_counter() - started:.1f}s', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/splits/seed_0'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, nargs='+', default=[64, 16])
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--max-epochs', type=int, default=600)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--input-clip', type=float, default=5., help='0 disables clipping')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if min(args.members, args.max_epochs, args.patience, *args.hidden) < 1:
        parser.error('Members, epochs, patience, and hidden widths must be positive.')
    if args.input_clip < 0:
        parser.error('Input clipping bound must be nonnegative.')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    features, targets, metadata = load_split(project_path(args.data_dir))
    output = project_path(args.output_dir or Path(f'results/multitask_clip_{args.input_clip:g}_cv5_seed_{args.seed}'))
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': version('numpy'), 'pandas': version('pandas'),
        'scikit-learn': version('scikit-learn'), 'device': args.device,
        'seed': args.seed, 'folds': 5, 'samples': len(features),
        'features': len(features.columns), 'targets': len(targets.columns),
        'hidden': '-'.join(map(str, args.hidden)), 'dropout': args.dropout,
        'members': args.members, 'max_epochs': args.max_epochs, 'patience': args.patience,
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'input_clip': args.input_clip, 'missing_values': 'learned_per_feature_token',
        'early_stopping': 'inner_80_20_then_refit_outer_training',
        'data_dir': args.data_dir.as_posix(), 'status': 'running',
    }
    pd.Series(settings, name='value').rename_axis('setting').to_csv(output / 'run.csv', encoding='utf-8-sig')
    print(pd.Series(settings).to_string(), flush=True)
    run_cv(features, targets, metadata, output, args)
    settings['status'] = 'completed'
    pd.Series(settings, name='value').rename_axis('setting').to_csv(output / 'run.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
