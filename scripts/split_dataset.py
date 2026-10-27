from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(seed: int = 0) -> Path:
    project = Path(__file__).resolve().parents[1]
    raw = project / 'raw_data'
    output = project / 'data' / 'splits' / f'seed_{seed}'
    chemical = pd.read_csv(raw / '1.csv', dtype=str, keep_default_na=False)
    sensory = pd.read_csv(raw / '4.csv', dtype=str, keep_default_na=False)
    assert chemical['beer'].is_unique and sensory['beer'].is_unique
    assert set(chemical['beer']) == set(sensory['beer'])
    sensory = sensory.set_index('beer').loc[chemical['beer']].reset_index()
    assert chemical['beer'].equals(sensory['beer'])
    assert chemical['tasting_category_fine'].equals(sensory['tasting_category_fine'])

    positions = np.arange(len(chemical))
    train_pos, test_pos = train_test_split(
        positions, test_size=0.30, random_state=seed,
        shuffle=True, stratify=chemical['tasting_category_fine'],
    )
    repeat_train, repeat_test = train_test_split(
        positions, test_size=0.30, random_state=seed,
        shuffle=True, stratify=chemical['tasting_category_fine'],
    )
    assert np.array_equal(train_pos, repeat_train)
    assert np.array_equal(test_pos, repeat_test)
    assert not (set(train_pos) & set(test_pos))
    assert set(train_pos) | set(test_pos) == set(positions)

    train_pos, test_pos = np.sort(train_pos), np.sort(test_pos)
    subsets = {'train': train_pos, 'test': test_pos}
    plan = chemical[['beer', 'tasting_category_fine']].copy()
    plan.insert(0, 'source_row', positions + 1)
    plan['split'] = 'train'
    plan.loc[test_pos, 'split'] = 'test'
    style_counts = pd.crosstab(plan['tasting_category_fine'], plan['split'])
    style_counts = style_counts.reindex(columns=['train', 'test'])
    style_counts['total'] = style_counts.sum(axis=1)
    assert (style_counts[['train', 'test']] > 0).all().all()

    output.mkdir(parents=True, exist_ok=True)
    for label, pos in subsets.items():
        for number, source in [('1', chemical), ('4', sensory)]:
            target = output / f'{number}_{label}.csv'
            selected = source.iloc[pos].reset_index(drop=True)
            selected.to_csv(target, index=False, encoding='utf-8-sig')
            reread = pd.read_csv(target, dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(reread, selected)
    plan.to_csv(output / 'split_manifest.csv', index=False, encoding='utf-8-sig')
    style_counts.to_csv(output / 'style_counts.csv', encoding='utf-8-sig')

    def na_count(frame: pd.DataFrame) -> int:
        return int(frame.isin(['', 'NA']).to_numpy().sum())

    missing = {}
    for number, frame in [('1', chemical), ('4', sensory)]:
        missing[number] = {'all': na_count(frame)}
        missing[number].update({name: na_count(frame.iloc[pos]) for name, pos in subsets.items()})
        assert missing[number]['all'] == missing[number]['train'] + missing[number]['test']
    print(f'seed={seed}, train={len(train_pos)}, test={len(test_pos)}')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    split_dataset(parser.parse_args().seed)
