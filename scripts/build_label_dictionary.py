from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_label_dictionary() -> Path:
    project = Path(__file__).resolve().parents[1]
    raw = project / 'data/raw'
    output = project / 'data' / 'interim' / 'label_dictionary_v1.csv'

    labels_raw = pd.read_csv(raw / '4.csv', encoding='utf-8-sig')
    label_columns = list(labels_raw.columns[3:])
    y = labels_raw.set_index('beer')[label_columns].apply(pd.to_numeric, errors='coerce')
    assert y.shape == (250, 50)
    assert y.isna().sum().sum() == 0

    sensory_names = pd.read_excel(raw / '8.xlsx', sheet_name='Sensory names')
    panel_meta = (
        sensory_names[sensory_names['dataset'].eq('Trained panel')]
        .drop_duplicates('R_name')
        .set_index('R_name')
    )
    assert set(label_columns) == set(panel_meta.index)

    label_dictionary = panel_meta.reindex(label_columns)[[
        'Category_Paper', 'Category', 'CleanName_Paper', 'CleanName_PaperLong',
        'CleanName_PanelFeedback', 'DataType']].copy()
    label_dictionary.index.name = 'label'
    label_dictionary = label_dictionary.rename(columns={
        'Category_Paper': 'sensory_dimension',
        'Category': 'chemical_sensory_family',
        'CleanName_Paper': 'paper_name',
        'CleanName_PaperLong': 'paper_name_long',
        'CleanName_PanelFeedback': 'panel_form_name',
        'DataType': 'source_item_type',
    })
    label_dictionary['model_target_type'] = 'continuous_regression'

    output.parent.mkdir(parents=True, exist_ok=True)
    label_dictionary.to_csv(output, encoding='utf-8-sig')
    print(f'labels={len(label_dictionary)}')
    print(output)
    return output


if __name__ == '__main__':
    build_label_dictionary()
