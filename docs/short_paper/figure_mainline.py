from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent / 'figures'
out.mkdir(exist_ok=True)

# Q1: show where the model improves on the no-chemistry baseline.
families = pd.read_csv(root / 'reports/modeling/q1_mainline_retrain_v1/family_rmse_summary.csv').set_index('model')
base = families.loc['global_mean']
main = families.loc['grouped_extra_trees']
gain = 100 * (base - main) / base
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=180)
names = ['Aroma', 'Flavour', 'Mouthfeel', 'Fault / special', 'Overall']
ax.barh(names[::-1], gain.to_numpy()[::-1], color='#2474A6')
ax.set_xlabel('RMSE reduction vs mean baseline (%)')
ax.set_xlim(0, 38)
for i, v in enumerate(gain.to_numpy()[::-1]):
    ax.text(v + .5, i, f'{v:.1f}%', va='center', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out / 'q1_family_gain.png')
plt.close(fig)

# Q3: the two objectives move in different directions under source guidance.
comparison = pd.read_csv(root / 'reports/modeling/q3_source_guided_v2/source_guidance_comparison.csv')
comparison = comparison[comparison.status == 'compared']
groups = comparison.groupby('target_style').mean(numeric_only=True)
colors = {'Tripel': '#2474A6', 'Kriek': '#C97928', 'Lambic': '#5C9A67'}
fig, ax = plt.subplots(figsize=(5.9, 3.5), dpi=180)
for style, row in groups.iterrows():
    color = colors[style]
    x0, y0 = row.target_only_typicality_distance, row.target_only_audit_distance
    x1, y1 = row.guided_typicality_distance, row.guided_audit_distance
    ax.scatter(x0, y0, marker='s', s=75, color=color)
    ax.scatter(x1, y1, marker='o', s=75, color=color)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.6})
    ax.annotate(style, (x1, y1), xytext=(5, -11), textcoords='offset points', color=color, fontsize=9)
ax.set_xlabel('Distance to target-style real samples')
ax.set_ylabel('Distance to source traits')
ax.set_title('Squares: target only    Circles: source guided', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out / 'q3_tradeoff.png')
