# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(r"F:\cs\mothercup\matho-rtri-mind-2026\scripts")))
import numpy as np, pandas as pd
import data_contract as dc
cp = pd.read_csv(Path(r"F:\cs\mothercup\matho-rtri-mind-2026\data\processed\q1_final\proxy_grade_cutpoints.csv"),
                 encoding="utf-8-sig", index_col=0)
print("切点表形状:", cp.shape, "列:", list(cp.columns))
print(cp.head(3).to_string())
print("索引前5:", list(cp.index[:5]))
# 与 q1_evaluation 的 training_quantile_cutpoints 对账：训练集 175 条的七分位
raw = dc.load_raw(); y = dc.load_targets(raw); styles = dc.load_styles(raw)
split = dc.load_split(); train_ids = split.loc[split["split"]=="train","beer"].tolist()
y = y.loc[y.index.intersection(train_ids)]
ref = np.quantile(y.to_numpy(), np.arange(1,7)/7, axis=0).T   # (50,6)
mine = cp.to_numpy()
print("\n切点与 training_quantile_cutpoints 重算最大偏差:", float(np.max(np.abs(ref-mine))))
print("维度名与 50 目标一致:", list(cp.index) == list(y.columns))
# 验证 apply_cutpoints 口径
import q1_evaluation as q1e
grades = q1e.apply_cutpoints(y.to_numpy(), ref)
print("真实目标离散化后 grade 取值范围:", grades.min(), grades.max(), "行数", grades.shape)
