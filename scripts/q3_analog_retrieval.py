#!/usr/bin/env python3
"""Q3: Retrieve real Tripel analogs and audit defect feasibility."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/modeling/q3_analog_v1'
DEFECTS = ['DMS', 'X2.3.butanedione', 'X4VG', 'X.E..2.nonenal']
THRESHOLDS = {'DMS': np.log10(.019), 'X2.3.butanedione': np.log10(.09), 'X4VG': np.log10(.2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(ROOT/'data/interim/chem_feature_sets_v1.csv')
    features = reg.loc[reg.q3_primary_generation,'feature'].tolist()
    chem = pd.read_csv(ROOT/'data/interim/chem_log10_matrix_v1.csv').set_index('beer')
    labels = pd.read_csv(ROOT/'data/interim/beer_style_v1.csv').set_index('beer').loc[chem.index,'style'].to_numpy()
    sensory = pd.read_csv(ROOT/'data/interim/sensory_targets_v1.csv').set_index('beer').loc[chem.index]
    x = chem[features].to_numpy(float)
    idx = {f:features.index(f) for f in DEFECTS}
    rows, profiles = [], []
    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=20261027)
    for fold,(train,test) in enumerate(cv.split(x,labels),1):
        scale = StandardScaler().fit(x[train])
        portrait = Ridge(alpha=100).fit(scale.transform(x[train]),sensory.iloc[train].to_numpy(float))
        stale_col = list(sensory.columns).index('stale_hops')
        refs = train[labels[train]=='Tripel']
        for si in test[labels[test]=='Pils/Lager']:
            source = x[si]
            source_stale = portrait.predict(scale.transform(source[None,:]))[0,stale_col]
            candidates = []
            for ti in refs:
                ref = x[ti]
                strict = all(ref[j] <= source[j]+1e-12 for j in idx.values())
                allowed = {f:max(source[idx[f]],THRESHOLDS[f]) if f in THRESHOLDS else source[idx[f]] for f in DEFECTS}
                threshold_feasible = all(ref[idx[f]] <= allowed[f]+1e-12 for f in DEFECTS)
                corrected = ref.copy()
                for f,j in idx.items():
                    corrected[j] = min(corrected[j],allowed[f])
                edit = float(np.linalg.norm(scale.transform(corrected[None,:])-scale.transform(ref[None,:])))
                scripts_dist = float(np.linalg.norm(scale.transform(corrected[None,:])-scale.transform(source[None,:])))
                stale = portrait.predict(scale.transform(corrected[None,:]))[0,stale_col]
                candidates.append((ti,corrected,strict,threshold_feasible,edit,scripts_dist,stale))
            feasible = [v for v in candidates if v[6] <= source_stale]
            chosen = min(feasible or candidates,key=lambda v:v[5]+.5*v[4])
            ti,g,strict,threshold_feasible,edit,scripts_dist,stale = chosen
            rows.append({'fold':fold,'source_beer':int(chem.index[si]),'tripel_reference_beer':int(chem.index[ti]),
                         'n_strict_real_refs':sum(v[2] for v in candidates),
                         'n_threshold_real_refs':sum(v[3] for v in candidates),
                         'reference_defect_edit_distance':edit,'source_to_candidate_distance':scripts_dist,
                         'n_defect_columns_corrected':sum(g[j]<x[ti,j]-1e-10 for j in idx.values()),
                         'predicted_stale_hops_delta':float(stale-source_stale),
                         'predicted_stale_hops_nonworsening':bool(stale<=source_stale)})
            profiles.append({'source_beer':int(chem.index[si]),'tripel_reference_beer':int(chem.index[ti]),**dict(zip(features,g))})
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'analog_audit.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'analog_candidate_profiles_logscale.csv',index=False)
    summary={'n_sources':len(d),'strict_real_reference_feasible_sources':int((d.n_strict_real_refs>0).sum()),
             'threshold_real_reference_feasible_sources':int((d.n_threshold_real_refs>0).sum()),
             'selected_predicted_stale_nonworsening':int(d.predicted_stale_hops_nonworsening.sum()),
             'mean_source_to_candidate_distance':float(d.source_to_candidate_distance.mean()),
             'mean_reference_defect_edit_distance':float(d.reference_defect_edit_distance.mean())}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)


if __name__=='__main__':
    main()
