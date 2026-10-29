#!/usr/bin/env python3
"""Source-guided design with defect and target-style gates, plus separate audits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist

from q3_copula import ConditionalCopula

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/modeling/q3_source_guided_v2'
TARGETS = ['Tripel', 'Kriek', 'Lambic']
M, K = 2000, 3
MIN_POOL = 30
MATCH_BINS = 3
KNOWN_LIMITS = {'DMS': np.log10(.019), 'X2.3.butanedione': np.log10(.09), 'X4VG': np.log10(.2)}
RELATIVE_ONLY = 'X.E..2.nonenal'
EXCLUDED_SENSORY = {'overall','diacetyl','dms','t2n','X4vg','stale_hops','metallic'}


def axes_for_source(y0, source_style, labels, ytrain, names, r2):
    local = ytrain[labels == source_style]
    if len(local) < 5:
        local = ytrain
    median = np.median(local, axis=0)
    spread = np.maximum(np.std(ytrain, axis=0), .1)
    salience = np.abs((y0 - median) / spread)
    allowed = [j for j, name in enumerate(names) if name not in EXCLUDED_SENSORY and r2[j] > 0]
    selected, groups = [], set()
    for j in sorted(allowed, key=lambda j: salience[j], reverse=True):
        name = names[j]
        group = name[2:] if name.startswith(('A_','F_')) else name
        if group in groups:
            continue
        selected.append(j); groups.add(group)
        if len(selected) == 3:
            break
    return selected


def choose_guided(typicality, source_distance, transformed, diversity_threshold, count=3):
    # A single candidate pool supplies all three design positions.
    rank_t = pd.Series(typicality).rank(pct=True).to_numpy()
    rank_s = pd.Series(source_distance).rank(pct=True).to_numpy()
    policies = [rank_s + .25*rank_t, rank_t + .25*rank_s, rank_t + rank_s]
    picked = []
    for score in policies[:count]:
        order = np.argsort(score)
        choice = next((int(i) for i in order if int(i) not in picked and
                       (not picked or np.min(cdist(transformed[i:i+1],transformed[picked])) >= diversity_threshold)),
                      next(int(i) for i in order if int(i) not in picked))
        picked.append(choice)
    return picked


def choose_typicality_only(typicality, transformed, diversity_threshold, count=3):
    picked=[]
    order=np.argsort(typicality)
    for _ in range(count):
        choice=next((int(i) for i in order if int(i) not in picked and
                     (not picked or np.min(cdist(transformed[i:i+1],transformed[picked])) >= diversity_threshold)),
                    next(int(i) for i in order if int(i) not in picked))
        picked.append(choice)
    return picked


def pareto_mask(typicality, source_distance):
    mask=np.zeros(len(typicality),dtype=bool)
    best=np.inf
    for i in np.argsort(typicality):
        if source_distance[i] < best-1e-12:
            mask[i]=True
            best=source_distance[i]
    return mask


def min_pairwise(z):
    if len(z)<2:
        return np.nan
    d=cdist(z,z)
    return float(np.min(d[np.triu_indices(len(z),1)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(ROOT/'data/interim/chem_feature_sets_v1.csv')
    features = reg.loc[reg.q3_primary_generation,'feature'].tolist()
    chem = pd.read_csv(ROOT/'data/interim/chem_log10_matrix_v1.csv').set_index('beer')
    status = pd.read_csv(ROOT/'data/interim/chem_observation_status_v1.csv').set_index('beer').loc[chem.index,features].to_numpy(int)
    styles = pd.read_csv(ROOT/'data/interim/beer_style_v1.csv').set_index('beer').loc[chem.index,'style'].to_numpy()
    sensory = pd.read_csv(ROOT/'data/interim/sensory_targets_v1.csv').set_index('beer').loc[chem.index]
    names = sensory.columns.tolist(); x=chem[features].to_numpy(float); y=sensory.to_numpy(float)
    manifest = pd.read_csv(ROOT/'data/splits/seed_0/split_manifest.csv').set_index('beer').loc[chem.index]
    train=np.where(manifest.split.to_numpy()=='train')[0]
    test=np.where(manifest.split.to_numpy()=='test')[0]
    scale=StandardScaler().fit(x[train]); xt=scale.transform(x[train]); xv=scale.transform(x[test])
    # Train-only out-of-fold error and reliability; folds follow the supplied training order.
    original_order = np.argsort(manifest.iloc[train].source_row.to_numpy())
    ordered_train = train[original_order]
    oof=np.empty((len(train),len(names)))
    position={int(idx):i for i,idx in enumerate(train)}
    for fit_pos,val_pos in KFold(n_splits=5,shuffle=False).split(ordered_train):
        fit_idx,val_idx=ordered_train[fit_pos],ordered_train[val_pos]
        fold_scale=StandardScaler().fit(x[fit_idx])
        model=Ridge(alpha=100).fit(fold_scale.transform(x[fit_idx]),y[fit_idx])
        pred=model.predict(fold_scale.transform(x[val_idx]))
        for idx,row in zip(val_idx,pred): oof[position[int(idx)]]=row
    pooled_mae=np.mean(np.abs(oof-y[train]),axis=0)
    reliable_r2=np.array([r2_score(y[train,j],oof[:,j]) for j in range(len(names))])
    error_rows=[]
    style_error={}; style_local_error={}
    for target in TARGETS:
        mask=styles[train]==target
        local=np.mean(np.abs(oof[mask]-y[train][mask]),axis=0)
        style_local_error[target]=np.maximum(local,.1)
        weight=mask.sum()/(mask.sum()+15)
        style_error[target]=np.maximum(weight*local+(1-weight)*pooled_mae,.1)
        error_rows.extend({'target_style':target,'sensory_feature':n,'pooled_oof_mae':pooled_mae[j],
                           'target_shrunk_oof_mae':style_error[target][j], 'oof_r2':reliable_r2[j]}
                          for j,n in enumerate(names))
    pd.DataFrame(error_rows).to_csv(OUT/'sensory_reliability.csv',index=False)
    select_model=Ridge(alpha=100).fit(xt,y[train])
    audit_model=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=3,max_features=.6,
                                    random_state=20261028,n_jobs=-1).fit(xt,y[train])
    heldout={'ridge_mae':float(np.mean(np.abs(select_model.predict(xv)-y[test]))),
             'audit_extra_trees_mae':float(np.mean(np.abs(audit_model.predict(xv)-y[test])))}
    generator=ConditionalCopula(prior=8,seed=20261028).fit(x[train],status[train],styles[train])
    style_audit=RandomForestClassifier(n_estimators=250,min_samples_leaf=2,max_features='sqrt',
                                       random_state=20261028,n_jobs=-1).fit(xt,styles[train])
    pca=PCA(n_components=20,random_state=0).fit(xt)
    style_check=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(pca.transform(xt),styles[train])
    target_pools={}; generation_checks=[]
    for target in TARGETS:
        g,gs=generator.sample(target,M)
        gz=scale.transform(g)
        z=pca.transform(gz)
        real_z=pca.transform(xt[styles[train]==target])
        typicality=cdist(z,real_z).min(axis=1)
        real_pairwise=cdist(real_z,real_z)
        diversity_threshold=float(np.quantile(real_pairwise[np.triu_indices(len(real_z),1)],.25))
        target_pools[target]={'g':g,'status':gs,'z':z,'typicality':typicality,
                              'select_y':select_model.predict(gz),'audit_y':audit_model.predict(gz),
                              'style_gate':style_audit.predict(gz)==target,
                              'style_check':style_check.predict(z)==target,
                              'diversity_threshold':diversity_threshold}
        held_idx=test[styles[test]==target]
        held_z=pca.transform(scale.transform(x[held_idx]))
        active=np.any(status[train]!=0,axis=0)
        generation_checks.append({'target_style':target,'train_n':int((styles[train]==target).sum()),
                                  'heldout_n':len(held_idx),
                                  'generated_style_hit':float(np.mean(style_audit.predict(gz)==target)),
                                  'heldout_style_hit':float(np.mean(style_audit.predict(scale.transform(x[held_idx]))==target)),
                                  'heldout_independent_style_hit':float(np.mean(style_check.predict(held_z)==target)),
                                  'generated_nearest_target_median':float(np.median(typicality)),
                                  'heldout_nearest_target_median':float(np.median(cdist(held_z,real_z).min(axis=1))),
                                  'diversity_threshold':diversity_threshold,
                                  'detection_rate_mae':float(np.mean(np.abs((gs[:,active]==0).mean(axis=0)-
                                                                         (status[held_idx][:,active]==0).mean(axis=0))))})
        pd.DataFrame(g,columns=features).assign(candidate_id=[f'{target}-{i:04d}' for i in range(M)],
                                                target_style=target).to_csv(OUT/f'pool_{target}_logscale.csv',index=False)
    selected_rows=[]; profile_rows=[]; state_rows=[]; portrait_rows=[]; defect_rows=[]; comparison=[]
    rng=np.random.default_rng(20261028)
    fidx={f:features.index(f) for f in features}
    for si in test:
        source=x[si]; y0=y[si]; source_style=styles[si]
        axes=axes_for_source(y0,source_style,styles[train],y[train],names,reliable_r2)
        if not axes:
            for target in TARGETS:
                if target!=source_style:
                    comparison.append({'source_beer':int(chem.index[si]),'source_style':source_style,
                                       'target_style':target,'source_axes_count':0,'n_feasible':np.nan,
                                       'status':'no_reliable_source_axes'})
            continue
        for target in TARGETS:
            if target==source_style:
                continue
            pool=target_pools[target]; g=pool['g']
            feasible=pool['style_gate'].copy()
            for f,threshold in KNOWN_LIMITS.items():
                feasible &= g[:,fidx[f]] <= max(source[fidx[f]],threshold)+1e-12
            feasible &= g[:,fidx[RELATIVE_ONLY]] <= source[fidx[RELATIVE_ONLY]]+1e-12
            ids=np.where(feasible)[0]
            if len(ids)<MIN_POOL:
                comparison.append({'source_beer':int(chem.index[si]),'source_style':source_style,
                                   'target_style':target,'source_axes_count':len(axes),
                                   'n_feasible':len(ids),'status':'insufficient_feasible_pool'})
                continue
            err=style_error[target][axes]
            source_dist=np.mean(np.abs(pool['select_y'][ids][:,axes]-y0[axes])/err,axis=1)
            source_dist_pooled=np.mean(np.abs(pool['select_y'][ids][:,axes]-y0[axes])/
                                       np.maximum(pooled_mae[axes],.1),axis=1)
            source_dist_local=np.mean(np.abs(pool['select_y'][ids][:,axes]-y0[axes])/
                                      style_local_error[target][axes],axis=1)
            audit_dist=np.mean(np.abs(pool['audit_y'][ids][:,axes]-y0[axes])/err,axis=1)
            typicality=pool['typicality'][ids]
            diversity_threshold=pool['diversity_threshold']
            chosen=choose_guided(typicality,source_dist,pool['z'][ids],diversity_threshold)
            target_only=choose_typicality_only(typicality,pool['z'][ids],diversity_threshold)
            pooled_chosen=choose_guided(typicality,source_dist_pooled,pool['z'][ids],diversity_threshold)
            local_chosen=choose_guided(typicality,source_dist_local,pool['z'][ids],diversity_threshold)
            frontier=pareto_mask(typicality,source_dist)
            rank_t=pd.Series(typicality).rank(pct=True).to_numpy()
            rank_s=pd.Series(source_dist).rank(pct=True).to_numpy()
            # Feasible-pool tertiles; fallback is explicit if an exact bin is exhausted.
            bins=pd.qcut(pd.Series(typicality).rank(method='first'),q=MATCH_BINS,labels=False).to_numpy()
            controls=[]; expected_audit=[]; expected_typicality=[]; exact_matches=[]
            for ci in chosen:
                difference=np.abs(typicality-typicality[ci])
                choices=[int(v) for v in np.argsort(difference) if bins[v]==bins[ci]
                         and int(v) not in chosen and int(v) not in controls]
                exact_matches.append(bool(choices))
                if not choices:
                    choices=[int(v) for v in np.argsort(difference) if int(v) not in chosen+controls]
                expected_audit.append(float(np.mean(audit_dist[choices])))
                expected_typicality.append(float(np.mean(typicality[choices])))
                controls.append(int(rng.choice(choices)))
            for arm,local_indices in [('guided',chosen),('control',controls),('target_only',target_only)]:
                for rank,li in enumerate(local_indices,1):
                    i=int(ids[li]); sid=f'{target}-{i:04d}'
                    row={'source_beer':int(chem.index[si]),'source_style':source_style,'target_style':target,
                         'arm':arm,'rank':rank,'candidate_id':sid,'n_feasible':len(ids),
                         'source_axes':','.join(names[a] for a in axes),
                         'selector_source_distance':float(source_dist[li]),
                         'audit_source_distance':float(audit_dist[li]),
                         'target_nearest_latent_distance':float(typicality[li]),
                         'independent_style_check_pass':bool(pool['style_check'][i]),
                         'predicted_stale_hops_delta':float(pool['audit_y'][i,names.index('stale_hops')]-y0[names.index('stale_hops')])}
                    selected_rows.append(row)
                    profile_rows.append({**{k:row[k] for k in ['source_beer','target_style','arm','rank','candidate_id']},
                                         **dict(zip(features,g[i]))})
                    state_rows.append({**{k:row[k] for k in ['source_beer','target_style','arm','rank','candidate_id']},
                                       **dict(zip(features,pool['status'][i]))})
                    portrait_rows.append({**{k:row[k] for k in ['source_beer','target_style','arm','rank','candidate_id']},
                                          **dict(zip(names,pool['select_y'][i]))})
                    for f,threshold in KNOWN_LIMITS.items():
                        j=fidx[f]
                        defect_rows.append({'source_beer':int(chem.index[si]),'target_style':target,'arm':arm,
                                            'candidate_id':sid,'feature':f,'source_logscale':source[j],
                                            'candidate_logscale':g[i,j], 'allowed_logscale':max(source[j],threshold),
                                            'generated_status':int(pool['status'][i,j]),
                                            'passes_numeric_proxy':bool(g[i,j]<=max(source[j],threshold)+1e-12),
                                            'conclusive_concentration_check':bool(pool['status'][i,j]==0),
                                            'reference_threshold_available':True})
                    j=fidx[RELATIVE_ONLY]
                    defect_rows.append({'source_beer':int(chem.index[si]),'target_style':target,'arm':arm,
                                        'candidate_id':sid,'feature':RELATIVE_ONLY,'source_logscale':source[j],
                                        'candidate_logscale':g[i,j],'allowed_logscale':source[j],
                                        'generated_status':int(pool['status'][i,j]),
                                        'passes_numeric_proxy':bool(g[i,j]<=source[j]+1e-12),
                                        'conclusive_concentration_check':bool(pool['status'][i,j]==0),
                                        'reference_threshold_available':False})
            comparison.append({'source_beer':int(chem.index[si]),'source_style':source_style,'target_style':target,
                               'n_feasible':len(ids),'status':'compared',
                               'guided_audit_distance':float(np.mean(audit_dist[chosen])),
                               'control_audit_distance':float(np.mean(audit_dist[controls])),
                               'control_audit_distance_expected':float(np.mean(expected_audit)),
                               'target_only_audit_distance':float(np.mean(audit_dist[target_only])),
                               'guided_typicality_distance':float(np.mean(typicality[chosen])),
                               'control_typicality_distance':float(np.mean(typicality[controls])),
                               'control_typicality_distance_expected':float(np.mean(expected_typicality)),
                               'target_only_typicality_distance':float(np.mean(typicality[target_only]))})
    pd.DataFrame(selected_rows).to_csv(OUT/'selected_candidates.csv',index=False)
    pd.DataFrame(generation_checks).to_csv(OUT/'generation_holdout_diagnostics.csv',index=False)
    pd.DataFrame(profile_rows).to_csv(OUT/'selected_profiles_logscale.csv',index=False)
    pd.DataFrame(state_rows).to_csv(OUT/'selected_detection_status.csv',index=False)
    pd.DataFrame(portrait_rows).to_csv(OUT/'selected_sensory_predictions.csv',index=False)
    pd.DataFrame(defect_rows).to_csv(OUT/'selected_defect_audit.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'source_guidance_comparison.csv',index=False)
    cmp=pd.DataFrame(comparison); valid=cmp[cmp.status=='compared']
    per_source=(valid.assign(delta=valid.target_only_audit_distance-valid.guided_audit_distance)
                .groupby('source_beer').delta.mean().to_numpy())
    boot_rng=np.random.default_rng(20261028)
    bootstrap=np.mean(boot_rng.choice(per_source,size=(5000,len(per_source)),replace=True),axis=1)
    summary={'train_n':len(train),'test_n':len(test),'features':len(features),'targets':TARGETS,
             'pool_size_per_target':M,'selected_per_arm':K,'n_source_target_pairs':len(cmp),
             'n_compared_pairs':len(valid),'n_insufficient_pairs':int((cmp.status!='compared').sum()),
             'sensory_heldout_mae':heldout,
             'mean_guided_audit_source_distance':float(valid.guided_audit_distance.mean()) if len(valid) else None,
             'mean_control_audit_source_distance':float(valid.control_audit_distance_expected.mean()) if len(valid) else None,
             'mean_target_only_audit_source_distance':float(valid.target_only_audit_distance.mean()) if len(valid) else None,
             'guided_better_than_target_only_pairs':int((valid.guided_audit_distance<valid.target_only_audit_distance).sum()),
             'selected_independent_style_check_rate':float(pd.DataFrame(selected_rows).query("arm == 'guided'").independent_style_check_pass.mean()) if selected_rows else None,
             'target_only_minus_guided_source_cluster_mean':float(per_source.mean()),
             'target_only_minus_guided_source_cluster_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
             'note':'Random forest target-style recognition is a candidate-selection gate. LDA is a separate model audit, not independent sensory or brewing validation. Evaluation is retrospective because earlier Q3 work inspected all 250 beers. E-2-nonenal uses a relative cap; stale_hops remains sensory-predicted.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
