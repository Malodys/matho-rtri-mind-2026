#!/usr/bin/env python3
"""Q3 style-conditional two-part Gaussian copula benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/modeling/q3_copula_v1'
STYLES = ['Tripel', 'Kriek', 'Lambic']


def nearest(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :])**2).sum(axis=2)).min(axis=1)


class ConditionalCopula:
    def __init__(self, prior=8, seed=0):
        self.prior, self.rng = prior, np.random.default_rng(seed)

    def fit(self, x, status, styles):
        self.x, self.status, self.styles = x, status, np.asarray(styles)
        n, p = x.shape
        # Rank scores retain observed association while Ledoit-Wolf regularizes
        # the covariance in p≈n dimensions. Jitter resolves common LOD ties.
        latent = np.empty_like(x)
        for j in range(p):
            v = x[:, j] + self.rng.normal(0, 1e-8, n)
            latent[:, j] = norm.ppf((rankdata(v) - .5) / n)
        covariance = LedoitWolf().fit(latent).covariance_
        sd = np.sqrt(np.diag(covariance))
        self.corr = covariance / sd[:, None] / sd[None, :]
        self.corr += np.eye(p)*1e-6
        self.floors = np.array([np.median(x[status[:,j] != 0,j]) if np.any(status[:,j] != 0)
                                else np.min(x[:,j]) for j in range(p)])
        return self

    def sample(self, style, n):
        z = self.rng.multivariate_normal(np.zeros(self.x.shape[1]), self.corr, size=n, check_valid='ignore')
        u = norm.cdf(z)
        pool = self.x[self.styles == style]
        state = self.status[self.styles == style]
        m, p = pool.shape
        result = np.empty((n,p))
        generated_status = np.zeros((n,p), dtype=int)
        for j in range(p):
            global_observed = self.x[self.status[:,j] == 0,j]
            local_observed = pool[state[:,j] == 0,j]
            # Beta-binomial pooling for the detection probability.
            p_global = len(global_observed)/len(self.x)
            p_local = (len(local_observed)+self.prior*p_global)/(m+self.prior)
            censored = u[:,j] < 1-p_local
            generated_status[censored,j] = 1
            result[censored,j] = self.floors[j]
            if np.any(~censored):
                q = np.clip((u[~censored,j]-(1-p_local))/max(p_local,1e-6),0,1)
                # Style-specific quantile estimate shrunk toward pooled data.
                if len(local_observed):
                    w = len(local_observed)/(len(local_observed)+self.prior)
                    local_q = np.quantile(local_observed,q)
                    global_q = np.quantile(global_observed,q)
                    result[~censored,j] = w*local_q+(1-w)*global_q
                else:
                    result[~censored,j] = np.quantile(global_observed,q) if len(global_observed) else self.floors[j]
        return result, generated_status


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reg = pd.read_csv(ROOT/'data/interim/chem_feature_sets_v1.csv')
    features = reg.loc[reg.q3_primary_generation,'feature'].tolist()
    chem = pd.read_csv(ROOT/'data/interim/chem_log10_matrix_v1.csv').set_index('beer')
    status = pd.read_csv(ROOT/'data/interim/chem_observation_status_v1.csv').set_index('beer').loc[chem.index,features].to_numpy(int)
    styles = pd.read_csv(ROOT/'data/interim/beer_style_v1.csv').set_index('beer').loc[chem.index,'style'].to_numpy()
    sensory = pd.read_csv(ROOT/'data/interim/sensory_targets_v1.csv').set_index('beer').loc[chem.index]
    x = chem[features].to_numpy(float)
    y = sensory.to_numpy(float)
    checks=[]
    sensory_errors=[]
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=20261027)
    for fold,(train,test) in enumerate(cv.split(x,styles),1):
        scale=StandardScaler().fit(x[train]); xt=scale.transform(x[train]); xv=scale.transform(x[test])
        pca=PCA(n_components=20,random_state=0).fit(xt)
        lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(pca.transform(xt),styles[train])
        forest=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features='sqrt',random_state=fold,n_jobs=-1).fit(xt,styles[train])
        model=ConditionalCopula(prior=8,seed=fold).fit(x[train],status[train],styles[train])
        sensory_model=Ridge(alpha=100).fit(xt,y[train])
        sensory_errors.append(np.abs(sensory_model.predict(xv)-y[test]))
        for style in STYLES:
            held=x[test][styles[test]==style]
            held_s=status[test][styles[test]==style]
            g,gs=model.sample(style,100)
            gz=scale.transform(g)
            # Weighted detection-rate error avoids treating shared floors as exact concentrations.
            active=np.any(status[train] != 0,axis=0)
            real_detection=(held_s[:,active]==0).mean(axis=0)
            gen_detection=(gs[:,active]==0).mean(axis=0)
            checks.append({'fold':fold,'style':style,'heldout_n':len(held),
                           'generated_lda_hit':float(np.mean(lda.predict(pca.transform(gz))==style)),
                           'heldout_lda_hit':float(np.mean(lda.predict(pca.transform(scale.transform(held)))==style)),
                           'generated_forest_hit':float(np.mean(forest.predict(gz)==style)),
                           'heldout_forest_hit':float(np.mean(forest.predict(scale.transform(held))==style)),
                           'generated_nearest_median':float(np.median(nearest(gz,xt))),
                           'heldout_nearest_median':float(np.median(nearest(scale.transform(held),xt))),
                           'detection_rate_mae':float(np.mean(np.abs(gen_detection-real_detection))),
                           'generated_internal_nearest_median':float(np.median(nearest(gz[:50],gz[50:])))})
    pd.DataFrame(checks).to_csv(OUT/'cv_diagnostics.csv',index=False)
    mae=np.vstack(sensory_errors).mean(axis=0)
    pd.DataFrame({'sensory_feature':sensory.columns,'cv_mae':mae}).to_csv(OUT/'sensory_cv_error.csv',index=False)
    model=ConditionalCopula(prior=8,seed=20261028).fit(x,status,styles)
    full_scale=StandardScaler().fit(x)
    sensory_model=Ridge(alpha=100).fit(full_scale.transform(x),y)
    rows=[]; states=[]
    portraits=[]
    for style in STYLES:
        g,gs=model.sample(style,20)
        for i,(v,st) in enumerate(zip(g,gs)):
            sid=f'{style}-{i+1:02d}'
            rows.append({'sample_id':sid,'requested_style':style,**dict(zip(features,v))})
            states.append({'sample_id':sid,'requested_style':style,**dict(zip(features,st))})
            predicted=sensory_model.predict(full_scale.transform(v[None,:]))[0]
            portraits.append({'sample_id':sid,'requested_style':style,**dict(zip(sensory.columns,predicted))})
    pd.DataFrame(rows).to_csv(OUT/'generated_profiles_logscale.csv',index=False)
    pd.DataFrame(states).to_csv(OUT/'generated_detection_status.csv',index=False)
    pd.DataFrame(portraits).to_csv(OUT/'generated_sensory_predictions.csv',index=False)
    summary={'n_beers':len(x),'features':len(features),'styles_evaluated':STYLES,
             'model':'pooled shrinkage Gaussian copula, style-shrunk detection probability and observed quantiles',
             'sensory_cv_mae_mean':float(mae.mean()),
             'caution':'Generated status 1 is a censored/undetected state. Its numeric floor is a placeholder, not a chemical concentration.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(pd.read_csv(OUT/'cv_diagnostics.csv').groupby('style').mean(numeric_only=True).round(3).to_string())


if __name__=='__main__':
    main()
