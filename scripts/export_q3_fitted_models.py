"""Export the fitted components used by Q3 source-guided design v2."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from q3_copula import ConditionalCopula  # noqa: E402


def main():
    out = ROOT / "reports/modeling/q3_source_guided_v2/fitted_models"
    out.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(ROOT / "data/interim/chem_feature_sets_v1.csv")
    features = reg.loc[reg.q3_primary_generation, "feature"].tolist()
    chem = pd.read_csv(ROOT / "data/interim/chem_log10_matrix_v1.csv").set_index("beer")
    status = pd.read_csv(ROOT / "data/interim/chem_observation_status_v1.csv").set_index("beer").loc[chem.index, features].to_numpy(int)
    styles = pd.read_csv(ROOT / "data/interim/beer_style_v1.csv").set_index("beer").loc[chem.index, "style"].to_numpy()
    sensory = pd.read_csv(ROOT / "data/interim/sensory_targets_v1.csv").set_index("beer").loc[chem.index]
    plan = pd.read_csv(ROOT / "data/splits/seed_0/split_manifest.csv").set_index("beer").loc[chem.index]
    train = np.where(plan.split.to_numpy() == "train")[0]
    x, y = chem[features].to_numpy(float), sensory.to_numpy(float)
    scaler = StandardScaler().fit(x[train])
    xt = scaler.transform(x[train])
    generator = ConditionalCopula(prior=8, seed=20261028).fit(x[train], status[train], styles[train])
    selector = Ridge(alpha=100).fit(xt, y[train])
    auditor = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, max_features=.6,
                                  random_state=20261028, n_jobs=-1).fit(xt, y[train])
    gate = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt",
                                  random_state=20261028, n_jobs=-1).fit(xt, styles[train])
    pca = PCA(n_components=20, random_state=0).fit(xt)
    checker = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(pca.transform(xt), styles[train])
    objects = {"generator": generator, "scaler": scaler, "sensory_selector": selector,
               "sensory_auditor": auditor, "style_gate": gate,
               "pca": pca, "style_checker": checker,
               "features": features, "sensory_targets": list(sensory.columns),
               "training_beers": chem.index[train].tolist()}
    joblib.dump(objects, out / "q3_source_guided_v2.joblib", compress=3)
    print("exported Q3 fitted components", len(features), len(train))


if __name__ == "__main__":
    main()
